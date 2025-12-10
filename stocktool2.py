#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
stocktool2.py

量化回测系统 V2.0
- 通达信公式批量回测
- 多策略 & 多图形展示
- 扩展功能：多周期信号、指标评分、止盈止损建议、仓位计划、情景压力测试、收益热力图、复盘摘要、自然语言转公式
"""

from __future__ import annotations

import json
import os
import threading
from queue import Queue, Empty
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import dates as mdates, font_manager
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText

from formula_engine import TdxFormulaEngine
from backtesting import (
    BacktestResult,
    backtest_fixed_period,
    backtest_take_profit_stop_loss,
    backtest_dca_simple,
    backtest_grid_simple,
)
from plotting_utils import (
    plot_strategy_stats,
    plot_trade_chart,
    plot_multi_timeframe,
    plot_heatmap,
)
from analytics import (
    indicator_scoring,
    atr_based_stop,
    position_rebalance_plan,
    run_stress_test,
    holding_return_heatmap,
    generate_multi_timeframe_signals,
    generate_daily_brief,
    simple_rule_based_formula,
)


def _setup_matplotlib_font():
    candidate_fonts = [
        "Microsoft YaHei",
        "PingFang SC",
        "SimHei",
        "Source Han Sans CN",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in candidate_fonts if f in available), None)
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen]
    plt.rcParams["axes.unicode_minus"] = False


_setup_matplotlib_font()


class QuantGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("量化回测系统 V2.0")
        root.geometry("980x900")

        menubar = tk.Menu(root)
        menu_option = tk.Menu(menubar, tearoff=0)
        menu_option.add_command(label="退出", command=root.destroy)
        menubar.add_cascade(label="选项", menu=menu_option)

        menu_action = tk.Menu(menubar, tearoff=0)
        menu_action.add_command(label="运行回测", command=self.run_backtest_thread)
        menubar.add_cascade(label="操作", menu=menu_action)

        menu_help = tk.Menu(menubar, tearoff=0)
        menu_help.add_command(label="关于", command=self.show_about)
        menubar.add_cascade(label="帮助", menu=menu_help)
        root.config(menu=menubar)

        top_frame = ttk.Frame(root, padding=5)
        top_frame.pack(fill=tk.X)
        ttk.Label(top_frame, text="量化回测系统 V2.0", font=("Microsoft YaHei", 16, "bold")).pack()

        formula_mgmt = ttk.LabelFrame(root, text="通达信策略指标公式管理", padding=5)
        formula_mgmt.pack(fill=tk.X, padx=5, pady=5)

        btn_row = ttk.Frame(formula_mgmt)
        btn_row.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row, text="打开公式", command=self.open_formula_file).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="粘贴公式", command=self.paste_formula).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="转换公式", command=self.check_formula).pack(side=tk.LEFT, padx=3)

        self.csv_path_var = tk.StringVar()
        path_row = ttk.Frame(formula_mgmt)
        path_row.pack(fill=tk.X, pady=2)
        ttk.Label(path_row, text="行情CSV:").pack(side=tk.LEFT)
        ttk.Entry(path_row, textvariable=self.csv_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(path_row, text="选择文件", command=self.choose_csv).pack(side=tk.LEFT)

        self.formula_text = ScrolledText(formula_mgmt, height=8, wrap=tk.NONE)
        self.formula_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.formula_text.bind("<<Modified>>", self._on_formula_text_modified)
        self.formula_text.edit_modified(False)

        self.initial_capital_var = tk.StringVar(value="100000")
        self.fee_var = tk.StringVar(value="0.0005")
        self.multi_timeframe_var = tk.StringVar(value="D,W,M")
        self.strategy_text_var = tk.StringVar()

        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_config.json")
        self._config_loaded = False
        self._formula_save_job: Optional[str] = None

        common_frame = ttk.Frame(root, padding=5)
        common_frame.pack(fill=tk.X)
        ttk.Label(common_frame, text="初始资金:").pack(side=tk.LEFT)
        ttk.Entry(common_frame, width=10, textvariable=self.initial_capital_var).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Label(common_frame, text="单边手续费:").pack(side=tk.LEFT)
        ttk.Entry(common_frame, width=8, textvariable=self.fee_var).pack(side=tk.LEFT)
        ttk.Label(common_frame, text=" (如 0.0005)").pack(side=tk.LEFT)

        self.fixed_enabled = tk.BooleanVar(value=True)
        self.fixed_periods_var = tk.StringVar(value="5,10,15,20")
        fixed_frame = ttk.LabelFrame(root, text="固定周期", padding=5)
        fixed_frame.pack(fill=tk.X, padx=5, pady=3)
        ttk.Checkbutton(fixed_frame, text="", variable=self.fixed_enabled).pack(side=tk.LEFT)
        ttk.Label(fixed_frame, text="持有周期数:").pack(side=tk.LEFT)
        ttk.Entry(fixed_frame, width=20, textvariable=self.fixed_periods_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(fixed_frame, text="(逗号分隔，如 5,10,20)").pack(side=tk.LEFT)

        self.tpsl_enabled = tk.BooleanVar(value=True)
        self.tp_var = tk.StringVar(value="0.1")
        self.sl_var = tk.StringVar(value="0.05")
        tpsl_frame = ttk.LabelFrame(root, text="止盈止损", padding=5)
        tpsl_frame.pack(fill=tk.X, padx=5, pady=3)
        ttk.Checkbutton(tpsl_frame, variable=self.tpsl_enabled).pack(side=tk.LEFT)
        ttk.Label(tpsl_frame, text="止盈比例%:").pack(side=tk.LEFT)
        ttk.Entry(tpsl_frame, width=10, textvariable=self.tp_var).pack(side=tk.LEFT, padx=3)
        ttk.Label(tpsl_frame, text="止损比例%:").pack(side=tk.LEFT)
        ttk.Entry(tpsl_frame, width=10, textvariable=self.sl_var).pack(side=tk.LEFT, padx=3)

        self.dca_enabled = tk.BooleanVar(value=False)
        self.dca_size_var = tk.StringVar(value="0.05")
        self.dca_target_var = tk.StringVar(value="0.2")
        dca_frame = ttk.LabelFrame(root, text="定投策略", padding=5)
        dca_frame.pack(fill=tk.X, padx=5, pady=3)
        ttk.Checkbutton(dca_frame, variable=self.dca_enabled).pack(side=tk.LEFT)
        ttk.Label(dca_frame, text="定投尺寸%:").pack(side=tk.LEFT)
        ttk.Entry(dca_frame, width=10, textvariable=self.dca_size_var).pack(side=tk.LEFT, padx=3)
        ttk.Label(dca_frame, text="目标收益率%:").pack(side=tk.LEFT)
        ttk.Entry(dca_frame, width=10, textvariable=self.dca_target_var).pack(side=tk.LEFT, padx=3)

        self.grid_enabled = tk.BooleanVar(value=False)
        self.grid_size_var = tk.StringVar(value="0.05")
        self.grid_cash_var = tk.StringVar(value="1000")
        self.grid_limit_var = tk.StringVar(value="None")
        self.grid_accumulate_var = tk.StringVar(value="True")
        grid_frame = ttk.LabelFrame(root, text="网格策略", padding=5)
        grid_frame.pack(fill=tk.X, padx=5, pady=3)
        ttk.Checkbutton(grid_frame, variable=self.grid_enabled).pack(side=tk.LEFT)
        ttk.Label(grid_frame, text="网格尺寸%:").pack(side=tk.LEFT)
        ttk.Entry(grid_frame, width=10, textvariable=self.grid_size_var).pack(side=tk.LEFT, padx=3)
        ttk.Label(grid_frame, text="单网资金:").pack(side=tk.LEFT)
        ttk.Entry(grid_frame, width=10, textvariable=self.grid_cash_var).pack(side=tk.LEFT, padx=3)
        ttk.Label(grid_frame, text="网格数限制:").pack(side=tk.LEFT)
        ttk.Entry(grid_frame, width=10, textvariable=self.grid_limit_var).pack(side=tk.LEFT, padx=3)
        ttk.Label(grid_frame, text="累积份额(True/False):").pack(side=tk.LEFT)
        ttk.Entry(grid_frame, width=10, textvariable=self.grid_accumulate_var).pack(side=tk.LEFT, padx=3)

        analytics_frame = ttk.LabelFrame(root, text="扩展分析功能", padding=5)
        analytics_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(analytics_frame, text="多周期(如 D,W,M):").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(analytics_frame, width=15, textvariable=self.multi_timeframe_var).grid(row=0, column=1, padx=3, pady=2)
        ttk.Button(analytics_frame, text="多周期信号", command=self.show_multi_timeframe_signals).grid(row=0, column=2, padx=3)
        ttk.Button(analytics_frame, text="指标评分曲线", command=self.show_indicator_scores).grid(row=0, column=3, padx=3)
        ttk.Button(analytics_frame, text="止盈/止损建议", command=self.show_stop_suggestion).grid(row=0, column=4, padx=3)
        ttk.Button(analytics_frame, text="仓位/补仓计划", command=self.show_position_plan).grid(row=0, column=5, padx=3)

        ttk.Button(analytics_frame, text="情景压力测试", command=self.show_stress_test).grid(row=1, column=0, padx=3, pady=2)
        ttk.Button(analytics_frame, text="收益热力图", command=self.show_heatmap).grid(row=1, column=1, padx=3)
        ttk.Button(analytics_frame, text="复盘摘要", command=self.show_daily_brief).grid(row=1, column=2, padx=3)
        ttk.Entry(analytics_frame, width=30, textvariable=self.strategy_text_var).grid(row=1, column=3, padx=3)
        ttk.Button(analytics_frame, text="自然语言转公式", command=self.convert_nlp_strategy).grid(row=1, column=4, padx=3)

        status_frame = ttk.Frame(root, padding=5)
        status_frame.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="状态：未运行")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(status_frame, mode="indeterminate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        btn_frame = ttk.Frame(root, padding=5)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="运行回测", command=self.run_backtest_thread).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="停止回测", command=self.not_implemented).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="绘制统计图", command=self.plot_stats).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="绘制交易图", command=self.plot_trade_chart_window).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="打开结果文件夹", command=self.open_results_folder).pack(side=tk.LEFT, padx=3)

        log_frame = ttk.LabelFrame(root, text="运行日志", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text = ScrolledText(log_frame, height=12, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.csv_path_var.trace_add("write", lambda *args: self._on_csv_path_change())
        self._load_user_config()

        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.worker_queue: Optional[Queue] = None
        self.analysis_thread: Optional[threading.Thread] = None
        self.analysis_queue: Optional[Queue] = None
        self.analysis_busy = False
        self._analysis_callback: Optional[Callable[[object], None]] = None
        self.last_df: Optional[pd.DataFrame] = None
        self.last_buy_signals: Optional[pd.Series] = None
        self.last_sell_signals: Optional[pd.Series] = None
        self.last_results: List[Tuple[str, BacktestResult]] = []
        self.last_scores: Optional[pd.DataFrame] = None
        self.last_formula: Optional[str] = None

    # ---------- UI 辅助 ----------

    def log(self, text: str):
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def show_about(self):
        messagebox.showinfo("关于", "量化回测系统 V2.0\n支持多策略与扩展分析功能。")

    def not_implemented(self):
        messagebox.showinfo("提示", "该功能暂未实现。")

    def choose_csv(self):
        path = filedialog.askopenfilename(
            title="选择行情 CSV 文件",
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
        )
        if path:
            self.csv_path_var.set(path)

    def open_formula_file(self):
        path = filedialog.askopenfilename(
            title="打开通达信公式文本",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.formula_text.delete("1.0", tk.END)
            self.formula_text.insert(tk.END, content)
        except Exception as e:
            messagebox.showerror("错误", f"读取公式失败：\n{e}")

    def paste_formula(self):
        try:
            text = self.root.clipboard_get()
            if text:
                self.formula_text.delete("1.0", tk.END)
                self.formula_text.insert(tk.END, text)
        except Exception:
            messagebox.showwarning("提示", "剪贴板为空或无法获取内容。")

    def check_formula(self):
        content = self.formula_text.get("1.0", tk.END)
        if "B_COND" in content:
            messagebox.showinfo("检查结果", "公式中包含 B_COND，具体错误将运行时提示。")
        else:
            messagebox.showwarning("检查结果", "未找到 B_COND 定义，请加入：B_COND := 你的买入条件;")

    def open_results_folder(self):
        folder = os.path.abspath("results")
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        os.system(f'open "{folder}"' if os.name == "posix" else f'start "" "{folder}"')

    def _load_user_config(self):
        if not os.path.exists(self.config_path):
            self._config_loaded = True
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.log(f"加载配置失败：{e}")
            self._config_loaded = True
            return

        csv_path = data.get("csv_path", "")
        formula = data.get("formula", "")
        if csv_path:
            self.csv_path_var.set(csv_path)
        if formula:
            self.formula_text.delete("1.0", tk.END)
            self.formula_text.insert(tk.END, formula)
            self.formula_text.edit_modified(False)
        self._config_loaded = True

    def _save_user_config(self):
        if not self._config_loaded:
            return
        data = {
            "csv_path": self.csv_path_var.get().strip(),
            "formula": self.formula_text.get("1.0", tk.END).strip(),
        }
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log(f"保存配置失败：{e}")
        finally:
            if self._formula_save_job:
                self.root.after_cancel(self._formula_save_job)
                self._formula_save_job = None

    def _on_csv_path_change(self):
        if not self._config_loaded:
            return
        self._save_user_config()

    def _on_formula_text_modified(self, _event=None):
        if not self._config_loaded:
            self.formula_text.edit_modified(False)
            return
        if self.formula_text.edit_modified():
            self.formula_text.edit_modified(False)
            if self._formula_save_job:
                self.root.after_cancel(self._formula_save_job)
            self._formula_save_job = self.root.after(800, self._save_user_config)

    # ---------- 扩展功能 ----------

    def _ensure_data_ready(self) -> bool:
        if self.last_df is None:
            messagebox.showinfo("提示", "请先运行一次回测。")
            return False
        return True

    def plot_stats(self):
        if not self.last_results:
            messagebox.showinfo("提示", "暂无结果，请先运行回测。")
            return
        plot_strategy_stats(self.last_results)

    def plot_trade_chart_window(self):
        if not self.last_results or self.last_df is None:
            messagebox.showinfo("提示", "请先运行回测。")
            return
        idx = 0
        if len(self.last_results) > 1:
            options = "\n".join([f"{i + 1}. {name}" for i, (name, _) in enumerate(self.last_results)])
            choice = simpledialog.askinteger("选择策略", f"请选择策略编号：\n{options}", minvalue=1, maxvalue=len(self.last_results))
            if choice is None:
                return
            idx = choice - 1
        name, result = self.last_results[idx]
        plot_trade_chart(self.last_df, result, name, self.last_buy_signals, self.last_sell_signals)

    def show_multi_timeframe_signals(self):
        if not self._ensure_data_ready() or not self.last_formula:
            return
        freqs = [f.strip() for f in self.multi_timeframe_var.get().split(",") if f.strip()]
        if not freqs:
            freqs = ["D", "W", "M"]
        try:
            signals, frames = generate_multi_timeframe_signals(self.last_df, self.last_formula, freqs)
        except Exception as e:
            messagebox.showerror("错误", f"多周期信号生成失败：\n{e}")
            return
        plot_multi_timeframe(signals, frames)

    def show_indicator_scores(self):
        if not self._ensure_data_ready():
            return
        scores = indicator_scoring(self.last_df)
        self.last_scores = scores
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(scores.index, scores["total_score"], label="综合评分")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("指标综合评分")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show()

    def show_stop_suggestion(self):
        if not self._ensure_data_ready():
            return
        info = atr_based_stop(self.last_df)
        messagebox.showinfo(
            "止盈止损建议",
            f"最新价: {info['last_price']:.2f}\nATR: {info['atr']:.2f}\n"\
            f"建议止损: {info['suggest_stop_loss']:.2f}\n"\
            f"建议止盈: {info['suggest_take_profit']:.2f}\n"\
            f"跟踪止损: {info['trailing_stop']:.2f}",
        )

    def show_position_plan(self):
        if not self._ensure_data_ready():
            return
        try:
            capital = float(self.initial_capital_var.get())
        except Exception:
            messagebox.showerror("错误", "初始资金格式错误。")
            return
        plan = position_rebalance_plan(self.last_df, capital)
        if plan.empty:
            messagebox.showinfo("提示", "资金不足以生成补仓计划。")
            return
        self.show_table_window("仓位/补仓计划", plan)

    def show_stress_test(self):
        if not self._ensure_data_ready():
            return
        def task():
            return run_stress_test(self.last_df)

        def on_success(df: pd.DataFrame):
            if df.empty:
                messagebox.showinfo("提示", "样本区间不足以覆盖预设情景。")
                return
            self.show_table_window("情景压力测试", df)

        self._start_async_analysis("情景压力测试", task, on_success)

    def show_heatmap(self):
        if not self._ensure_data_ready():
            return
        def task():
            return holding_return_heatmap(self.last_df)

        def on_success(heatmap_df: pd.DataFrame):
            if heatmap_df.empty:
                messagebox.showinfo("提示", "数据不足以生成热力图。")
                return
            plot_heatmap(heatmap_df.tail(200))

        self._start_async_analysis("收益热力图", task, on_success)

    def show_daily_brief(self):
        if not self._ensure_data_ready():
            return
        result = self.last_results[0][1] if self.last_results else None
        text = generate_daily_brief(self.last_df, self.last_scores, result, self.last_buy_signals, self.last_sell_signals)
        messagebox.showinfo("复盘摘要", text)

    def convert_nlp_strategy(self):
        text = self.strategy_text_var.get().strip()
        if not text:
            messagebox.showinfo("提示", "请输入策略描述。")
            return
        try:
            script = simple_rule_based_formula(text)
        except Exception as e:
            messagebox.showerror("错误", f"转换失败：{e}")
            return
        self.formula_text.delete("1.0", tk.END)
        self.formula_text.insert(tk.END, script)
        messagebox.showinfo("完成", "已生成基础公式，可根据需要调整。")

    def show_table_window(self, title: str, df: pd.DataFrame):
        win = tk.Toplevel(self.root)
        win.title(title)
        columns = list(df.columns)
        tree = ttk.Treeview(win, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor=tk.CENTER)
        vsb = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        for _, row in df.iterrows():
            values = [row[col] for col in columns]
            tree.insert("", tk.END, values=values)

    def _start_async_analysis(self, task_name: str, func: Callable[[], object], on_success: Callable[[object], None]):
        if self.analysis_busy:
            messagebox.showwarning("提示", "已有分析任务在执行，请稍候。")
            return
        self.analysis_busy = True
        self.analysis_queue = Queue()
        self._analysis_callback = on_success
        self.status_var.set(f"状态：{task_name}处理中……")

        def worker():
            try:
                result = func()
                self.analysis_queue.put(("success", result))
            except Exception as e:
                self.analysis_queue.put(("error", str(e)))
            finally:
                self.analysis_queue.put(("done", None))

        self.analysis_thread = threading.Thread(target=worker, daemon=True)
        self.analysis_thread.start()
        self.root.after(100, self._poll_analysis_queue)

    def _poll_analysis_queue(self):
        if not self.analysis_queue:
            return
        try:
            while True:
                event, payload = self.analysis_queue.get_nowait()
                if event == "success" and self._analysis_callback:
                    callback = self._analysis_callback
                    self._analysis_callback = None
                    callback(payload)
                elif event == "error":
                    messagebox.showerror("错误", payload)
                elif event == "done":
                    self._finish_analysis_task()
                    return
        except Empty:
            self.root.after(100, self._poll_analysis_queue)

    def _finish_analysis_task(self):
        self.analysis_busy = False
        self.analysis_queue = None
        self.analysis_thread = None
        self._analysis_callback = None
        if not self.is_running:
            self.status_var.set("状态：空闲")

    # ---------- 回测主流程 ----------

    def run_backtest_thread(self):
        if self.is_running:
            messagebox.showwarning("提示", "回测正在运行中。")
            return
        csv_path = self.csv_path_var.get().strip()
        if not csv_path:
            messagebox.showwarning("提示", "请先选择行情 CSV 文件。")
            return

        formula = self.formula_text.get("1.0", tk.END).strip()
        if not formula:
            messagebox.showwarning("提示", "请粘贴通达信公式（至少定义 B_COND）。")
            return

        try:
            initial_capital = float(self.initial_capital_var.get())
            fee_rate = float(self.fee_var.get())
        except Exception:
            messagebox.showerror("错误", "初始资金或手续费格式错误。")
            return

        params: Dict = {
            "csv_path": csv_path,
            "formula": formula,
            "initial_capital": initial_capital,
            "fee_rate": fee_rate,
            "strategies": {},
        }

        if self.fixed_enabled.get():
            try:
                periods = [int(x) for x in self.fixed_periods_var.get().split(",") if x.strip()]
            except Exception:
                messagebox.showerror("错误", "固定周期中的持有天数格式错误。")
                return
            if periods:
                params["strategies"]["fixed"] = periods

        if self.tpsl_enabled.get():
            try:
                tp_raw = float(self.tp_var.get())
                sl_raw = float(self.sl_var.get())
                tp = tp_raw / 100 if tp_raw > 1 else tp_raw
                sl = sl_raw / 100 if sl_raw > 1 else sl_raw
            except Exception:
                messagebox.showerror("错误", "止盈/止损比例格式错误。")
                return
            params["strategies"]["tpsl"] = {"tp": tp, "sl": sl}

        if self.dca_enabled.get():
            try:
                dca_size = float(self.dca_size_var.get())
                target = float(self.dca_target_var.get())
                if dca_size > 1:
                    dca_size /= 100
                if target > 1:
                    target /= 100
            except Exception:
                messagebox.showerror("错误", "定投参数格式错误。")
                return
            params["strategies"]["dca"] = {"size": dca_size, "target": target}

        if self.grid_enabled.get():
            try:
                grid_pct = float(self.grid_size_var.get())
                if grid_pct > 1:
                    grid_pct /= 100
                single_cash = float(self.grid_cash_var.get())
                limit_raw = self.grid_limit_var.get().strip()
                max_grids = None if (not limit_raw or limit_raw.lower() == "none") else int(limit_raw)
                accumulate = self.grid_accumulate_var.get().strip().lower() == "true"
            except Exception:
                messagebox.showerror("错误", "网格参数格式错误。")
                return
            params["strategies"]["grid"] = {
                "grid_pct": grid_pct,
                "single_cash": single_cash,
                "max_grids": max_grids,
                "accumulate": accumulate,
            }

        self.status_var.set("状态：正在运行……")
        self.progress.start(50)
        self.log_text.delete("1.0", tk.END)
        self.last_results = []
        self.last_df = None
        self.last_buy_signals = None
        self.last_sell_signals = None
        self.last_scores = None

        self.is_running = True
        self.worker_queue = Queue()
        self.worker_thread = threading.Thread(target=self._run_backtest_worker, args=(params,), daemon=True)
        self.worker_thread.start()
        self.root.after(100, self._poll_worker_queue)

    def _run_backtest_worker(self, params: Dict):
        queue = self.worker_queue
        if queue is None:
            return

        def log(msg: str):
            queue.put(("log", msg))

        try:
            df = pd.read_csv(params["csv_path"])
        except Exception as e:
            queue.put(("error", f"读取 CSV 失败：\n{e}"))
            queue.put(("done", None))
            return

        required_cols = {"date", "open", "high", "low", "close"}
        if not required_cols.issubset(df.columns):
            queue.put(("error", f"CSV 至少需要列：{', '.join(required_cols)}"))
            queue.put(("done", None))
            return

        df_datetime = df.copy()
        df_datetime["date"] = pd.to_datetime(df_datetime["date"])
        df_datetime.set_index("date", inplace=True)

        log(f"加载行情数据：{params['csv_path']}")
        log(f"数据行数：{len(df)}")
        log("正在解析公式并生成信号……")

        try:
            engine = TdxFormulaEngine(df)
            buy, sell = engine.run(params["formula"])
        except Exception as e:
            queue.put(("error", f"解析/执行通达信公式失败：\n{e}"))
            queue.put(("done", None))
            return

        if not buy.any():
            log("警告：全样本中没有任何买入信号。")

        entries: List[Dict] = []
        strategies = params["strategies"]
        initial_capital = params["initial_capital"]
        fee_rate = params["fee_rate"]

        if "fixed" in strategies:
            for p in strategies["fixed"]:
                log("")
                log(f"=== 固定周期策略：持有 {p} 天 ===")
                res = backtest_fixed_period(df, buy, sell, p, initial_capital, fee_rate)
                entries.append({"name": f"fixed_{p}", "title": f"=== 固定周期策略：持有 {p} 天 ===", "result": res})

        if "tpsl" in strategies:
            tp = strategies["tpsl"]["tp"]
            sl = strategies["tpsl"]["sl"]
            log("")
            log(f"=== 止盈止损策略：TP={tp*100:.1f}%, SL={sl*100:.1f}% ===")
            res = backtest_take_profit_stop_loss(df, buy, sell, tp, sl, initial_capital, fee_rate)
            entries.append({
                "name": f"tpsl_{tp}_{sl}",
                "title": f"=== 止盈止损策略：TP={tp*100:.1f}%, SL={sl*100:.1f}% ===",
                "result": res,
            })

        if "dca" in strategies:
            dca_size = strategies["dca"]["size"]
            target = strategies["dca"]["target"]
            log("")
            log(f"=== 定投策略：尺寸 {dca_size*100:.1f}%, 目标收益 {target*100:.1f}% ===")
            res = backtest_dca_simple(df, buy, target, dca_size, initial_capital, fee_rate)
            entries.append({
                "name": f"dca_{dca_size}_{target}",
                "title": f"=== 定投策略：尺寸 {dca_size*100:.1f}%, 目标收益 {target*100:.1f}% ===",
                "result": res,
            })

        if "grid" in strategies:
            cfg = strategies["grid"]
            log("")
            log(
                f"=== 网格策略：间距 {cfg['grid_pct']*100:.1f}%, 单网资金 {cfg['single_cash']}, "
                f"网格数限制 {cfg['max_grids']}, 累积={cfg['accumulate']} ==="
            )
            res = backtest_grid_simple(
                df,
                cfg["grid_pct"],
                cfg["single_cash"],
                cfg["max_grids"],
                cfg["accumulate"],
                initial_capital,
                fee_rate,
            )
            entries.append({
                "name": f"grid_{cfg['grid_pct']}",
                "title": f"=== 网格策略：间距 {cfg['grid_pct']*100:.1f}%, 单网资金 {cfg['single_cash']}, "
                f"网格数限制 {cfg['max_grids']}, 累积={cfg['accumulate']} ===",
                "result": res,
            })

        if entries:
            os.makedirs("results", exist_ok=True)
            for entry in entries:
                name = entry["name"]
                res = entry["result"]
                res.equity_curve.to_csv(os.path.join("results", f"{name}_equity.csv"))
                trades_df = pd.DataFrame([t.__dict__ for t in res.trades])
                trades_df.to_csv(os.path.join("results", f"{name}_trades.csv"), index=False)
            log("")
            log("结果已保存到 ./results 目录下。")

        queue.put(
            (
                "result",
                {
                    "df": df_datetime,
                    "buy": buy,
                    "sell": sell,
                    "formula": params["formula"],
                    "entries": entries,
                },
            )
        )
        queue.put(("done", None))

    def _poll_worker_queue(self):
        if not self.worker_queue:
            return
        try:
            while True:
                event, payload = self.worker_queue.get_nowait()
                if event == "log":
                    self.log(payload)
                elif event == "error":
                    messagebox.showerror("错误", payload)
                elif event == "result":
                    self._apply_worker_result(payload)
                elif event == "done":
                    self._end_run()
                    self.worker_thread = None
                    self.worker_queue = None
                    return
        except Empty:
            pass
        if self.is_running:
            self.root.after(100, self._poll_worker_queue)

    def _apply_worker_result(self, payload: Dict):
        self.last_df = payload["df"]
        self.last_buy_signals = payload["buy"]
        self.last_sell_signals = payload["sell"]
        self.last_formula = payload["formula"]
        entries: List[Dict] = payload.get("entries", [])
        self.last_results = [(entry["name"], entry["result"]) for entry in entries]
        for entry in entries:
            self.log("")
            self.log(entry["title"])
            self.show_result(entry["result"])

    def _end_run(self):
        self.progress.stop()
        self.status_var.set("状态：空闲")
        self.is_running = False

    def show_result(self, res: BacktestResult):
        self.log(f"总收益率: {res.total_return*100:.2f}%")
        self.log(f"年化收益: {res.annualized_return*100:.2f}%")
        self.log(f"最大回撤: {res.max_drawdown*100:.2f}%")
        self.log(f"交易次数: {len(res.trades)}")
        self.log(f"胜率: {res.win_rate*100:.2f}%")
        self.log(f"平均盈利: {res.avg_win*100:.2f}%")
        self.log(f"平均亏损: {res.avg_loss*100:.2f}%")
        if res.trades:
            self.log("最近 5 笔交易：")
            for t in res.trades[-5:]:
                self.log(
                    f"{t.entry_date.date()} 买入 {t.entry_price:.2f} -> "
                    f"{t.exit_date.date()} 卖出 {t.exit_price:.2f} "
                    f"收益 {t.return_pct*100:.2f}%, 持有 {t.holding_days} 天, {t.note}"
                )


def main():
    root = tk.Tk()
    app = QuantGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
