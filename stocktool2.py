#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Modernized StockTool GUI with modular services and responsive task management."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText

from config_manager import ConfigManager, UserConfig
from task_runner import TaskRunner
from backtest_service import BacktestParams, run_backtests, BacktestEntry, BacktestPayload
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
from plotting_utils import (
    plot_strategy_stats,
    plot_trade_chart,
    plot_multi_timeframe,
    plot_heatmap,
)


# ----------------------------------------------------------------------------
# Matplotlib font setup
# ----------------------------------------------------------------------------

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
    for name in candidate_fonts:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


_setup_matplotlib_font()


# ----------------------------------------------------------------------------
# 应用状态
# ----------------------------------------------------------------------------

@dataclass
class AppState:
    df: Optional[pd.DataFrame] = None
    buy: Optional[pd.Series] = None
    sell: Optional[pd.Series] = None
    results: List[BacktestEntry] = field(default_factory=list)
    scores: Optional[pd.DataFrame] = None
    formula: str = ""


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------


class QuantGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("量化回测系统 V2.0")
        self.root.geometry("1250x900")

        self.state = AppState()
        self.config_mgr = ConfigManager()
        self.user_config = self.config_mgr.load()
        self._formula_save_job: Optional[str] = None

        self._build_menu()
        self.body_frame = ttk.Frame(self.root)
        self.body_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.content_parent = ttk.Frame(self.body_frame)
        self.content_parent.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_top_panel(self.content_parent)
        self._build_strategy_sections(self.content_parent)
        self._build_extensions(self.content_parent)
        self._build_status_bar(self.content_parent)
        self._build_log_area(self.body_frame)

        self.task_runner = TaskRunner(
            root=self.root,
            status_callback=self._set_status,
            log_callback=self.log,
        )

        self._apply_user_config()

    # ------------------------------------------------------------------
    # GUI 构建
    # ------------------------------------------------------------------

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        menu_option = tk.Menu(menubar, tearoff=0)
        menu_option.add_command(label="退出", command=self.root.destroy)
        menubar.add_cascade(label="选项", menu=menu_option)

        menu_action = tk.Menu(menubar, tearoff=0)
        menu_action.add_command(label="运行回测", command=self.run_backtest)
        menubar.add_cascade(label="操作", menu=menu_action)

        menu_help = tk.Menu(menubar, tearoff=0)
        menu_help.add_command(label="关于", command=self.show_about)
        menubar.add_cascade(label="帮助", menu=menu_help)
        self.root.config(menu=menubar)

    def _build_top_panel(self, parent):
        frame = ttk.Frame(parent, padding=5)
        frame.pack(fill=tk.X)
        ttk.Label(frame, text="量化回测系统 V2.0", font=("Microsoft YaHei", 16, "bold")).pack()

        formula_frame = ttk.LabelFrame(parent, text="通达信策略指标公式管理", padding=5)
        formula_frame.pack(fill=tk.X, padx=5, pady=5)

        btn_row = ttk.Frame(formula_frame)
        btn_row.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row, text="打开公式", command=self.open_formula_file).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="粘贴公式", command=self.paste_formula).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="转换公式", command=self.check_formula).pack(side=tk.LEFT, padx=3)

        path_row = ttk.Frame(formula_frame)
        path_row.pack(fill=tk.X, pady=2)
        ttk.Label(path_row, text="行情CSV:").pack(side=tk.LEFT)
        self.csv_path_var = tk.StringVar()
        ttk.Entry(path_row, textvariable=self.csv_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(path_row, text="选择文件", command=self.choose_csv).pack(side=tk.LEFT)
        self.csv_path_var.trace_add("write", lambda *_: self._schedule_config_save())

        self.formula_text = ScrolledText(formula_frame, height=8, wrap=tk.NONE)
        self.formula_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.formula_text.bind("<<Modified>>", self._on_formula_changed)

        common_frame = ttk.Frame(parent, padding=5)
        common_frame.pack(fill=tk.X)
        self.initial_capital_var = tk.StringVar(value="100000")
        self.fee_var = tk.StringVar(value="0.0005")
        self.multi_timeframe_var = tk.StringVar(value="D,W,M")
        self.strategy_text_var = tk.StringVar()

        ttk.Label(common_frame, text="初始资金:").pack(side=tk.LEFT)
        ttk.Entry(common_frame, width=10, textvariable=self.initial_capital_var).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Label(common_frame, text="单边手续费:").pack(side=tk.LEFT)
        ttk.Entry(common_frame, width=8, textvariable=self.fee_var).pack(side=tk.LEFT)
        ttk.Label(common_frame, text=" (如 0.0005)").pack(side=tk.LEFT)

    def _build_strategy_sections(self, parent):
        self.fixed_enabled = tk.BooleanVar(value=True)
        self.fixed_periods_var = tk.StringVar(value="5,10,15,20")
        frame = ttk.LabelFrame(parent, text="固定周期", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=3)
        ttk.Checkbutton(frame, variable=self.fixed_enabled).pack(side=tk.LEFT)
        ttk.Label(frame, text="持有周期数:").pack(side=tk.LEFT)
        ttk.Entry(frame, width=20, textvariable=self.fixed_periods_var).pack(side=tk.LEFT, padx=5)

        self.tpsl_enabled = tk.BooleanVar(value=True)
        self.tp_var = tk.StringVar(value="0.1")
        self.sl_var = tk.StringVar(value="0.05")
        frame = ttk.LabelFrame(parent, text="止盈止损", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=3)
        ttk.Checkbutton(frame, variable=self.tpsl_enabled).pack(side=tk.LEFT)
        ttk.Label(frame, text="止盈比例%:").pack(side=tk.LEFT)
        ttk.Entry(frame, width=10, textvariable=self.tp_var).pack(side=tk.LEFT, padx=3)
        ttk.Label(frame, text="止损比例%:").pack(side=tk.LEFT)
        ttk.Entry(frame, width=10, textvariable=self.sl_var).pack(side=tk.LEFT, padx=3)

        self.dca_enabled = tk.BooleanVar(value=False)
        self.dca_size_var = tk.StringVar(value="0.05")
        self.dca_target_var = tk.StringVar(value="0.2")
        frame = ttk.LabelFrame(parent, text="定投策略", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=3)
        ttk.Checkbutton(frame, variable=self.dca_enabled).pack(side=tk.LEFT)
        ttk.Label(frame, text="定投尺寸%:").pack(side=tk.LEFT)
        ttk.Entry(frame, width=10, textvariable=self.dca_size_var).pack(side=tk.LEFT, padx=3)
        ttk.Label(frame, text="目标收益率%:").pack(side=tk.LEFT)
        ttk.Entry(frame, width=10, textvariable=self.dca_target_var).pack(side=tk.LEFT, padx=3)

        self.grid_enabled = tk.BooleanVar(value=False)
        self.grid_size_var = tk.StringVar(value="0.05")
        self.grid_cash_var = tk.StringVar(value="1000")
        self.grid_limit_var = tk.StringVar(value="None")
        self.grid_accumulate_var = tk.StringVar(value="True")
        frame = ttk.LabelFrame(parent, text="网格策略", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=3)
        ttk.Checkbutton(frame, variable=self.grid_enabled).pack(side=tk.LEFT)
        ttk.Label(frame, text="网格尺寸%:").pack(side=tk.LEFT)
        ttk.Entry(frame, width=10, textvariable=self.grid_size_var).pack(side=tk.LEFT, padx=3)
        ttk.Label(frame, text="单网资金:").pack(side=tk.LEFT)
        ttk.Entry(frame, width=10, textvariable=self.grid_cash_var).pack(side=tk.LEFT, padx=3)
        ttk.Label(frame, text="网格数限制:").pack(side=tk.LEFT)
        ttk.Entry(frame, width=10, textvariable=self.grid_limit_var).pack(side=tk.LEFT, padx=3)
        ttk.Label(frame, text="累积份额(True/False):").pack(side=tk.LEFT)
        ttk.Entry(frame, width=8, textvariable=self.grid_accumulate_var).pack(side=tk.LEFT, padx=3)

    def _build_extensions(self, parent):
        frame = ttk.LabelFrame(parent, text="扩展分析功能", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frame, text="多周期(如 D,W,M):").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(frame, width=15, textvariable=self.multi_timeframe_var).grid(row=0, column=1, padx=3)
        ttk.Button(frame, text="多周期信号", command=self.show_multi_timeframe_signals).grid(row=0, column=2, padx=3)
        ttk.Button(frame, text="指标评分曲线", command=self.show_indicator_scores).grid(row=0, column=3, padx=3)
        ttk.Button(frame, text="止盈/止损建议", command=self.show_stop_suggestion).grid(row=0, column=4, padx=3)
        ttk.Button(frame, text="仓位/补仓计划", command=self.show_position_plan).grid(row=0, column=5, padx=3)

        ttk.Button(frame, text="情景压力测试", command=self.show_stress_test).grid(row=1, column=0, padx=3, pady=2)
        ttk.Button(frame, text="收益热力图", command=self.show_heatmap).grid(row=1, column=1, padx=3)
        ttk.Button(frame, text="复盘摘要", command=self.show_daily_brief).grid(row=1, column=2, padx=3)
        ttk.Entry(frame, width=30, textvariable=self.strategy_text_var).grid(row=1, column=3, padx=3)
        ttk.Button(frame, text="自然语言转公式", command=self.convert_nlp_strategy).grid(row=1, column=4, padx=3)

    def _build_status_bar(self, parent):
        frame = ttk.Frame(parent, padding=5)
        frame.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="状态：未运行")
        ttk.Label(frame, textvariable=self.status_var).pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        btn_frame = ttk.Frame(parent, padding=5)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="运行回测", command=self.run_backtest).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="绘制统计图", command=self.plot_stats).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="绘制交易图", command=self.plot_trade_chart_window).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="打开结果文件夹", command=self.open_results_folder).pack(side=tk.LEFT, padx=3)

    def _build_log_area(self, parent):
        log_frame = ttk.LabelFrame(parent, text="运行日志", padding=5)
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        self.log_text = ScrolledText(log_frame, width=40, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # 基础操作
    # ------------------------------------------------------------------

    def log(self, text: str):
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def _set_status(self, text: str):
        self.status_var.set(text)
        if "运行" in text or "处理中" in text:
            self.progress.start(40)
        else:
            self.progress.stop()

    def show_about(self):
        messagebox.showinfo("关于", "量化回测系统 V2.0\n模块化设计，响应更快。")

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
            self._schedule_config_save()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("错误", f"读取公式失败：\n{exc}")

    def paste_formula(self):
        try:
            text = self.root.clipboard_get()
        except Exception:
            messagebox.showwarning("提示", "无法获取剪贴板内容。")
            return
        self.formula_text.delete("1.0", tk.END)
        self.formula_text.insert(tk.END, text)
        self._schedule_config_save()

    def check_formula(self):
        content = self.formula_text.get("1.0", tk.END)
        if "B_COND" in content:
            messagebox.showinfo("检查结果", "公式看起来包含 B_COND，可尝试运行。")
        else:
            messagebox.showwarning("检查结果", "未找到 B_COND 定义，请添加：B_COND := ...;")

    def open_results_folder(self):
        folder = os.path.abspath("results")
        os.makedirs(folder, exist_ok=True)
        os.system(f'open "{folder}"' if os.name == "posix" else f'start "" "{folder}"')

    # ------------------------------------------------------------------
    # 配置保存
    # ------------------------------------------------------------------

    def _apply_user_config(self):
        if self.user_config.csv_path:
            self.csv_path_var.set(self.user_config.csv_path)
        if self.user_config.formula:
            self.formula_text.delete("1.0", tk.END)
            self.formula_text.insert(tk.END, self.user_config.formula)
        self.formula_text.edit_modified(False)

    def _schedule_config_save(self):
        if self._formula_save_job:
            self.root.after_cancel(self._formula_save_job)
        self._formula_save_job = self.root.after(800, self._save_config)

    def _on_formula_changed(self, _event=None):
        if self.formula_text.edit_modified():
            self.formula_text.edit_modified(False)
            self._schedule_config_save()

    def _save_config(self):
        self.user_config = UserConfig(
            csv_path=self.csv_path_var.get().strip(),
            formula=self.formula_text.get("1.0", tk.END).strip(),
        )
        self.config_mgr.save(self.user_config)
        self._formula_save_job = None

    # ------------------------------------------------------------------
    # 回测逻辑
    # ------------------------------------------------------------------

    def run_backtest(self):
        params = self._collect_backtest_params()
        if not params:
            return

        def worker(stop_event, emit):
            run_backtests(params, stop_event, emit)

        self.task_runner.request(
            name="回测",
            status_text="状态：正在运行回测…",
            worker=worker,
            on_result=self._on_backtest_result,
            on_error=self._handle_task_error,
            on_final=lambda: self._set_status("状态：空闲"),
        )

    def _collect_backtest_params(self) -> Optional[BacktestParams]:
        csv_path = self.csv_path_var.get().strip()
        if not csv_path:
            messagebox.showwarning("提示", "请选择行情 CSV 文件。")
            return None

        formula = self.formula_text.get("1.0", tk.END).strip()
        if not formula:
            messagebox.showwarning("提示", "请粘贴通达信公式（至少定义 B_COND）。")
            return None

        try:
            initial_capital = float(self.initial_capital_var.get())
            fee_rate = float(self.fee_var.get())
        except Exception:
            messagebox.showerror("错误", "初始资金或手续费格式错误。")
            return None

        strategies: Dict = {}
        if self.fixed_enabled.get():
            try:
                periods = [int(x) for x in self.fixed_periods_var.get().split(",") if x.strip()]
                if periods:
                    strategies["fixed"] = periods
            except Exception:
                messagebox.showerror("错误", "固定周期输入格式有误。")
                return None

        if self.tpsl_enabled.get():
            try:
                tp = float(self.tp_var.get())
                sl = float(self.sl_var.get())
                if tp > 1:
                    tp /= 100
                if sl > 1:
                    sl /= 100
                strategies["tpsl"] = {"tp": tp, "sl": sl}
            except Exception:
                messagebox.showerror("错误", "止盈/止损输入格式有误。")
                return None

        if self.dca_enabled.get():
            try:
                size = float(self.dca_size_var.get())
                target = float(self.dca_target_var.get())
                if size > 1:
                    size /= 100
                if target > 1:
                    target /= 100
                strategies["dca"] = {"size": size, "target": target}
            except Exception:
                messagebox.showerror("错误", "定投参数格式有误。")
                return None

        if self.grid_enabled.get():
            try:
                grid_pct = float(self.grid_size_var.get())
                if grid_pct > 1:
                    grid_pct /= 100
                single_cash = float(self.grid_cash_var.get())
                limit_raw = self.grid_limit_var.get().strip()
                max_grids = None if (not limit_raw or limit_raw.lower() == "none") else int(limit_raw)
                accumulate = self.grid_accumulate_var.get().strip().lower() == "true"
                strategies["grid"] = {
                    "grid_pct": grid_pct,
                    "single_cash": single_cash,
                    "max_grids": max_grids,
                    "accumulate": accumulate,
                }
            except Exception:
                messagebox.showerror("错误", "网格参数格式有误。")
                return None

        if not strategies:
            messagebox.showwarning("提示", "请至少启用一个策略模块。")
            return None

        return BacktestParams(
            csv_path=csv_path,
            formula=formula,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            strategies=strategies,
        )

    def _on_backtest_result(self, payload: BacktestPayload):
        self.state.df = payload.df
        self.state.buy = payload.buy
        self.state.sell = payload.sell
        self.state.results = payload.entries
        self.state.formula = payload.formula
        self.state.scores = None

        for entry in payload.entries:
            self.log("")
            self.log(entry.title)
            self._log_result(entry.result)

    def _log_result(self, res):
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

    def _handle_task_error(self, exc: Exception):
        if str(exc).lower() == "cancelled":
            self.log("任务已取消。")
            return
        messagebox.showerror("错误", str(exc))

    def _run_async_action(self, name: str, status_text: str, compute: Callable[[], object], on_result: Callable[[object], None]):
        def worker(stop_event, emit):
            if stop_event.is_set():
                emit("log", f"{name}已取消")
                return
            result = compute()
            if stop_event.is_set():
                emit("log", f"{name}已取消")
                return
            emit("result", result)

        self.task_runner.request(
            name=name,
            status_text=status_text,
            worker=worker,
            on_result=on_result,
            on_error=self._handle_task_error,
            on_final=lambda: self._set_status("状态：空闲"),
        )

    # ------------------------------------------------------------------
    # 功能按钮
    # ------------------------------------------------------------------

    def _ensure_data(self) -> bool:
        if self.state.df is None:
            messagebox.showinfo("提示", "请先运行一次回测。")
            return False
        return True

    def plot_stats(self):
        if not self.state.results:
            messagebox.showinfo("提示", "暂无回测结果可绘制。")
            return
        data = [(entry.name, entry.result) for entry in self.state.results]
        self._run_async_action(
            name="绘制统计图",
            status_text="状态：绘制统计图…",
            compute=lambda: data,
            on_result=lambda payload: plot_strategy_stats(payload),
        )

    def plot_trade_chart_window(self):
        if not self.state.results or self.state.df is None:
            messagebox.showinfo("提示", "暂无回测结果。")
            return
        index = 0
        if len(self.state.results) > 1:
            options = "\n".join([f"{i+1}. {entry.name}" for i, entry in enumerate(self.state.results)])
            choice = simpledialog.askinteger("选择策略", f"请选择策略编号：\n{options}", minvalue=1, maxvalue=len(self.state.results))
            if choice is None:
                return
            index = choice - 1
        entry = self.state.results[index]
        df = self.state.df.copy()
        buy = self.state.buy.copy() if self.state.buy is not None else None
        sell = self.state.sell.copy() if self.state.sell is not None else None

        self._run_async_action(
            name="绘制交易图",
            status_text="状态：绘制交易图…",
            compute=lambda: (df, entry.result, entry.name, buy, sell),
            on_result=lambda payload: plot_trade_chart(*payload),
        )

    def show_multi_timeframe_signals(self):
        if not self._ensure_data():
            return
        df = self.state.df.copy()
        formula = self.state.formula
        freqs = [f.strip() for f in self.multi_timeframe_var.get().split(",") if f.strip()]
        if not freqs:
            freqs = ["D", "W", "M"]

        self._run_async_action(
            name="多周期信号",
            status_text="状态：多周期信号计算中…",
            compute=lambda: generate_multi_timeframe_signals(df, formula, freqs),
            on_result=lambda payload: plot_multi_timeframe(*payload),
        )

    def show_indicator_scores(self):
        if not self._ensure_data():
            return
        df = self.state.df.copy()

        def handler(scores):
            self.state.scores = scores
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(scores.index, scores["total_score"], label="综合评分")
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_title("指标综合评分")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.autofmt_xdate()
            fig.tight_layout()
            plt.show(block=False)

        self._run_async_action(
            name="指标评分",
            status_text="状态：指标评分计算中…",
            compute=lambda: indicator_scoring(df),
            on_result=handler,
        )

    def show_stop_suggestion(self):
        if not self._ensure_data():
            return
        info = atr_based_stop(self.state.df)
        messagebox.showinfo(
            "止盈止损建议",
            f"最新价: {info['last_price']:.2f}\nATR: {info['atr']:.2f}\n"
            f"建议止损: {info['suggest_stop_loss']:.2f}\n" f"建议止盈: {info['suggest_take_profit']:.2f}\n" f"跟踪止损: {info['trailing_stop']:.2f}"
        )

    def show_position_plan(self):
        if not self._ensure_data():
            return
        try:
            capital = float(self.initial_capital_var.get())
        except Exception:
            messagebox.showerror("错误", "初始资金格式有误。")
            return
        df = self.state.df.copy()

        def handler(plan):
            if plan.empty:
                messagebox.showinfo("提示", "资金不足或价格不适合生成计划。")
                return
            self._show_table_window("仓位/补仓计划", plan)

        self._run_async_action(
            name="仓位计划",
            status_text="状态：仓位计划计算中…",
            compute=lambda: position_rebalance_plan(df, capital),
            on_result=handler,
        )

    def show_stress_test(self):
        if not self._ensure_data():
            return
        df = self.state.df.copy()

        def handler(result):
            if result.empty:
                messagebox.showinfo("提示", "样本区间不足以覆盖预设情景。")
            else:
                self._show_table_window("情景压力测试", result)

        self._run_async_action(
            name="情景压力测试",
            status_text="状态：情景压力测试处理中…",
            compute=lambda: run_stress_test(df),
            on_result=handler,
        )

    def show_heatmap(self):
        if not self._ensure_data():
            return
        df = self.state.df.copy()

        def handler(result):
            if result.empty:
                messagebox.showinfo("提示", "数据不足以生成热力图。")
                return
            plot_heatmap(result.tail(200))

        self._run_async_action(
            name="收益热力图",
            status_text="状态：收益热力图处理中…",
            compute=lambda: holding_return_heatmap(df),
            on_result=handler,
        )

    def show_daily_brief(self):
        if not self._ensure_data():
            return
        first_result = self.state.results[0].result if self.state.results else None
        text = generate_daily_brief(self.state.df, self.state.scores, first_result, self.state.buy, self.state.sell)
        messagebox.showinfo("复盘摘要", text)

    def convert_nlp_strategy(self):
        desc = self.strategy_text_var.get().strip()
        if not desc:
            messagebox.showinfo("提示", "请输入策略描述。")
            return
        try:
            script = simple_rule_based_formula(desc)
        except Exception as exc:
            messagebox.showerror("错误", str(exc))
            return
        self.formula_text.delete("1.0", tk.END)
        self.formula_text.insert(tk.END, script)
        self._schedule_config_save()
        messagebox.showinfo("完成", "已生成基础公式，请按需调整。")

    # ------------------------------------------------------------------
    # 结果呈现
    # ------------------------------------------------------------------

    def _show_table_window(self, title: str, df: pd.DataFrame):
        top = tk.Toplevel(self.root)
        top.title(title)
        columns = list(df.columns)
        tree = ttk.Treeview(top, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor=tk.CENTER)
        vsb = ttk.Scrollbar(top, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        for _, row in df.iterrows():
            tree.insert("", tk.END, values=[row[col] for col in columns])


def main():
    root = tk.Tk()
    app = QuantGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
