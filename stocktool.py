#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stocktool.py

简易“通达信公式 + 回测”可视化小工具

依赖：
    pip install pandas numpy

使用方式：
    直接运行本文件：
        python stocktool.py

功能概览：
    1. 从本地选择一个 CSV 行情文件（列：date,open,high,low,close,volume）
    2. 在大文本框中粘贴通达信公式：
        - 支持函数：EMA, REF, COUNT, LLV, HHV, CROSS
        - 支持变量：C,O,H,L,V
        - 支持 AND / OR / NOT 逻辑
        - 需要在公式中定义：
            B_COND := ... ;   // 最终买入条件
        - 可选：
            S_COND := ... ;   // 卖出条件，没有则默认从不卖出（仅按策略规则出场）
    3. 在界面中选择策略类型和参数（固定周期 or 止盈止损）
    4. 点击“运行回测”，在下方输出结果。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


# ==============================
#   通达信函数实现
# ==============================

def EMA(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def REF(series: pd.Series, n: int) -> pd.Series:
    return series.shift(n)


def COUNT(cond: pd.Series, n: int) -> pd.Series:
    return cond.astype(int).rolling(window=n, min_periods=1).sum()


def LLV(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(window=n, min_periods=1).min()


def HHV(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(window=n, min_periods=1).max()


def CROSS(s1: pd.Series, s2: pd.Series) -> pd.Series:
    diff = s1 - s2
    prev = diff.shift(1)
    return (prev <= 0) & (diff > 0)


# ==============================
#   通达信公式执行引擎
# ==============================

class TdxFormulaEngine:
    """
    极简通达信公式执行器，仅支持常见语法：

        E1:=EMA(C,13);
        A1:=COUNT(E1<REF(E1,1),5)>=3 AND E1>REF(E1,1);
        B_COND:=A1;

    支持：
        - 变量：C,O,H,L,V
        - 函数：EMA, REF, COUNT, LLV, HHV, CROSS
        - 逻辑：AND/OR/NOT (大小写都行)
        - 注释：以 { / } / // / # 开头 的行一律忽略

    公式执行完后从上下文中取：
        B_COND  买入条件（必须）
        S_COND  卖出条件（可选）
    """

    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        self.df = df

        self.ctx: Dict[str, object] = {
            # 行情序列
            "C": df["close"],
            "O": df["open"],
            "H": df["high"],
            "L": df["low"],
            "V": df.get("volume", pd.Series(np.nan, index=df.index)),
            # 函数
            "EMA": EMA,
            "REF": REF,
            "COUNT": COUNT,
            "LLV": LLV,
            "HHV": HHV,
            "CROSS": CROSS,
            # 允许用到的库（少量）
            "np": np,
            "pd": pd,
        }

    @staticmethod
    def _convert_expr(expr: str) -> str:
        expr = expr.strip()
        # 通达信逻辑运算符 → pandas 按位运算
        expr = expr.replace("AND", "&").replace("and", "&")
        expr = expr.replace("OR", "|").replace("or", "|")
        expr = expr.replace("NOT", "~").replace("not", "~")
        return expr

    def run(self, script: str) -> Tuple[pd.Series, pd.Series]:
        for raw_line in script.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # 简单过滤注释
            if line.startswith(("{", "}", "//", "#", "(*", "*)")):
                continue
            if line.endswith(";"):
                line = line[:-1]

            if ":=" in line:
                name, expr = line.split(":=", 1)
                name = name.strip()
                expr = self._convert_expr(expr)
                value = eval(expr, {}, self.ctx)
                self.ctx[name] = value
            else:
                expr = self._convert_expr(line)
                _ = eval(expr, {}, self.ctx)

        buy = self.ctx.get("B_COND", None)
        sell = self.ctx.get("S_COND", None)

        if buy is None:
            raise ValueError("公式中没有定义 B_COND（买入条件）。")

        if sell is None:
            sell = pd.Series(False, index=self.df.index)

        buy = buy.astype(bool).reindex(self.df.index).fillna(False)
        sell = sell.astype(bool).reindex(self.df.index).fillna(False)

        return buy, sell


# ==============================
#           回测逻辑
# ==============================

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    return_pct: float
    holding_days: int


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: List[Trade]
    total_return: float
    annualized_return: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["equity_curve"] = self.equity_curve.to_dict()
        d["trades"] = [asdict(t) for t in self.trades]
        return d


def _calc_stats(equity: pd.Series, trades: List[Trade]) -> BacktestResult:
    initial_capital = float(equity.iloc[0])
    total_return = equity.iloc[-1] / initial_capital - 1.0

    days = max(1, (equity.index[-1] - equity.index[0]).days)
    years = days / 365.0
    if years > 0:
        annualized_return = (1 + total_return) ** (1 / years) - 1.0
    else:
        annualized_return = 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min())

    if trades:
        wins = [t.return_pct for t in trades if t.return_pct > 0]
        losses = [t.return_pct for t in trades if t.return_pct <= 0]
        win_rate = len(wins) / len(trades) if trades else 0.0
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
    else:
        win_rate = avg_win = avg_loss = 0.0

    return BacktestResult(
        equity_curve=equity,
        trades=trades,
        total_return=float(total_return),
        annualized_return=float(annualized_return),
        max_drawdown=max_drawdown,
        win_rate=float(win_rate),
        avg_win=avg_win,
        avg_loss=avg_loss,
    )


def backtest_fixed_period(
    df: pd.DataFrame,
    buy: pd.Series,
    sell: pd.Series,
    hold_days: int,
    initial_capital: float = 100000.0,
    fee_rate: float = 0.0005,
) -> BacktestResult:
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    close = df["close"]
    idx = df.index

    cash = initial_capital
    position = 0  # 股数
    entry_price = 0.0
    entry_idx: Optional[int] = None

    trades: List[Trade] = []
    equity = pd.Series(index=idx, dtype=float)

    for i, date in enumerate(idx):
        price = float(close.iloc[i])

        if position == 0:
            # 无持仓，看是否买入
            if buy.iloc[i]:
                # 全仓买入（考虑手续费）
                buy_cash = cash * (1 - fee_rate)
                position = int(buy_cash // price)
                cost = position * price
                fee = cost * fee_rate
                cash = cash - cost - fee
                entry_price = price
                entry_idx = i
        else:
            # 有持仓，判断是否应该卖出
            should_exit = False
            if entry_idx is not None:
                days_hold = (date - idx[entry_idx]).days
                if days_hold >= hold_days:
                    should_exit = True
            if sell.iloc[i]:
                should_exit = True

            if should_exit and entry_idx is not None:
                value = position * price
                fee = value * fee_rate
                cash = cash + value - fee
                ret = (price - entry_price) / entry_price
                days_hold = (date - idx[entry_idx]).days
                trades.append(
                    Trade(
                        entry_date=idx[entry_idx],
                        exit_date=date,
                        entry_price=entry_price,
                        exit_price=price,
                        return_pct=float(ret),
                        holding_days=int(days_hold),
                    )
                )
                position = 0
                entry_price = 0.0
                entry_idx = None

        # 每日权益
        if position > 0 and entry_price > 0:
            equity.iloc[i] = cash + position * price
        else:
            equity.iloc[i] = cash

    # 末尾如果还有仓位，按最后价强平
    if position > 0 and entry_idx is not None:
        price = float(close.iloc[-1])
        value = position * price
        fee = value * fee_rate
        cash = cash + value - fee
        ret = (price - entry_price) / entry_price
        days_hold = (idx[-1] - idx[entry_idx]).days
        trades.append(
            Trade(
                entry_date=idx[entry_idx],
                exit_date=idx[-1],
                entry_price=entry_price,
                exit_price=price,
                return_pct=float(ret),
                holding_days=int(days_hold),
            )
        )
        equity.iloc[-1] = cash

    return _calc_stats(equity, trades)


def backtest_take_profit_stop_loss(
    df: pd.DataFrame,
    buy: pd.Series,
    sell: pd.Series,
    tp: float,
    sl: float,
    initial_capital: float = 100000.0,
    fee_rate: float = 0.0005,
) -> BacktestResult:
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    close = df["close"]
    idx = df.index

    cash = initial_capital
    position = 0
    entry_price = 0.0
    entry_idx: Optional[int] = None

    trades: List[Trade] = []
    equity = pd.Series(index=idx, dtype=float)

    for i, date in enumerate(idx):
        price = float(close.iloc[i])

        if position == 0:
            if buy.iloc[i]:
                buy_cash = cash * (1 - fee_rate)
                position = int(buy_cash // price)
                cost = position * price
                fee = cost * fee_rate
                cash = cash - cost - fee
                entry_price = price
                entry_idx = i
        else:
            assert entry_idx is not None
            ret = (price - entry_price) / entry_price
            should_exit = False
            if ret >= tp:
                should_exit = True
            if ret <= -sl:
                should_exit = True
            if sell.iloc[i]:
                should_exit = True

            if should_exit:
                value = position * price
                fee = value * fee_rate
                cash = cash + value - fee
                days_hold = (date - idx[entry_idx]).days
                trades.append(
                    Trade(
                        entry_date=idx[entry_idx],
                        exit_date=date,
                        entry_price=entry_price,
                        exit_price=price,
                        return_pct=float(ret),
                        holding_days=int(days_hold),
                    )
                )
                position = 0
                entry_price = 0.0
                entry_idx = None

        if position > 0 and entry_price > 0:
            equity.iloc[i] = cash + position * price
        else:
            equity.iloc[i] = cash

    if position > 0 and entry_idx is not None:
        price = float(close.iloc[-1])
        value = position * price
        fee = value * fee_rate
        cash = cash + value - fee
        ret = (price - entry_price) / entry_price
        days_hold = (idx[-1] - idx[entry_idx]).days
        trades.append(
            Trade(
                entry_date=idx[entry_idx],
                exit_date=idx[-1],
                entry_price=entry_price,
                exit_price=price,
                return_pct=float(ret),
                holding_days=int(days_hold),
            )
        )
        equity.iloc[-1] = cash

    return _calc_stats(equity, trades)


# ==============================
#           图形界面
# ==============================

class StockToolGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("简易量化回测系统 - stocktool.py")
        root.geometry("900x700")

        self.csv_path_var = tk.StringVar()
        self.strategy_var = tk.StringVar(value="fixed")
        self.hold_days_var = tk.StringVar(value="5,10,20")
        self.tp_var = tk.StringVar(value="0.1")
        self.sl_var = tk.StringVar(value="0.05")
        self.capital_var = tk.StringVar(value="100000")
        self.fee_var = tk.StringVar(value="0.0005")

        self._build_ui()

    def _build_ui(self):
        # 行情文件选择
        file_frame = ttk.Frame(self.root, padding=5)
        file_frame.pack(fill=tk.X)

        ttk.Label(file_frame, text="行情CSV文件:").pack(side=tk.LEFT)
        entry = ttk.Entry(file_frame, textvariable=self.csv_path_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(file_frame, text="浏览...", command=self.browse_csv).pack(side=tk.LEFT)

        # 参数区域
        param_frame = ttk.LabelFrame(self.root, text="回测参数", padding=5)
        param_frame.pack(fill=tk.X, padx=5, pady=5)

        # 初始资金 & 手续费
        row1 = ttk.Frame(param_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="初始资金:").pack(side=tk.LEFT)
        ttk.Entry(row1, width=10, textvariable=self.capital_var).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(row1, text="单边手续费:").pack(side=tk.LEFT)
        ttk.Entry(row1, width=8, textvariable=self.fee_var).pack(side=tk.LEFT)
        ttk.Label(row1, text="(如 0.0005)").pack(side=tk.LEFT, padx=(2, 0))

        # 策略选择
        row2 = ttk.Frame(param_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="策略类型:").pack(side=tk.LEFT)

        ttk.Radiobutton(
            row2, text="固定持有周期", value="fixed", variable=self.strategy_var, command=self.update_strategy_state
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            row2, text="止盈止损", value="tpsl", variable=self.strategy_var, command=self.update_strategy_state
        ).pack(side=tk.LEFT)

        # 固定周期参数
        row3 = ttk.Frame(param_frame)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="固定持有天数(逗号分隔):").pack(side=tk.LEFT)
        self.hold_entry = ttk.Entry(row3, width=20, textvariable=self.hold_days_var)
        self.hold_entry.pack(side=tk.LEFT)

        # 止盈止损参数
        row4 = ttk.Frame(param_frame)
        row4.pack(fill=tk.X, pady=2)
        ttk.Label(row4, text="止盈:").pack(side=tk.LEFT)
        self.tp_entry = ttk.Entry(row4, width=8, textvariable=self.tp_var)
        self.tp_entry.pack(side=tk.LEFT)
        ttk.Label(row4, text="止损:").pack(side=tk.LEFT, padx=(10, 0))
        self.sl_entry = ttk.Entry(row4, width=8, textvariable=self.sl_var)
        self.sl_entry.pack(side=tk.LEFT)
        ttk.Label(row4, text="(如 0.1 表示 10%)").pack(side=tk.LEFT, padx=(5, 0))

        # 公式文本框
        formula_frame = ttk.LabelFrame(self.root, text="通达信公式（需定义 B_COND / 可选 S_COND）", padding=5)
        formula_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.formula_text = ScrolledText(formula_frame, height=12, wrap=tk.NONE)
        self.formula_text.pack(fill=tk.BOTH, expand=True)

        # 按钮
        btn_frame = ttk.Frame(self.root, padding=5)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="运行回测", command=self.run_backtest).pack(side=tk.RIGHT)

        # 输出区域
        output_frame = ttk.LabelFrame(self.root, text="回测结果", padding=5)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.output_text = ScrolledText(output_frame, height=14, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)

        self.update_strategy_state()

    # ---------- 事件处理 ----------

    def browse_csv(self):
        path = filedialog.askopenfilename(
            title="选择行情 CSV 文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.csv_path_var.set(path)

    def update_strategy_state(self):
        strategy = self.strategy_var.get()
        if strategy == "fixed":
            self.hold_entry.configure(state="normal")
            self.tp_entry.configure(state="disabled")
            self.sl_entry.configure(state="disabled")
        else:
            self.hold_entry.configure(state="disabled")
            self.tp_entry.configure(state="normal")
            self.sl_entry.configure(state="normal")

    def append_output(self, text: str):
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)

    def run_backtest(self):
        self.output_text.delete("1.0", tk.END)

        csv_path = self.csv_path_var.get().strip()
        if not csv_path:
            messagebox.showwarning("提示", "请先选择行情 CSV 文件。")
            return

        formula = self.formula_text.get("1.0", tk.END).strip()
        if not formula:
            messagebox.showwarning("提示", "请粘贴通达信公式（至少定义 B_COND）。")
            return

        try:
            capital = float(self.capital_var.get())
            fee = float(self.fee_var.get())
        except Exception:
            messagebox.showerror("错误", "初始资金或手续费格式错误。")
            return

        # 读取数据
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            messagebox.showerror("错误", f"读取 CSV 失败：\n{e}")
            return

        required_cols = {"date", "open", "high", "low", "close"}
        if not required_cols.issubset(df.columns):
            messagebox.showerror("错误", f"CSV 至少需要列：{', '.join(required_cols)}")
            return

        self.append_output(f"加载行情数据：{csv_path}")
        self.append_output(f"数据行数：{len(df)}")
        self.append_output("正在解析公式并生成信号……")

        try:
            engine = TdxFormulaEngine(df)
            buy, sell = engine.run(formula)
        except Exception as e:
            messagebox.showerror("公式错误", f"解析/执行通达信公式失败：\n{e}")
            return

        if not buy.any():
            self.append_output("警告：在全样本中没有任何买入信号（B_COND 全 False）。")

        strategy = self.strategy_var.get()

        try:
            if strategy == "fixed":
                hold_str = self.hold_days_var.get().strip()
                days_list = [int(x) for x in hold_str.split(",") if x.strip()]
                if not days_list:
                    raise ValueError("请填写至少一个持有天数。")

                for hd in days_list:
                    self.append_output("")
                    self.append_output(f"=== 固定持有 {hd} 天 回测 ===")
                    res = backtest_fixed_period(df, buy, sell, hold_days=hd,
                                                initial_capital=capital, fee_rate=fee)
                    self._show_result(res)

            else:
                tp = float(self.tp_var.get())
                sl = float(self.sl_var.get())
                self.append_output("")
                self.append_output(f"=== 止盈止损 回测 (止盈 {tp*100:.1f}%, 止损 {sl*100:.1f}%) ===")
                res = backtest_take_profit_stop_loss(df, buy, sell, tp=tp, sl=sl,
                                                     initial_capital=capital, fee_rate=fee)
                self._show_result(res)

        except Exception as e:
            messagebox.showerror("回测错误", f"回测过程中出现错误：\n{e}")
            return

    def _show_result(self, res: BacktestResult):
        self.append_output(f"总收益率: {res.total_return*100:.2f}%")
        self.append_output(f"年化收益: {res.annualized_return*100:.2f}%")
        self.append_output(f"最大回撤: {res.max_drawdown*100:.2f}%")
        self.append_output(f"交易次数: {len(res.trades)}")
        self.append_output(f"胜率: {res.win_rate*100:.2f}%")
        self.append_output(f"平均盈利: {res.avg_win*100:.2f}%")
        self.append_output(f"平均亏损: {res.avg_loss*100:.2f}%")

        if res.trades:
            self.append_output("")
            self.append_output("最近 5 笔交易：")
            for t in res.trades[-5:]:
                self.append_output(
                    f"{t.entry_date.date()} 买入 {t.entry_price:.2f}  -> "
                    f"{t.exit_date.date()} 卖出 {t.exit_price:.2f} "
                    f"收益 {t.return_pct*100:.2f}%, 持有 {t.holding_days} 天"
                )


def main():
    root = tk.Tk()
    app = StockToolGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()