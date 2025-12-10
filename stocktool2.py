#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
stocktool.py

量化回测系统 V1.0（简化仿通达信回测工具）

功能：
- 从 CSV 读取单标的日线行情（date,open,high,low,close,volume）
- 通达信公式解析：支持 EMA / REF / COUNT / LLV / HHV / CROSS，变量 C,O,H,L,V
- B_COND：买入条件（必须定义）
- S_COND：卖出条件（可选）

界面模块：
- 固定周期策略
- 止盈止损策略
- 定投策略（简化版）
- 网格策略（简化版）
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates, font_manager

# GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText

# ==============================
# Matplotlib 字体配置（避免中文乱码）
# ==============================

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

# ==============================
# 通达信函数实现
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
# 通达信公式执行引擎
# ==============================

class TdxFormulaEngine:
    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        self.df = df

        self.ctx: Dict[str, object] = {
            "C": df["close"],
            "O": df["open"],
            "H": df["high"],
            "L": df["low"],
            "V": df.get("volume", pd.Series(np.nan, index=df.index)),
            "EMA": EMA,
            "REF": REF,
            "COUNT": COUNT,
            "LLV": LLV,
            "HHV": HHV,
            "CROSS": CROSS,
            "np": np,
            "pd": pd,
        }

    @staticmethod
    def _convert_expr(expr: str) -> str:
        expr = expr.strip()
        expr = expr.replace("AND", "&").replace("and", "&")
        expr = expr.replace("OR", "|").replace("or", "|")
        expr = expr.replace("NOT", "~").replace("not", "~")
        return expr

    def run(self, script: str) -> Tuple[pd.Series, pd.Series]:
        for raw_line in script.splitlines():
            line = raw_line.strip()
            if not line:
                continue
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

        buy = self.ctx.get("B_COND")
        if buy is None:
            raise ValueError("公式中未定义 B_COND（买入条件）。")

        sell = self.ctx.get("S_COND")
        if sell is None:
            sell = pd.Series(False, index=self.df.index)

        buy = buy.astype(bool).reindex(self.df.index).fillna(False)
        sell = sell.astype(bool).reindex(self.df.index).fillna(False)
        return buy, sell

# ==============================
# 回测基础结构
# ==============================

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    return_pct: float
    holding_days: int
    note: str = ""

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
    annualized_return = (1 + total_return) ** (1 / years) - 1.0 if years > 0 else 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min())

    if trades:
        wins = [t.return_pct for t in trades if t.return_pct > 0]
        losses = [t.return_pct for t in trades if t.return_pct <= 0]
        win_rate = len(wins) / len(trades)
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

# ==============================
# 各策略回测
# ==============================

def backtest_fixed_period(
    df: pd.DataFrame,
    buy: pd.Series,
    sell: pd.Series,
    hold_days: int,
    initial_capital: float,
    fee_rate: float,
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
                cash -= cost + fee
                entry_price = price
                entry_idx = i
        else:
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
                cash += value - fee
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
                        note=f"固定{hold_days}天",
                    )
                )
                position = 0
                entry_price = 0.0
                entry_idx = None

        equity.iloc[i] = cash + position * price

    if position > 0 and entry_idx is not None:
        price = float(close.iloc[-1])
        value = position * price
        fee = value * fee_rate
        cash += value - fee
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
                note=f"到样本末尾强平(固定{hold_days}天)",
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
    initial_capital: float,
    fee_rate: float,
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
                cash -= cost + fee
                entry_price = price
                entry_idx = i
        else:
            assert entry_idx is not None
            ret = (price - entry_price) / entry_price
            should_exit = False
            reason = ""

            if ret >= tp:
                should_exit = True
                reason = "止盈"
            if ret <= -sl:
                should_exit = True
                reason = "止损"
            if sell.iloc[i]:
                should_exit = True
                if not reason:
                    reason = "卖出信号"

            if should_exit:
                value = position * price
                fee = value * fee_rate
                cash += value - fee
                days_hold = (date - idx[entry_idx]).days
                trades.append(
                    Trade(
                        entry_date=idx[entry_idx],
                        exit_date=date,
                        entry_price=entry_price,
                        exit_price=price,
                        return_pct=float(ret),
                        holding_days=int(days_hold),
                        note=reason,
                    )
                )
                position = 0
                entry_price = 0.0
                entry_idx = None

        equity.iloc[i] = cash + position * price

    if position > 0 and entry_idx is not None:
        price = float(close.iloc[-1])
        value = position * price
        fee = value * fee_rate
        cash += value - fee
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
                note="样本末尾强平(止盈止损)",
            )
        )
        equity.iloc[-1] = cash

    return _calc_stats(equity, trades)

def backtest_dca_simple(
    df: pd.DataFrame,
    buy: pd.Series,
    target_return: float,
    dca_size_pct: float,
    initial_capital: float,
    fee_rate: float,
) -> BacktestResult:
    """
    非严格“教科书定投”，简化版：
    - 每次 B_COND 为 True 时，用当前“可用现金 * dca_size_pct” 买入
    - 不看 S_COND；当总资产达到 initial*(1+target_return) 时卖出全部
    - 若样本结束仍未达标，就在最后一天卖出全部
    """
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    close = df["close"]
    idx = df.index
    cash = initial_capital
    position = 0
    trades: List[Trade] = []
    equity = pd.Series(index=idx, dtype=float)

    first_buy_date: Optional[pd.Timestamp] = None
    first_buy_price: float = 0.0

    for i, date in enumerate(idx):
        price = float(close.iloc[i])

        # 定投买入
        if buy.iloc[i] and cash > 0:
            invest_cash = cash * dca_size_pct * (1 - fee_rate)
            shares = int(invest_cash // price)
            if shares > 0:
                cost = shares * price
                fee = cost * fee_rate
                cash -= cost + fee
                position += shares
                if first_buy_date is None:
                    first_buy_date = date
                    first_buy_price = price

        equity.iloc[i] = cash + position * price

        # 检查目标收益
        if equity.iloc[i] >= initial_capital * (1 + target_return) and position > 0:
            value = position * price
            fee = value * fee_rate
            cash += value - fee
            ret = cash / initial_capital - 1.0
            if first_buy_date is None:
                first_buy_date = idx[0]
                first_buy_price = float(close.iloc[0])
            days_hold = (date - first_buy_date).days
            trades.append(
                Trade(
                    entry_date=first_buy_date,
                    exit_date=date,
                    entry_price=first_buy_price,
                    exit_price=price,
                    return_pct=float(ret),
                    holding_days=int(days_hold),
                    note="达到目标收益率，全部卖出",
                )
            )
            position = 0
            first_buy_date = None
            first_buy_price = 0.0
            equity.iloc[i] = cash  # 卖出后净值

    # 样本结束仍有仓位 -> 清仓
    if position > 0:
        price = float(close.iloc[-1])
        value = position * price
        fee = value * fee_rate
        cash += value - fee
        ret = cash / initial_capital - 1.0
        if first_buy_date is None:
            first_buy_date = idx[0]
            first_buy_price = float(close.iloc[0])
        days_hold = (idx[-1] - first_buy_date).days
        trades.append(
            Trade(
                entry_date=first_buy_date,
                exit_date=idx[-1],
                entry_price=first_buy_price,
                exit_price=price,
                return_pct=float(ret),
                holding_days=int(days_hold),
                note="样本末尾全部卖出(定投)",
            )
        )
        equity.iloc[-1] = cash

    return _calc_stats(equity, trades)

def backtest_grid_simple(
    df: pd.DataFrame,
    grid_pct: float,
    single_grid_cash: float,
    max_grids: Optional[int],
    accumulative: bool,
    initial_capital: float,
    fee_rate: float,
) -> BacktestResult:
    """
    简化版单标的网格交易，不依赖 B_COND / S_COND：
    - 起始日按 current price 建网格中心价 center
    - 每跌 grid_pct 买一个网格（single_grid_cash）
    - 每涨 grid_pct 卖一个网格（已有仓位时）
    - 累积份额：True 表示收益滚入；False 表示每次按固定 single_grid_cash
    """
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    close = df["close"]
    idx = df.index
    cash = initial_capital
    position = 0
    base_price = float(close.iloc[0])
    last_trade_price = base_price
    grids_opened = 0

    trades: List[Trade] = []
    equity = pd.Series(index=idx, dtype=float)

    for i, date in enumerate(idx):
        price = float(close.iloc[i])
        equity.iloc[i] = cash + position * price

        # 向下开多网格
        down_threshold = last_trade_price * (1 - grid_pct)
        up_threshold = last_trade_price * (1 + grid_pct)

        trade_note = ""

        # 下跌开仓
        if price <= down_threshold and (max_grids is None or grids_opened < max_grids):
            # 计算本次投入资金
            invest_cash = single_grid_cash if not accumulative else min(cash, equity.iloc[i] * grid_pct)
            invest_cash = invest_cash * (1 - fee_rate)
            shares = int(invest_cash // price)
            if shares > 0:
                cost = shares * price
                fee = cost * fee_rate
                if cash >= cost + fee:
                    cash -= cost + fee
                    position += shares
                    last_trade_price = price
                    grids_opened += 1
                    trade_note = "网格买入"

        # 上涨减仓
        elif price >= up_threshold and position > 0:
            shares = int(single_grid_cash // price)
            if shares <= 0:
                shares = position  # 一次清仓
            shares = min(shares, position)
            value = shares * price
            fee = value * fee_rate
            cash += value - fee
            position -= shares
            last_trade_price = price
            trade_note = "网格卖出"

        # 记录成交
        if trade_note:
            trades.append(
                Trade(
                    entry_date=date,
                    exit_date=date,
                    entry_price=price,
                    exit_price=price,
                    return_pct=0.0,
                    holding_days=0,
                    note=trade_note,
                )
            )
        equity.iloc[i] = cash + position * price

    return _calc_stats(equity, trades)

# ==============================
# GUI
# ==============================

class QuantGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("量化回测系统 V1.0")
        root.geometry("900x800")

        # ---------------- 顶部菜单（装饰用） ----------------
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

        # ---------------- 顶部：标题和公式管理 ----------------
        top_frame = ttk.Frame(root, padding=5)
        top_frame.pack(fill=tk.X)

        title_label = ttk.Label(top_frame, text="量化回测系统 V1.0", font=("Microsoft YaHei", 16, "bold"))
        title_label.pack()

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

        # 公式文本框
        self.formula_text = ScrolledText(formula_mgmt, height=8, wrap=tk.NONE)
        self.formula_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # ---------------- 中部：四个策略块 ----------------
        self.initial_capital_var = tk.StringVar(value="100000")
        self.fee_var = tk.StringVar(value="0.0005")

        common_frame = ttk.Frame(root, padding=5)
        common_frame.pack(fill=tk.X)

        ttk.Label(common_frame, text="初始资金:").pack(side=tk.LEFT)
        ttk.Entry(common_frame, width=10, textvariable=self.initial_capital_var).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Label(common_frame, text="单边手续费:").pack(side=tk.LEFT)
        ttk.Entry(common_frame, width=8, textvariable=self.fee_var).pack(side=tk.LEFT)
        ttk.Label(common_frame, text=" (如 0.0005)").pack(side=tk.LEFT)

        # 固定周期
        self.fixed_enabled = tk.BooleanVar(value=True)
        self.fixed_periods_var = tk.StringVar(value="5,10,15,20")

        fixed_frame = ttk.LabelFrame(root, text="固定周期", padding=5)
        fixed_frame.pack(fill=tk.X, padx=5, pady=3)

        ttk.Checkbutton(fixed_frame, text="", variable=self.fixed_enabled).pack(side=tk.LEFT)
        ttk.Label(fixed_frame, text="持有周期数:").pack(side=tk.LEFT)
        ttk.Entry(fixed_frame, width=20, textvariable=self.fixed_periods_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(fixed_frame, text="(用逗号分隔，如 5,10,20)").pack(side=tk.LEFT)

        # 止盈止损
        self.tpsl_enabled = tk.BooleanVar(value=True)
        self.tp_var = tk.StringVar(value="0.1")
        self.sl_var = tk.StringVar(value="0.05")
        self.dd_var = tk.StringVar(value="0.2")

        tpsl_frame = ttk.LabelFrame(root, text="止盈止损", padding=5)
        tpsl_frame.pack(fill=tk.X, padx=5, pady=3)

        ttk.Checkbutton(tpsl_frame, variable=self.tpsl_enabled).pack(side=tk.LEFT)
        ttk.Label(tpsl_frame, text="止盈比例%:").pack(side=tk.LEFT)
        ttk.Entry(tpsl_frame, width=10, textvariable=self.tp_var).pack(side=tk.LEFT, padx=3)
        ttk.Label(tpsl_frame, text="止损比例%:").pack(side=tk.LEFT)
        ttk.Entry(tpsl_frame, width=10, textvariable=self.sl_var).pack(side=tk.LEFT, padx=3)
        ttk.Label(tpsl_frame, text="回撤比例%(暂未用):").pack(side=tk.LEFT)
        ttk.Entry(tpsl_frame, width=10, textvariable=self.dd_var).pack(side=tk.LEFT, padx=3)

        # 定投策略
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

        # 网格策略
        self.grid_enabled = tk.BooleanVar(value=False)
        self.grid_size_var = tk.StringVar(value="0.05")
        self.grid_cash_var = tk.StringVar(value="1000")
        self.grid_factor_var = tk.StringVar(value="1")
        self.grid_accumulate_var = tk.StringVar(value="True")
        self.grid_limit_var = tk.StringVar(value="None")

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
        ttk.Entry(grid_frame, width=8, textvariable=self.grid_accumulate_var).pack(side=tk.LEFT, padx=3)

        # ---------------- 状态 + 按钮 + 日志 ----------------
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
        ttk.Button(btn_frame, text="绘制交易图", command=self.plot_trade_chart).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="打开结果文件夹", command=self.open_results_folder).pack(side=tk.LEFT, padx=3)

        log_frame = ttk.LabelFrame(root, text="运行日志", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = ScrolledText(log_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.running_thread: Optional[threading.Thread] = None
        self.last_df: Optional[pd.DataFrame] = None
        self.last_buy_signals: Optional[pd.Series] = None
        self.last_sell_signals: Optional[pd.Series] = None
        self.last_results: List[Tuple[str, BacktestResult]] = []

    # ---------- UI 辅助 ----------

    def log(self, text: str):
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def show_about(self):
        messagebox.showinfo("关于", "简易量化回测系统 V1.0\n仿通达信批量回测界面，用 Python 实现。")

    def not_implemented(self):
        messagebox.showinfo("提示", "该功能目前仅为占位，还没实现图表绘制/停止功能。")

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
        # 简单检查是否包含 B_COND
        content = self.formula_text.get("1.0", tk.END)
        if "B_COND" in content:
            messagebox.showinfo("检查结果", "公式中包含 B_COND，基本格式看起来没问题。\n具体错误会在运行时报告。")
        else:
            messagebox.showwarning("检查结果", "未找到 B_COND 定义，请在公式中加入：\nB_COND := 你的买入条件;")

    def open_results_folder(self):
        folder = os.path.abspath("results")
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        os.system(f'open "{folder}"' if os.name == "posix" else f'start "" "{folder}"')

    def plot_stats(self):
        if not self.last_results:
            messagebox.showinfo("提示", "请先运行回测后再绘制统计图。")
            return

        names = [name for name, _ in self.last_results]
        total_returns = [res.total_return * 100 for _, res in self.last_results]
        drawdowns = [res.max_drawdown * 100 for _, res in self.last_results]
        win_rates = [res.win_rate * 100 for _, res in self.last_results]

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        bar_positions = np.arange(len(names))

        axes[0].bar(bar_positions, total_returns, color="#2ca02c")
        axes[0].set_ylabel("总收益率(%)")
        axes[0].set_title("策略表现对比")

        axes[1].bar(bar_positions, drawdowns, color="#d62728")
        axes[1].set_ylabel("最大回撤(%)")

        axes[2].bar(bar_positions, win_rates, color="#1f77b4")
        axes[2].set_ylabel("胜率(%)")
        axes[2].set_xticks(bar_positions)
        axes[2].set_xticklabels(names, rotation=45, ha="right")

        for ax in axes:
            ax.grid(axis="y", linestyle="--", alpha=0.4)

        fig.tight_layout()
        plt.show()

    def plot_trade_chart(self):
        if self.last_df is None or not self.last_results:
            messagebox.showinfo("提示", "请先运行回测并确保有结果。")
            return

        idx = 0
        if len(self.last_results) > 1:
            options = "\n".join([
                f"{i + 1}. {name}" for i, (name, _) in enumerate(self.last_results)
            ])
            choice = simpledialog.askinteger(
                "选择策略",
                f"请选择要绘制交易图的策略编号：\n{options}",
                minvalue=1,
                maxvalue=len(self.last_results),
            )
            if choice is None:
                return
            idx = choice - 1

        strategy_name, result = self.last_results[idx]
        df = self.last_df

        fig, (ax_price, ax_equity) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

        ax_price.plot(df.index, df["close"], label="收盘价", color="black")

        if self.last_buy_signals is not None:
            buy_mask = self.last_buy_signals.reindex(df.index).fillna(False)
            buy_dates = df.index[buy_mask.to_numpy(dtype=bool)]
            ax_price.scatter(buy_dates, df.loc[buy_dates, "close"], marker="^", color="#2ca02c", s=30, label="信号买入")

        if self.last_sell_signals is not None:
            sell_mask = self.last_sell_signals.reindex(df.index).fillna(False)
            sell_dates = df.index[sell_mask.to_numpy(dtype=bool)]
            ax_price.scatter(sell_dates, df.loc[sell_dates, "close"], marker="v", color="#d62728", s=30, label="信号卖出")

        if result.trades:
            entry_dates = [t.entry_date for t in result.trades]
            entry_prices = [t.entry_price for t in result.trades]
            exit_dates = [t.exit_date for t in result.trades]
            exit_prices = [t.exit_price for t in result.trades]
            ax_price.scatter(entry_dates, entry_prices, marker="^", color="green", s=80, label="实际买入")
            ax_price.scatter(exit_dates, exit_prices, marker="v", color="red", s=80, label="实际卖出")

        ax_price.set_title(f"{strategy_name} 交易示意")
        ax_price.set_ylabel("价格")
        ax_price.grid(alpha=0.3)
        ax_price.legend(loc="upper left")

        ax_equity.plot(result.equity_curve.index, result.equity_curve.values, color="#1f77b4", label="权益曲线")
        ax_equity.set_ylabel("资金")
        ax_equity.set_xlabel("日期")
        ax_equity.legend(loc="upper left")
        ax_equity.grid(alpha=0.3)
        ax_equity.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show()

    # ---------- 回测主流程 ----------

    def run_backtest_thread(self):
        if self.running_thread and self.running_thread.is_alive():
            messagebox.showwarning("提示", "回测正在运行中，请稍等。")
            return
        self.running_thread = threading.Thread(target=self.run_backtest, daemon=True)
        self.running_thread.start()

    def run_backtest(self):
        self.status_var.set("状态：正在运行……")
        self.progress.start(50)
        self.log_text.delete("1.0", tk.END)
        self.last_results = []
        self.last_df = None
        self.last_buy_signals = None
        self.last_sell_signals = None

        csv_path = self.csv_path_var.get().strip()
        if not csv_path:
            messagebox.showwarning("提示", "请先选择行情 CSV 文件。")
            self._end_run()
            return

        formula = self.formula_text.get("1.0", tk.END).strip()
        if not formula:
            messagebox.showwarning("提示", "请粘贴通达信公式（至少定义 B_COND）。")
            self._end_run()
            return

        try:
            initial_capital = float(self.initial_capital_var.get())
            fee_rate = float(self.fee_var.get())
        except Exception:
            messagebox.showerror("错误", "初始资金或手续费格式错误。")
            self._end_run()
            return

        # 读取行情
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            messagebox.showerror("错误", f"读取 CSV 失败：\n{e}")
            self._end_run()
            return

        required_cols = {"date", "open", "high", "low", "close"}
        if not required_cols.issubset(df.columns):
            messagebox.showerror("错误", f"CSV 至少需要列：{', '.join(required_cols)}")
            self._end_run()
            return

        df_datetime = df.copy()
        df_datetime["date"] = pd.to_datetime(df_datetime["date"])
        df_datetime.set_index("date", inplace=True)
        self.last_df = df_datetime

        self.log(f"加载行情数据：{csv_path}")
        self.log(f"数据行数：{len(df)}")
        self.log("正在解析公式并生成信号……")

        try:
            engine = TdxFormulaEngine(df)
            buy, sell = engine.run(formula)
        except Exception as e:
            messagebox.showerror("公式错误", f"解析/执行通达信公式失败：\n{e}")
            self._end_run()
            return

        self.last_buy_signals = buy
        self.last_sell_signals = sell

        if not buy.any():
            self.log("警告：在全样本中没有任何买入信号（B_COND 全 False）。")

        # 选择策略并执行
        results: List[Tuple[str, BacktestResult]] = []

        # 固定周期
        if self.fixed_enabled.get():
            try:
                periods = [
                    int(x) for x in self.fixed_periods_var.get().split(",") if x.strip()
                ]
            except Exception:
                messagebox.showerror("错误", "固定周期中的持有天数格式错误。")
                self._end_run()
                return

            for p in periods:
                self.log("")
                self.log(f"=== 固定周期策略：持有 {p} 天 ===")
                res = backtest_fixed_period(df, buy, sell, p, initial_capital, fee_rate)
                self.show_result(res)
                results.append((f"fixed_{p}", res))

        # 止盈止损
        if self.tpsl_enabled.get():
            try:
                tp = float(self.tp_var.get()) / 100 if float(self.tp_var.get()) > 1 else float(self.tp_var.get())
                sl = float(self.sl_var.get()) / 100 if float(self.sl_var.get()) > 1 else float(self.sl_var.get())
            except Exception:
                messagebox.showerror("错误", "止盈/止损比例格式错误。")
                self._end_run()
                return

            self.log("")
            self.log(f"=== 止盈止损策略：TP={tp*100:.1f}%, SL={sl*100:.1f}% ===")
            res = backtest_take_profit_stop_loss(df, buy, sell, tp, sl, initial_capital, fee_rate)
            self.show_result(res)
            results.append((f"tpsl_{tp}_{sl}", res))

        # 定投策略
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
                self._end_run()
                return

            self.log("")
            self.log(f"=== 定投策略：尺寸 {dca_size*100:.1f}%, 目标收益 {target*100:.1f}% ===")
            res = backtest_dca_simple(df, buy, target, dca_size, initial_capital, fee_rate)
            self.show_result(res)
            results.append((f"dca_{dca_size}_{target}", res))

        # 网格策略
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
                self._end_run()
                return

            self.log("")
            self.log(f"=== 网格策略：间距 {grid_pct*100:.1f}%, 单网资金 {single_cash}, "
                     f"网格数限制 {max_grids}, 累积={accumulate} ===")
            res = backtest_grid_simple(df, grid_pct, single_cash, max_grids, accumulate, initial_capital, fee_rate)
            self.show_result(res)
            results.append((f"grid_{grid_pct}", res))

        self.last_results = results

        # 保存结果
        if results:
            os.makedirs("results", exist_ok=True)
            for name, res in results:
                # equity
                res.equity_curve.to_csv(os.path.join("results", f"{name}_equity.csv"))
                # trades
                trades_df = pd.DataFrame([asdict(t) for t in res.trades])
                trades_df.to_csv(os.path.join("results", f"{name}_trades.csv"), index=False)
            self.log("")
            self.log("结果已保存到 ./results 目录下。")

        self._end_run()

    def _end_run(self):
        self.progress.stop()
        self.status_var.set("状态：空闲")

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
