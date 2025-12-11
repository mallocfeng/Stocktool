from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from formula_engine import TdxFormulaEngine
from data_loader import load_price_csv
from backtesting import (
    BacktestResult,
    backtest_fixed_period,
    backtest_take_profit_stop_loss,
    backtest_dca_simple,
    backtest_grid_simple,
    backtest_dynamic_capital,
)


@dataclass
class BacktestParams:
    csv_path: str
    formula: str
    initial_capital: float
    fee_rate: float
    strategies: Dict


@dataclass
class BacktestEntry:
    name: str
    title: str
    result: BacktestResult


@dataclass
class BacktestPayload:
    df: pd.DataFrame
    buy: pd.Series
    sell: pd.Series
    entries: List[BacktestEntry]
    formula: str


class CancelledError(Exception):
    """Raised internally when a background task is cancelled."""


def run_backtests(params: BacktestParams, stop_event, emit) -> None:
    try:
        _run_backtests(params, stop_event, emit)
    except CancelledError:
        emit("log", "回测任务已取消")


def _run_backtests(params: BacktestParams, stop_event, emit) -> None:
    def check_cancel():
        if stop_event.is_set():
            raise CancelledError

    emit("log", f"加载行情数据：{params.csv_path}")
    try:
        df = load_price_csv(params.csv_path)
    except Exception as exc:  # noqa: BLE001
        emit("error", RuntimeError(f"读取 CSV 失败：{exc}"))
        raise

    check_cancel()

    required_cols = {"date", "open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 至少需要列：{', '.join(required_cols)}")

    df_datetime = df.copy()
    df_datetime["date"] = pd.to_datetime(df_datetime["date"])
    df_datetime.set_index("date", inplace=True)
    emit("log", f"数据行数：{len(df)}")
    emit("log", "正在解析公式并生成信号……")

    engine = TdxFormulaEngine(df)
    buy, sell = engine.run(params.formula)
    check_cancel()

    if not buy.any():
        emit("log", "警告：全样本中没有任何买入信号。")

    entries: List[BacktestEntry] = []
    strategies = params.strategies
    initial_capital = params.initial_capital
    fee_rate = params.fee_rate

    def append_result(name: str, title: str, result: BacktestResult):
        entries.append(BacktestEntry(name=name, title=title, result=result))

    if "fixed" in strategies:
        for p in strategies["fixed"]:
            check_cancel()
            title = f"=== 固定周期策略：持有 {p} 天 ==="
            emit("log", "")
            emit("log", title)
            res = backtest_fixed_period(df, buy, sell, p, initial_capital, fee_rate)
            append_result(f"fixed_{p}", title, res)

    if "tpsl" in strategies:
        check_cancel()
        cfg = strategies["tpsl"]
        title = f"=== 止盈止损策略：TP={cfg['tp']*100:.1f}%, SL={cfg['sl']*100:.1f}% ==="
        emit("log", "")
        emit("log", title)
        res = backtest_take_profit_stop_loss(df, buy, sell, cfg["tp"], cfg["sl"], initial_capital, fee_rate)
        append_result(f"tpsl_{cfg['tp']}_{cfg['sl']}", title, res)

    if "dca" in strategies:
        check_cancel()
        cfg = strategies["dca"]
        title = f"=== 定投策略：尺寸 {cfg['size']*100:.1f}%, 目标收益 {cfg['target']*100:.1f}% ==="
        emit("log", "")
        emit("log", title)
        res = backtest_dca_simple(df, buy, cfg["target"], cfg["size"], initial_capital, fee_rate)
        append_result(f"dca_{cfg['size']}_{cfg['target']}", title, res)

    if "grid" in strategies:
        check_cancel()
        cfg = strategies["grid"]
        title = (
            f"=== 网格策略：间距 {cfg['grid_pct']*100:.1f}%, 单网资金 {cfg['single_cash']}, "
            f"网格数限制 {cfg['max_grids']}, 累积={cfg['accumulate']} ==="
        )
        emit("log", "")
        emit("log", title)
        res = backtest_grid_simple(
            df,
            cfg["grid_pct"],
            cfg["single_cash"],
            cfg["max_grids"],
            cfg["accumulate"],
            initial_capital,
            fee_rate,
        )
        append_result(f"grid_{cfg['grid_pct']}", title, res)
    if "dynamic" in strategies:
        check_cancel()
        cfg = strategies["dynamic"]
        title = "=== 动态资金管理策略 ==="
        emit("log", "")
        emit("log", title)
        res = backtest_dynamic_capital(df, buy, sell, cfg, initial_capital, fee_rate)
        append_result("dynamic_capital", title, res)

    if entries:
        import os  # local import to avoid circular issues

        os.makedirs("results", exist_ok=True)
        for entry in entries:
            equity_path = os.path.join("results", f"{entry.name}_equity.csv")
            trades_path = os.path.join("results", f"{entry.name}_trades.csv")
            entry.result.equity_curve.to_csv(equity_path)
            trades_df = pd.DataFrame([t.__dict__ for t in entry.result.trades])
            trades_df.to_csv(trades_path, index=False)
        emit("log", "")
        emit("log", "结果已保存到 ./results 目录下。")

    payload = BacktestPayload(df=df_datetime, buy=buy, sell=sell, entries=entries, formula=params.formula)
    emit("result", payload)
