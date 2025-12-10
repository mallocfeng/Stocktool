from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import pandas as pd


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

    def to_dict(self):
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
            equity.iloc[i] = cash

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

        down_threshold = last_trade_price * (1 - grid_pct)
        up_threshold = last_trade_price * (1 + grid_pct)

        trade_note = ""

        if price <= down_threshold and (max_grids is None or grids_opened < max_grids):
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

        elif price >= up_threshold and position > 0:
            shares = int(single_grid_cash // price)
            if shares <= 0:
                shares = position
            shares = min(shares, position)
            value = shares * price
            fee = value * fee_rate
            cash += value - fee
            position -= shares
            last_trade_price = price
            trade_note = "网格卖出"

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
