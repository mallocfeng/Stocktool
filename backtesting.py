from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

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
    investment_amount: Optional[float] = None
    loss_streak: Optional[int] = None
    adjusted_quantity: Optional[int] = None
    pnl_with_dynamic_fund: Optional[float] = None
    hedge_investment_amount: Optional[float] = None
    hedge_loss_streak: Optional[int] = None
    hedge_adjusted_quantity: Optional[int] = None
    hedge_pnl_with_dynamic_fund: Optional[float] = None


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
    dynamic_equity_curve: Optional[pd.Series] = None
    investment_curve_main: Optional[List[List[Any]]] = None
    investment_curve_hedge: Optional[List[List[Any]]] = None
    max_loss_streak_used: Optional[int] = None
    max_investment_used: Optional[float] = None
    hedge_max_loss_streak_used: Optional[int] = None
    hedge_max_investment_used: Optional[float] = None
    dynamic_force_stop: bool = False
    position_details: Optional[List[Dict[str, Any]]] = None
    dynamic_summary: Optional[Dict[str, Any]] = None

    def to_dict(self):
        d = asdict(self)
        d["equity_curve"] = self.equity_curve.to_dict()
        d["trades"] = [asdict(t) for t in self.trades]
        if self.dynamic_equity_curve is not None:
            d["dynamic_equity_curve"] = self.dynamic_equity_curve.to_dict()
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


def backtest_dynamic_capital(
    df: pd.DataFrame,
    buy: pd.Series,
    sell: pd.Series,
    config: Dict[str, Any],
    initial_capital: float,
    fee_rate: float,
) -> BacktestResult:
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    close = df["close"]
    idx = df.index
    equity = pd.Series(index=idx, dtype=float)

    def _to_float(value, default=0.0):
        if value is None:
            return default
        if isinstance(value, str):
            txt = value.strip()
            if not txt:
                return default
            if txt.endswith("%"):
                try:
                    return float(txt.rstrip("%")) / 100.0
                except ValueError:
                    return default
            try:
                return float(txt)
            except ValueError:
                return default
        return float(value)

    initial_investment = max(0.0, _to_float(config.get("initialInvestment"), initial_capital))
    loss_step_amount = max(0.0, _to_float(config.get("lossStepAmount"), 0.0))
    max_add_steps_raw = config.get("maxAddSteps", 3)
    max_add_steps = int(max_add_steps_raw) if isinstance(max_add_steps_raw, (int, float)) else 0
    max_limit = _to_float(config.get("maxInvestmentLimit"), initial_capital)
    reset_on_win = bool(config.get("resetOnWin", True))
    drawdown_limit_raw = _to_float(config.get("maxDrawdownLimit"), 0.0)
    hedge_enabled = bool(config.get("enableHedge"))
    hedge_initial = max(0.0, _to_float(config.get("hedgeInitialInvestment"), 0.0))
    hedge_loss_step = max(0.0, _to_float(config.get("hedgeLossStepAmount"), 0.0))
    hedge_max_steps_raw = config.get("hedgeMaxAddSteps", 2)
    hedge_max_steps = int(hedge_max_steps_raw) if isinstance(hedge_max_steps_raw, (int, float)) else 0

    if max_limit <= 0:
        max_limit = initial_capital
    if initial_investment <= 0:
        initial_investment = max_limit
    drawdown_limit_value = None
    if drawdown_limit_raw > 0:
        drawdown_limit_value = initial_capital * drawdown_limit_raw if drawdown_limit_raw <= 1 else drawdown_limit_raw

    cash_main = initial_capital
    position = 0
    entry_price = 0.0
    entry_idx: Optional[int] = None
    entry_cost = 0.0
    entry_fee = 0.0
    entry_shares = 0
    entry_loss_state = 0
    entry_investment_amount = 0.0

    hedge_cash = 0.0
    hedge_position = 0
    hedge_entry_price = 0.0
    hedge_entry_fee = 0.0
    hedge_entry_shares = 0
    hedge_entry_loss_state = 0
    hedge_entry_amount = 0.0

    loss_streak = 0
    hedge_loss_streak = 0
    next_amount_main = initial_investment
    next_amount_hedge = min(max_limit, hedge_initial) if hedge_initial > 0 else min(max_limit, initial_investment)
    last_allocation_main = next_amount_main
    last_allocation_hedge = next_amount_hedge

    trades: List[Trade] = []
    investment_series: List[List[Any]] = []
    hedge_invest_series: List[List[Any]] = []
    position_details: List[Dict[str, Any]] = []

    peak_equity = initial_capital
    force_stop = False
    max_loss_streak_used = 0
    max_investment_used = 0.0
    hedge_max_loss_streak_used = 0
    hedge_max_investment_used = 0.0

    def clamp_amount(value: float) -> float:
        if max_limit > 0:
            return min(value, max_limit)
        return value

    def allowed_steps(streak: int, max_steps: int) -> int:
        if streak <= 0 or max_steps is None or max_steps < 1:
            return 0
        return min(streak, max_steps)

    for i, date in enumerate(idx):
        price = float(close.iloc[i])

        if position == 0 and buy.iloc[i]:
            target_amount = clamp_amount(next_amount_main if not force_stop else initial_investment)
            if target_amount <= 0 or cash_main <= 0:
                pass
            else:
                loss_state = loss_streak
                affordable_shares = int((cash_main / price) if fee_rate == 0 else cash_main / (price * (1 + fee_rate)))
                desired_shares = int(target_amount // price)
                shares = min(affordable_shares, desired_shares)
                if shares > 0:
                    cost = shares * price
                    fee = cost * fee_rate
                    total_cost = cost + fee
                    if total_cost > cash_main:
                        shares = int(cash_main / (price * (1 + fee_rate)))
                        if shares <= 0:
                            shares = 0
                    if shares > 0:
                        cost = shares * price
                        fee = cost * fee_rate
                        cash_main -= cost + fee
                        position = shares
                        entry_price = price
                        entry_idx = i
                        entry_cost = cost
                        entry_fee = fee
                        entry_shares = shares
                        entry_loss_state = int(loss_state)
                        entry_investment_amount = float(cost)
                        actual_invest = entry_investment_amount
                        last_allocation_main = actual_invest
                        max_investment_used = max(max_investment_used, actual_invest)
                        max_loss_streak_used = max(max_loss_streak_used, loss_state)
                        ts = str(date)
                        investment_series.append([ts, actual_invest])
                        hedge_amount_used = 0.0
                        hedge_loss_state = hedge_loss_streak
                        hedge_entry_shares = 0
                        hedge_entry_amount = 0.0
                        hedge_entry_loss_state = int(hedge_loss_state)
                        if hedge_enabled and next_amount_hedge > 0:
                            hedge_target = clamp_amount(next_amount_hedge if not force_stop else hedge_initial)
                            hedge_shares = int(hedge_target // price)
                            if hedge_shares > 0:
                                proceeds = hedge_shares * price
                                hedge_entry_fee = proceeds * fee_rate
                                hedge_cash += proceeds - hedge_entry_fee
                                hedge_position = hedge_shares
                                hedge_entry_price = price
                                hedge_amount_used = float(proceeds)
                                last_allocation_hedge = hedge_amount_used
                                hedge_invest_series.append([ts, hedge_amount_used])
                                hedge_entry_shares = hedge_shares
                                hedge_entry_amount = hedge_amount_used
                                hedge_entry_loss_state = int(hedge_loss_state)
                                hedge_max_investment_used = max(hedge_max_investment_used, hedge_amount_used)
                                hedge_max_loss_streak_used = max(hedge_max_loss_streak_used, hedge_loss_state)
                        position_details.append(
                            {
                                "date": ts,
                                "investmentAmount": actual_invest,
                                "lossStreak": int(loss_state),
                                "hedgeInvestmentAmount": hedge_amount_used,
                                "hedgeLossStreak": int(hedge_loss_state),
                                "forceStop": force_stop,
                                "quantity": entry_shares,
                                "hedgeQuantity": hedge_entry_shares,
                            }
                        )

        elif position > 0 and sell.iloc[i]:
            exit_price = price
            value = position * exit_price
            exit_fee = value * fee_rate
            cash_main += value - exit_fee
            pnl = value - exit_fee - entry_cost - entry_fee
            days_hold = (date - idx[entry_idx]).days if entry_idx is not None else 0
            hedge_realized_pnl = None
            hedge_record_amount = hedge_entry_amount or 0.0
            hedge_record_loss_state = hedge_entry_loss_state
            hedge_record_quantity = hedge_entry_shares

            if hedge_enabled and hedge_position > 0:
                cover_cost = hedge_position * exit_price
                cover_fee = cover_cost * fee_rate
                hedge_cash -= cover_cost + cover_fee
                hedge_pnl = hedge_cash
                hedge_realized_pnl = float(hedge_pnl)
                cash_main += hedge_cash
                hedge_cash = 0.0
                hedge_position = 0
                if hedge_pnl >= 0:
                    hedge_loss_streak = 0
                    next_amount_hedge = clamp_amount(hedge_initial if reset_on_win else last_allocation_hedge)
                else:
                    hedge_loss_streak += 1
                    hedge_max_loss_streak_used = max(hedge_max_loss_streak_used, hedge_loss_streak)
                    hedge_steps = allowed_steps(hedge_loss_streak, hedge_max_steps)
                    if force_stop:
                        next_amount_hedge = hedge_initial
                    else:
                        next_amount_hedge = clamp_amount(hedge_initial + hedge_loss_step * hedge_steps)
                hedge_entry_shares = 0
                hedge_entry_amount = 0.0
                hedge_entry_loss_state = 0

            trades.append(
                Trade(
                    entry_date=idx[entry_idx] if entry_idx is not None else date,
                    exit_date=date,
                    entry_price=float(entry_price),
                    exit_price=float(exit_price),
                    return_pct=float((exit_price - entry_price) / entry_price) if entry_price else 0.0,
                    holding_days=int(days_hold),
                    note="动态资金管理",
                    investment_amount=entry_investment_amount if entry_investment_amount else None,
                    loss_streak=int(entry_loss_state),
                    adjusted_quantity=int(entry_shares) if entry_shares else None,
                    pnl_with_dynamic_fund=float(pnl),
                    hedge_investment_amount=hedge_record_amount or None,
                    hedge_loss_streak=int(hedge_record_loss_state) if hedge_record_amount else None,
                    hedge_adjusted_quantity=int(hedge_record_quantity) if hedge_record_quantity else None,
                    hedge_pnl_with_dynamic_fund=hedge_realized_pnl,
                )
            )
            position = 0
            entry_idx = None
            entry_shares = 0
            entry_investment_amount = 0.0
            entry_loss_state = 0

            if pnl >= 0:
                loss_streak = 0
                next_amount_main = clamp_amount(initial_investment if reset_on_win else last_allocation_main)
            else:
                loss_streak += 1
                max_loss_streak_used = max(max_loss_streak_used, loss_streak)
                steps = allowed_steps(loss_streak, max_add_steps)
                if force_stop:
                    next_amount_main = initial_investment
                else:
                    next_amount_main = clamp_amount(initial_investment + loss_step_amount * steps)

        equity_value = cash_main + position * price + hedge_cash - hedge_position * price
        equity.iloc[i] = equity_value
        if equity_value > peak_equity:
            peak_equity = equity_value
        drawdown_value = peak_equity - equity_value
        if drawdown_limit_value is not None and drawdown_value >= drawdown_limit_value:
            force_stop = True

    if position > 0 and entry_idx is not None:
        price = float(close.iloc[-1])
        value = position * price
        exit_fee = value * fee_rate
        cash_main += value - exit_fee
        days_hold = (idx[-1] - idx[entry_idx]).days
        pnl = value - exit_fee - entry_cost - entry_fee
        hedge_realized_pnl = None
        hedge_record_amount = hedge_entry_amount or 0.0
        hedge_record_loss_state = hedge_entry_loss_state
        hedge_record_quantity = hedge_entry_shares

        if hedge_enabled and hedge_position > 0:
            cover_cost = hedge_position * price
            cover_fee = cover_cost * fee_rate
            hedge_cash -= cover_cost + cover_fee
            hedge_pnl = hedge_cash
            hedge_realized_pnl = float(hedge_pnl)
            cash_main += hedge_cash
            hedge_cash = 0.0
            hedge_position = 0
            if hedge_pnl >= 0:
                hedge_loss_streak = 0
            else:
                hedge_loss_streak += 1
                hedge_max_loss_streak_used = max(hedge_max_loss_streak_used, hedge_loss_streak)
            hedge_entry_shares = 0
            hedge_entry_amount = 0.0
            hedge_entry_loss_state = 0

        trades.append(
            Trade(
                entry_date=idx[entry_idx],
                exit_date=idx[-1],
                entry_price=float(entry_price),
                exit_price=float(price),
                return_pct=float((price - entry_price) / entry_price) if entry_price else 0.0,
                holding_days=int(days_hold),
                note="动态资金管理-样本结束强平",
                investment_amount=entry_investment_amount if entry_investment_amount else None,
                loss_streak=int(entry_loss_state),
                adjusted_quantity=int(entry_shares) if entry_shares else None,
                pnl_with_dynamic_fund=float(pnl),
                hedge_investment_amount=hedge_record_amount or None,
                hedge_loss_streak=int(hedge_record_loss_state) if hedge_record_amount else None,
                hedge_adjusted_quantity=int(hedge_record_quantity) if hedge_record_quantity else None,
                hedge_pnl_with_dynamic_fund=hedge_realized_pnl,
            )
        )
        position = 0
        entry_idx = None
        entry_shares = 0
        entry_investment_amount = 0.0
        entry_loss_state = 0
        if pnl >= 0:
            loss_streak = 0
        else:
            loss_streak += 1
            max_loss_streak_used = max(max_loss_streak_used, loss_streak)

    if hedge_enabled and hedge_position > 0:
        price = float(close.iloc[-1])
        cover_cost = hedge_position * price
        cover_fee = cover_cost * fee_rate
        hedge_cash -= cover_cost + cover_fee
        cash_main += hedge_cash
        hedge_cash = 0.0
        hedge_position = 0

    equity.iloc[-1] = cash_main

    result = _calc_stats(equity, trades)
    result.dynamic_equity_curve = equity
    result.investment_curve_main = investment_series or None
    result.investment_curve_hedge = hedge_invest_series or None
    result.dynamic_force_stop = force_stop
    result.position_details = position_details or None
    result.max_loss_streak_used = int(max_loss_streak_used) if max_loss_streak_used else 0
    result.max_investment_used = float(max_investment_used)
    result.hedge_max_loss_streak_used = (
        int(hedge_max_loss_streak_used) if hedge_enabled and hedge_max_loss_streak_used else None
    )
    result.hedge_max_investment_used = (
        float(hedge_max_investment_used) if hedge_enabled and hedge_max_investment_used else None
    )
    result.dynamic_summary = {
        "initialInvestment": float(initial_investment),
        "lossStepAmount": float(loss_step_amount),
        "maxAddSteps": max_add_steps,
        "maxInvestmentLimit": float(max_limit),
        "resetOnWin": reset_on_win,
        "maxLossStreakUsed": int(max_loss_streak_used),
        "maxInvestmentUsed": float(max_investment_used),
        "enableHedge": hedge_enabled,
        "hedgeInitialInvestment": float(hedge_initial),
        "hedgeLossStepAmount": float(hedge_loss_step),
        "hedgeMaxAddSteps": hedge_max_steps,
        "hedgeMaxLossStreakUsed": int(hedge_max_loss_streak_used) if hedge_enabled else None,
        "hedgeMaxInvestmentUsed": float(hedge_max_investment_used) if hedge_enabled else None,
        "forceStopByDrawdown": force_stop,
        "maxDrawdownLimitInput": config.get("maxDrawdownLimit"),
        "maxDrawdownLimitValue": float(drawdown_limit_value) if drawdown_limit_value is not None else None,
    }
    return result


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
