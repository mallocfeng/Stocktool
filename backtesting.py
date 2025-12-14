from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple

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
    baseline_equity_curve: Optional[pd.Series] = None
    investment_curve_main: Optional[List[List[Any]]] = None
    investment_curve_hedge: Optional[List[List[Any]]] = None
    max_loss_streak_used: Optional[int] = None
    max_investment_used: Optional[float] = None
    hedge_max_loss_streak_used: Optional[int] = None
    hedge_max_investment_used: Optional[float] = None
    dynamic_force_stop: bool = False
    position_details: Optional[List[Dict[str, Any]]] = None
    dynamic_summary: Optional[Dict[str, Any]] = None
    buy_hedge_summary: Optional[Dict[str, Any]] = None
    buy_hedge_trades: Optional[List[Dict[str, Any]]] = None
    buy_hedge_events: Optional[List[Dict[str, Any]]] = None

    def to_dict(self):
        d = asdict(self)
        d["equity_curve"] = self.equity_curve.to_dict()
        d["trades"] = [asdict(t) for t in self.trades]
        if self.dynamic_equity_curve is not None:
            d["dynamic_equity_curve"] = self.dynamic_equity_curve.to_dict()
        return d


class TPlusOneGuard:
    """Tracks individual lots so sells obey the T+1 constraint."""

    def __init__(self) -> None:
        self._lots: List[Tuple[pd.Timestamp, int]] = []

    @staticmethod
    def _normalize_day(ts: pd.Timestamp) -> pd.Timestamp:
        if isinstance(ts, pd.Timestamp):
            return pd.Timestamp(ts.date())
        return pd.Timestamp(ts).normalize()

    def add(self, date: pd.Timestamp, shares: int) -> None:
        if shares <= 0:
            return
        self._lots.append((self._normalize_day(date), int(shares)))

    def available(self, date: pd.Timestamp) -> int:
        trade_day = self._normalize_day(date)
        return int(sum(shares for lot_day, shares in self._lots if trade_day > lot_day))

    def can_sell(self, date: pd.Timestamp, shares: int) -> bool:
        if shares <= 0:
            return True
        return self.available(date) >= shares

    def consume(self, date: pd.Timestamp, shares: int) -> int:
        if shares <= 0:
            return 0
        trade_day = self._normalize_day(date)
        allowed = min(int(shares), self.available(date))
        if allowed <= 0:
            return 0
        remaining = allowed
        updated: List[Tuple[pd.Timestamp, int]] = []
        for lot_day, lot_shares in self._lots:
            if trade_day > lot_day and remaining > 0:
                if lot_shares <= remaining:
                    remaining -= lot_shares
                    continue
                updated.append((lot_day, lot_shares - remaining))
                remaining = 0
            else:
                updated.append((lot_day, lot_shares))
        self._lots = updated
        return allowed

    def reset(self) -> None:
        self._lots.clear()

    def force_clear(self) -> None:
        self._lots.clear()

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
    t1_guard = TPlusOneGuard()
    pending_exit = False
    pending_exit_reason = ""

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
                t1_guard.add(date, position)
                pending_exit = False
                pending_exit_reason = ""
        else:
            should_exit = pending_exit
            reason = pending_exit_reason
            if entry_idx is not None:
                days_hold = (date - idx[entry_idx]).days
                if not should_exit and days_hold >= hold_days:
                    should_exit = True
                    reason = f"固定{hold_days}天"
            if not should_exit and sell.iloc[i]:
                should_exit = True
                reason = "卖出信号"

            if should_exit and entry_idx is not None:
                if not t1_guard.can_sell(date, position):
                    pending_exit = True
                    pending_exit_reason = reason or "卖出信号"
                else:
                    t1_guard.consume(date, position)
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
                            note=reason or f"固定{hold_days}天",
                        )
                    )
                    position = 0
                    entry_price = 0.0
                    entry_idx = None
                    pending_exit = False
                    pending_exit_reason = ""

        equity.iloc[i] = cash + position * price

    if position > 0 and entry_idx is not None:
        price = float(close.iloc[-1])
        exit_date = idx[-1]
        if t1_guard.can_sell(exit_date, position):
            t1_guard.consume(exit_date, position)
            value = position * price
            fee = value * fee_rate
            cash += value - fee
            ret = (price - entry_price) / entry_price
            days_hold = (exit_date - idx[entry_idx]).days
            trades.append(
                Trade(
                    entry_date=idx[entry_idx],
                    exit_date=exit_date,
                    entry_price=entry_price,
                    exit_price=price,
                    return_pct=float(ret),
                    holding_days=int(days_hold),
                    note=f"到样本末尾强平(固定{hold_days}天)",
                )
            )
            equity.iloc[-1] = cash
        else:
            equity.iloc[-1] = cash + position * price

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
    t1_guard = TPlusOneGuard()
    pending_exit = False
    pending_exit_reason = ""

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
                t1_guard.add(date, position)
                pending_exit = False
                pending_exit_reason = ""
        else:
            assert entry_idx is not None
            ret = (price - entry_price) / entry_price
            should_exit = pending_exit
            reason = pending_exit_reason

            if not should_exit and ret >= tp:
                should_exit = True
                reason = "止盈"
            if not should_exit and ret <= -sl:
                should_exit = True
                reason = "止损"
            if not should_exit and sell.iloc[i]:
                should_exit = True
                reason = reason or "卖出信号"

            if should_exit:
                if not t1_guard.can_sell(date, position):
                    pending_exit = True
                    pending_exit_reason = reason or "卖出信号"
                else:
                    t1_guard.consume(date, position)
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
                            note=reason or "卖出信号",
                        )
                    )
                    position = 0
                    entry_price = 0.0
                    entry_idx = None
                    pending_exit = False
                    pending_exit_reason = ""

        equity.iloc[i] = cash + position * price

    if position > 0 and entry_idx is not None:
        price = float(close.iloc[-1])
        exit_date = idx[-1]
        if t1_guard.can_sell(exit_date, position):
            t1_guard.consume(exit_date, position)
            value = position * price
            fee = value * fee_rate
            cash += value - fee
            ret = (price - entry_price) / entry_price
            days_hold = (exit_date - idx[entry_idx]).days
            trades.append(
                Trade(
                    entry_date=idx[entry_idx],
                    exit_date=exit_date,
                    entry_price=entry_price,
                    exit_price=price,
                    return_pct=float(ret),
                    holding_days=int(days_hold),
                    note="样本末尾强平(止盈止损)",
                )
            )
            equity.iloc[-1] = cash
        else:
            equity.iloc[-1] = cash + position * price

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
    baseline_equity = pd.Series(index=idx, dtype=float)

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

    LOT_SIZE = 100

    initial_investment = max(0.0, _to_float(config.get("initialInvestment"), initial_capital))
    loss_step_lots = max(0.0, _to_float(config.get("lossStepAmount"), 0.0))
    max_add_steps_raw = config.get("maxAddSteps", 3)
    max_add_steps = int(max_add_steps_raw) if isinstance(max_add_steps_raw, (int, float)) else 0
    max_limit = _to_float(config.get("maxInvestmentLimit"), initial_capital)
    reset_on_win = bool(config.get("resetOnWin", True))
    drawdown_limit_raw = _to_float(config.get("maxDrawdownLimit"), 0.0)
    hedge_enabled = bool(config.get("enableHedge"))
    hedge_initial = max(0.0, _to_float(config.get("hedgeInitialInvestment"), 0.0))
    hedge_loss_step_lots = max(0.0, _to_float(config.get("hedgeLossStepAmount"), 0.0))
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
    pending_loss_steps_main = 0
    pending_loss_steps_hedge = 0

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
    t1_guard = TPlusOneGuard()
    pending_exit = False
    pending_exit_reason = ""

    baseline_cash = float(initial_capital)
    baseline_position = 0
    baseline_entry_price = 0.0
    baseline_entry_idx: Optional[int] = None
    baseline_t1_guard = TPlusOneGuard()
    baseline_pending_exit = False

    def clamp_amount(value: float) -> float:
        if max_limit > 0:
            return min(value, max_limit)
        return value

    def allowed_steps(streak: int, max_steps: int) -> int:
        if streak <= 0 or max_steps is None or max_steps < 1:
            return 0
        return min(streak, max_steps)

    def floor_lot(shares: int) -> int:
        if shares <= 0:
            return 0
        return (shares // LOT_SIZE) * LOT_SIZE

    for i, date in enumerate(idx):
        price = float(close.iloc[i])

        if baseline_position == 0:
            if buy.iloc[i]:
                buy_cash = baseline_cash * (1 - fee_rate)
                baseline_position = floor_lot(int(buy_cash // price))
                if baseline_position > 0:
                    cost = baseline_position * price
                    fee = cost * fee_rate
                    baseline_cash -= cost + fee
                    baseline_entry_price = price
                    baseline_entry_idx = i
                    baseline_t1_guard.add(date, baseline_position)
                    baseline_pending_exit = False
        else:
            should_exit = baseline_pending_exit or bool(sell.iloc[i])
            if should_exit:
                if not baseline_t1_guard.can_sell(date, baseline_position):
                    baseline_pending_exit = True
                else:
                    baseline_t1_guard.consume(date, baseline_position)
                    value = baseline_position * price
                    fee = value * fee_rate
                    baseline_cash += value - fee
                    baseline_position = 0
                    baseline_entry_price = 0.0
                    baseline_entry_idx = None
                    baseline_pending_exit = False

        if position == 0 and buy.iloc[i]:
            base_budget = clamp_amount(next_amount_main if not force_stop else initial_investment)
            base_shares = floor_lot(int(base_budget // price))
            extra_shares = 0
            if not force_stop and pending_loss_steps_main > 0 and loss_step_lots > 0:
                extra_shares = int(loss_step_lots * LOT_SIZE * pending_loss_steps_main)
            if max_limit > 0:
                max_shares_limit = floor_lot(int(max_limit // price))
            else:
                max_shares_limit = None
            desired_shares = base_shares + extra_shares
            if max_shares_limit is not None:
                desired_shares = min(desired_shares, max_shares_limit)
            if desired_shares <= 0 or cash_main <= 0:
                pass
            else:
                loss_state = loss_streak
                affordable_shares = floor_lot(
                    int((cash_main / price) if fee_rate == 0 else cash_main / (price * (1 + fee_rate)))
                )
                shares = floor_lot(min(affordable_shares, desired_shares))
                if shares > 0:
                    cost = shares * price
                    fee = cost * fee_rate
                    total_cost = cost + fee
                    if total_cost > cash_main:
                        shares = floor_lot(int(cash_main / (price * (1 + fee_rate))))
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
                            hedge_base = clamp_amount(next_amount_hedge if not force_stop else hedge_initial)
                            hedge_base_shares = floor_lot(int(hedge_base // price))
                            hedge_extra_shares = 0
                            if not force_stop and pending_loss_steps_hedge > 0 and hedge_loss_step_lots > 0:
                                hedge_extra_shares = int(hedge_loss_step_lots * LOT_SIZE * pending_loss_steps_hedge)
                            if max_limit > 0:
                                hedge_max_shares_limit = floor_lot(int(max_limit // price))
                            else:
                                hedge_max_shares_limit = None
                            hedge_shares = hedge_base_shares + hedge_extra_shares
                            if hedge_max_shares_limit is not None:
                                hedge_shares = min(hedge_shares, hedge_max_shares_limit)
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
                                pending_loss_steps_hedge = 0
                        pending_loss_steps_main = 0
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
                        t1_guard.add(date, shares)
                        pending_exit = False
                        pending_exit_reason = ""

        elif position > 0:
            should_exit = pending_exit or sell.iloc[i]
            if should_exit:
                if not t1_guard.can_sell(date, position):
                    pending_exit = True
                    pending_exit_reason = "卖出信号"
                else:
                    t1_guard.consume(date, position)
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
                            pending_loss_steps_hedge = 0
                            next_amount_hedge = clamp_amount(hedge_initial if reset_on_win else last_allocation_hedge)
                        else:
                            hedge_loss_streak += 1
                            hedge_max_loss_streak_used = max(hedge_max_loss_streak_used, hedge_loss_streak)
                            hedge_steps = allowed_steps(hedge_loss_streak, hedge_max_steps)
                            if force_stop:
                                pending_loss_steps_hedge = 0
                                next_amount_hedge = clamp_amount(hedge_initial)
                            else:
                                pending_loss_steps_hedge = hedge_steps
                                next_amount_hedge = clamp_amount(hedge_initial)
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
                    pending_exit = False
                    pending_exit_reason = ""

                    if pnl >= 0:
                        loss_streak = 0
                        pending_loss_steps_main = 0
                        next_amount_main = clamp_amount(initial_investment if reset_on_win else last_allocation_main)
                    else:
                        loss_streak += 1
                        max_loss_streak_used = max(max_loss_streak_used, loss_streak)
                        steps = allowed_steps(loss_streak, max_add_steps)
                        if force_stop:
                            pending_loss_steps_main = 0
                            next_amount_main = clamp_amount(initial_investment)
                        else:
                            pending_loss_steps_main = steps
                            next_amount_main = clamp_amount(initial_investment)

        equity_value = cash_main + position * price + hedge_cash - hedge_position * price
        equity.iloc[i] = equity_value
        baseline_equity.iloc[i] = baseline_cash + baseline_position * price
        if equity_value > peak_equity:
            peak_equity = equity_value
        drawdown_value = peak_equity - equity_value
        if drawdown_limit_value is not None and drawdown_value >= drawdown_limit_value:
            force_stop = True

    if position > 0 and entry_idx is not None:
        price = float(close.iloc[-1])
        exit_date = idx[-1]
        if t1_guard.can_sell(exit_date, position):
            t1_guard.consume(exit_date, position)
            value = position * price
            exit_fee = value * fee_rate
            cash_main += value - exit_fee
            days_hold = (exit_date - idx[entry_idx]).days
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
                    exit_date=exit_date,
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
            pending_exit = False
            pending_exit_reason = ""
            if pnl >= 0:
                loss_streak = 0
            else:
                loss_streak += 1
                max_loss_streak_used = max(max_loss_streak_used, loss_streak)
        else:
            pending_exit = True
            pending_exit_reason = "样本结束未满足T+1"

    if hedge_enabled and hedge_position > 0 and position == 0:
        price = float(close.iloc[-1])
        cover_cost = hedge_position * price
        cover_fee = cover_cost * fee_rate
        hedge_cash -= cover_cost + cover_fee
        cash_main += hedge_cash
        hedge_cash = 0.0
        hedge_position = 0

    final_price = float(close.iloc[-1])
    equity.iloc[-1] = cash_main + position * final_price + hedge_cash - hedge_position * final_price
    baseline_equity.iloc[-1] = baseline_cash + baseline_position * final_price

    result = _calc_stats(equity, trades)
    result.dynamic_equity_curve = equity
    result.baseline_equity_curve = baseline_equity
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
        "lossStepAmount": float(loss_step_lots),
        "maxAddSteps": max_add_steps,
        "maxInvestmentLimit": float(max_limit),
        "resetOnWin": reset_on_win,
        "maxLossStreakUsed": int(max_loss_streak_used),
        "maxInvestmentUsed": float(max_investment_used),
        "enableHedge": hedge_enabled,
        "hedgeInitialInvestment": float(hedge_initial),
        "hedgeLossStepAmount": float(hedge_loss_step_lots),
        "hedgeMaxAddSteps": hedge_max_steps,
        "hedgeMaxLossStreakUsed": int(hedge_max_loss_streak_used) if hedge_enabled else None,
        "hedgeMaxInvestmentUsed": float(hedge_max_investment_used) if hedge_enabled else None,
        "forceStopByDrawdown": force_stop,
        "maxDrawdownLimitInput": config.get("maxDrawdownLimit"),
        "maxDrawdownLimitValue": float(drawdown_limit_value) if drawdown_limit_value is not None else None,
    }
    return result


def backtest_buy_hedge(
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

    def to_int(value, default=0):
        try:
            if value is None:
                return default
            return int(float(value))
        except (ValueError, TypeError):
            return default

    def to_float(value, default=0.0):
        try:
            if value is None:
                return default
            return float(value)
        except (ValueError, TypeError):
            return default

    LOT_SIZE = 100

    def lot_align(value: int) -> int:
        if value <= 0:
            return 0
        return max(0, (value // LOT_SIZE) * LOT_SIZE)

    def shares_to_hands(value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        return int(value // LOT_SIZE)

    step_type = str(config.get("step_type") or "percent").lower()
    step_pct = to_float(config.get("step_pct"), 0.0)
    if step_type in {"percent", "percentage"} and step_pct > 1:
        step_pct /= 100.0
    step_pct = max(0.0, float(step_pct))
    start_position_shares = lot_align(max(0, to_int(config.get("start_position"), 0)))
    increment_unit_shares = lot_align(max(0, to_int(config.get("increment_unit"), 0)))
    start_position_hands = shares_to_hands(start_position_shares) or 0
    increment_unit_hands = shares_to_hands(increment_unit_shares) or 0
    mode = str(config.get("mode") or "equal").lower()
    max_adds = to_int(config.get("max_adds"), 0)
    reference = str(config.get("reference") or "last").lower()
    max_capital_value = to_float(config.get("max_capital"), 0.0)
    max_capital_ratio = to_float(config.get("max_capital_ratio"), 0.0)
    max_capital_input = config.get("max_capital_input")
    if (max_capital_value <= 0) and max_capital_ratio > 0:
        max_capital_value = initial_capital * max_capital_ratio

    cash = float(initial_capital)
    position = 0
    position_cost = 0.0
    entry_idx: Optional[int] = None
    entry_price0 = 0.0
    last_add_price = 0.0
    add_count = 0
    current_trade_id = 0
    can_add_more = True

    trades: List[Trade] = []
    buy_hedge_trades: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []

    trade_add_records: List[int] = []
    max_layers = 0
    trade_count = 0
    avg_cost_reduction_sum = 0.0
    max_capital_used = 0.0
    skipped_by_cash = 0
    skipped_by_limit = 0
    skipped_by_rule = 0
    t1_guard = TPlusOneGuard()
    pending_exit = False
    pending_exit_reason = ""

    def compute_trigger_price() -> Optional[float]:
        if position <= 0 or step_pct <= 0 or entry_price0 <= 0:
            return None
        if reference == "first":
            target_level = add_count + 1
            trigger = entry_price0 * (1 - step_pct * target_level)
        else:
            ref_price = last_add_price if last_add_price > 0 else entry_price0
            trigger = ref_price * (1 - step_pct)
        return max(trigger, 0.0)

    def quantity_for_add(layer_index: int) -> int:
        if mode == "increment":
            qty = start_position_shares + layer_index * increment_unit_shares
        elif mode == "double":
            qty = int(round(start_position_shares * (2 ** layer_index)))
        else:
            qty = start_position_shares
        qty = lot_align(int(qty))
        return max(0, int(qty))

    def append_event(event: Dict[str, Any]) -> None:
        payload = event.copy()
        payload["trade_id"] = current_trade_id if current_trade_id else None
        payload["date"] = str(event.get("date"))
        payload["type"] = event.get("type", "add")
        if "shares" in payload and payload["shares"] is not None:
            payload["shares"] = shares_to_hands(int(payload["shares"]))
        if "total_shares" in payload and payload["total_shares"] is not None:
            payload["total_shares"] = shares_to_hands(int(payload["total_shares"]))
        events.append(payload)

    for i, date in enumerate(idx):
        raw_price = close.iloc[i]
        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            price = float("nan")
        if not np.isfinite(price) or price <= 0:
            prev = equity.iloc[i - 1] if i > 0 else cash
            equity.iloc[i] = prev
            continue

        added_this_bar = False

        should_exit = position > 0 and (pending_exit or sell.iloc[i])
        if should_exit:
            if not t1_guard.can_sell(date, position):
                pending_exit = True
                pending_exit_reason = pending_exit_reason or "卖出信号"
            else:
                t1_guard.consume(date, position)
                exit_price = price
                value = position * exit_price
                exit_fee = value * fee_rate
                cash += value - exit_fee
                avg_cost = position_cost / position if position > 0 else 0.0
                pnl = value - exit_fee - position_cost
                entry_date = idx[entry_idx] if entry_idx is not None else date
                days_hold = (date - entry_date).days if entry_idx is not None else 0
                trades.append(
                    Trade(
                        entry_date=entry_date,
                        exit_date=date,
                        entry_price=float(entry_price0),
                        exit_price=float(exit_price),
                        return_pct=float((exit_price - entry_price0) / entry_price0) if entry_price0 else 0.0,
                        holding_days=int(days_hold),
                        note="买入对冲",
                    )
                )
                cost_reduction = ((entry_price0 - avg_cost) / entry_price0) if entry_price0 else 0.0
                buy_hedge_trades.append(
                    {
                        "trade_id": current_trade_id or trade_count + 1,
                        "entry_date": str(entry_date),
                        "exit_date": str(date),
                        "entry_price": float(entry_price0),
                        "exit_price": float(exit_price),
                        "avg_cost": float(avg_cost),
                        "total_shares": shares_to_hands(int(position)),
                        "adds": int(add_count),
                        "capital_used": float(position_cost),
                        "pnl": float(pnl),
                        "return_pct": float(pnl / position_cost) if position_cost else 0.0,
                        "avg_cost_delta_pct": float(cost_reduction),
                    }
                )
                max_layers = max(max_layers, add_count)
                trade_add_records.append(add_count)
                trade_count += 1
                avg_cost_reduction_sum += float(cost_reduction)
                position = 0
                position_cost = 0.0
                entry_idx = None
                entry_price0 = 0.0
                last_add_price = 0.0
                add_count = 0
                current_trade_id = 0
                can_add_more = True
                pending_exit = False
                pending_exit_reason = ""

        if position == 0 and buy.iloc[i] and start_position_shares > 0:
            desired = start_position_shares
            affordable = int(
                cash / (price * (1 + fee_rate)) if fee_rate >= 0 else cash // price
            )
            affordable = lot_align(affordable)
            shares = min(desired, affordable)
            if shares > 0:
                cost = shares * price
                fee = cost * fee_rate
                cash -= cost + fee
                position = shares
                position_cost = cost + fee
                entry_idx = i
                entry_price0 = price
                last_add_price = price
                add_count = 0
                current_trade_id = trade_count + 1
                can_add_more = True
                avg_cost = position_cost / position
                append_event(
                    {
                        "date": date,
                        "price": float(price),
                        "shares": int(shares),
                        "total_shares": int(position),
                        "avg_cost": float(avg_cost),
                        "cost": float(cost + fee),
                        "type": "entry",
                        "layer": 0,
                    }
                )
                max_capital_used = max(max_capital_used, position_cost)
                t1_guard.add(date, shares)
                pending_exit = False
                pending_exit_reason = ""

        elif position > 0 and step_pct > 0 and not added_this_bar and can_add_more and not pending_exit:
            trigger_price = compute_trigger_price()
            if trigger_price is not None and price <= trigger_price:
                if max_adds > 0 and add_count >= max_adds:
                    skipped_by_rule += 1
                    can_add_more = False
                    append_event(
                        {
                            "date": date,
                            "price": float(price),
                            "shares": 0,
                            "total_shares": int(position),
                            "avg_cost": float(position_cost / position if position else 0.0),
                            "type": "skip",
                            "layer": int(add_count),
                            "note": "已达到最大加仓次数",
                            "trigger_price": float(trigger_price),
                        }
                    )
                else:
                    desired = quantity_for_add(add_count)
                    if desired <= 0:
                        skipped_by_rule += 1
                        can_add_more = False
                        append_event(
                            {
                                "date": date,
                                "price": float(price),
                                "shares": 0,
                                "total_shares": int(position),
                                "avg_cost": float(position_cost / position if position else 0.0),
                                "type": "skip",
                                "layer": int(add_count),
                                "note": "加仓数量为 0",
                                "trigger_price": float(trigger_price),
                            }
                        )
                    else:
                        affordable = int(cash / (price * (1 + fee_rate))) if fee_rate >= 0 else int(cash // price)
                        affordable = lot_align(affordable)
                        limit_shares = desired
                        if max_capital_value > 0:
                            remaining = max(0.0, max_capital_value - position_cost)
                            limit_by_cap = (
                                int(remaining / (price * (1 + fee_rate))) if fee_rate >= 0 else int(remaining // price)
                            )
                            limit_by_cap = max(0, limit_by_cap)
                            limit_shares = min(limit_shares, limit_by_cap)
                        limit_shares = lot_align(limit_shares)
                        shares = min(desired, affordable, limit_shares)
                        reason = None
                        if shares <= 0:
                            if affordable <= 0:
                                skipped_by_cash += 1
                                reason = "现金不足"
                            else:
                                skipped_by_limit += 1
                                reason = "资金占用受限"
                            can_add_more = False
                            append_event(
                                {
                                    "date": date,
                                    "price": float(price),
                                    "shares": 0,
                                    "total_shares": int(position),
                                    "avg_cost": float(position_cost / position if position else 0.0),
                                    "type": "skip",
                                    "layer": int(add_count),
                                    "note": reason,
                                    "trigger_price": float(trigger_price),
                                }
                            )
                        else:
                            cost = shares * price
                            fee = cost * fee_rate
                            cash -= cost + fee
                            position += shares
                            position_cost += cost + fee
                            add_count += 1
                            added_this_bar = True
                            last_add_price = price
                            avg_cost = position_cost / position if position > 0 else 0.0
                            append_event(
                                {
                                    "date": date,
                                    "price": float(price),
                                    "shares": int(shares),
                                    "total_shares": int(position),
                                    "avg_cost": float(avg_cost),
                                    "cost": float(cost + fee),
                                    "type": "add",
                                    "layer": int(add_count),
                                    "trigger_price": float(trigger_price),
                                }
                            )
                            max_capital_used = max(max_capital_used, position_cost)
                            t1_guard.add(date, shares)
                            if max_adds > 0 and add_count >= max_adds:
                                can_add_more = False

        equity.iloc[i] = cash + position * price

    if position > 0 and entry_idx is not None:
        date = idx[-1]
        price = float(close.iloc[-1])
        if t1_guard.can_sell(date, position):
            t1_guard.consume(date, position)
            value = position * price
            exit_fee = value * fee_rate
            cash += value - exit_fee
            avg_cost = position_cost / position if position > 0 else 0.0
            entry_date = idx[entry_idx]
            days_hold = (date - entry_date).days
            trades.append(
                Trade(
                    entry_date=entry_date,
                    exit_date=date,
                    entry_price=float(entry_price0),
                    exit_price=float(price),
                    return_pct=float((price - entry_price0) / entry_price0) if entry_price0 else 0.0,
                    holding_days=int(days_hold),
                    note="买入对冲-样本结束强平",
                )
            )
            pnl = value - exit_fee - position_cost
            cost_reduction = ((entry_price0 - avg_cost) / entry_price0) if entry_price0 else 0.0
            buy_hedge_trades.append(
                {
                    "trade_id": current_trade_id or trade_count + 1,
                    "entry_date": str(entry_date),
                    "exit_date": str(date),
                    "entry_price": float(entry_price0),
                    "exit_price": float(price),
                    "avg_cost": float(avg_cost),
                    "total_shares": shares_to_hands(int(position)),
                    "adds": int(add_count),
                    "capital_used": float(position_cost),
                    "pnl": float(pnl),
                    "return_pct": float(pnl / position_cost) if position_cost else 0.0,
                    "avg_cost_delta_pct": float(cost_reduction),
                }
            )
            max_layers = max(max_layers, add_count)
            trade_add_records.append(add_count)
            trade_count += 1
            avg_cost_reduction_sum += float(cost_reduction)
            pending_exit = False
            pending_exit_reason = ""
        else:
            pending_exit = True
            pending_exit_reason = "样本结束未满足T+1"

    final_price = float(close.iloc[-1]) if len(close) else 0.0
    equity.iloc[-1] = cash + position * final_price
    result = _calc_stats(equity, trades)

    total_adds = sum(trade_add_records)
    avg_adds_per_trade = float(total_adds / trade_count) if trade_count else 0.0
    avg_cost_reduction = float(avg_cost_reduction_sum / trade_count) if trade_count else 0.0
    summary = {
        "enabled": bool(config),
        "trade_count": trade_count,
        "total_adds": int(total_adds),
        "avg_adds_per_trade": avg_adds_per_trade,
        "max_layers": int(max_layers),
        "max_capital_used": float(max_capital_used),
        "avg_cost_reduction_pct": avg_cost_reduction,
        "step_type": step_type,
        "step_pct": float(step_pct),
        "mode": mode,
        "start_position": int(start_position_hands),
        "increment_unit": int(increment_unit_hands),
        "max_adds": int(max_adds),
        "reference": reference,
        "max_capital_value": float(max_capital_value) if max_capital_value > 0 else None,
        "max_capital_ratio": float(max_capital_ratio) if max_capital_ratio > 0 else None,
        "max_capital_input": max_capital_input,
        "skipped_by_cash": int(skipped_by_cash),
        "skipped_by_limit": int(skipped_by_limit),
        "skipped_by_rule": int(skipped_by_rule),
    }

    result.buy_hedge_summary = summary
    result.buy_hedge_trades = buy_hedge_trades or []
    result.buy_hedge_events = events or []
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
    t1_guard = TPlusOneGuard()
    pending_exit = False
    pending_exit_reason = ""

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
                t1_guard.add(date, shares)
                if first_buy_date is None:
                    first_buy_date = date
                    first_buy_price = price

        equity.iloc[i] = cash + position * price

        target_reached = equity.iloc[i] >= initial_capital * (1 + target_return) and position > 0
        should_exit = position > 0 and (pending_exit or target_reached)
        exit_note = pending_exit_reason or "达到目标收益率，全部卖出"
        if should_exit:
            if not t1_guard.can_sell(date, position):
                pending_exit = True
                pending_exit_reason = exit_note
            else:
                t1_guard.consume(date, position)
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
                        note=exit_note,
                    )
                )
                position = 0
                first_buy_date = None
                first_buy_price = 0.0
                pending_exit = False
                pending_exit_reason = ""
                equity.iloc[i] = cash

    if position > 0:
        price = float(close.iloc[-1])
        exit_date = idx[-1]
        if t1_guard.can_sell(exit_date, position):
            t1_guard.consume(exit_date, position)
            value = position * price
            fee = value * fee_rate
            cash += value - fee
            ret = cash / initial_capital - 1.0
            if first_buy_date is None:
                first_buy_date = idx[0]
                first_buy_price = float(close.iloc[0])
            days_hold = (exit_date - first_buy_date).days
            trades.append(
                Trade(
                    entry_date=first_buy_date,
                    exit_date=exit_date,
                    entry_price=first_buy_price,
                    exit_price=price,
                    return_pct=float(ret),
                    holding_days=int(days_hold),
                    note="样本末尾全部卖出(定投)",
                )
            )
            position = 0
            first_buy_date = None
            first_buy_price = 0.0
            pending_exit = False
            pending_exit_reason = ""
        else:
            pending_exit = True
            pending_exit_reason = pending_exit_reason or "样本结束未满足T+1"

    final_price = float(close.iloc[-1]) if len(close) else 0.0
    equity.iloc[-1] = cash + position * final_price

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
    t1_guard = TPlusOneGuard()

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
                    t1_guard.add(date, shares)

        elif price >= up_threshold and position > 0:
            shares = int(single_grid_cash // price)
            if shares <= 0:
                shares = position
            shares = min(shares, position)
            sellable = t1_guard.consume(date, shares)
            if sellable > 0:
                value = sellable * price
                fee = value * fee_rate
                cash += value - fee
                position -= sellable
                last_trade_price = price
                trade_note = "网格卖出"
            else:
                trade_note = ""

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
