from __future__ import annotations

from dataclasses import dataclass, asdict
import math
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
    buy_hedge_hedge_trades: Optional[List[Dict[str, Any]]] = None
    buy_hedge_hedge_events: Optional[List[Dict[str, Any]]] = None

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
    high = df["high"]
    low = df["low"]
    idx = df.index

    cash = initial_capital
    position = 0
    entry_price = 0.0
    entry_idx: Optional[int] = None
    entry_investment_amount = 0.0
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
                entry_investment_amount = cost
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
                            investment_amount=entry_investment_amount if entry_investment_amount else None,
                            note=reason or f"固定{hold_days}天",
                        )
                    )
                    position = 0
                    entry_price = 0.0
                    entry_idx = None
                    pending_exit = False
                    pending_exit_reason = ""
                    entry_investment_amount = 0.0

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
                    investment_amount=entry_investment_amount if entry_investment_amount else None,
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
    entry_investment_amount = 0.0
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
                entry_investment_amount = cost
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
                            investment_amount=entry_investment_amount if entry_investment_amount else None,
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
                    investment_amount=entry_investment_amount if entry_investment_amount else None,
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
    single_limit_value = max(0.0, _to_float(config.get("singleInvestmentLimit"), 0.0))
    force_one_lot_first = bool(config.get("forceOneLotEntry"))
    allow_single_limit_override = bool(config.get("allowSingleLimitOverride"))
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
    position_cost = 0.0
    position_fee_total = 0.0

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
            base_budget_raw = next_amount_main if not force_stop else initial_investment
            base_budget = (
                base_budget_raw if force_one_lot_first and position == 0 else clamp_amount(base_budget_raw)
            )
            base_shares = floor_lot(int(base_budget // price))
            extra_shares = 0
            if not force_stop and pending_loss_steps_main > 0 and loss_step_lots > 0:
                extra_shares = int(loss_step_lots * LOT_SIZE * pending_loss_steps_main)
            total_limit_shares = floor_lot(int(max_limit // price)) if max_limit > 0 else None
            if force_one_lot_first:
                base_target_shares = LOT_SIZE
            else:
                base_target_shares = base_shares
                if total_limit_shares is not None:
                    base_target_shares = min(base_target_shares, total_limit_shares)
            additional_shares = extra_shares
            if total_limit_shares is not None:
                remaining_shares = max(0, total_limit_shares - base_target_shares)
                additional_shares = min(additional_shares, remaining_shares)
            addition_note = None
            if single_limit_value > 0 and additional_shares > 0:
                amount_with_fee = additional_shares * price * (1 + fee_rate)
                while additional_shares >= LOT_SIZE and amount_with_fee > single_limit_value:
                    additional_shares -= LOT_SIZE
                    amount_with_fee = additional_shares * price * (1 + fee_rate)
                if additional_shares < LOT_SIZE:
                    if additional_shares <= 0:
                        if allow_single_limit_override:
                            additional_shares = LOT_SIZE
                            addition_note = "单笔上限允许忽略"
                        else:
                            addition_note = "单笔上限限制"
                    else:
                        # Keep whatever is left above zero
                        pass
                    if additional_shares < LOT_SIZE and not allow_single_limit_override:
                        additional_shares = 0
            desired_shares = base_target_shares + additional_shares
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
                        position_cost += cost
                        position_fee_total += fee
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
                    position_cost = 0.0
                    position_fee_total = 0.0
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
    floating_value = position * final_price
    floating_pnl = floating_value - position_cost - position_fee_total
    result.dynamic_summary = {
        "initialInvestment": float(initial_investment),
        "lossStepAmount": float(loss_step_lots),
        "maxAddSteps": max_add_steps,
        "maxInvestmentLimit": float(max_limit),
        "singleInvestmentLimit": float(single_limit_value),
        "forceOneLotEntry": force_one_lot_first,
        "allowSingleLimitOverride": allow_single_limit_override,
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
        "currentFloatingPnl": float(floating_pnl),
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

    step_mode = str(config.get("step_mode") or "fixed").lower()
    step_type = str(config.get("step_type") or "percent").lower()
    step_pct = to_float(config.get("step_pct"), 0.0)
    step_abs = to_float(config.get("step_abs"), 0.0)
    step_rounding = str(config.get("step_rounding") or "round").lower()
    if step_type in {"percent", "percentage"} and step_pct > 1:
        step_pct /= 100.0
    step_pct = max(0.0, float(step_pct))
    step_abs = max(0.0, float(step_abs))

    def round_price(value: float) -> float:
        if step_rounding == "floor":
            return math.floor(value * 100) / 100
        if step_rounding == "ceil":
            return math.ceil(value * 100) / 100
        return round(value, 2)

    if step_type == "absolute" and step_abs > 0:
        step_abs = round_price(step_abs)

    start_position_shares_raw = max(0, to_int(config.get("start_position"), 0))
    increment_unit_shares_raw = max(0, to_int(config.get("increment_unit"), 0))
    start_position_shares = lot_align(start_position_shares_raw)
    increment_unit_shares = lot_align(increment_unit_shares_raw)
    start_position_hands = shares_to_hands(start_position_shares) or 0
    increment_unit_hands = shares_to_hands(increment_unit_shares) or 0

    growth_cfg = config.get("growth", {}) or {}
    growth_mode = str(growth_cfg.get("mode") or config.get("mode") or "equal").lower()

    position_cfg = config.get("position", {}) or {}
    position_mode = str(position_cfg.get("mode") or "fixed").lower()
    position_fixed_pct = to_float(position_cfg.get("fixed_pct"), 0.0)
    position_inc_start_pct = to_float(position_cfg.get("inc_start_pct"), 0.0)
    position_inc_step_pct = to_float(position_cfg.get("inc_step_pct"), 0.0)

    capital_cfg = config.get("capital", {}) or {}
    capital_enabled = capital_cfg.get("enabled")
    if capital_enabled is None:
        capital_enabled = True
    capital_mode = str(capital_cfg.get("mode") or "unlimited").lower()
    capital_fixed_amount = to_float(capital_cfg.get("fixed_amount"), 0.0)
    capital_fixed_percent = to_float(capital_cfg.get("fixed_percent"), 0.0)
    capital_increment_start = to_float(capital_cfg.get("increment_start"), 0.0)
    capital_increment_step = to_float(capital_cfg.get("increment_step"), 0.0)
    if not capital_enabled:
        capital_mode = "unlimited"
        capital_fixed_amount = 0.0
        capital_fixed_percent = 0.0
        capital_increment_start = 0.0
        capital_increment_step = 0.0

    limits_cfg = config.get("limits", {}) or {}
    limit_buy_price = to_float(limits_cfg.get("limit_buy_price"), 0.0)
    limit_sell_price = to_float(limits_cfg.get("limit_sell_price"), 0.0)
    min_price = to_float(limits_cfg.get("min_price"), 0.0)
    stop_adding_at_min = bool(limits_cfg.get("stop_adding_at_min"))

    base_cfg = config.get("base", {}) or {}
    base_initial_hands = max(0, to_int(base_cfg.get("initial_hands"), 0))
    base_initial_shares = lot_align(base_initial_hands * LOT_SIZE)
    base_reference_price = to_float(base_cfg.get("reference_price"), 0.0)
    base_reference_source = str(base_cfg.get("reference_source") or "first").lower()

    reference_mode = str(config.get("reference") or "first").lower()
    if reference_mode == "last":
        reference_mode = "first"

    hedge_cfg = config.get("hedge", {}) or {}
    hedge_enabled = bool(hedge_cfg.get("enabled"))
    hedge_mode = str(hedge_cfg.get("mode") or "full").lower()
    hedge_size_mode = str(hedge_cfg.get("size_mode") or "ratio").lower()
    hedge_size_ratio = to_float(hedge_cfg.get("size_ratio"), 0.0)
    if hedge_size_ratio > 1:
        hedge_size_ratio = hedge_size_ratio / 100.0
    hedge_size_hands = max(0, to_int(hedge_cfg.get("size_hands"), 0))
    hedge_size_amount = max(0.0, to_float(hedge_cfg.get("size_amount"), 0.0))
    hedge_exit_cfg = hedge_cfg.get("exit", {}) or {}
    hedge_exit_on_main = hedge_exit_cfg.get("on_main_exit")
    if hedge_exit_on_main is None:
        hedge_exit_on_main = True
    hedge_exit_on_profit = bool(hedge_exit_cfg.get("on_profit"))
    hedge_exit_profit_mode = str(hedge_exit_cfg.get("profit_mode") or "percent").lower()
    hedge_exit_profit_value = to_float(hedge_exit_cfg.get("profit_value"), 0.0)
    hedge_exit_on_loss = bool(hedge_exit_cfg.get("on_loss"))
    hedge_exit_loss_mode = str(hedge_exit_cfg.get("loss_mode") or "percent").lower()
    hedge_exit_loss_value = to_float(hedge_exit_cfg.get("loss_value"), 0.0)
    hedge_exit_on_reverse = bool(hedge_exit_cfg.get("on_reverse"))
    allow_repeat = bool(config.get("allow_repeat"))
    max_adds = to_int(config.get("max_adds"), 0)
    exit_cfg = config.get("exit", {}) or {}
    exit_enabled = exit_cfg.get("enabled")
    if exit_enabled is None:
        exit_enabled = True
    exit_mode = str(exit_cfg.get("mode") or "batch").lower()
    exit_batch_pct = to_float(exit_cfg.get("batch_pct"), 0.0)
    exit_batch_strategy = str(exit_cfg.get("batch_strategy") or "per_batch").lower()
    exit_batch_step_pct = to_float(exit_cfg.get("batch_step_pct"), 0.0)
    if not exit_enabled:
        exit_mode = "batch"
        exit_batch_pct = 0.0
        exit_batch_step_pct = 0.0

    entry_cfg = config.get("entry", {}) or {}
    entry_mode = str(entry_cfg.get("mode") or "none").lower()
    entry_ma_fast = max(0, to_int(entry_cfg.get("ma_fast"), 0))
    entry_ma_slow = max(0, to_int(entry_cfg.get("ma_slow"), 0))
    entry_ma_period = max(0, to_int(entry_cfg.get("ma_period"), 0))
    entry_progressive_count = max(1, to_int(entry_cfg.get("progressive_count"), 1))

    profit_cfg = config.get("profit", {}) or {}
    profit_mode = str(profit_cfg.get("mode") or "percent").lower()
    profit_target_pct = to_float(profit_cfg.get("target_pct"), 0.0)
    if profit_target_pct > 1:
        profit_target_pct /= 100.0
    profit_target_pct = max(0.0, profit_target_pct)
    profit_target_abs = max(0.0, to_float(profit_cfg.get("target_abs"), 0.0))
    profit_reference = str(profit_cfg.get("reference") or "overall").lower()
    profit_per_batch = bool(profit_cfg.get("per_batch"))

    reverse_cfg = config.get("reverse", {}) or {}
    reverse_enabled = bool(reverse_cfg.get("enabled"))
    reverse_indicator = str(reverse_cfg.get("indicator") or "rsi").lower()
    reverse_interval = max(1, to_int(reverse_cfg.get("interval"), 0)) or 14
    reverse_filter_mode = str(reverse_cfg.get("filter_mode") or "consecutive").lower()
    reverse_filter_value = max(1, to_int(reverse_cfg.get("filter_value"), 0))
    reverse_min_hits = max(1, to_int(reverse_cfg.get("min_hits"), 0))
    reverse_threshold = to_float(reverse_cfg.get("threshold"), 0.0)
    reverse_action = str(reverse_cfg.get("action") or "exit").lower()
    reverse_profit_type = str(reverse_cfg.get("profit_type") or "percent").lower()
    reverse_profit_value = to_float(reverse_cfg.get("profit_value"), 0.0)

    step_auto_cfg = config.get("step_auto", {}) or {}
    step_auto_method = str(step_auto_cfg.get("method") or "atr").lower()
    step_auto_atr_period = max(1, to_int(step_auto_cfg.get("atr_period"), 0)) or 14
    step_auto_atr_multiplier = to_float(step_auto_cfg.get("atr_multiplier"), 1.0) or 1.0
    step_auto_avg_len = max(1, to_int(step_auto_cfg.get("avg_range_length"), 0)) or 10
    step_auto_avg_multiplier = to_float(step_auto_cfg.get("avg_range_multiplier"), 1.0) or 1.0
    step_auto_std_period = max(1, to_int(step_auto_cfg.get("std_period"), 0)) or 20
    step_auto_std_multiplier = to_float(step_auto_cfg.get("std_multiplier"), 1.0) or 1.0
    step_auto_ma_fast = max(1, to_int(step_auto_cfg.get("ma_fast"), 0)) or 5
    step_auto_ma_slow = max(1, to_int(step_auto_cfg.get("ma_slow"), 0)) or 20
    step_auto_ma_gap_pct = to_float(step_auto_cfg.get("ma_gap_pct"), 0.0)

    close_series = close.astype(float)
    high_series = high.astype(float)
    low_series = low.astype(float)

    def compute_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(0.0)

    def compute_kdj(close_s: pd.Series, high_s: pd.Series, low_s: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
        low_n = low_s.rolling(period).min()
        high_n = high_s.rolling(period).max()
        rsv = ((close_s - low_n) / (high_n - low_n).replace(0, np.nan)) * 100.0
        rsv = rsv.fillna(50.0)
        k_vals = []
        d_vals = []
        k_prev = 50.0
        d_prev = 50.0
        for val in rsv:
            k_prev = (2.0 / 3.0) * k_prev + (1.0 / 3.0) * float(val)
            d_prev = (2.0 / 3.0) * d_prev + (1.0 / 3.0) * k_prev
            k_vals.append(k_prev)
            d_vals.append(d_prev)
        return pd.Series(k_vals, index=close_s.index), pd.Series(d_vals, index=close_s.index)

    entry_fast_ma = close_series.rolling(entry_ma_fast).mean() if entry_ma_fast > 0 else None
    entry_slow_ma = close_series.rolling(entry_ma_slow).mean() if entry_ma_slow > 0 else None
    entry_single_ma = close_series.rolling(entry_ma_period).mean() if entry_ma_period > 0 else None
    entry_base = pd.Series(False, index=idx)
    if entry_mode != "none":
        if entry_fast_ma is not None and entry_slow_ma is not None:
            entry_base = entry_fast_ma > entry_slow_ma
        elif entry_single_ma is not None:
            entry_base = close_series > entry_single_ma

    auto_step_abs_series = None
    auto_step_pct_series = None
    if step_mode == "auto":
        if step_auto_method == "atr":
            prev_close = close_series.shift(1)
            tr = pd.concat(
                [
                    (high_series - low_series).abs(),
                    (high_series - prev_close).abs(),
                    (low_series - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr = tr.rolling(step_auto_atr_period).mean()
            auto_step_abs_series = atr * step_auto_atr_multiplier
        elif step_auto_method == "avg_range":
            avg_range = (high_series - low_series).abs().rolling(step_auto_avg_len).mean()
            auto_step_abs_series = avg_range * step_auto_avg_multiplier
        elif step_auto_method == "stddev":
            stddev = close_series.rolling(step_auto_std_period).std()
            auto_step_abs_series = stddev * step_auto_std_multiplier
        elif step_auto_method == "ma_gap":
            ma_fast = close_series.rolling(step_auto_ma_fast).mean()
            ma_slow = close_series.rolling(step_auto_ma_slow).mean()
            gap_pct = abs(step_auto_ma_gap_pct) / 100.0
            auto_step_pct_series = (ma_fast - ma_slow).abs() / ma_slow.replace(0, np.nan)
            auto_step_pct_series = auto_step_pct_series.fillna(0.0).clip(lower=gap_pct)

    reverse_base = pd.Series(False, index=idx)
    if reverse_enabled:
        if reverse_indicator == "rsi":
            rsi = compute_rsi(close_series, reverse_interval)
            if reverse_threshold > 50:
                reverse_base = rsi >= reverse_threshold
            else:
                reverse_base = rsi <= reverse_threshold
        elif reverse_indicator == "macd":
            ema_fast = close_series.ewm(span=12, adjust=False).mean()
            ema_slow = close_series.ewm(span=26, adjust=False).mean()
            dif = ema_fast - ema_slow
            dea = dif.ewm(span=9, adjust=False).mean()
            reverse_base = (dif > dea) & (dif.shift(1) <= dea.shift(1))
        elif reverse_indicator == "kdj":
            k_val, d_val = compute_kdj(close_series, high_series, low_series, reverse_interval)
            reverse_base = (k_val > d_val) & (k_val.shift(1) <= d_val.shift(1))
        elif reverse_indicator == "ma_turn":
            ma_val = close_series.rolling(reverse_interval).mean()
            reverse_base = (ma_val > ma_val.shift(1)) & (ma_val.shift(1) <= ma_val.shift(2))
        else:
            rolling_low = low_series.rolling(reverse_interval).min()
            reverse_base = (low_series <= rolling_low) & (close_series > close_series.shift(1))

    cash = float(initial_capital)
    position = 0
    position_cost = 0.0
    reference_price: Optional[float] = None
    if base_reference_source == "custom" and base_reference_price > 0:
        reference_price = float(base_reference_price)
    grid_index = 0

    trades: List[Trade] = []
    buy_hedge_trades: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    hedge_trades: List[Dict[str, Any]] = []
    hedge_events: List[Dict[str, Any]] = []

    trade_count = 0
    total_adds = 0
    max_layers = 0
    avg_cost_reduction_sum = 0.0
    max_capital_used = 0.0
    skipped_by_cash = 0
    skipped_by_limit = 0
    skipped_by_rule = 0
    current_trade_id = 0
    current_hedge_trade_id = 0
    hedge_trade_count = 0

    cycle_entry_idx: Optional[int] = None
    cycle_entry_price = 0.0
    cycle_buy_shares = 0
    cycle_cost_total = 0.0
    cycle_sell_total = 0.0
    cycle_add_count = 0
    cycle_buy_count = 0
    cycle_max_shares = 0
    cycle_max_cost = 0.0

    t1_guard = TPlusOneGuard()
    entry_ready = entry_mode == "none"
    entry_hits = 0
    trading_active = entry_ready
    base_init_done = base_initial_shares <= 0
    last_buy_price = 0.0
    profit_override: Optional[Tuple[str, float]] = None
    reverse_hits = 0
    reverse_window: List[bool] = []
    hedge_position = 0
    hedge_avg_price = 0.0
    hedge_cash = 0.0
    hedge_entry_idx: Optional[int] = None
    hedge_entry_price = 0.0
    hedge_entry_value = 0.0
    hedge_max_shares = 0
    hedge_paused = False

    def resolve_position_pct(layer_index: int) -> float:
        if position_mode == "increment":
            pct = position_inc_start_pct + layer_index * position_inc_step_pct
        else:
            pct = position_fixed_pct
        return max(0.0, float(pct)) / 100.0

    def resolve_capital_base(layer_index: int) -> float:
        if capital_mode == "fixed":
            if capital_fixed_amount > 0:
                return float(capital_fixed_amount)
            if capital_fixed_percent > 0:
                return float(initial_capital) * (capital_fixed_percent / 100.0)
        elif capital_mode == "increment":
            pct = capital_increment_start + layer_index * capital_increment_step
            if pct > 0:
                return float(initial_capital) * (pct / 100.0)
        return float(initial_capital)

    def quantity_for_add(layer_index: int) -> int:
        if growth_mode == "increment":
            qty = start_position_shares + layer_index * increment_unit_shares
        elif growth_mode == "double":
            qty = int(round(start_position_shares * (2 ** layer_index)))
        else:
            qty = start_position_shares
        return lot_align(int(qty))

    def desired_shares(layer_index: int, price: float) -> int:
        pct = resolve_position_pct(layer_index)
        if pct > 0:
            capital_base = resolve_capital_base(layer_index)
            target_amount = capital_base * pct
            return lot_align(int(target_amount // price))
        return quantity_for_add(layer_index)

    def exit_shares_for_sell(price: float) -> int:
        if exit_mode == "single":
            return int(position)
        pct = 0.0
        if exit_batch_strategy == "per_step" and exit_batch_step_pct > 0:
            pct = exit_batch_step_pct
        elif exit_batch_pct > 0:
            pct = exit_batch_pct
        if pct <= 0:
            return desired_shares(max(cycle_buy_count - 1, 0), price)
        return lot_align(int(position * (pct / 100.0)))

    def compute_triggers(
        base_price: float,
        grid_idx: int,
        step_type_value: str,
        step_pct_value: float,
        step_abs_value: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        if step_type_value == "absolute":
            if step_abs_value <= 0:
                return None, None
            if reference_mode == "last":
                up = base_price + step_abs_value
                down = base_price - step_abs_value
            else:
                up = base_price + step_abs_value * (grid_idx + 1)
                down = base_price + step_abs_value * (grid_idx - 1)
            return round_price(up), round_price(down)
        if step_pct_value <= 0:
            return None, None
        if reference_mode == "last":
            up = base_price * (1 + step_pct_value)
            down = base_price * (1 - step_pct_value)
        else:
            up = base_price * (1 + step_pct_value * (grid_idx + 1))
            down = base_price * (1 + step_pct_value * (grid_idx - 1))
        return up, down

    def resolve_step_values(index: int) -> Tuple[str, float, float]:
        if step_mode != "auto":
            return step_type, step_pct, step_abs
        if auto_step_pct_series is not None:
            val = float(auto_step_pct_series.iloc[index])
            if not np.isfinite(val) or val <= 0:
                return "percent", 0.0, 0.0
            return "percent", max(0.0, val), 0.0
        if auto_step_abs_series is not None:
            val = float(auto_step_abs_series.iloc[index])
            if not np.isfinite(val) or val <= 0:
                return "absolute", 0.0, 0.0
            return "absolute", 0.0, round_price(val)
        return step_type, step_pct, step_abs

    def append_event(event: Dict[str, Any]) -> None:
        payload = event.copy()
        payload["trade_id"] = current_trade_id if current_trade_id else None
        payload["date"] = str(event.get("date"))
        payload["type"] = event.get("type", "record")
        if "shares" in payload and payload["shares"] is not None:
            payload["shares"] = shares_to_hands(int(payload["shares"]))
        if "total_shares" in payload and payload["total_shares"] is not None:
            payload["total_shares"] = shares_to_hands(int(payload["total_shares"]))
        events.append(payload)

    def append_hedge_event(event: Dict[str, Any]) -> None:
        payload = event.copy()
        payload["trade_id"] = current_hedge_trade_id if current_hedge_trade_id else None
        payload["date"] = str(event.get("date"))
        payload["type"] = event.get("type", "record")
        if "shares" in payload and payload["shares"] is not None:
            payload["shares"] = shares_to_hands(int(payload["shares"]))
        if "total_shares" in payload and payload["total_shares"] is not None:
            payload["total_shares"] = shares_to_hands(int(payload["total_shares"]))
        hedge_events.append(payload)

    def resolve_hedge_target(price: float) -> int:
        if price <= 0 or position <= 0:
            return 0
        target_shares = 0.0
        if hedge_size_mode == "ratio":
            ratio = hedge_size_ratio
            if ratio > 1:
                ratio = ratio / 100.0
            ratio = max(0.0, ratio)
            target_shares = position * ratio
        elif hedge_size_mode == "fixed_hands":
            target_shares = hedge_size_hands * LOT_SIZE
        elif hedge_size_mode == "fixed_amount":
            if hedge_size_amount > 0:
                target_shares = hedge_size_amount // price
        return lot_align(int(target_shares))

    def hedge_profit_trigger() -> Optional[float]:
        if hedge_avg_price <= 0 or hedge_position <= 0:
            return None
        value = hedge_exit_profit_value
        if hedge_exit_profit_mode == "percent" and value > 1:
            value /= 100.0
        if value <= 0:
            return None
        if hedge_exit_profit_mode == "percent":
            return hedge_avg_price * (1 - value)
        return hedge_avg_price - value

    def hedge_loss_trigger() -> Optional[float]:
        if hedge_avg_price <= 0 or hedge_position <= 0:
            return None
        value = hedge_exit_loss_value
        if hedge_exit_loss_mode == "percent" and value > 1:
            value /= 100.0
        if value <= 0:
            return None
        if hedge_exit_loss_mode == "percent":
            return hedge_avg_price * (1 + value)
        return hedge_avg_price + value

    def hedge_open(exec_date: pd.Timestamp, exec_price: float, shares: int, note: str, trigger_price: Optional[float]) -> None:
        nonlocal hedge_position, hedge_avg_price, hedge_cash, hedge_entry_idx
        nonlocal hedge_entry_price, hedge_entry_value, hedge_max_shares, current_hedge_trade_id
        if shares <= 0:
            return
        proceeds = shares * exec_price
        fee = proceeds * fee_rate
        hedge_cash += proceeds - fee
        total_shares = hedge_position + shares
        hedge_avg_price = (
            (hedge_avg_price * hedge_position + exec_price * shares) / total_shares if hedge_position > 0 else exec_price
        )
        hedge_position = total_shares
        if hedge_entry_idx is None:
            hedge_entry_idx = i
            current_hedge_trade_id = hedge_trade_count + 1
        hedge_entry_price = hedge_avg_price
        hedge_entry_value = max(hedge_entry_value, hedge_position * hedge_entry_price)
        hedge_max_shares = max(hedge_max_shares, hedge_position)
        append_hedge_event(
            {
                "date": exec_date,
                "price": float(exec_price),
                "shares": int(shares),
                "total_shares": int(hedge_position),
                "avg_price": float(hedge_avg_price),
                "type": "hedge_open" if hedge_position == shares else "hedge_add",
                "note": note,
                "trigger_price": float(trigger_price) if trigger_price is not None else None,
            }
        )

    def hedge_cover(
        exec_date: pd.Timestamp, exec_price: float, shares: int, note: str, trigger_price: Optional[float]
    ) -> None:
        nonlocal hedge_position, hedge_avg_price, hedge_cash, hedge_entry_idx
        nonlocal hedge_entry_price, hedge_entry_value, hedge_max_shares, current_hedge_trade_id, hedge_trade_count, cash
        if shares <= 0 or hedge_position <= 0:
            return
        cover_shares = min(hedge_position, shares)
        cost = cover_shares * exec_price
        fee = cost * fee_rate
        hedge_cash -= cost + fee
        hedge_position -= cover_shares
        append_hedge_event(
            {
                "date": exec_date,
                "price": float(exec_price),
                "shares": int(cover_shares),
                "total_shares": int(hedge_position),
                "avg_price": float(hedge_avg_price) if hedge_avg_price > 0 else None,
                "type": "hedge_close" if hedge_position == 0 else "hedge_reduce",
                "note": note,
                "trigger_price": float(trigger_price) if trigger_price is not None else None,
            }
        )
        if hedge_position == 0:
            entry_date = idx[hedge_entry_idx] if hedge_entry_idx is not None else exec_date
            entry_price = hedge_entry_price if hedge_entry_price > 0 else exec_price
            entry_value = hedge_entry_value if hedge_entry_value > 0 else entry_price * cover_shares
            pnl = hedge_cash
            return_pct = pnl / entry_value if entry_value > 0 else 0.0
            hedge_trades.append(
                {
                    "trade_id": current_hedge_trade_id or hedge_trade_count + 1,
                    "entry_date": str(entry_date),
                    "exit_date": str(exec_date),
                    "entry_price": float(entry_price),
                    "exit_price": float(exec_price),
                    "total_shares": shares_to_hands(int(hedge_max_shares)) if hedge_max_shares > 0 else 0,
                    "pnl": float(pnl),
                    "return_pct": float(return_pct),
                    "mode": hedge_mode,
                    "size_mode": hedge_size_mode,
                    "size_ratio": float(hedge_size_ratio),
                    "size_hands": int(hedge_size_hands),
                    "size_amount": float(hedge_size_amount),
                    "note": note,
                }
            )
            hedge_trade_count += 1
            cash += hedge_cash
            hedge_cash = 0.0
            hedge_avg_price = 0.0
            hedge_entry_idx = None
            hedge_entry_price = 0.0
            hedge_entry_value = 0.0
            hedge_max_shares = 0
            current_hedge_trade_id = 0

    for i, date in enumerate(idx):
        raw_close = close.iloc[i]
        raw_high = high.iloc[i]
        raw_low = low.iloc[i]
        try:
            close_price = float(raw_close)
            high_price = float(raw_high)
            low_price = float(raw_low)
        except (TypeError, ValueError):
            close_price = float("nan")
            high_price = float("nan")
            low_price = float("nan")
        if not np.isfinite(close_price) or close_price <= 0:
            prev = equity.iloc[i - 1] if i > 0 else cash
            equity.iloc[i] = prev
            continue
        if not np.isfinite(high_price) or high_price <= 0:
            high_price = close_price
        if not np.isfinite(low_price) or low_price <= 0:
            low_price = close_price

        main_exit = False
        main_exit_price: Optional[float] = None
        main_exit_reason: Optional[str] = None

        entry_signal = False
        if not entry_ready:
            if entry_mode == "ma":
                prev = bool(entry_base.iloc[i - 1]) if i > 0 else False
                entry_signal = bool(entry_base.iloc[i]) and not prev
            elif entry_mode == "ma_progressive":
                if bool(entry_base.iloc[i]):
                    entry_hits += 1
                else:
                    entry_hits = 0
                entry_signal = entry_hits >= entry_progressive_count
            else:
                entry_signal = bool(entry_base.iloc[i])
            if entry_signal:
                entry_ready = True
                trading_active = True
        if entry_ready and reference_price is None:
            reference_price = close_price
            grid_index = 0

        if entry_ready and not base_init_done and base_initial_shares > 0:
            if limit_buy_price > 0 and close_price > limit_buy_price:
                skipped_by_limit += 1
                append_event(
                    {
                        "date": date,
                        "price": float(close_price),
                        "shares": 0,
                        "total_shares": int(position),
                        "avg_cost": position_cost / position if position > 0 else None,
                        "type": "skip",
                        "layer": grid_index,
                        "note": "超过限买价",
                        "trigger_price": None,
                    }
                )
            elif stop_adding_at_min and min_price > 0 and close_price <= min_price:
                append_event(
                    {
                        "date": date,
                        "price": float(close_price),
                        "shares": 0,
                        "total_shares": int(position),
                        "avg_cost": position_cost / position if position > 0 else None,
                        "type": "skip",
                        "layer": grid_index,
                        "note": "达到最低价",
                        "trigger_price": None,
                    }
                )
            else:
                affordable = (
                    int(cash / (close_price * (1 + fee_rate))) if fee_rate >= 0 else int(cash // close_price)
                )
                affordable = lot_align(affordable)
                shares = min(base_initial_shares, affordable)
                if shares > 0:
                    cost = shares * close_price
                    fee = cost * fee_rate
                    cash -= cost + fee
                    position += shares
                    position_cost += cost + fee
                    t1_guard.add(date, shares)
                    if reference_mode == "last":
                        reference_price = close_price
                    current_trade_id = trade_count + 1
                    cycle_entry_idx = i
                    cycle_entry_price = close_price
                    cycle_buy_shares = shares
                    cycle_cost_total = cost + fee
                    cycle_sell_total = 0.0
                    cycle_add_count = 0
                    cycle_buy_count = 1
                    cycle_max_shares = max(cycle_max_shares, position)
                    cycle_max_cost = max(cycle_max_cost, position_cost)
                    max_capital_used = max(max_capital_used, position_cost)
                    append_event(
                        {
                            "date": date,
                            "price": float(close_price),
                            "shares": int(shares),
                            "total_shares": int(position),
                            "avg_cost": float(position_cost / position),
                            "type": "buy",
                            "layer": grid_index,
                            "note": "底仓",
                            "trigger_price": None,
                        }
                    )
                    last_buy_price = float(close_price)
                else:
                    skipped_by_cash += 1
                    append_event(
                        {
                            "date": date,
                            "price": float(close_price),
                            "shares": 0,
                            "total_shares": int(position),
                            "avg_cost": position_cost / position if position > 0 else None,
                            "type": "skip",
                            "layer": grid_index,
                            "note": "现金不足",
                            "trigger_price": None,
                        }
                    )
            base_init_done = True

        step_type_value, step_pct_value, step_abs_value = resolve_step_values(i)
        up_trigger = None
        down_trigger = None
        if reference_price is not None:
            up_trigger, down_trigger = compute_triggers(
                reference_price, grid_index, step_type_value, step_pct_value, step_abs_value
            )

        reverse_signal = False
        if reverse_enabled:
            base_signal = bool(reverse_base.iloc[i])
            if reverse_filter_mode == "at_least":
                reverse_window.append(base_signal)
                if len(reverse_window) > reverse_filter_value:
                    reverse_window.pop(0)
                reverse_signal = sum(reverse_window) >= max(reverse_min_hits, 1)
            else:
                reverse_hits = reverse_hits + 1 if base_signal else 0
                reverse_signal = reverse_hits >= reverse_filter_value
            if reverse_signal and reverse_action == "adjust" and position > 0:
                if reverse_profit_value > 0:
                    value = reverse_profit_value
                    if reverse_profit_type == "percent" and value > 1:
                        value /= 100.0
                    profit_override = (reverse_profit_type, value)
                append_event(
                    {
                        "date": date,
                        "price": float(close_price),
                        "shares": 0,
                        "total_shares": int(position),
                        "avg_cost": position_cost / position if position > 0 else None,
                        "type": "reverse",
                        "layer": grid_index,
                        "note": "反转信号触发止盈调整",
                        "trigger_price": None,
                    }
                )

        profit_trigger = None
        profit_note = None
        if position > 0:
            base_price = 0.0
            if profit_reference == "last" or profit_per_batch:
                base_price = last_buy_price or (position_cost / position if position > 0 else 0.0)
            else:
                base_price = position_cost / position if position > 0 else 0.0
            active_mode = profit_mode
            active_value = profit_target_pct if profit_mode == "percent" else profit_target_abs
            if profit_override:
                active_mode, active_value = profit_override
            if base_price > 0 and active_value > 0:
                target = base_price * (1 + active_value) if active_mode == "percent" else base_price + active_value
                if high_price >= target:
                    profit_trigger = target
                    profit_note = "止盈触发"

        action = None
        trigger_price = None
        action_note = None
        if reverse_signal and reverse_action == "exit" and position > 0:
            action = "sell"
            trigger_price = close_price
            action_note = "反转信号"
        elif profit_trigger is not None:
            action = "sell"
            trigger_price = profit_trigger
            action_note = profit_note
        elif up_trigger is not None and position > 0 and high_price >= up_trigger:
            action = "sell"
            trigger_price = up_trigger
        elif down_trigger is not None and trading_active and low_price <= down_trigger:
            action = "buy"
            trigger_price = down_trigger

        if action == "buy":
            exec_price = float(trigger_price) if trigger_price is not None else close_price
            if max_adds > 0 and cycle_add_count >= max_adds:
                skipped_by_rule += 1
                append_event(
                    {
                        "date": date,
                        "price": float(exec_price),
                        "shares": 0,
                        "total_shares": int(position),
                        "avg_cost": position_cost / position if position > 0 else None,
                        "type": "skip",
                        "layer": grid_index,
                        "note": "已达到最大加仓次数",
                        "trigger_price": float(trigger_price) if trigger_price is not None else None,
                    }
                )
            elif limit_buy_price > 0 and exec_price > limit_buy_price:
                skipped_by_limit += 1
                append_event(
                    {
                        "date": date,
                        "price": float(exec_price),
                        "shares": 0,
                        "total_shares": int(position),
                        "avg_cost": position_cost / position if position > 0 else None,
                        "type": "skip",
                        "layer": grid_index,
                        "note": "超过限买价",
                        "trigger_price": float(trigger_price) if trigger_price is not None else None,
                    }
                )
            elif stop_adding_at_min and min_price > 0 and low_price <= min_price:
                skipped_by_rule += 1
                append_event(
                    {
                        "date": date,
                        "price": float(exec_price),
                        "shares": 0,
                        "total_shares": int(position),
                        "avg_cost": position_cost / position if position > 0 else None,
                        "type": "skip",
                        "layer": grid_index,
                        "note": "达到最低价",
                        "trigger_price": float(trigger_price) if trigger_price is not None else None,
                    }
                )
            else:
                layer_index = cycle_buy_count
                desired = desired_shares(layer_index, exec_price)
                if desired <= 0:
                    skipped_by_rule += 1
                    append_event(
                        {
                            "date": date,
                            "price": float(exec_price),
                            "shares": 0,
                            "total_shares": int(position),
                            "avg_cost": position_cost / position if position > 0 else None,
                            "type": "skip",
                            "layer": grid_index,
                            "note": "仓位不足（不足 1 手）",
                            "trigger_price": float(trigger_price) if trigger_price is not None else None,
                        }
                    )
                else:
                    affordable = (
                        int(cash / (exec_price * (1 + fee_rate))) if fee_rate >= 0 else int(cash // exec_price)
                    )
                    affordable = lot_align(affordable)
                    shares = min(desired, affordable)
                    if shares <= 0:
                        skipped_by_cash += 1
                        append_event(
                            {
                                "date": date,
                                "price": float(exec_price),
                                "shares": 0,
                                "total_shares": int(position),
                                "avg_cost": position_cost / position if position > 0 else None,
                                "type": "skip",
                                "layer": grid_index,
                                "note": "现金不足",
                                "trigger_price": float(trigger_price) if trigger_price is not None else None,
                            }
                        )
                    else:
                        cost = shares * exec_price
                        fee = cost * fee_rate
                        cash -= cost + fee
                        position += shares
                        position_cost += cost + fee
                        t1_guard.add(date, shares)
                        last_buy_price = float(exec_price)
                        if cycle_buy_shares == 0:
                            current_trade_id = trade_count + 1
                            cycle_entry_idx = i
                            cycle_entry_price = exec_price
                            cycle_buy_shares = shares
                            cycle_cost_total = cost + fee
                            cycle_sell_total = 0.0
                            cycle_add_count = 0
                            cycle_buy_count = 1
                        else:
                            cycle_buy_shares += shares
                            cycle_cost_total += cost + fee
                            cycle_add_count += 1
                            cycle_buy_count += 1
                        cycle_max_shares = max(cycle_max_shares, position)
                        cycle_max_cost = max(cycle_max_cost, position_cost)
                        max_capital_used = max(max_capital_used, position_cost)
                        append_event(
                            {
                                "date": date,
                                "price": float(exec_price),
                                "shares": int(shares),
                                "total_shares": int(position),
                                "avg_cost": float(position_cost / position),
                                "type": "buy",
                                "layer": grid_index - 1 if reference_mode != "last" else grid_index,
                                "note": None,
                                "trigger_price": float(trigger_price) if trigger_price is not None else None,
                            }
                        )
                        if reference_mode == "last":
                            reference_price = exec_price
                        else:
                            grid_index -= 1

        elif action == "sell":
            exec_price = float(trigger_price) if trigger_price is not None else close_price
            if limit_sell_price > 0 and exec_price < limit_sell_price:
                skipped_by_limit += 1
                append_event(
                    {
                        "date": date,
                        "price": float(exec_price),
                        "shares": 0,
                        "total_shares": int(position),
                        "avg_cost": position_cost / position if position > 0 else None,
                        "type": "skip",
                        "layer": grid_index,
                        "note": "低于限平价",
                        "trigger_price": float(trigger_price) if trigger_price is not None else None,
                    }
                )
            else:
                desired = exit_shares_for_sell(exec_price)
                if desired <= 0:
                    skipped_by_rule += 1
                    append_event(
                        {
                            "date": date,
                            "price": float(exec_price),
                            "shares": 0,
                            "total_shares": int(position),
                            "avg_cost": position_cost / position if position > 0 else None,
                            "type": "skip",
                            "layer": grid_index,
                            "note": "卖出比例不足（不足 1 手）",
                            "trigger_price": float(trigger_price) if trigger_price is not None else None,
                        }
                    )
                else:
                    sellable = min(position, desired)
                    if exit_mode == "single":
                        if not t1_guard.can_sell(date, position):
                            skipped_by_rule += 1
                            append_event(
                                {
                                    "date": date,
                                    "price": float(exec_price),
                                    "shares": 0,
                                    "total_shares": int(position),
                                    "avg_cost": position_cost / position if position > 0 else None,
                                    "type": "skip",
                                    "layer": grid_index,
                                    "note": "T+1限制",
                                    "trigger_price": float(trigger_price) if trigger_price is not None else None,
                                }
                            )
                            sellable = 0
                    if sellable <= 0 or t1_guard.available(date) < sellable:
                        if sellable > 0:
                            sellable = min(sellable, t1_guard.available(date))
                        if sellable <= 0:
                            skipped_by_rule += 1
                            append_event(
                                {
                                    "date": date,
                                    "price": float(exec_price),
                                    "shares": 0,
                                    "total_shares": int(position),
                                    "avg_cost": position_cost / position if position > 0 else None,
                                    "type": "skip",
                                    "layer": grid_index,
                                    "note": "T+1限制",
                                    "trigger_price": float(trigger_price) if trigger_price is not None else None,
                                }
                            )
                    if sellable > 0:
                        avg_cost = position_cost / position if position > 0 else 0.0
                        value = sellable * exec_price
                        fee = value * fee_rate
                        cash += value - fee
                        position -= sellable
                        position_cost = max(0.0, position_cost - avg_cost * sellable)
                        cycle_sell_total += value - fee
                        t1_guard.consume(date, sellable)
                        append_event(
                            {
                                "date": date,
                                "price": float(exec_price),
                                "shares": int(sellable),
                                "total_shares": int(position),
                                "avg_cost": float(position_cost / position) if position > 0 else None,
                                "type": "sell",
                                "layer": grid_index + 1 if reference_mode != "last" else grid_index,
                                "note": action_note,
                                "trigger_price": float(trigger_price) if trigger_price is not None else None,
                            }
                        )
                        if reference_mode == "last":
                            reference_price = exec_price
                        else:
                            grid_index += 1
                        if position == 0 and cycle_buy_shares > 0 and cycle_entry_idx is not None:
                            main_exit = True
                            main_exit_price = exec_price
                            main_exit_reason = action_note or "主仓清仓"
                            entry_date = idx[cycle_entry_idx]
                            pnl = cycle_sell_total - cycle_cost_total
                            avg_cost_cycle = cycle_cost_total / cycle_buy_shares if cycle_buy_shares > 0 else 0.0
                            cost_reduction = (
                                (cycle_entry_price - avg_cost_cycle) / cycle_entry_price if cycle_entry_price else 0.0
                            )
                            days_hold = (date - entry_date).days
                            trades.append(
                                Trade(
                                    entry_date=entry_date,
                                    exit_date=date,
                                    entry_price=float(cycle_entry_price),
                                    exit_price=float(exec_price),
                                    return_pct=float(pnl / cycle_cost_total) if cycle_cost_total else 0.0,
                                    holding_days=int(days_hold),
                                    note="买入对冲-区间离场",
                                    investment_amount=float(cycle_max_cost) if cycle_max_cost else None,
                                )
                            )
                            buy_hedge_trades.append(
                                {
                                    "trade_id": current_trade_id or trade_count + 1,
                                    "entry_date": str(entry_date),
                                    "exit_date": str(date),
                                    "entry_price": float(cycle_entry_price),
                                    "exit_price": float(exec_price),
                                    "avg_cost": float(avg_cost_cycle),
                                    "total_shares": shares_to_hands(int(cycle_max_shares)),
                                    "adds": int(cycle_add_count),
                                    "capital_used": float(cycle_max_cost),
                                    "pnl": float(pnl),
                                    "return_pct": float(pnl / cycle_cost_total) if cycle_cost_total else 0.0,
                                    "avg_cost_delta_pct": float(cost_reduction),
                                    "hedge_active": hedge_enabled,
                                    "hedge_mode": hedge_mode,
                                    "allow_repeat": allow_repeat,
                                }
                            )
                            trade_count += 1
                            total_adds += cycle_add_count
                            max_layers = max(max_layers, cycle_add_count)
                            avg_cost_reduction_sum += float(cost_reduction)
                            current_trade_id = 0
                            cycle_entry_idx = None
                            cycle_entry_price = 0.0
                            cycle_buy_shares = 0
                            cycle_cost_total = 0.0
                            cycle_sell_total = 0.0
                            cycle_add_count = 0
                            cycle_buy_count = 0
                            cycle_max_shares = 0
                            cycle_max_cost = 0.0
                            last_buy_price = 0.0
                            profit_override = None
                            reverse_hits = 0
                            reverse_window = []
                            if allow_repeat:
                                entry_ready = entry_mode == "none"
                                trading_active = entry_ready
                                entry_hits = 0
                                base_init_done = base_initial_shares <= 0
                                if base_reference_source == "custom" and base_reference_price > 0:
                                    reference_price = float(base_reference_price)
                                else:
                                    reference_price = None
                                grid_index = 0
                                hedge_paused = False
                            else:
                                trading_active = False

        if hedge_enabled:
            hedge_exit_price = None
            hedge_exit_note = None
            if hedge_position > 0:
                if hedge_exit_on_main and main_exit:
                    hedge_exit_price = main_exit_price or close_price
                    hedge_exit_note = main_exit_reason or "主仓平仓"
                else:
                    loss_target = hedge_loss_trigger()
                    if hedge_exit_on_loss and loss_target is not None and high_price >= loss_target:
                        hedge_exit_price = loss_target
                        hedge_exit_note = "对冲止损"
                    profit_target = hedge_profit_trigger()
                    if hedge_exit_price is None and hedge_exit_on_profit and profit_target is not None and low_price <= profit_target:
                        hedge_exit_price = profit_target
                        hedge_exit_note = "对冲止盈"
                    if hedge_exit_price is None and hedge_exit_on_reverse and reverse_signal:
                        hedge_exit_price = close_price
                        hedge_exit_note = "反转信号"
            if hedge_exit_price is not None and hedge_position > 0:
                hedge_cover(date, float(hedge_exit_price), hedge_position, hedge_exit_note or "对冲结束", hedge_exit_price)
                hedge_paused = True
            if not hedge_paused and position > 0:
                target_hedge = resolve_hedge_target(close_price)
                if hedge_mode == "weak":
                    if hedge_position == 0 and target_hedge > 0:
                        hedge_open(date, close_price, target_hedge, "固定对冲", None)
                else:
                    if target_hedge > hedge_position:
                        hedge_open(date, close_price, target_hedge - hedge_position, "跟随对冲", None)
                    elif target_hedge < hedge_position:
                        hedge_cover(date, close_price, hedge_position - target_hedge, "对冲回补", None)

        equity.iloc[i] = cash + position * close_price + hedge_cash - hedge_position * close_price

    if position > 0 and cycle_entry_idx is not None:
        date = idx[-1]
        price = float(close.iloc[-1])
        if t1_guard.can_sell(date, position):
            sell_shares = position
            avg_cost = position_cost / position if position > 0 else 0.0
            value = sell_shares * price
            fee = value * fee_rate
            cash += value - fee
            cycle_sell_total += value - fee
            t1_guard.consume(date, sell_shares)
            position = 0
            position_cost = 0.0
            entry_date = idx[cycle_entry_idx]
            pnl = cycle_sell_total - cycle_cost_total
            avg_cost_cycle = cycle_cost_total / cycle_buy_shares if cycle_buy_shares > 0 else 0.0
            cost_reduction = (
                (cycle_entry_price - avg_cost_cycle) / cycle_entry_price if cycle_entry_price else 0.0
            )
            days_hold = (date - entry_date).days
            trades.append(
                Trade(
                    entry_date=entry_date,
                    exit_date=date,
                    entry_price=float(cycle_entry_price),
                    exit_price=float(price),
                    return_pct=float(pnl / cycle_cost_total) if cycle_cost_total else 0.0,
                    holding_days=int(days_hold),
                    note="买入对冲-样本结束强平",
                    investment_amount=float(cycle_max_cost) if cycle_max_cost else None,
                )
            )
            buy_hedge_trades.append(
                {
                    "trade_id": current_trade_id or trade_count + 1,
                    "entry_date": str(entry_date),
                    "exit_date": str(date),
                    "entry_price": float(cycle_entry_price),
                    "exit_price": float(price),
                    "avg_cost": float(avg_cost_cycle),
                    "total_shares": shares_to_hands(int(cycle_max_shares)),
                    "adds": int(cycle_add_count),
                    "capital_used": float(cycle_max_cost),
                    "pnl": float(pnl),
                    "return_pct": float(pnl / cycle_cost_total) if cycle_cost_total else 0.0,
                    "avg_cost_delta_pct": float(cost_reduction),
                    "hedge_active": hedge_enabled,
                    "hedge_mode": hedge_mode,
                    "allow_repeat": allow_repeat,
                }
            )
            trade_count += 1
            total_adds += cycle_add_count
            max_layers = max(max_layers, cycle_add_count)
            avg_cost_reduction_sum += float(cost_reduction)
        else:
            append_event(
                {
                    "date": date,
                    "price": float(price),
                    "shares": 0,
                    "total_shares": int(position),
                    "avg_cost": position_cost / position if position > 0 else None,
                    "type": "skip",
                    "layer": grid_index,
                    "note": "样本结束未满足T+1",
                    "trigger_price": None,
                }
            )

    if hedge_position > 0 and len(idx):
        date = idx[-1]
        price = float(close.iloc[-1])
        hedge_cover(date, price, hedge_position, "样本结束对冲平仓", price)

    final_price = float(close.iloc[-1]) if len(close) else 0.0
    equity.iloc[-1] = cash + position * final_price + hedge_cash - hedge_position * final_price
    result = _calc_stats(equity, trades)

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
        "mode": growth_mode,
        "start_position": int(start_position_hands),
        "increment_unit": int(increment_unit_hands),
        "max_adds": int(config.get("max_adds") or 0),
        "reference": reference_mode,
        "max_capital_value": None,
        "max_capital_ratio": None,
        "max_capital_input": None,
        "skipped_by_cash": int(skipped_by_cash),
        "skipped_by_limit": int(skipped_by_limit),
        "skipped_by_rule": int(skipped_by_rule),
        "hedge_trade_count": int(hedge_trade_count),
        "hedge": {
            "enabled": hedge_enabled,
            "mode": hedge_mode,
            "size_mode": hedge_size_mode,
            "size_ratio": float(hedge_size_ratio),
            "size_hands": int(hedge_size_hands),
            "size_amount": float(hedge_size_amount),
            "exit": {
                "on_main_exit": bool(hedge_exit_on_main),
                "on_profit": bool(hedge_exit_on_profit),
                "profit_mode": hedge_exit_profit_mode,
                "profit_value": float(hedge_exit_profit_value),
                "on_loss": bool(hedge_exit_on_loss),
                "loss_mode": hedge_exit_loss_mode,
                "loss_value": float(hedge_exit_loss_value),
                "on_reverse": bool(hedge_exit_on_reverse),
            },
        },
        "allow_repeat": allow_repeat,
        "step_mode": step_mode,
        "step_abs": config.get("step_abs"),
        "step_rounding": config.get("step_rounding"),
        "step_auto": config.get("step_auto"),
        "growth": config.get("growth"),
        "position": config.get("position"),
        "entry": config.get("entry"),
        "profit": config.get("profit"),
        "reverse": config.get("reverse"),
        "capital": config.get("capital"),
        "exit": config.get("exit"),
        "limits": config.get("limits"),
        "base": config.get("base"),
    }

    result.buy_hedge_summary = summary
    result.buy_hedge_trades = buy_hedge_trades or []
    result.buy_hedge_events = events or []
    result.buy_hedge_hedge_trades = hedge_trades or []
    result.buy_hedge_hedge_events = hedge_events or []
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
    entry_investment_amount = 0.0

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
                entry_investment_amount += cost

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
                        investment_amount=entry_investment_amount if entry_investment_amount else None,
                    )
                )
                position = 0
                first_buy_date = None
                first_buy_price = 0.0
                pending_exit = False
                pending_exit_reason = ""
                equity.iloc[i] = cash
                entry_investment_amount = 0.0

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
                    investment_amount=entry_investment_amount if entry_investment_amount else None,
                )
            )
            position = 0
            first_buy_date = None
            first_buy_price = 0.0
            pending_exit = False
            pending_exit_reason = ""
            entry_investment_amount = 0.0
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
        trade_investment_amount = None

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
                    trade_investment_amount = float(cost)
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
                trade_investment_amount = float(value)
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
                    investment_amount=trade_investment_amount,
                )
            )
        equity.iloc[i] = cash + position * price

    return _calc_stats(equity, trades)
