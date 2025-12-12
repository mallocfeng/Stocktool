from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from formula_engine import TdxFormulaEngine
from backtesting import BacktestResult


def _resample_ohlc(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return df.resample(freq).agg(agg).dropna()


def normalize_timeframe_token(freq: str) -> Tuple[str, str]:
    token = (freq or "").strip()
    if not token:
        raise ValueError("empty freq")
    cleaned = token.lower().strip()
    cleaned = cleaned.replace("分钟k", "分钟").replace("分k", "分")
    cleaned = cleaned.rstrip("k").strip()
    base_aliases = {
        "d": ("D", "1D"),
        "day": ("D", "1D"),
        "daily": ("D", "1D"),
        "w": ("W", "7D"),
        "week": ("W", "7D"),
        "weekly": ("W", "7D"),
        "m": ("M", "30D"),
        "mon": ("M", "30D"),
        "month": ("M", "30D"),
        "monthly": ("M", "30D"),
        "h": ("1H", "1H"),
        "hour": ("1H", "1H"),
        "hourly": ("1H", "1H"),
    }
    if cleaned in base_aliases:
        return base_aliases[cleaned]

    pattern = re.compile(r"(\d+)\s*([a-z]+|[\u4e00-\u9fff]+)?")
    match = pattern.fullmatch(cleaned)
    if not match:
        raise ValueError(f"无法解析周期：{freq}")
    num = int(match.group(1))
    unit = (match.group(2) or "").lower()

    def unit_key(name: str) -> Optional[str]:
        if not name:
            return None
        minute_alias = {"m", "min", "mins", "minute", "minutes", "分", "分钟"}
        hour_alias = {"h", "hr", "hrs", "hour", "hours", "小时"}
        day_alias = {"d", "day", "days", "天", "日"}
        week_alias = {"w", "wk", "week", "weeks", "周"}
        month_alias = {"mo", "mon", "month", "months", "月"}
        if name in minute_alias:
            return "minute"
        if name in hour_alias:
            return "hour"
        if name in day_alias:
            return "day"
        if name in week_alias:
            return "week"
        if name in month_alias:
            return "month"
        return None

    resolved = unit_key(unit)
    if resolved == "minute":
        return (f"{num}m", f"{num}T")
    if resolved == "hour":
        return (f"{num}H", f"{num}H")
    if resolved == "day":
        label = "D" if num == 1 else f"{num}D"
        return (label, f"{num}D")
    if resolved == "week":
        label = "W" if num == 1 else f"{num}W"
        return (label, f"{num * 7}D")
    if resolved == "month":
        label = "M" if num == 1 else f"{num}M"
        return (label, f"{num * 30}D")
    raise ValueError(f"无法识别的周期单位：{freq}")


def _infer_base_interval_ns(index: pd.DatetimeIndex) -> Optional[int]:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return None
    values = index.sort_values().view("i8")
    diffs = np.diff(values)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    return int(np.median(diffs))


def _format_duration(ns: Optional[int]) -> Optional[str]:
    if not ns:
        return None
    seconds = ns / 1_000_000_000
    if seconds < 60:
        return f"{int(round(seconds))}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(round(minutes))}m"
    hours = minutes / 60
    if hours < 24:
        return f"{int(round(hours))}H"
    days = hours / 24
    if days < 7:
        return f"{int(round(days))}D"
    weeks = days / 7
    if weeks < 4:
        return f"{int(round(weeks))}W"
    months = days / 30
    return f"{int(round(months))}M"


def _recommended_freqs(base_ns: Optional[int]) -> List[str]:
    candidates = [
        ("1m", "1T"),
        ("3m", "3T"),
        ("5m", "5T"),
        ("15m", "15T"),
        ("30m", "30T"),
        ("1H", "1H"),
        ("2H", "2H"),
        ("4H", "4H"),
        ("D", "1D"),
        ("W", "1W"),
        ("M", "1M"),
    ]
    result: List[str] = []
    for label, alias in candidates:
        try:
            offset = to_offset(alias)
            nanos = getattr(offset, "nanos", None)
            if nanos is None:
                # 非固定周期（例如以周几结尾），跳过
                continue
        except ValueError:
            continue
        if base_ns is None or nanos >= base_ns:
            result.append(label)
    return result or ["D", "W", "M"]


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    frame = df.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
        frame.set_index("date", inplace=True)
    else:
        frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def generate_multi_timeframe_signals(
    df: pd.DataFrame, formula: str, freqs: Sequence[str]
) -> Tuple[Dict[str, Dict[str, pd.Series]], Dict[str, pd.DataFrame], Dict[str, object]]:
    results: Dict[str, Dict[str, pd.Series]] = {}
    frames: Dict[str, pd.DataFrame] = {}
    df_work = _prepare_dataframe(df)
    base_ns = _infer_base_interval_ns(df_work.index)
    default_freqs = _recommended_freqs(base_ns)
    freq_inputs = list(freqs) if freqs else default_freqs
    seen_labels: set[str] = set()
    skipped: List[Dict[str, str]] = []

    for freq in freq_inputs:
        try:
            label, alias = normalize_timeframe_token(freq)
        except ValueError as exc:
            skipped.append({"freq": str(freq), "reason": str(exc)})
            continue
        if label in seen_labels:
            continue
        seen_labels.add(label)
        try:
            offset = to_offset(alias)
            nanos = getattr(offset, "nanos", None)
            if nanos is None:
                raise ValueError("non-fixed frequency")
        except ValueError:
            skipped.append({"freq": label, "reason": "无法转换该周期"})
            continue

        if base_ns is not None and nanos < base_ns:
            skipped.append({"freq": label, "reason": "数据粒度不足"})
            continue
        if nanos == base_ns:
            df_freq = df_work
        else:
            df_freq = _resample_ohlc(df_work, alias)
        if df_freq.empty:
            skipped.append({"freq": label, "reason": "周期数据为空"})
            continue
        frames[label] = df_freq
        engine = TdxFormulaEngine(df_freq.reset_index().rename(columns={"index": "date"}))
        buy, sell = engine.run(formula)
        results[label] = {"buy": buy, "sell": sell}

    meta = {
        "base_interval": _format_duration(base_ns),
        "recommended_freqs": default_freqs,
        "used_freqs": list(frames.keys()),
        "skipped_freqs": skipped,
    }
    return results, frames, meta


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def indicator_scoring(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    scores = pd.DataFrame(index=df.index)

    ma_fast = _ema(close, 5)
    ma_mid = _ema(close, 10)
    ma_slow = _ema(close, 20)
    scores["MA"] = 0
    scores.loc[close > ma_fast, "MA"] += 1
    scores.loc[ma_fast > ma_mid, "MA"] += 1
    scores.loc[ma_mid > ma_slow, "MA"] += 1
    scores.loc[close < ma_slow, "MA"] -= 1

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    dif = ema12 - ema26
    dea = _ema(dif, 9)
    macd = (dif - dea) * 2
    scores["MACD"] = np.where(macd > 0, 1, -1)

    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14, min_periods=1).mean()
    roll_down = down.rolling(14, min_periods=1).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - 100 / (1 + rs)
    scores["RSI"] = 0
    scores.loc[rsi > 70, "RSI"] = -1
    scores.loc[rsi < 30, "RSI"] = 1

    rolling_mean = close.rolling(20, min_periods=1).mean()
    rolling_std = close.rolling(20, min_periods=1).std(ddof=0)
    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std
    norm = (close - rolling_mean) / (2 * rolling_std + 1e-9)
    scores["BOLL"] = np.clip(norm, -1, 1)

    volume = df.get("volume")
    if volume is not None:
        vol_ratio = volume / (volume.rolling(5, min_periods=1).mean() + 1e-9)
        scores["VOL"] = np.where(vol_ratio > 1.5, 1, np.where(vol_ratio < 0.8, -1, 0))
    else:
        scores["VOL"] = 0

    scores["total_score"] = scores.sum(axis=1)
    return scores.assign(MA_fast=ma_fast, MA_mid=ma_mid, MA_slow=ma_slow, MACD=macd, RSI=rsi, BOLL_mid=rolling_mean, BOLL_upper=upper, BOLL_lower=lower)


def atr_based_stop(df: pd.DataFrame, period: int = 14) -> Dict[str, float]:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    last_price = float(close.iloc[-1])
    last_atr = float(atr.iloc[-1])
    return {
        "last_price": last_price,
        "atr": last_atr,
        "suggest_stop_loss": max(last_price - 2 * last_atr, 0),
        "suggest_take_profit": last_price + 2 * last_atr,
        "trailing_stop": max(last_price - 1.5 * last_atr, 0),
    }


def position_rebalance_plan(
    df: pd.DataFrame,
    capital: float,
    step_pct: float = 0.05,
    max_steps: int = 4,
    min_lot: int = 100,
) -> pd.DataFrame:
    last_price = float(df["close"].iloc[-1])
    plan_rows = []
    remaining_cash = capital
    position = 0

    for i in range(max_steps):
        target_price = last_price * (1 - step_pct * i)
        invest_cash = capital * (0.25 if i == 0 else step_pct)
        shares = max(int(invest_cash // target_price / min_lot) * min_lot, 0)
        cost = shares * target_price
        if cost > remaining_cash or shares <= 0:
            continue
        remaining_cash -= cost
        position += shares
        plan_rows.append(
            {
                "step": i + 1,
                "target_price": round(target_price, 2),
                "buy_shares": shares,
                "cost": round(cost, 2),
                "remaining_cash": round(remaining_cash, 2),
                "avg_cost": round(cost / shares if shares else 0, 2),
            }
        )

    return pd.DataFrame(plan_rows)


def stress_test_windows() -> List[Tuple[str, str, str]]:
    return [
        ("2015股灾", "2015-06-01", "2015-09-30"),
        ("2018贸易战", "2018-01-01", "2018-12-31"),
        ("2020疫情", "2020-01-01", "2020-06-30"),
        ("2022反弹", "2022-03-01", "2022-12-31"),
    ]


def run_stress_test(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    rows = []
    for name, start, end in stress_test_windows():
        period = df.loc[start:end]
        if period.empty:
            continue
        start_price = float(period["close"].iloc[0])
        end_price = float(period["close"].iloc[-1])
        ret = end_price / start_price - 1
        rolling_max = period["close"].cummax()
        drawdown = period["close"] / rolling_max - 1
        rows.append(
            {
                "window": name,
                "start": start,
                "end": end,
                "return_pct": round(ret * 100, 2),
                "max_drawdown_pct": round(drawdown.min() * 100, 2),
                "volatility_pct": round(period["close"].pct_change().std() * np.sqrt(252) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def holding_return_heatmap(df: pd.DataFrame, max_hold: int = 30) -> pd.DataFrame:
    close = df["close"].reset_index(drop=True)
    data = {}
    for hold in range(1, max_hold + 1):
        future = close.shift(-hold)
        ret = future / close - 1
        data[f"持有{hold}天"] = ret
    heatmap_df = pd.DataFrame(data)
    heatmap_df.index = df.index
    return heatmap_df.dropna(how="all")


def generate_daily_brief(
    df: pd.DataFrame,
    scoring: Optional[pd.DataFrame],
    result: Optional[BacktestResult],
    buy: Optional[pd.Series],
    sell: Optional[pd.Series],
) -> str:
    if df.empty or "close" not in df.columns:
        return "【复盘摘要】当前数据不足，无法生成复盘内容。"

    frame = df.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
        frame.set_index("date", inplace=True)

    close = frame["close"].astype(float)
    last_date = close.index[-1]
    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) > 1 else last_close
    change_pct = (last_close / prev_close - 1) * 100 if prev_close else 0.0
    high = float(frame.get("high", close).iloc[-1])
    low = float(frame.get("low", close).iloc[-1])
    intraday_range = (high / low - 1) * 100 if low else 0.0
    volume = float(frame.get("volume", pd.Series(dtype=float)).iloc[-1]) if "volume" in frame else 0.0
    avg_volume = float(frame["volume"].tail(5).mean()) if "volume" in frame else 0.0
    volume_desc = ""
    if avg_volume > 0:
        ratio = volume / avg_volume if avg_volume else 1.0
        volume_desc = f"；成交量 {volume/1e4:.1f} 万手，较5日均{'放大' if ratio >= 1 else '缩量'} {abs(ratio-1)*100:.0f}%"

    ma_short = close.ewm(span=5, adjust=False).mean().iloc[-1]
    ma_long = close.ewm(span=20, adjust=False).mean().iloc[-1]
    trend_desc = (
        "5日均线在20日均线上方，短线保持多头结构"
        if ma_short > ma_long
        else "5日均线跌破20日均线，短线偏弱"
    )

    base = [
        f"【复盘摘要】{last_date.date()} 收盘 {last_close:.2f}（{change_pct:+.2f}%），日内波幅 {intraday_range:.2f}%{volume_desc}",
        f"趋势观察：{trend_desc}，近10日涨跌 {((last_close / close.iloc[-10]) - 1) * 100:.2f}%"
        if len(close) >= 10
        else f"趋势观察：{trend_desc}",
    ]

    if scoring is not None and not scoring.empty:
        latest_score = scoring.iloc[-1]
        total = latest_score.get("total_score", 0.0)
        components = []
        for col in latest_score.index:
            if col in {"total_score", "date"}:
                continue
            components.append((col, latest_score[col]))
        components.sort(key=lambda item: abs(item[1]), reverse=True)
        detail = "，".join([f"{name} {value:+.1f}" for name, value in components[:3]])
        base.append(f"指标评分：{total:+.1f}（{detail}）")

    if result is not None:
        equity_last = float(result.equity_curve.iloc[-1])
        total_return = result.total_return * 100
        annualized = result.annualized_return * 100
        max_dd = result.max_drawdown * 100
        win_rate = result.win_rate * 100
        avg_win = result.avg_win * 100
        avg_loss = result.avg_loss * 100
        base.append(
            "策略表现：净值 {equity:.0f}（{ret:+.1f}%），年化 {ann:+.1f}%，最大回撤 {dd:.1f}%，胜率 {win:.1f}%（平均盈 {avg_win:+.1f}% / 亏 {avg_loss:+.1f}%）".format(
                equity=equity_last,
                ret=total_return,
                ann=annualized,
                dd=max_dd,
                win=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
            )
        )

    signals = []
    if buy is not None and not buy.empty:
        last_buy_idx = buy[buy].index[-1] if buy.any() else None
        if buy.iloc[-1]:
            signals.append("今日触发买入信号")
        elif last_buy_idx is not None:
            signals.append(f"最近买入信号：{last_buy_idx.date()}")
    if sell is not None and not sell.empty:
        last_sell_idx = sell[sell].index[-1] if sell.any() else None
        if sell.iloc[-1]:
            signals.append("今日触发卖出信号")
        elif last_sell_idx is not None:
            signals.append(f"最近卖出信号：{last_sell_idx.date()}")
    if signals:
        base.append("信号追踪：" + "，".join(signals))

    return "\n".join(base)


def simple_rule_based_formula(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("请输入策略描述")

    lower = text.lower()
    lines = []
    used = False

    ma_pattern = re.findall(r"(\d+)日?均线.*?(上穿|向上|金叉).*?(\d+)日?均线", text)
    if ma_pattern:
        short, _, long = ma_pattern[0]
        lines.append(f"MA_S:=EMA(C,{short});")
        lines.append(f"MA_L:=EMA(C,{long});")
        lines.append("B_COND:=CROSS(MA_S,MA_L);")
        used = True

    if "量" in text or "成交" in text:
        lines.append("VOL_RATIO:=V/REF(V,1);")
        lines.append("B_COND:=B_COND AND VOL_RATIO>1.2;" if used else "B_COND:=VOL_RATIO>1.2;")
        used = True

    if "macd" in lower or "dif" in lower:
        lines.extend(
            [
                "DIFF:=EMA(C,12)-EMA(C,26);",
                "DEA:=EMA(DIFF,9);",
            ]
        )
        lines.append("B_COND:=B_COND AND CROSS(DIFF,DEA);" if used else "B_COND:=CROSS(DIFF,DEA);")
        used = True

    if not used:
        lines.append("B_COND:=CROSS(EMA(C,5),EMA(C,20));")

    lines.append("S_COND:=CROSS(EMA(C,20),EMA(C,5));")
    return "\n".join(lines)
