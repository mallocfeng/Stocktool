from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def EMA(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def REF(series: pd.Series, n: int) -> pd.Series:
    return series.shift(n)


def COUNT(cond: pd.Series, n) -> pd.Series:
    cond_values = cond.astype(int)
    if isinstance(n, pd.Series):
        window_sizes = n.fillna(0).astype(int)
        values = cond_values.to_numpy()
        results = []
        for idx, window in enumerate(window_sizes):
            window = int(window)
            if window <= 0:
                results.append(values[idx])
                continue
            start = max(0, idx - window + 1)
            results.append(values[start : idx + 1].sum())
        return pd.Series(results, index=cond.index)
    window = max(int(n), 1)
    return cond_values.rolling(window=window, min_periods=1).sum()


def MA(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(window=n, min_periods=1).mean()


def LLV(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(window=n, min_periods=1).min()


def HHV(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(window=n, min_periods=1).max()


def CROSS(s1: pd.Series, s2: pd.Series) -> pd.Series:
    diff = s1 - s2
    prev = diff.shift(1)
    return (prev <= 0) & (diff > 0)


def BARSLAST(cond: pd.Series) -> pd.Series:
    cond_bool = cond.astype(bool)
    result = []
    last = np.nan
    for val in cond_bool:
        if val:
            last = 0
        elif np.isnan(last):
            last = np.nan
        else:
            last += 1
        result.append(last)
    return pd.Series(result, index=cond.index)


def IF(cond: pd.Series, true_val, false_val):
    cond_series = cond.astype(bool)
    index = cond_series.index

    def to_series(val):
        if isinstance(val, pd.Series):
            return val.reindex(index)
        return pd.Series(val, index=index)

    true_series = to_series(true_val)
    false_series = to_series(false_val)
    return pd.Series(np.where(cond_series, true_series, false_series), index=index)


class TdxFormulaEngine:
    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        self.df = df

        price_series = {
            "C": df["close"],
            "O": df["open"],
            "H": df["high"],
            "L": df["low"],
            "V": df.get("volume", pd.Series(np.nan, index=df.index)),
        }
        aliases = {
            "CLOSE": price_series["C"],
            "OPEN": price_series["O"],
            "HIGH": price_series["H"],
            "LOW": price_series["L"],
            "VOL": price_series["V"],
            "SH": price_series["C"],
            "SL": price_series["L"],
            "SC": price_series["C"],
            "SO": price_series["O"],
        }
        self.ctx: Dict[str, object] = {
            **price_series,
            **aliases,
            "EMA": EMA,
            "REF": REF,
            "COUNT": COUNT,
            "MA": MA,
            "LLV": LLV,
            "HHV": HHV,
            "CROSS": CROSS,
            "BARSLAST": BARSLAST,
            "IF": IF,
            "np": np,
            "pd": pd,
        }

    def _convert_expr(self, expr: str) -> str:
        expr = expr.strip()
        expr = self._normalize_equality(expr)
        return self._handle_or(expr)

    def _normalize_colon_assignment(self, line: str) -> str:
        if ":=" in line or ":" not in line:
            return line
        match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)", line)
        if match:
            name, expr = match.groups()
            return f"{name} := {expr}"
        return line

    def _normalize_equality(self, expr: str) -> str:
        buf = []
        i = 0
        while i < len(expr):
            ch = expr[i]
            if ch == "=":
                prev = expr[i - 1] if i - 1 >= 0 else ""
                nxt = expr[i + 1] if i + 1 < len(expr) else ""
                if prev not in "<>!=:" and nxt != "=":
                    buf.append("==")
                    i += 1
                    continue
            buf.append(ch)
            i += 1
        return "".join(buf)

    def _handle_or(self, expr: str) -> str:
        parts = self._split_keyword(expr, "OR")
        if len(parts) == 1:
            return self._handle_and(parts[0])
        return "(" + ") | (".join(self._handle_and(part) for part in parts) + ")"

    def _handle_and(self, expr: str) -> str:
        parts = self._split_keyword(expr, "AND")
        if len(parts) == 1:
            return self._handle_not(parts[0])
        return "(" + ") & (".join(self._handle_not(part) for part in parts) + ")"

    def _handle_not(self, expr: str) -> str:
        expr = expr.strip()
        upper = expr.upper()
        if upper.startswith("NOT") and self._is_end_boundary(expr, 3):
            remainder = expr[3:].strip()
            return f"~({self._handle_not(remainder)})"
        if expr.startswith("(") and expr.endswith(")") and self._is_wrapped(expr):
            inner = expr[1:-1]
            return f"({self._handle_or(inner)})"
        return expr

    def _split_keyword(self, expr: str, keyword: str):
        parts = []
        buf = []
        depth = 0
        i = 0
        kw = keyword.upper()
        while i < len(expr):
            ch = expr[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if (
                depth == 0
                and expr[i:].upper().startswith(kw)
                and self._is_start_boundary(expr, i)
                and self._is_end_boundary(expr, i + len(keyword))
            ):
                parts.append("".join(buf))
                buf = []
                i += len(keyword)
                continue
            buf.append(ch)
            i += 1
        parts.append("".join(buf))
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _is_start_boundary(expr: str, idx: int) -> bool:
        if idx <= 0:
            return True
        prev = expr[idx - 1]
        return not (prev.isalnum() or prev == "_")

    @staticmethod
    def _is_end_boundary(expr: str, idx: int) -> bool:
        if idx >= len(expr):
            return True
        nxt = expr[idx]
        return not (nxt.isalnum() or nxt == "_")

    @staticmethod
    def _is_wrapped(expr: str) -> bool:
        depth = 0
        for i, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(expr) - 1:
                    return False
        return depth == 0

    def run(self, script: str) -> Tuple[pd.Series, pd.Series]:
        for raw_line in script.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(("{", "}", "//", "#", "(*", "*)")):
                continue
            if line.endswith(";"):
                line = line[:-1]

            line = self._normalize_colon_assignment(line)

            if ":=" in line:
                name, expr = line.split(":=", 1)
                name = name.strip()
                expr = self._convert_expr(expr)
                expr = expr.strip()
                value = eval(expr, {}, dict(self.ctx))
                self.ctx[name] = value
                if name.upper() != name:
                    self.ctx[name.upper()] = value
            else:
                expr = self._convert_expr(line)
                _ = eval(expr, {}, dict(self.ctx))

        buy = self.ctx.get("B_COND")
        if buy is None:
            raise ValueError("公式中未定义 B_COND（买入条件）。")

        sell = self.ctx.get("S_COND")
        if sell is None:
            sell = pd.Series(False, index=self.df.index)

        if not isinstance(buy, pd.Series):
            buy = pd.Series(bool(buy), index=self.df.index)
        if not isinstance(sell, pd.Series):
            sell = pd.Series(bool(sell), index=self.df.index)

        buy = buy.astype(bool).reindex(self.df.index).fillna(False)
        sell = sell.astype(bool).reindex(self.df.index).fillna(False)
        return buy, sell
