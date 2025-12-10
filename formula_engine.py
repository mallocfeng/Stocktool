from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


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
