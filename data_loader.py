from __future__ import annotations

import csv
from typing import Optional, Set

import pandas as pd


REQUIRED_COLUMNS: Set[str] = {"date", "open", "high", "low", "close"}
NUMERIC_COLUMNS: Set[str] = {"open", "high", "low", "close", "volume"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def _read_dataframe(csv_path: str, skiprows: Optional[int] = None) -> pd.DataFrame:
    last_exc: Optional[Exception] = None
    for encoding in (None, "utf-8-sig", "gbk"):
        try:
            kwargs = {}
            if skiprows not in (None, 0):
                kwargs["skiprows"] = skiprows
            if encoding:
                kwargs["encoding"] = encoding
            df = pd.read_csv(csv_path, **kwargs)
            return _normalize_columns(df)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_exc = exc
            continue
    if last_exc:
        raise last_exc
    raise RuntimeError("无法解析 CSV 文件")


def _find_header_row(csv_path: str, required_columns: Set[str]) -> Optional[int]:
    try:
        with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as fh:
            reader = csv.reader(fh)
            for idx, row in enumerate(reader):
                normalized = {cell.strip().strip('"').lower() for cell in row if cell.strip()}
                if required_columns.issubset(normalized):
                    return idx
    except FileNotFoundError:
        raise
    return None


def load_price_csv(csv_path: str) -> pd.DataFrame:
    """Load OHLCV CSV data while tolerating banner rows and encodings."""

    def finalize(data: pd.DataFrame) -> pd.DataFrame:
        cleaned = data.copy()
        if "date" in cleaned.columns:
            cleaned["date"] = cleaned["date"].astype(str).str.strip()
            cleaned = cleaned[cleaned["date"].astype(bool)]
        else:
            cleaned.insert(0, "date", "")
        for col in NUMERIC_COLUMNS & set(cleaned.columns):
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        required_subset = (REQUIRED_COLUMNS - {"date"}) & set(cleaned.columns)
        if required_subset:
            cleaned.dropna(subset=required_subset, inplace=True)
        cleaned.reset_index(drop=True, inplace=True)
        return cleaned

    df = _read_dataframe(csv_path)
    df = finalize(df)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if not missing:
        return df

    header_row = _find_header_row(csv_path, REQUIRED_COLUMNS)
    if header_row is not None:
        df = _read_dataframe(csv_path, skiprows=header_row)
        df = finalize(df)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if not missing:
            return df

    raise ValueError(f"CSV 至少需要列：{', '.join(sorted(REQUIRED_COLUMNS))}")
