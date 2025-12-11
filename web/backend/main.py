import sys
import os
import shutil
import json
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests

# Add project root to sys.path to import existing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backtest_service import BacktestParams, run_backtests, BacktestPayload, BacktestEntry
from analytics import (
    indicator_scoring,
    atr_based_stop,
    position_rebalance_plan,
    run_stress_test,
    holding_return_heatmap,
    generate_multi_timeframe_signals,
    generate_daily_brief,
    simple_rule_based_formula,
)
# We might need to adjust plotting functions to return data instead of calling plt.show()
# For now, let's just focus on data endpoints.

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- State Management (Single User) ---
class AppState:
    df: Optional[pd.DataFrame] = None
    buy: Optional[pd.Series] = None
    sell: Optional[pd.Series] = None
    results: List[BacktestEntry] = []
    scores: Optional[pd.DataFrame] = None
    formula: str = ""

state = AppState()

# --- Models ---

class StrategyConfig(BaseModel):
    fixed: Optional[List[int]] = None
    tpsl: Optional[Dict[str, float]] = None
    dca: Optional[Dict[str, float]] = None
    grid: Optional[Dict[str, Any]] = None
    dynamic: Optional[Dict[str, Any]] = None

class BacktestRequest(BaseModel):
    csv_path: str
    formula: str
    initial_capital: float
    fee_rate: float
    strategies: StrategyConfig

class PlotRequest(BaseModel):
    strategy_index: int

class MultiTimeframeRequest(BaseModel):
    freqs: List[str]


class StrategyTextRequest(BaseModel):
    text: str

class SinaImportRequest(BaseModel):
    symbol: str
    scale: int = 240
    datalen: int = 500

# --- External Data Helpers ---

SINA_API_URLS = [
    "https://quotes.sina.cn/cn/api/openapi.php/CN_MarketDataService.getKLineData",
    "https://finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketDataService.getKLineData",
    "http://finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketDataService.getKLineData",
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketDataService.getKLineData",
    "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketDataService.getKLineData",
]
SINA_QUOTE_URL = "https://hq.sinajs.cn/list={symbol}"
SINA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://finance.sina.com.cn/",
}

def normalize_symbol(symbol: str) -> str:
    code = (symbol or "").strip().lower()
    if not code:
        raise ValueError("股票代码不能为空")
    if code.startswith(("sh", "sz")):
        return code
    if code[0] in ("6", "9"):
        return f"sh{code}"
    if code[0] in ("0", "3"):
        return f"sz{code}"
    raise ValueError("无法识别的股票代码，请输入如 600519 或 sh600519")

def fetch_sina_klines(symbol: str, scale: int, datalen: int) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "scale": max(1, int(scale)),
        "ma": "no",
        "datalen": max(50, min(int(datalen), 2000)),
    }
    payload = None
    last_exc = None
    for url in SINA_API_URLS:
        try:
            resp = requests.get(url, params=params, timeout=10, headers=SINA_HEADERS)
            resp.raise_for_status()
            payload = resp.json()
            break
        except requests.RequestException as exc:
            last_exc = exc
        except json.JSONDecodeError as exc:  # noqa: PERF203
            last_exc = exc
    if payload is None:
        raise RuntimeError(f"新浪接口请求失败：{last_exc}") from last_exc

    if isinstance(payload, dict) and "result" in payload:
        payload = payload.get("result", {}).get("data")

    if not isinstance(payload, list) or not payload:
        raise RuntimeError("未从新浪获取到行情数据")

    records = []
    for row in payload:
        day = row.get("day")
        if not day:
            continue
        date_str = str(day).split(" ")[0]
        try:
            records.append(
                {
                    "date": date_str,
                    "open": float(row.get("open", 0)),
                    "high": float(row.get("high", 0)),
                    "low": float(row.get("low", 0)),
                    "close": float(row.get("close", 0)),
                    "volume": float(row.get("volume", 0)),
                }
            )
        except (TypeError, ValueError):
            continue

    if not records:
        raise RuntimeError("新浪行情解析失败，记录为空")

    df = pd.DataFrame(records)
    df.sort_values("date", inplace=True)
    df.drop_duplicates(subset="date", inplace=True)
    return df

def fetch_sina_name(symbol: str) -> str:
    try:
        resp = requests.get(SINA_QUOTE_URL.format(symbol=symbol), timeout=5, headers=SINA_HEADERS)
        resp.encoding = "gbk"
        if resp.status_code != 200:
            return symbol.upper()
        text = resp.text.strip()
        if "=" not in text:
            return symbol.upper()
        _, value = text.split("=", 1)
        value = value.strip().strip('";')
        name = value.split(",")[0].strip()
        return name if name else symbol.upper()
    except Exception:
        return symbol.upper()

# --- WebSocket ---

@app.websocket("/ws")
async def noop_ws(socket: WebSocket):
    """Accept and immediately close stray websocket requests (e.g. from dev tooling)."""
    await socket.accept()
    await socket.close()

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"path": os.path.abspath(file_path)}

@app.post("/import/sina")
def import_from_sina(req: SinaImportRequest):
    try:
        symbol = normalize_symbol(req.symbol)
    except ValueError as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        df = fetch_sina_klines(symbol, req.scale, req.datalen)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"请求新浪接口失败：{exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    label = fetch_sina_name(symbol)
    upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    filename = f"{symbol}_{int(time.time())}.csv"
    file_path = os.path.join(upload_dir, filename)
    df.to_csv(file_path, index=False)
    return {
        "path": os.path.abspath(file_path),
        "symbol": symbol,
        "label": label,
        "rows": len(df),
    }

# --- Helpers ---

class MockStopEvent:
    def is_set(self):
        return False

def serialize_backtest_result(res) -> Dict:
    # Convert equity curve to list of [timestamp/str, value]
    equity_data = []
    if hasattr(res.equity_curve, "index"):
        # index is datetime
        times = res.equity_curve.index.astype(str).tolist()
        values = res.equity_curve.values.tolist()
        equity_data = list(zip(times, values))
    
    trades_data = []
    if res.trades:
        for t in res.trades:
            trade_payload = {
                "entry_date": str(t.entry_date),
                "entry_price": float(t.entry_price),
                "exit_date": str(t.exit_date),
                "exit_price": float(t.exit_price),
                "return_pct": float(t.return_pct),
                "holding_days": int(t.holding_days),
                "note": str(t.note)
            }
            if getattr(t, "investment_amount", None) is not None:
                trade_payload["investment_amount"] = float(t.investment_amount)
            if getattr(t, "loss_streak", None) is not None:
                trade_payload["loss_streak"] = int(t.loss_streak)
            if getattr(t, "adjusted_quantity", None) is not None:
                trade_payload["adjusted_quantity"] = int(t.adjusted_quantity)
            if getattr(t, "pnl_with_dynamic_fund", None) is not None:
                trade_payload["pnl_with_dynamic_fund"] = float(t.pnl_with_dynamic_fund)
            if getattr(t, "hedge_investment_amount", None) is not None:
                trade_payload["hedge_investment_amount"] = float(t.hedge_investment_amount)
            if getattr(t, "hedge_loss_streak", None) is not None:
                trade_payload["hedge_loss_streak"] = int(t.hedge_loss_streak)
            if getattr(t, "hedge_adjusted_quantity", None) is not None:
                trade_payload["hedge_adjusted_quantity"] = int(t.hedge_adjusted_quantity)
            if getattr(t, "hedge_pnl_with_dynamic_fund", None) is not None:
                trade_payload["hedge_pnl_with_dynamic_fund"] = float(t.hedge_pnl_with_dynamic_fund)
            trades_data.append(trade_payload)

    payload = {
        "total_return": float(res.total_return),
        "annualized_return": float(res.annualized_return),
        "max_drawdown": float(res.max_drawdown),
        "win_rate": float(res.win_rate),
        "avg_win": float(res.avg_win),
        "avg_loss": float(res.avg_loss),
        "equity_curve": equity_data,
        "trades": trades_data
    }
    if getattr(res, "dynamic_equity_curve", None) is not None:
        dyn_times = res.dynamic_equity_curve.index.astype(str).tolist()
        dyn_values = res.dynamic_equity_curve.values.tolist()
        payload["equityCurveWithDynamicFund"] = list(zip(dyn_times, dyn_values))
    if getattr(res, "investment_curve_main", None):
        payload["investmentCurveMain"] = res.investment_curve_main
        payload["investmentAmount"] = res.investment_curve_main  # backward compatibility
    if getattr(res, "investment_curve_hedge", None):
        payload["investmentCurveHedge"] = res.investment_curve_hedge
        payload["hedgeInvestmentAmount"] = res.investment_curve_hedge
    payload["forceStop"] = bool(getattr(res, "dynamic_force_stop", False))
    payload["forceStopByDrawdown"] = bool(getattr(res, "dynamic_force_stop", False))
    if getattr(res, "position_details", None):
        payload["positionDetail"] = res.position_details
    if getattr(res, "max_loss_streak_used", None) is not None:
        payload["maxLossStreakUsed"] = int(res.max_loss_streak_used)
    if getattr(res, "max_investment_used", None) is not None:
        payload["maxInvestmentUsed"] = float(res.max_investment_used)
    if getattr(res, "hedge_max_loss_streak_used", None) is not None:
        payload["hedgeMaxLossStreakUsed"] = res.hedge_max_loss_streak_used
    if getattr(res, "hedge_max_investment_used", None) is not None:
        payload["hedgeMaxInvestmentUsed"] = res.hedge_max_investment_used
    if getattr(res, "dynamic_summary", None):
        payload["dynamicSummary"] = res.dynamic_summary
    return payload

@app.post("/run_backtest")
def api_run_backtest(req: BacktestRequest):
    strategies = {}
    if req.strategies.fixed:
        strategies["fixed"] = req.strategies.fixed
    if req.strategies.tpsl:
        strategies["tpsl"] = req.strategies.tpsl
    if req.strategies.dca:
        strategies["dca"] = req.strategies.dca
    if req.strategies.grid:
        strategies["grid"] = req.strategies.grid
    if req.strategies.dynamic:
        strategies["dynamic"] = req.strategies.dynamic

    if not strategies:
        raise HTTPException(status_code=400, detail="No strategies selected")

    params = BacktestParams(
        csv_path=req.csv_path,
        formula=req.formula,
        initial_capital=req.initial_capital,
        fee_rate=req.fee_rate,
        strategies=strategies,
    )

    logs = []
    final_payload: Optional[BacktestPayload] = None

    def emit_mock(event_type, data):
        if event_type == "log":
            logs.append(data)
        elif event_type == "result":
            nonlocal final_payload
            final_payload = data
        elif event_type == "error":
            logs.append(f"ERROR: {data}")

    stop_event = MockStopEvent()
    
    try:
        run_backtests(params, stop_event, emit_mock)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    if not final_payload:
        return {"logs": logs, "status": "failed", "error": "No result returned"}

    # Update global state
    state.df = final_payload.df
    state.buy = final_payload.buy
    state.sell = final_payload.sell
    state.results = final_payload.entries
    state.formula = final_payload.formula
    state.scores = None # Reset scores on new backtest

    # Serialize results
    serialized_entries = []
    for entry in final_payload.entries:
        serialized_entries.append({
            "name": entry.name,
            "title": entry.title,
            "result": serialize_backtest_result(entry.result)
        })

    return {
        "status": "success",
        "logs": logs,
        "entries": serialized_entries
    }

@app.get("/market_data_chart")
def get_market_data():
    """Return OHLCV data for charting."""
    if state.df is None:
        raise HTTPException(status_code=400, detail="No data loaded. Run backtest first.")
    
    # Reset index to get date column if it's strictly index
    df = state.df.reset_index()
    
    # Adding buy/sell markers
    buys = []
    if state.buy is not None:
         # boolean series
         buys = df[state.buy.values]["date"].astype(str).tolist()

    sells = []
    if state.sell is not None:
         sells = df[state.sell.values]["date"].astype(str).tolist()

    records = df[["date", "open", "high", "low", "close"]].copy()
    records["date"] = records["date"].astype(str)
    
    return {
        "kline": records.to_dict(orient="records"),
        "buy_signals": buys,
        "sell_signals": sells
    }

@app.post("/analytics/scores")
def get_scores():
    if state.df is None:
        raise HTTPException(status_code=400, detail="Run backtest first")
    scores = indicator_scoring(state.df)
    state.scores = scores
    # Return data for plotting
    data = []
    for date, row in scores.iterrows():
        data.append({"date": str(date), "total_score": float(row["total_score"])})
    return data

@app.post("/analytics/stress")
def get_stress_test():
    if state.df is None:
        raise HTTPException(status_code=400, detail="Run backtest first")
    res = run_stress_test(state.df)
    if res.empty:
        return {"message": "Insufficient data"}
    # DataFrame to list of dicts, careful with NaN
    return res.fillna(0).reset_index().to_dict(orient="records")

@app.post("/analytics/daily_brief")
def get_daily_brief():
    if state.df is None:
        raise HTTPException(status_code=400, detail="Run backtest first")
    first_result = state.results[0].result if state.results else None
    text = generate_daily_brief(state.df, state.scores, first_result, state.buy, state.sell)
    return {"text": text}

@app.post("/analytics/position_plan")
def get_position_plan(capital: float = Body(..., embed=True)):
    if state.df is None:
        raise HTTPException(status_code=400, detail="Run backtest first")
    try:
        plan = position_rebalance_plan(state.df, capital)
        if plan.empty:
            return {"message": "条件不足，无法生成计划"}
        return plan.reset_index().to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analytics/heatmap")
def get_heatmap():
    if state.df is None:
        raise HTTPException(status_code=400, detail="Run backtest first")
    try:
        # We need to return data suitable for a heatmap chart.
        # holding_return_heatmap returns a DataFrame where index=entry_date, columns=holding_days, value=return
        res = holding_return_heatmap(state.df)
        if res.empty:
            return {"message": "数据不足"}
        
        # Convert to ECharts heatmap format: [ [x, y, value], ... ]
        # X axis: Holding Days (columns)
        # Y axis: Entry Date (index)
        
        # Limit to last 50 rows to avoid too much data if needed, or all. 
        # The user wants "cool", so let's try to return a reasonable amount.
        subset = res.tail(100) 
        
        data = []
        y_labels = subset.index.astype(str).tolist()
        x_labels = subset.columns.astype(str).tolist()
        
        for i, date_idx in enumerate(subset.index):
            for j, hold_col in enumerate(subset.columns):
                val = subset.iloc[i, j]
                # ECharts heatmap data: [x_index, y_index, value]
                if pd.notna(val):
                    data.append([j, i, float(val)])
        
        return {
            "x_labels": x_labels,
            "y_labels": y_labels,
            "data": data
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics/stop_suggestion")
def get_stop_suggestion():
    if state.df is None:
        raise HTTPException(status_code=400, detail="Run backtest first")
    info = atr_based_stop(state.df)
    # Convert numpy floats to native
    return {k: float(v) for k, v in info.items()}


@app.post("/analytics/multi_timeframe")
def get_multi_timeframe(req: MultiTimeframeRequest):
    if state.df is None or not state.formula:
        raise HTTPException(status_code=400, detail="Run backtest first")
    freqs = req.freqs or ["D", "W", "M"]
    signals, frames = generate_multi_timeframe_signals(state.df, state.formula, freqs)
    payload = []
    for freq_key, frame in frames.items():
        frame_reset = frame.reset_index().rename(columns={"index": "date"})
        frame_reset["date"] = frame_reset["date"].astype(str)
        kline = frame_reset[["date", "open", "high", "low", "close"]].to_dict(orient="records")
        freq_signals = signals.get(freq_key, {})
        buy_series = freq_signals.get("buy")
        sell_series = freq_signals.get("sell")
        buys = buy_series[buy_series].index.astype(str).tolist() if buy_series is not None else []
        sells = sell_series[sell_series].index.astype(str).tolist() if sell_series is not None else []
        payload.append({
            "freq": freq_key,
            "kline": kline,
            "buy_signals": buys,
            "sell_signals": sells,
        })
    return payload


@app.post("/analytics/nlp_formula")
def generate_formula(req: StrategyTextRequest):
    try:
        formula = simple_rule_based_formula(req.text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))
    return {"formula": formula}
