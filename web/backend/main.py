import sys
import os
import shutil
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

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
            trades_data.append({
                "entry_date": str(t.entry_date),
                "entry_price": float(t.entry_price),
                "exit_date": str(t.exit_date),
                "exit_price": float(t.exit_price),
                "return_pct": float(t.return_pct),
                "holding_days": int(t.holding_days),
                "note": str(t.note)
            })

    return {
        "total_return": float(res.total_return),
        "annualized_return": float(res.annualized_return),
        "max_drawdown": float(res.max_drawdown),
        "win_rate": float(res.win_rate),
        "avg_win": float(res.avg_win),
        "avg_loss": float(res.avg_loss),
        "equity_curve": equity_data,
        "trades": trades_data
    }

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
