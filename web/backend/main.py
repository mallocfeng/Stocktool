import sys
import os
import shutil
import json
import time
import hashlib
import sqlite3
from datetime import datetime
from typing import List, Optional, Dict, Any, Generator
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, WebSocket, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from argon2.low_level import Type

# Add project root to sys.path to import existing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backtest_service import BacktestParams, run_backtests, BacktestPayload, BacktestEntry
from data_loader import load_price_csv
from formula_engine import TdxFormulaEngine
from analytics import (
    indicator_scoring,
    atr_based_stop,
    position_rebalance_plan,
    run_stress_test,
    holding_return_heatmap,
    generate_multi_timeframe_signals,
    generate_daily_brief,
    simple_rule_based_formula,
    normalize_timeframe_token,
    generate_performance_report,
)
# We might need to adjust plotting functions to return data instead of calling plt.show()
# For now, let's just focus on data endpoints.

AI_PROVIDERS = {
    "trial": {
        "baseURL": "https://mallocfeng1982.win/v1",
        "model_extract": "deepseek-chat",
        "model_summarize": "deepseek-chat",
        "apiKey": "trial",
        "lockFields": True,
    }
}

def _resolve_ai_config():
    provider_key = os.environ.get("STOCKTOOL_AI_PROVIDER", "trial")
    provider = AI_PROVIDERS.get(provider_key, {})
    base = os.environ.get("STOCKTOOL_AI_BASE", provider.get("baseURL", "https://mallocfeng1982.win/v1"))
    model_summary = os.environ.get(
        "STOCKTOOL_AI_MODEL_SUMMARY",
        provider.get("model_summarize", provider.get("model_extract", "deepseek-chat")),
    )
    model_extract = os.environ.get(
        "STOCKTOOL_AI_MODEL_EXTRACT",
        provider.get("model_extract", provider.get("model_summarize", "deepseek-chat")),
    )
    api_key = os.environ.get("STOCKTOOL_AI_KEY", provider.get("apiKey", ""))
    timeout = float(os.environ.get("STOCKTOOL_AI_TIMEOUT", "90"))
    return {
        "base_url": base.rstrip("/"),
        "model_summary": model_summary,
        "model_extract": model_extract,
        "api_key": api_key,
        "timeout": timeout,
        "provider": provider_key,
    }

AI_CONFIG = _resolve_ai_config()
AI_BASE_URL = AI_CONFIG["base_url"]
AI_MODEL = AI_CONFIG["model_summary"]
AI_MODEL_EXTRACT = AI_CONFIG["model_extract"]
AI_API_KEY = AI_CONFIG["api_key"]
AI_TIMEOUT = AI_CONFIG["timeout"]

BACKEND_ROOT = os.path.dirname(__file__)
USER_DB_PATH = os.path.join(BACKEND_ROOT, "users.db")
SESSION_SECRET = os.environ.get("STOCKTOOL_SESSION_SECRET", "stocktool-session-secret")
SESSION_COOKIE_SECURE = os.environ.get("STOCKTOOL_SESSION_COOKIE_SECURE", "false").lower() in ("1", "true", "yes")
SESSION_SAME_SITE = os.environ.get("STOCKTOOL_SESSION_SAME_SITE", "lax").lower()
if SESSION_SAME_SITE not in {"lax", "strict", "none"}:
    SESSION_SAME_SITE = "lax"
SESSION_MAX_AGE = int(os.environ.get("STOCKTOOL_SESSION_MAX_AGE", str(7 * 24 * 3600)))
PASSWORD_HASHER = PasswordHasher(time_cost=2, memory_cost=102400, parallelism=8, hash_len=32, type=Type.ID)
DISABLE_AUTH = os.environ.get("STOCKTOOL_DISABLE_AUTH", "true").lower() in ("1", "true", "yes")
_BYPASS_USERNAME = os.environ.get("STOCKTOOL_BYPASS_USERNAME", "admin")
_BYPASS_ROLE = os.environ.get("STOCKTOOL_BYPASS_ROLE", "admin")

if SESSION_SECRET == "stocktool-session-secret":
    print("WARNING: using default session secret; set STOCKTOOL_SESSION_SECRET for production safety")

DEFAULT_ALLOW_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://104.207.154.5",
    "http://xpq.qazwsx123.uk",
    "https://104.207.154.5",
    "https://xpq.qazwsx123.uk",
]
allow_origins_env = os.environ.get("STOCKTOOL_ALLOW_ORIGINS")
ALLOW_ORIGINS = (
    [origin.strip() for origin in allow_origins_env.split(",") if origin.strip()]
    if allow_origins_env
    else DEFAULT_ALLOW_ORIGINS
)
# Default to permissive regex so any client IP can submit requests when credentials are needed.
# Use STOCKTOOL_ALLOW_ORIGIN_REGEX to tighten this in production.
allow_origin_regex = os.environ.get(
    "STOCKTOOL_ALLOW_ORIGIN_REGEX",
    r"^https?://.*$",
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=allow_origin_regex,
)

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="stocktool_session",
    max_age=SESSION_MAX_AGE,
    same_site=SESSION_SAME_SITE,
    https_only=SESSION_COOKIE_SECURE,
)

# --- State Management (Single User) ---
class AppState:
    df: Optional[pd.DataFrame] = None
    buy: Optional[pd.Series] = None
    sell: Optional[pd.Series] = None
    results: List[BacktestEntry] = []
    scores: Optional[pd.DataFrame] = None
    formula: str = ""
    ai_cache: Optional[Dict[str, Any]] = None

state = AppState()

# --- Authentication helpers ---

def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(USER_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _parse_iso_datetime(value: Optional[str], field_name: str = "datetime") -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} 格式非法，需为 ISO 8601") from exc


def _serialize_datetime(value: Optional[datetime]) -> Optional[str]:
    if not value:
        return None
    return value.replace(microsecond=0).isoformat()


def _row_to_user(row: sqlite3.Row) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "is_active": bool(row["is_active"]),
        "disabled_until": row["disabled_until"],
        "created_at": row["created_at"],
    }


def _ensure_users_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            disabled_until TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _hash_password(raw: str) -> str:
    if not raw:
        raise ValueError("密码不能为空")
    return PASSWORD_HASHER.hash(raw)


def _create_user_record(
    conn: sqlite3.Connection,
    username: str,
    password: str,
    role: str = "user",
    is_active: bool = True,
    disabled_until: Optional[str] = None,
) -> Dict[str, Any]:
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat()
    password_hash = _hash_password(password)
    conn.execute(
        """
        INSERT INTO users (username, password_hash, role, is_active, disabled_until, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (username, password_hash, role, 1 if is_active else 0, disabled_until, now_iso),
    )
    conn.commit()
    user_row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    return _row_to_user(user_row)


def _active_admin_count(conn: sqlite3.Connection, exclude_id: Optional[int] = None) -> int:
    sql = "SELECT COUNT(1) AS cnt FROM users WHERE role = 'admin' AND is_active = 1"
    params: List[Any] = []
    if exclude_id:
        sql += " AND id != ?"
        params.append(exclude_id)
    row = conn.execute(sql, params).fetchone()
    return int(row["cnt"]) if row else 0


def init_user_store() -> None:
    os.makedirs(BACKEND_ROOT, exist_ok=True)
    with sqlite3.connect(USER_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_users_table(conn)
        has_any = conn.execute("SELECT COUNT(1) AS cnt FROM users").fetchone() or {"cnt": 0}
        if has_any["cnt"]:
            return
        admin_username = os.environ.get("STOCKTOOL_ADMIN_USERNAME", "admin")
        admin_password = os.environ.get("STOCKTOOL_ADMIN_PASSWORD")
        if not admin_password:
            admin_password = "admin"
            print(
                "WARNING: STOCKTOOL_ADMIN_PASSWORD 未设置，已自动创建用户名 admin 密码 admin，请尽快修改",
            )
        _create_user_record(conn, admin_username, admin_password, role="admin", is_active=True)


@app.on_event("startup")
def _startup_user_store():
    init_user_store()

# --- Models ---

class StrategyConfig(BaseModel):
    fixed: Optional[List[int]] = None
    tpsl: Optional[Dict[str, float]] = None
    dca: Optional[Dict[str, float]] = None
    grid: Optional[Dict[str, Any]] = None
    dynamic: Optional[Dict[str, Any]] = None
    buy_hedge: Optional[Dict[str, Any]] = None

class BacktestRequest(BaseModel):
    csv_path: str
    formula: str
    initial_capital: float
    fee_rate: float
    strategies: StrategyConfig
    date_start: Optional[str] = None
    date_end: Optional[str] = None


class FormulaValidateRequest(BaseModel):
    csv_path: str
    formula: str
    date_start: Optional[str] = None
    date_end: Optional[str] = None

class PlotRequest(BaseModel):
    strategy_index: int

class MultiTimeframeRequest(BaseModel):
    freqs: Optional[List[str]] = None
    pairs: Optional[List["TimeframePairRequest"]] = None
    labels: Optional[Dict[str, str]] = None


class StrategyTextRequest(BaseModel):
    text: str

class SinaImportRequest(BaseModel):
    symbol: str
    scale: int = 240
    datalen: int = 500

class AIAnalysisRequest(BaseModel):
    asset_label: Optional[str] = None
    limit_rows: int = 60
    extra_note: Optional[str] = None


class TimeframePairRequest(BaseModel):
    trend: str
    entry: str
    label: Optional[str] = None


MultiTimeframeRequest.model_rebuild()


class ReportRequest(BaseModel):
    strategy_index: int = 0

class LoginRequest(BaseModel):
    username: str
    password: str

class UserSummary(BaseModel):
    id: int
    username: str
    role: str
    is_active: bool
    disabled_until: Optional[str] = None
    created_at: str

class AdminCreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"
    is_active: bool = True
    disabled_until: Optional[str] = None

class AdminUpdateUserRequest(BaseModel):
    role: Optional[str] = None
    is_active: Optional[bool] = None
    disabled_until: Optional[str] = None

class TemporaryDisableRequest(BaseModel):
    disabled_until: str


class RegisterRequest(BaseModel):
    username: str
    password: str


def _bypass_user() -> UserSummary:
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat()
    return UserSummary(
        id=0,
        username=_BYPASS_USERNAME,
        role=_BYPASS_ROLE,
        is_active=True,
        disabled_until=None,
        created_at=now_iso,
    )


class AdminResetPasswordRequest(BaseModel):
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

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

# --- Authentication dependencies & endpoints ---

def _ensure_user_active(user: Dict[str, Any]) -> None:
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="账户已被禁用")
    disabled_until = _parse_iso_datetime(user["disabled_until"])
    if disabled_until and disabled_until > datetime.utcnow():
        raise HTTPException(
            status_code=403,
            detail=f"账户临时禁用至 {disabled_until.strftime('%Y-%m-%d %H:%M:%S')}",
        )


def get_current_user(request: Request, db: sqlite3.Connection = Depends(get_db)) -> UserSummary:
    if DISABLE_AUTH:
        return _bypass_user()
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="未登录")
    row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        request.session.clear()
        raise HTTPException(status_code=401, detail="用户不存在")
    user = _row_to_user(row)
    if not user:
        raise HTTPException(status_code=401, detail="用户信息异常")
    _ensure_user_active(user)
    return UserSummary(**user)


def require_role(role: str):
    def _dependency(user: UserSummary = Depends(get_current_user)) -> UserSummary:
        if user.role != role:
            raise HTTPException(status_code=403, detail="权限不足")
        return user

    return _dependency


@app.post("/register", response_model=UserSummary)
def register(payload: RegisterRequest, db: sqlite3.Connection = Depends(get_db)):
    username = (payload.username or "").strip()
    if not username:
        raise HTTPException(status_code=400, detail="用户名不能为空")
    password = payload.password or ""
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="密码至少需要 6 个字符")
    if db.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
        raise HTTPException(status_code=400, detail="用户名已存在")
    user = _create_user_record(db, username, password, role="user", is_active=True)
    return UserSummary(**user)


@app.post("/login", response_model=UserSummary)
def login(payload: LoginRequest, request: Request, db: sqlite3.Connection = Depends(get_db)):
    if DISABLE_AUTH:
        request.session.clear()
        return _bypass_user()
    row = db.execute("SELECT * FROM users WHERE username = ?", (payload.username,)).fetchone()
    if not row:
        time.sleep(0.3)
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    user = _row_to_user(row)
    if not user:
        raise HTTPException(status_code=401, detail="用户信息异常")
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="账户已被禁用")
    disabled_until = _parse_iso_datetime(user["disabled_until"])
    if disabled_until and disabled_until > datetime.utcnow():
        raise HTTPException(
            status_code=403,
            detail=f"账户临时禁用至 {disabled_until.strftime('%Y-%m-%d %H:%M:%S')}",
        )
    try:
        PASSWORD_HASHER.verify(row["password_hash"], payload.password)
    except VerifyMismatchError:
        time.sleep(0.3)
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    except Exception:
        time.sleep(0.3)
        raise HTTPException(status_code=401, detail="认证失败")

    if PASSWORD_HASHER.check_needs_rehash(row["password_hash"]):
        new_hash = PASSWORD_HASHER.hash(payload.password)
        db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, row["id"]))
        db.commit()

    request.session.clear()
    request.session["user_id"] = row["id"]
    request.session["role"] = row["role"]
    return UserSummary(**_row_to_user(row))


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return {"status": "ok"}


@app.get("/me", response_model=UserSummary)
def read_me(current_user: UserSummary = Depends(get_current_user)):
    return current_user


@app.post("/me/change-password")
def change_password_for_current_user(
    payload: ChangePasswordRequest,
    current_user: UserSummary = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
):
    current_password = payload.current_password or ""
    new_password = payload.new_password or ""
    if not current_password:
        raise HTTPException(status_code=400, detail="当前密码不能为空")
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="新密码至少需要 6 个字符")
    row = db.execute("SELECT * FROM users WHERE id = ?", (current_user.id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="用户不存在")
    try:
        PASSWORD_HASHER.verify(row["password_hash"], current_password)
    except VerifyMismatchError:
        raise HTTPException(status_code=403, detail="当前密码不正确")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"当前密码校验失败：{exc}") from exc
    db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (_hash_password(new_password), current_user.id))
    db.commit()
    return {"status": "ok"}


@app.get("/admin/users", response_model=List[UserSummary])
def list_users(current_user: UserSummary = Depends(require_role("admin")), db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    return [UserSummary(**_row_to_user(row)) for row in rows if row]


@app.post("/admin/users", response_model=UserSummary)
def create_user(payload: AdminCreateUserRequest, current_user: UserSummary = Depends(require_role("admin")), db: sqlite3.Connection = Depends(get_db)):
    if db.execute("SELECT 1 FROM users WHERE username = ?", (payload.username,)).fetchone():
        raise HTTPException(status_code=400, detail="用户名已存在")
    disabled_until = None
    if payload.disabled_until:
        disabled_until = _serialize_datetime(_parse_iso_datetime(payload.disabled_until, field_name="disabled_until"))
    return UserSummary(**_create_user_record(db, payload.username, payload.password, payload.role, payload.is_active, disabled_until))


def _update_user_state(conn: sqlite3.Connection, user_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
    if not updates:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _row_to_user(row)
    setters = ", ".join(f"{col} = ?" for col in updates.keys())
    params = list(updates.values()) + [user_id]
    conn.execute(f"UPDATE users SET {setters} WHERE id = ?", params)
    conn.commit()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _row_to_user(row)


@app.put("/admin/users/{user_id}", response_model=UserSummary)
def update_user(user_id: int, payload: AdminUpdateUserRequest, current_user: UserSummary = Depends(require_role("admin")), db: sqlite3.Connection = Depends(get_db)):
    row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="用户不存在")
    if current_user.id == user_id and payload.is_active is False:
        raise HTTPException(status_code=400, detail="不能禁用自己")
    updates: Dict[str, Any] = {}
    was_admin_active = row["role"] == "admin" and bool(row["is_active"])
    new_role = row["role"]
    new_is_active = bool(row["is_active"])
    if payload.role is not None:
        updates["role"] = payload.role
        new_role = payload.role
    if payload.is_active is not None:
        updates["is_active"] = 1 if payload.is_active else 0
        new_is_active = payload.is_active
    if payload.disabled_until is not None:
        if payload.disabled_until == "":
            updates["disabled_until"] = None
        else:
            parsed = _parse_iso_datetime(payload.disabled_until, field_name="disabled_until")
            updates["disabled_until"] = _serialize_datetime(parsed)
    # Prevent removing the last active admin
    if was_admin_active and (not new_is_active or new_role != "admin"):
        other_admins = _active_admin_count(db, exclude_id=user_id)
        if other_admins == 0:
            raise HTTPException(status_code=400, detail="至少需要一个管理员账号")
    user = _update_user_state(db, user_id, updates)
    if not user:
        raise HTTPException(status_code=500, detail="更新失败")
    return UserSummary(**user)


@app.post("/admin/users/{user_id}/disable")
def disable_user(user_id: int, current_user: UserSummary = Depends(require_role("admin")), db: sqlite3.Connection = Depends(get_db)):
    if current_user.id == user_id:
        raise HTTPException(status_code=400, detail="不能禁用自己")
    row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="用户不存在")
    if row["role"] == "admin":
        other_admins = _active_admin_count(db, exclude_id=user_id)
        if other_admins == 0:
            raise HTTPException(status_code=400, detail="至少需要一个管理员账号")
    db.execute("UPDATE users SET is_active = 0, disabled_until = NULL WHERE id = ?", (user_id,))
    db.commit()
    return {"status": "ok"}


@app.post("/admin/users/{user_id}/disable-temporary")
def disable_user_temporarily(user_id: int, payload: TemporaryDisableRequest, current_user: UserSummary = Depends(require_role("admin")), db: sqlite3.Connection = Depends(get_db)):
    if current_user.id == user_id:
        raise HTTPException(status_code=400, detail="不能禁用自己")
    parsed = _parse_iso_datetime(payload.disabled_until, field_name="disabled_until")
    db.execute(
        "UPDATE users SET disabled_until = ?, is_active = 1 WHERE id = ?",
        (_serialize_datetime(parsed), user_id),
    )
    db.commit()
    return {"status": "ok"}


@app.post("/admin/users/{user_id}/enable")
def enable_user(user_id: int, current_user: UserSummary = Depends(require_role("admin")), db: sqlite3.Connection = Depends(get_db)):
    row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="用户不存在")
    db.execute("UPDATE users SET is_active = 1, disabled_until = NULL WHERE id = ?", (user_id,))
    db.commit()
    return {"status": "ok"}


@app.post("/admin/users/{user_id}/reset-password")
def reset_user_password(user_id: int, payload: AdminResetPasswordRequest, current_user: UserSummary = Depends(require_role("admin")), db: sqlite3.Connection = Depends(get_db)):
    if len(payload.password or "") < 6:
        raise HTTPException(status_code=400, detail="密码至少需要 6 个字符")
    row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="用户不存在")
    db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (_hash_password(payload.password), user_id))
    db.commit()
    return {"status": "ok"}


@app.delete("/admin/users/{user_id}")
def delete_user(user_id: int, current_user: UserSummary = Depends(require_role("admin")), db: sqlite3.Connection = Depends(get_db)):
    if current_user.id == user_id:
        raise HTTPException(status_code=400, detail="不能删除自己")
    row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="用户不存在")
    if row["role"] == "admin":
        other_admins = _active_admin_count(db, exclude_id=user_id)
        if other_admins == 0:
            raise HTTPException(status_code=400, detail="删除前请先保留至少一名管理员")
    db.execute("DELETE FROM users WHERE id = ?", (user_id,))
    db.commit()
    return {"status": "ok"}

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


def _ensure_price_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to open/high/low/close/volume."""
    mapping_candidates = {
        "open": ["open", "Open", "OPEN", "o"],
        "high": ["high", "High", "HIGH", "h"],
        "low": ["low", "Low", "LOW", "l"],
        "close": ["close", "Close", "CLOSE", "c", "price"],
        "volume": ["volume", "Volume", "VOLUME", "vol", "Vol", "VOL"],
    }
    for target, candidates in mapping_candidates.items():
        if target in frame.columns:
            continue
        for cand in candidates:
            if cand in frame.columns:
                frame[target] = frame[cand]
                break
        else:
            frame[target] = 0.0
    return frame


def _align_signal_column(series, reference_index) -> pd.Series:
    if series is None:
        return pd.Series([False] * len(reference_index), index=reference_index)
    try:
        aligned = series.reindex(reference_index)
    except Exception:
        try:
            aligned = pd.Series(series).reindex(reference_index)
        except Exception:
            aligned = pd.Series([False] * len(reference_index), index=reference_index)
    return aligned.fillna(False).astype(bool)


def _summarize_recent_frame(frame: pd.DataFrame, asset_label: str) -> Dict[str, Any]:
    close_series = frame["close"].astype(float)
    volume_series = frame["volume"].astype(float)
    if close_series.empty:
        return {}
    start_close = close_series.iloc[0]
    end_close = close_series.iloc[-1]
    total_return = (end_close / start_close - 1) * 100 if start_close else 0
    high_val = float(close_series.max())
    low_val = float(close_series.min())
    returns = close_series.pct_change().dropna()
    volatility = float(returns.std() * (len(returns) ** 0.5)) * 100 if not returns.empty else 0
    avg_volume = float(volume_series.mean()) if not volume_series.empty else 0
    return {
        "asset": asset_label,
        "sample_count": len(frame),
        "date_range": f"{frame['date'].iloc[0]} 至 {frame['date'].iloc[-1]}",
        "start_close": float(start_close),
        "end_close": float(end_close),
        "total_return_pct": round(total_return, 2),
        "max_close": round(high_val, 2),
        "min_close": round(low_val, 2),
        "volatility_pct": round(volatility, 2),
        "avg_volume": round(avg_volume, 2),
    }

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
    if getattr(res, "baseline_equity_curve", None) is not None:
        base_times = res.baseline_equity_curve.index.astype(str).tolist()
        base_values = res.baseline_equity_curve.values.tolist()
        payload["equityCurveOriginal"] = list(zip(base_times, base_values))
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
    if getattr(res, "buy_hedge_summary", None):
        payload["buyHedgeSummary"] = res.buy_hedge_summary
    if getattr(res, "buy_hedge_trades", None):
        payload["buyHedgeTrades"] = res.buy_hedge_trades
    if getattr(res, "buy_hedge_events", None):
        payload["buyHedgeEvents"] = res.buy_hedge_events
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
    if req.strategies.buy_hedge:
        strategies["buy_hedge"] = req.strategies.buy_hedge

    if not strategies:
        raise HTTPException(status_code=400, detail="No strategies selected")

    params = BacktestParams(
        csv_path=req.csv_path,
        formula=req.formula,
        initial_capital=req.initial_capital,
        fee_rate=req.fee_rate,
        strategies=strategies,
        start_date=req.date_start,
        end_date=req.date_end,
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
    state.ai_cache = None

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


@app.post("/formula/validate")
def api_validate_formula(req: FormulaValidateRequest):
    logs: List[str] = []
    try:
        df = load_price_csv(req.csv_path)
        required_cols = {"date", "open", "high", "low", "close"}
        if not required_cols.issubset(df.columns):
            logs.append(f"ERROR: CSV 至少需要列：{', '.join(required_cols)}")
            return {"status": "failed", "logs": logs, "buy_count": 0, "sell_count": 0}

        df["date"] = pd.to_datetime(df["date"])
        if req.date_start or req.date_end:
            start_ts = pd.to_datetime(req.date_start) if req.date_start else None
            end_ts = pd.to_datetime(req.date_end) if req.date_end else None
            mask = pd.Series(True, index=df.index)
            if start_ts is not None:
                mask &= df["date"] >= start_ts
            if end_ts is not None:
                mask &= df["date"] <= end_ts
            df = df.loc[mask].copy()

        engine = TdxFormulaEngine(df.copy())
        buy, sell = engine.run(req.formula)
        logs.extend(getattr(engine, "logs", []) or [])

        buy_count = int(buy.astype(bool).sum()) if isinstance(buy, pd.Series) else 0
        sell_count = int(sell.astype(bool).sum()) if isinstance(sell, pd.Series) else 0
        return {
            "status": "success",
            "logs": logs,
            "buy_count": buy_count,
            "sell_count": sell_count,
        }
    except Exception as e:  # noqa: BLE001
        logs.append(f"ERROR: {type(e).__name__}: {e}")
        return {"status": "failed", "logs": logs, "buy_count": 0, "sell_count": 0}

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


def _frame_to_kline(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    reset = frame.reset_index()
    if "index" in reset.columns and "date" not in reset.columns:
        reset.rename(columns={"index": "date"}, inplace=True)
    if "date" not in reset.columns:
        reset["date"] = reset.index.astype(str)
    else:
        reset["date"] = reset["date"].astype(str)
    cols = [col for col in ["date", "open", "high", "low", "close"] if col in reset.columns]
    return reset[cols].to_dict(orient="records")


def _build_trend_bias(buy: pd.Series, sell: pd.Series) -> pd.Series:
    state_val = 0
    values = []
    index = buy.index
    for idx in index:
        if bool(buy.loc[idx]):
            state_val = 1
        elif bool(sell.loc[idx]):
            state_val = -1
        values.append(state_val)
    return pd.Series(values, index=index, dtype=float)


def _build_pair_dataset(
    trend_key: str,
    entry_key: str,
    frames: Dict[str, pd.DataFrame],
    signals: Dict[str, Dict[str, pd.Series]],
    custom_label: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    trend_frame = frames.get(trend_key)
    entry_frame = frames.get(entry_key)
    trend_signals = signals.get(trend_key)
    entry_signals = signals.get(entry_key)
    if (
        trend_frame is None
        or entry_frame is None
        or trend_signals is None
        or entry_signals is None
    ):
        return None

    trend_buy = trend_signals["buy"].reindex(trend_frame.index).fillna(False)
    trend_sell = trend_signals["sell"].reindex(trend_frame.index).fillna(False)
    entry_buy = entry_signals["buy"].reindex(entry_frame.index).fillna(False)
    entry_sell = entry_signals["sell"].reindex(entry_frame.index).fillna(False)

    trend_bias = _build_trend_bias(trend_buy, trend_sell)
    aligned_bias = trend_bias.reindex(entry_frame.index, method="ffill").fillna(0)
    valid_buy = entry_buy & (aligned_bias >= 0.5)
    valid_sell = entry_sell & (aligned_bias <= -0.5)
    label = custom_label or f"{trend_key}->{entry_key}"

    return {
        "freq": label,
        "label": custom_label or label,
        "trend": trend_key,
        "entry": entry_key,
        "pair": True,
        "kline": _frame_to_kline(entry_frame),
        "buy_signals": valid_buy[valid_buy].index.astype(str).tolist(),
        "sell_signals": valid_sell[valid_sell].index.astype(str).tolist(),
        "meta": {
            "pair_label": label,
            "trend_bias": [
                {"date": str(idx), "state": int(state)}
                for idx, state in aligned_bias.tail(120).items()
            ],
        },
    }


@app.post("/analytics/multi_timeframe")
def get_multi_timeframe(req: MultiTimeframeRequest):
    if state.df is None or not state.formula:
        raise HTTPException(status_code=400, detail="Run backtest first")
    freqs = req.freqs or []
    signals, frames, meta = generate_multi_timeframe_signals(state.df, state.formula, freqs)
    if not frames:
        raise HTTPException(status_code=400, detail="数据分辨率不足以生成所需周期")
    payload = []
    for freq_key, frame in frames.items():
        freq_signals = signals.get(freq_key, {})
        buy_series = freq_signals.get("buy", pd.Series(dtype=bool)).reindex(frame.index).fillna(False)
        sell_series = freq_signals.get("sell", pd.Series(dtype=bool)).reindex(frame.index).fillna(False)
        payload.append({
            "freq": freq_key,
            "label": (req.labels or {}).get(freq_key, freq_key),
            "kline": _frame_to_kline(frame),
            "buy_signals": buy_series[buy_series].index.astype(str).tolist(),
            "sell_signals": sell_series[sell_series].index.astype(str).tolist(),
        })

    pair_payload = []
    skipped_pairs: List[Dict[str, str]] = []
    for pair in req.pairs or []:
        try:
            trend_label, _ = normalize_timeframe_token(pair.trend)
            entry_label, _ = normalize_timeframe_token(pair.entry)
        except ValueError as exc:
            skipped_pairs.append({"pair": f"{pair.trend}->{pair.entry}", "reason": str(exc)})
            continue
        dataset = _build_pair_dataset(trend_label, entry_label, frames, signals, pair.label)
        if dataset:
            pair_payload.append(dataset)
        else:
            skipped_pairs.append({"pair": f"{trend_label}->{entry_label}", "reason": "周期数据缺失"})

    meta["skipped_pairs"] = skipped_pairs
    return {"series": payload, "pairs": pair_payload, "meta": meta}


@app.post("/analytics/nlp_formula")
def generate_formula(req: StrategyTextRequest):
    try:
        formula = simple_rule_based_formula(req.text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))
    return {"formula": formula}


@app.post("/analytics/performance_report")
def get_performance_report(req: ReportRequest):
    if not state.results:
        raise HTTPException(status_code=400, detail="请先运行回测")
    idx = max(0, min(req.strategy_index, len(state.results) - 1))
    entry = state.results[idx]
    try:
        report = generate_performance_report(entry.result)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"无法生成报告：{exc}") from exc
    report["strategy"] = {"name": entry.name, "title": entry.title}
    return report


@app.post("/analytics/ai_insight")
def generate_ai_insight(req: AIAnalysisRequest):
    if state.df is None:
        raise HTTPException(status_code=400, detail="Run backtest first")

    limit = req.limit_rows or 60
    limit = max(40, min(limit, 200))
    asset_label = (req.asset_label or "").strip() or "当前标的"

    recent = state.df.tail(limit).copy()
    if recent.empty:
        raise HTTPException(status_code=400, detail="数据不足，无法生成分析")

    if "date" in recent.columns:
        recent_frame = recent.reset_index(drop=True)
    else:
        recent_frame = recent.reset_index()
        if "index" in recent_frame.columns:
            recent_frame.rename(columns={"index": "date"}, inplace=True)
    if "date" not in recent_frame.columns:
        recent_frame["date"] = recent_frame.index.astype(str)
    else:
        recent_frame["date"] = recent_frame["date"].astype(str)

    # Align signals
    buy_series = _align_signal_column(state.buy, recent.index) if state.buy is not None else None
    sell_series = _align_signal_column(state.sell, recent.index) if state.sell is not None else None
    recent_frame.reset_index(drop=True, inplace=True)
    if buy_series is not None and len(buy_series) == len(recent_frame):
        recent_frame["buy_signal"] = list(buy_series)
    else:
        recent_frame["buy_signal"] = False
    if sell_series is not None and len(sell_series) == len(recent_frame):
        recent_frame["sell_signal"] = list(sell_series)
    else:
        recent_frame["sell_signal"] = False

    recent_frame = _ensure_price_columns(recent_frame)

    records = []
    for _, row in recent_frame.iterrows():
        records.append(
            {
                "date": str(row["date"]),
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": float(row.get("volume", 0)),
                "buy_signal": bool(row.get("buy_signal", False)),
                "sell_signal": bool(row.get("sell_signal", False)),
            }
        )

    stats_summary = _summarize_recent_frame(recent_frame, asset_label)
    if not stats_summary:
        raise HTTPException(status_code=400, detail="无法汇总行情数据")

    signature_payload = {
        "asset": asset_label,
        "records": records,
        "note": req.extra_note or "",
    }
    signature = hashlib.sha256(json.dumps(signature_payload, ensure_ascii=False).encode("utf-8")).hexdigest()
    cached_ai = state.ai_cache
    if cached_ai and cached_ai.get("signature") == signature:
        cached_response = cached_ai.get("response", {})
        if "signature" not in cached_response:
            cached_response = {**cached_response, "signature": signature}
            cached_ai["response"] = cached_response
        return cached_response

    system_prompt = (
        "你是资深证券分析师，擅长根据K线和成交量做出专业、审慎的中文研判。"
        "输出需包含：1) 行情背景与关键指标；2) 多角度走势解读；3) 风险提示；4) 未来短/中/长期展望。"
        "语言应正式、结构化，可使用有序列表，每个分点尽量控制在2-3句话。"
    )
    user_prompt = (
        f"标的：{asset_label}\\n"
        f"样本区间：{stats_summary['date_range']}，共 {stats_summary['sample_count']} 条K线。\\n"
        f"区间涨跌幅：{stats_summary['total_return_pct']}% ，波动率约 {stats_summary['volatility_pct']}%。\\n"
        f"数据（含买入/卖出布尔信号）JSON：{json.dumps(records, ensure_ascii=False)}\\n"
        "请严格结合数据阐述："
        "① 最近行情与量价特征；② 买卖信号或形态的含义；③ 风险点；④ 接下来短/中/长期的走势推演。"
    )
    if req.extra_note:
        user_prompt += f"附加背景：{req.extra_note}\\n"

    payload = {
        "model": AI_MODEL,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    try:
        resp = requests.post(
            f"{AI_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {AI_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=AI_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"AI 接口请求失败：{exc}") from exc

    try:
        data = resp.json()
        analysis_text = data["choices"][0]["message"]["content"].strip()
    except (ValueError, KeyError, IndexError, TypeError) as exc:  # noqa: PERF203
        raise HTTPException(status_code=500, detail=f"AI 返回格式异常：{exc}") from exc

    result = {
        "analysis": analysis_text,
        "stats": stats_summary,
        "generated_at": time.time(),
        "signature": signature,
    }
    state.ai_cache = {"signature": signature, "response": result}
    return result
