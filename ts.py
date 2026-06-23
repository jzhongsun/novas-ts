"""
Tushare A-share data pipeline — per-day, per-month, incremental.

Design (differs from baostock because tushare fetches ALL stocks per trade_date):
  - Fetch one trade date at a time: pro.daily + pro.daily_basic + pro.adj_factor,
    merged inline into one row-per-stock-per-day record.
  - Append directly into the month's parquet file (stock-day-YYYYMM.parquet).
  - Incremental: skip trade dates already present in the month file.
  - Parallelizable: each month is an independent worker unit (spawn context).
  - No all-in-memory merge; memory holds ~1 month at a time.

Subcommands: basic, klines, sw, sw-classify, calendar, forecast, index, dataset, upload

Setup:
    pip install tushare pandas pyarrow modelscope
    export TUSHARE_TOKEN=...        # https://tushare.pro
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterator

import pandas as pd

try:
    import pyarrow  # noqa: F401
except ImportError:
    raise ImportError("pip install pyarrow")


OUTPUT_DIR = "./output"
MAX_RETRIES = 5


# ── helpers ──────────────────────────────────────────────────────────────────

def _today() -> str:
    from datetime import date
    return date.today().strftime("%Y%m%d")


def _next_year_end() -> str:
    """Return Dec 31 of next year, e.g. 20261231 in 2026 — so calendar covers future dates."""
    from datetime import date
    return date(date.today().year + 1, 12, 31).strftime("%Y%m%d")


_PRO = None
_PRO_LOCK = None


def _pro():
    """Thread-safe lazy init of the shared tushare pro client."""
    global _PRO, _PRO_LOCK
    if _PRO is not None:
        return _PRO
    import threading
    if _PRO_LOCK is None:
        _PRO_LOCK = threading.Lock()
    with _PRO_LOCK:
        if _PRO is not None:
            return _PRO
        try:
            import tushare as ts
        except ImportError:
            raise ImportError("pip install tushare")
        token = os.environ.get("TUSHARE_TOKEN", "")
        if not token:
            raise RuntimeError("Set TUSHARE_TOKEN env var (https://tushare.pro)")
        base_url = os.environ.get("TUSHARE_BASEURL", "")
        _log(f"tushare: initializing pro_api (base_url={'custom' if base_url else 'default'})")
        ts.set_token(token)
        _PRO = ts.pro_api()
        if base_url:
            _PRO._DataApi__http_url = base_url
    return _PRO


def _fmt_date(d) -> str:
    """20180726 -> 2018-07-26."""
    s = str(d)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def _log(msg: str) -> None:
    """Timestamped one-line log to stdout (flushed). For milestone/summary lines."""
    from datetime import datetime
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def _call(fn, **kwargs):
    """Call a tushare pro function with retry on rate-limit."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(**kwargs)
        except Exception as e:
            msg = str(e)
            if attempt < MAX_RETRIES:
                wait = 10 * attempt
                print(f"  rate-limited, sleeping {wait}s ({attempt}/{MAX_RETRIES})", flush=True)
                time.sleep(wait)
            else:
                _log(f"  call {fn.__name__} failed: {msg}")
                raise


# ── low-level fetchers ───────────────────────────────────────────────────────

def fetch_stock_basic() -> pd.DataFrame:
    """All stock basic info (listed + delisted + paused), all documented fields."""
    fields = (
        "ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,"
        "exchange,curr_type,list_status,list_date,delist_date,is_hs,"
        "act_name,act_ent_type"
    )
    pro = _pro()
    frames = []
    for status in ("L", "D", "P"):
        try:
            df = _call(pro.stock_basic, exchange="", list_status=status, fields=fields)
            if df is not None and not df.empty:
                _log(f"stock_basic list_status={status}: {len(df)} rows")
                frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["ts_code"], keep="last").reset_index(drop=True)
    _log(f"stock_basic: {len(df)} unique stocks")
    return df


def fetch_calendar_dates(start_date: str, end_date: str) -> pd.DataFrame:
    """SSE/SZSE trade calendar. start/end as YYYYMMDD."""
    pro = _pro()
    df = _call(pro.trade_cal, exchange="SSE", start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        _log(f"trade_cal {start_date}~{end_date}: empty")
        return pd.DataFrame()
    n_open = int((df["is_open"] == 1).sum()) if "is_open" in df.columns else 0
    _log(f"trade_cal {start_date}~{end_date}: {len(df)} rows, {n_open} open days")
    return df


# ── per-day fetch + per-month store ──────────────────────────────────────────

def _fetch_day(pro, trade_date: str, with_basic: bool, with_adj: bool, with_st: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch all stocks for ONE trade date: daily + daily_basic + adj_factor + ST list."""

    df = _call(pro.daily, trade_date=trade_date)
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    daily_df: pd.DataFrame = df
    basic_df: pd.DataFrame = pd.DataFrame()
    adj_df: pd.DataFrame = pd.DataFrame()
    st_df: pd.DataFrame = pd.DataFrame()

    if with_basic:
        try:
            vb = _call(pro.daily_basic, trade_date=trade_date)
            if vb is not None and not vb.empty:
                keep = ["ts_code", "trade_date", "pe_ttm", "pb", "ps_ttm",
                        "turnover_rate", "dv_ratio", "circ_mv", "total_mv"]
                vb = vb[[c for c in keep if c in vb.columns]]
                basic_df = vb
        except Exception as err:
            _log(f"  daily_basic {trade_date} failed: {err}")
            pass

    if with_adj:
        try:
            af = _call(pro.adj_factor, trade_date=trade_date)
            if af is not None and not af.empty:
                adj_df = af
        except Exception as err:
            _log(f"  adj_factor {trade_date} failed: {err}")
            pass

    if with_st:
        try:
            # stock_st covers from 20160101; earlier dates simply return empty.
            # One request caps at 1000 rows — a single day's ST list is well
            # below that, so one call per trade_date is sufficient.
            st = _call(pro.stock_st, trade_date=trade_date)
            if st is not None and not st.empty:
                st_df = st
        except Exception as err:
            _log(f"  stock_st {trade_date} failed: {err}")
            pass

    return daily_df, basic_df, adj_df, st_df


def _process_month(
    yyyymm: str,
    days: list[str],
    output_dir: str,
    with_basic: bool,
    with_adj: bool,
    force: bool,
    workers: int = 1,
    with_st: bool = True,
) -> int:
    """Process one month: load existing daily/basic/adj/st files, fetch missing days, save.

    daily is the primary file (always fetched; drives incremental progress).
    basic/adj/st are optional side files fetched when their with_* flag is set.

    Missing trade dates are fetched in parallel (workers threads) — tushare is
    pure I/O HTTP, so concurrent per-day fetches are far more efficient than
    serializing within the month.

    Returns count of newly fetched day-records.
    """
    pro = _pro()
    daily_month_file = Path(output_dir) / "daily" / f"daily-{yyyymm}.parquet"
    basic_month_file = Path(output_dir) / "daily_basic" / f"daily_basic-{yyyymm}.parquet"
    adj_month_file = Path(output_dir) / "adj_factor" / f"adj_factor-{yyyymm}.parquet"
    st_month_file = Path(output_dir) / "stock_st" / f"stock_st-{yyyymm}.parquet"

    def _load(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(path, engine="pyarrow")
        except Exception:
            return pd.DataFrame()

    existing_daily = _load(daily_month_file)
    existing_basic = _load(basic_month_file)
    existing_adj = _load(adj_month_file)
    existing_st = _load(st_month_file)

    _log(f"[{yyyymm}] start: existing daily={len(existing_daily)} basic={len(existing_basic)} "
         f"adj={len(existing_adj)} st={len(existing_st)} rows; days={len(days)} "
         f"basic={with_basic} adj={with_adj} st={with_st}")

    # If requested side data is absent from existing files, refetch the whole month.
    need_refetch = force
    if not existing_daily.empty:
        if with_adj and existing_adj.empty:
            need_refetch = True
        if with_basic and existing_basic.empty:
            need_refetch = True
        if with_st and existing_st.empty and yyyymm >= "201601":
            need_refetch = True
    if need_refetch:
        reason = "forced" if force else "missing requested side data"
        _log(f"[{yyyymm}] full-month refetch ({reason})")

    if need_refetch or existing_daily.empty:
        present: set[str] = set()
    else:
        present = set(existing_daily["trade_date"].astype(str))

    missing = [d for d in days if str(d) not in present]
    if not missing:
        _log(f"[{yyyymm}] up to date ({len(days)} days)")
        return 0
    _log(f"[{yyyymm}] {len(missing)}/{len(days)} days to fetch (workers={workers})")

    new_daily: list[pd.DataFrame] = []
    new_basic: list[pd.DataFrame] = []
    new_adj: list[pd.DataFrame] = []
    new_st: list[pd.DataFrame] = []

    def _collect(dfs: tuple) -> None:
        """Append non-empty frames from a (daily, basic, adj, st) result."""
        daily_df, basic_df, adj_df, st_df = dfs
        if daily_df is not None and not daily_df.empty:
            new_daily.append(daily_df)
        if basic_df is not None and not basic_df.empty:
            new_basic.append(basic_df)
        if adj_df is not None and not adj_df.empty:
            new_adj.append(adj_df)
        if st_df is not None and not st_df.empty:
            new_st.append(st_df)

    def _fetch_one(d: str) -> tuple:
        """Fetch one trade date; returns (daily, basic, adj, st), all None on failure."""
        try:
            return _fetch_day(pro, str(d), with_basic, with_adj, with_st)
        except Exception as e:
            print(f"[{yyyymm}] {d} failed: {e}", flush=True)
            return (None, None, None, None)

    n = len(missing)
    if workers and workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_fetch_one, d): d for d in missing}
            done = 0
            for fut in as_completed(futures):
                _collect(fut.result())
                done += 1
                if done % 10 == 0 or done == n:
                    print(f"[{yyyymm}] {done}/{n} days fetched", flush=True)
    else:
        for i, d in enumerate(missing, start=1):
            _collect(_fetch_one(d))
            if i % 10 == 0 or i == n:
                print(f"[{yyyymm}] {i}/{n} days fetched", flush=True)

    if not new_daily:
        _log(f"[{yyyymm}] no daily rows fetched (all days empty/failed)")
        return 0

    _log(f"[{yyyymm}] fetched new daily={sum(len(f) for f in new_daily)} "
         f"basic={sum(len(f) for f in new_basic)} adj={sum(len(f) for f in new_adj)} "
         f"st={sum(len(f) for f in new_st)} rows")

    def _merge(existing: pd.DataFrame, frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Concat existing + new, dedup by (ts_code, trade_date), sort."""
        new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if need_refetch or existing.empty:
            combined = new  # refetched full month (or first write)
        else:
            combined = pd.concat([existing, new], ignore_index=True)
        if combined.empty:
            return combined
        combined = combined.drop_duplicates(
            subset=["ts_code", "trade_date"], keep="last")
        combined = combined.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        return combined

    def _save(path: Path, df: pd.DataFrame) -> None:
        if df.empty:
            return  # never write/clobber an empty side file
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, engine="pyarrow")

    combined_daily = _merge(existing_daily, new_daily)
    combined_basic = _merge(existing_basic, new_basic)
    combined_adj = _merge(existing_adj, new_adj)
    combined_st = _merge(existing_st, new_st)

    _save(daily_month_file, combined_daily)
    _save(basic_month_file, combined_basic)
    _save(adj_month_file, combined_adj)
    _save(st_month_file, combined_st)

    new_total = int(sum(len(f) for f in new_daily))
    print(
        f"[{yyyymm}] saved daily={len(combined_daily)} basic={len(combined_basic)} "
        f"adj={len(combined_adj)} st={len(combined_st)} rows ({len(missing)} new days)",
        flush=True,
    )
    return new_total


# Each kline kind is written to its own subdir with its own file prefix.
# flag None = always produced; otherwise gated by the named download flag.
_KLINE_KINDS = [
    {"kind": "daily",       "subdir": "daily",       "prefix": "daily",       "flag": None},
    {"kind": "daily_basic", "subdir": "daily_basic", "prefix": "daily_basic", "flag": "with_basic"},
    {"kind": "adj_factor",  "subdir": "adj_factor",  "prefix": "adj_factor",  "flag": "with_adj"},
    {"kind": "st",          "subdir": "stock_st",    "prefix": "stock_st",     "flag": "with_st"},
]


def _kline_uploads(output_dir: str, with_basic: bool, with_adj: bool, with_st: bool = True) -> list[dict]:
    """Build per-kind upload specs for files actually produced this run.

    A kind is included only when its flag is on AND its subdir contains at
    least one matching parquet file (so skipped flags or empty runs upload
    nothing for that kind).
    """
    flags = {None: True, "with_basic": with_basic, "with_adj": with_adj, "with_st": with_st}
    uploads: list[dict] = []
    for k in _KLINE_KINDS:
        if not flags[k["flag"]]:
            continue
        sub = Path(output_dir) / k["subdir"]
        if not sub.exists() or not any(sub.glob(f"{k['prefix']}-*.parquet")):
            continue
        uploads.append({
            "kind": k["kind"],
            "local_dir": str(sub),
            "allow_patterns": f"{k['prefix']}-*.parquet",
            "path_in_repo": k["subdir"],
        })
    return uploads


def download_all_stocks_klines(
    start_date: str,
    end_date: str,
    with_basic: bool = True,
    with_adj: bool = True,
    force: bool = False,
    workers: int = 1,
    with_st: bool = True,
) -> list[dict]:
    """Bulk-download daily (incl. valuation + adj_factor + ST list) per trade date, per month.

    Incremental by default: skips trade dates already stored in each month file.
    Months are processed serially; within each month the missing trade dates are
    fetched concurrently (workers threads) by _process_month.

    Returns a list of per-kind upload specs (daily / daily_basic / adj_factor / st)
    for the subdirectories that actually received files this run.
    """
    cal = fetch_calendar_dates(start_date, end_date)
    if cal.empty:
        raise RuntimeError("No trading calendar")
    days = cal[cal["is_open"] == 1]["cal_date"].astype(str).tolist()

    by_month: dict[str, list[str]] = {}
    for d in days:
        by_month.setdefault(d[:6], []).append(d)
    months = sorted(by_month.items())

    _log(f"klines: {len(months)} months, {len(days)} trade days, "
         f"workers={workers}, force={force}, basic={with_basic}, "
         f"adj={with_adj}, st={with_st}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Months are processed serially; parallelism lives inside _process_month,
    # which fetches each month's missing trade days concurrently (workers threads).
    total_new = 0
    for m, mdays in months:
        total_new += _process_month(
            m, mdays, OUTPUT_DIR, with_basic, with_adj, force, workers, with_st)

    _log(f"klines done: {total_new} new day-records across {len(months)} months")
    return _kline_uploads(OUTPUT_DIR, with_basic, with_adj, with_st)


# ── Shenwan industry-index daily (pro.sw_daily), per-day per-month ───────────

def _fetch_sw_day(pro, trade_date: str) -> pd.DataFrame | None:
    """Shenwan (2021) industry-index daily quotes for ONE trade date, all fields.

    A single trade_date returns ~400 indices, well under the 4000-row cap, so
    one call per day is sufficient.
    """
    try:
        df = _call(pro.sw_daily, trade_date=trade_date)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        print(f"  sw_daily {trade_date} failed: {e}", flush=True)
    return None


def _process_sw_month(
    pro,
    yyyymm: str,
    days: list[str],
    sw_dir: Path,
    force: bool,
    workers: int,
) -> int:
    """Process one month of sw_daily: load existing file, fetch missing days, save.

    Returns count of newly fetched rows.
    """
    month_file = sw_dir / f"sw_daily-{yyyymm}.parquet"

    existing = pd.DataFrame()
    if month_file.exists():
        try:
            existing = pd.read_parquet(month_file, engine="pyarrow")
        except Exception:
            existing = pd.DataFrame()

    _log(f"[sw {yyyymm}] start: existing={len(existing)} rows; days={len(days)} "
         f"force={force}")

    if force or existing.empty:
        present: set[str] = set()
    else:
        present = set(existing["trade_date"].astype(str))

    missing = [d for d in days if str(d) not in present]
    if not missing:
        _log(f"[sw {yyyymm}] up to date ({len(days)} days)")
        return 0
    _log(f"[sw {yyyymm}] {len(missing)}/{len(days)} days to fetch (workers={workers})")

    frames: list[pd.DataFrame] = []
    n = len(missing)
    if workers and workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_fetch_sw_day, pro, str(d)): d for d in missing}
            done = 0
            for fut in as_completed(futures):
                df = fut.result()
                if df is not None:
                    frames.append(df)
                done += 1
                if done % 10 == 0 or done == n:
                    print(f"[{yyyymm}] {done}/{n} days fetched", flush=True)
    else:
        for i, d in enumerate(missing, start=1):
            df = _fetch_sw_day(pro, str(d))
            if df is not None:
                frames.append(df)
            if i % 10 == 0 or i == n:
                print(f"[{yyyymm}] {i}/{n} days fetched", flush=True)

    if not frames:
        _log(f"[sw {yyyymm}] no rows fetched (all days empty/failed)")
        return 0

    new = pd.concat(frames, ignore_index=True)
    combined = new if (force or existing.empty) else pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    combined = combined.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    month_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(month_file, index=False, engine="pyarrow")
    _log(f"[sw {yyyymm}] saved {len(combined)} rows ({len(new)} new, {len(missing)} days)")
    return len(new)


def download_sw_daily(
    start_date: str,
    end_date: str,
    force: bool = False,
    workers: int = 1,
    output_dir: str = OUTPUT_DIR,
) -> list[dict]:
    """Bulk-download Shenwan industry-index daily (pro.sw_daily) per trade date, per month.

    Incremental by default: skips trade dates already stored in each month file.
    Months are processed serially; within each month the missing trade dates are
    fetched concurrently (workers threads).
    Files: <output_dir>/sw_daily/sw_daily-YYYYMM.parquet (all fields).

    Returns a one-element upload spec list for the sw_daily subdirectory.
    """
    cal = fetch_calendar_dates(start_date, end_date)
    if cal.empty:
        raise RuntimeError("No trading calendar")
    days = cal[cal["is_open"] == 1]["cal_date"].astype(str).tolist()

    by_month: dict[str, list[str]] = {}
    for d in days:
        by_month.setdefault(d[:6], []).append(d)
    months = sorted(by_month.items())

    print(f"sw_daily: months={len(months)}, trade days={len(days)}, "
          f"workers={workers}, force={force}")

    pro = _pro()
    sw_dir = Path(output_dir) / "sw_daily"
    sw_dir.mkdir(parents=True, exist_ok=True)

    total_new = 0
    for yyyymm, mdays in months:
        total_new += _process_sw_month(pro, yyyymm, mdays, sw_dir, force, workers)

    _log(f"sw_daily done: {total_new} new rows across {len(months)} months")
    return [{
        "kind": "sw_daily",
        "local_dir": str(sw_dir),
        "allow_patterns": "sw_daily-*.parquet",
        "path_in_repo": "sw_daily",
    }]


# ── dataset builder (tushare native, inline adj_factor) ──────────────────────

def _to_code(ts_code) -> str:
    """tushare ts_code '000001.SZ' -> 'SZ000001' (market prefix + 6-digit code)."""
    parts = str(ts_code).split(".")
    if len(parts) == 2:
        return f"{parts[1]}{parts[0]}".upper()
    return str(ts_code).upper()


def _load_kind(kline_dir: str, kind: str, months: list[str]) -> pd.DataFrame | None:
    """Load all existing monthly parquet files for a kline kind within `months`.

    Returns None if no file exists (caller degrades gracefully — missing source).
    """
    spec = next((k for k in _KLINE_KINDS if k["kind"] == kind), None)
    if spec is None:
        return None
    sub = Path(kline_dir) / spec["subdir"]
    files = [sub / f"{spec['prefix']}-{m}.parquet" for m in months]
    files = [f for f in files if f.exists()]
    if not files:
        return None
    return pd.concat(
        (pd.read_parquet(f, engine="pyarrow") for f in files), ignore_index=True)


def _check_nulls(df: pd.DataFrame, required: list[str], label: str) -> None:
    """Per-month data-quality check: warn (via _log) if any required column has nulls."""
    if df.empty:
        return
    n = len(df)
    problems = []
    for col in required:
        if col in df.columns:
            k = int(df[col].isna().sum())
            if k > 0:
                problems.append(f"{col}={k}({k / n * 100:.1f}%)")
    if problems:
        _log(f"  [{label}] null-check WARNING: {', '.join(problems)}")


def _apply_adjust(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Apply price/volume adjustment using tushare adj_factor (vectorized).

    tushare adj_factor is itself the back-adjust coefficient (anchored at the
    listing day, where adj_factor == 1), so no ratio/ref bookkeeping needed:
      back (后复权): price × adj_factor ; volume / adj_factor
      fore (前复权): price × adj_factor / adj_factor_latest ; volume inverse
    amount is unchanged (price × volume conserved).
    """
    if "adj_factor" not in df.columns:
        return df
    af = pd.to_numeric(df["adj_factor"], errors="coerce").astype("float32")
    if mode == "back":
        ratio = af
    else:  # fore — anchor at each stock's latest adj_factor (≈ today)
        latest = df.groupby("code")["adj_factor"].transform("max").astype("float32")
        ratio = af / latest
    for col in ("open", "high", "low", "close", "pre_close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32") * ratio
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("float32") / ratio
    return df


def build_stock_dataset(
    kline_dir: str,
    months: list[str],
    basic_path: str = "output/stock_basic.parquet",
    adjust: str = "none",
    include_mkts: str = "SH,SZ",
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Build standardized per-month stock datasets from the four sources.

    For each month, merges daily + daily_basic + adj_factor + stock_st into one
    row per (stock, trade_date), with naming/types aligned to baostock
    `normalize_klines_df` (code/date/volume/change_pct/pb_mrq/is_st...).

    Units (aligned to baostock): volume in shares (vol手×100), amount in yuan
    (千元×1000); turnover_rate/dv_ratio/change_pct in %; cap (流通市值) and
    total_mv (总市值, in yuan). All monetary columns in yuan. Does NOT drop IPO days; adds ipo_date column.

    Yields (yyyymm, DataFrame) per month — streaming so memory holds ~1 month.
    """
    # code -> ipo_date map from stock_basic (for ipo_date / days_since_ipo)
    ipo_date_map: dict[str, str] = {}
    if Path(basic_path).exists():
        bi = pd.read_parquet(basic_path, engine="pyarrow")
        bp = bi["ts_code"].astype(str).str.split(".")
        bi_code = (bp.str[1] + bp.str[0]).str.upper()
        ipo_date_map = dict(zip(bi_code, bi["list_date"].astype(str)))

    keys = ["ts_code", "trade_date"]
    for ym in months:
        daily = _load_kind(kline_dir, "daily", [ym])
        if daily is None or daily.empty:
            continue
        basic = _load_kind(kline_dir, "daily_basic", [ym])
        adj = _load_kind(kline_dir, "adj_factor", [ym])
        st = _load_kind(kline_dir, "st", [ym])

        # merge (pre-rename, on raw ts_code + trade_date)
        df = daily.copy()
        if basic is not None:
            df = df.merge(basic, on=keys, how="left")
        if adj is not None:
            df = df.merge(adj[keys + ["adj_factor"]], on=keys, how="left")
        else:
            df["adj_factor"] = pd.NA   # keep column in stock dataset even when adj source absent

        # is_st from per-day stock_st (sparse); default 0 when absent
        if st is not None:
            st_flag = st[keys].drop_duplicates()
            st_flag["is_st"] = 1
            df = df.merge(st_flag, on=keys, how="left")
            df["is_st"] = df["is_st"].fillna(0)
        else:
            df["is_st"] = 0

        # rename + unit conversion + types
        parts = df["ts_code"].astype(str).str.split(".")
        df["code"] = (parts.str[1] + parts.str[0]).str.upper()      # 000001.SZ -> SZ000001
        # filter by market (default SH,SZ — excludes BJ / new-third-board)
        mkts = {m.strip().upper() for m in include_mkts.split(",")}
        df = df[df["code"].astype(str).str[:2].isin(mkts)]
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
        df = df.rename(columns={"vol": "volume", "pb": "pb_mrq", "pct_chg": "change_pct", "circ_mv": "cap"})
        df = df.drop(columns=[c for c in ("ts_code", "trade_date") if c in df.columns])

        for c in ("open", "high", "low", "close", "pre_close", "change", "change_pct"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("float32") * 100.0  # 手 -> 股
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").astype("float32") * 1000.0  # 千元 -> 元
        for c in ("pe_ttm", "pb_mrq", "ps_ttm", "turnover_rate", "dv_ratio", "adj_factor"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        # cap & total_mv: tushare circ_mv/total_mv in 万元 -> convert to 元 (align with amount in yuan)
        for c in ("cap", "total_mv"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32") * 10000.0
        df["is_st"] = pd.to_numeric(df["is_st"], errors="coerce").fillna(0).astype("int8")

        # ipo_date (listing day); days_since_ipo omitted — derivable from ipo_date
        df["ipo_date"] = pd.to_datetime(df["code"].map(ipo_date_map), format="%Y%m%d", errors="coerce")

        # cap <-> turnover_rate fallback via amount:
        #   turnover_rate(%) = amount(yuan) / cap(yuan) × 100, so each fills the other
        if {"cap", "turnover_rate", "amount"} <= set(df.columns):
            m = df["cap"].isna() & df["turnover_rate"].gt(0)
            df.loc[m, "cap"] = df.loc[m, "amount"] * 100.0 / df.loc[m, "turnover_rate"]
            m = df["turnover_rate"].isna() & df["cap"].gt(0)
            df.loc[m, "turnover_rate"] = df.loc[m, "amount"] * 100.0 / df.loc[m, "cap"]

        # drop rows with unparseable date
        df = df.dropna(subset=["date"]).copy()

        # adjust (fore/back); recompute change/change_pct afterwards
        if adjust in ("fore", "back"):
            if "adj_factor" not in df.columns or df["adj_factor"].isna().all():
                raise RuntimeError(f"[{ym}] adj_factor missing — re-run klines with --adj")
            df = _apply_adjust(df, mode=adjust)
            df["change"] = (pd.to_numeric(df["close"], errors="coerce")
                            - pd.to_numeric(df["pre_close"], errors="coerce")).astype("float32")
            df["change_pct"] = ((df["close"].astype("float32") / df["pre_close"].astype("float32") - 1.0) * 100.0)
            df.loc[df["pre_close"] <= 0, "change_pct"] = float("nan")
        # non-adjusted: keep tushare's original change/change_pct (already renamed)

        # volume>0 filter + sort
        df = df[df["volume"] > 0]
        df = df.sort_values(["code", "date"]).reset_index(drop=True)
        _check_nulls(df, ["code", "date", "open", "high", "low", "close", "volume",
                          "turnover_rate", "cap"], f"{ym} stock")
        _log(f"[{ym}] stock dataset: rows={len(df)}, stocks={df['code'].nunique()}")
        yield ym, df


def build_industry_dataset(
    kline_dir: str,
    months: list[str],
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Build standardized per-month industry-index datasets from sw_daily.

    Normalizes pro.sw_daily output to align with the stock dataset naming/units:
    code (SI801010), date (datetime),
    volume (shares, vol is 万股→×10000), amount (yuan, 万元→×10000),
    pe→pe_ttm, pb→pb_mrq, float_mv→cap(万元→×10000), pct_change→change_pct.
    No pre_close/adj_factor/is_st (index has none); not adjusted.

    Yields (yyyymm, DataFrame) per month — streaming so memory holds ~1 month.
    """
    sub = Path(kline_dir) / "sw_daily"
    for ym in months:
        path = sub / f"sw_daily-{ym}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, engine="pyarrow")
        if df.empty:
            continue

        df["code"] = df["ts_code"].apply(_to_code)               # 801010.SI -> SI801010
        df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
        df = df.rename(columns={
            "vol": "volume", "pe": "pe_ttm", "pb": "pb_mrq",
            "float_mv": "cap", "pct_change": "change_pct",
        })
        df = df.drop(columns=[c for c in ("ts_code", "trade_date") if c in df.columns])

        for c in ("open", "high", "low", "close", "change", "change_pct"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("float32") * 10000.0  # 万股 -> 股
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").astype("float32") * 10000.0  # 万元 -> 元 (sw_daily amount is 万元, not 千元)
        for c in ("pe_ttm", "pb_mrq"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        # cap & total_mv: sw_daily float_mv/total_mv in 万元 -> convert to 元 (align with amount)
        for c in ("cap", "total_mv"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32") * 10000.0

        df = df.dropna(subset=["date"]).copy()
        df = df[df["volume"] > 0]
        df = df.sort_values(["code", "date"]).reset_index(drop=True)
        _check_nulls(df, ["code", "date", "open", "high", "low", "close", "volume"], f"{ym} industry")
        _log(f"[{ym}] industry dataset: rows={len(df)}, indices={df['code'].nunique()}")
        yield ym, df


# ── standalone downloads ─────────────────────────────────────────────────────

def download_stock_basic() -> pd.DataFrame:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = fetch_stock_basic()
    if df.empty:
        raise RuntimeError("Failed to fetch stock basic info")
    out = os.path.join(OUTPUT_DIR, "stock_basic.parquet")
    df.to_parquet(out, index=False, engine="pyarrow")
    print(f"Saved: {out}, rows={len(df)}, stocks={df['ts_code'].nunique()}")
    return df


def download_calendar_dates(start_date: str, end_date: str) -> pd.DataFrame:
    """Download trade calendar from tushare, incremental on existing file.

    If ``output/calendar_dates.parquet`` already exists and covers dates past
    ``end_date``, the download is skipped.  Otherwise only the uncovered range
    is fetched and merged in.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "calendar_dates.parquet")

    # ── incremental: check existing coverage ──
    existing: pd.DataFrame | None = None
    if Path(out).exists():
        try:
            existing = pd.read_parquet(out, engine="pyarrow")
            if "cal_date" in existing.columns and not existing.empty:
                max_existing = str(existing["cal_date"].max())
                if max_existing >= end_date:
                    _log(f"calendar: up to date (existing max={max_existing}, requested end={end_date})")
                    return existing
                # extend from day after max existing
                ext_start = str(int(max_existing) + 1)
                _log(f"calendar: extending from {ext_start} to {end_date} "
                     f"(existing covers through {max_existing})")
                start_date = ext_start
        except Exception:
            existing = None   # corrupt file → full re-fetch

    df = fetch_calendar_dates(start_date, end_date)
    if df.empty:
        if existing is not None and not existing.empty:
            _log("calendar: no new dates fetched — returning existing")
            return existing
        raise RuntimeError("Failed to fetch calendar")

    if existing is not None and not existing.empty:
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["cal_date"], keep="last")
        df = df.sort_values("cal_date").reset_index(drop=True)

    df.to_parquet(out, index=False, engine="pyarrow")
    _log(f"calendar: saved {out}, rows={len(df)}, "
         f"range={df['cal_date'].min()}~{df['cal_date'].max()}")
    return df


def download_forecast_reports(start_date: str, end_date: str) -> pd.DataFrame:
    import datetime as dt
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start = dt.date.fromisoformat(_fmt_date(start_date))
    end = dt.date.fromisoformat(_fmt_date(end_date))
    periods = []
    y, q = start.year, (start.month - 1) // 3 + 1
    while (y, q) <= (end.year, (end.month - 1) // 3 + 1):
        m = q * 3
        last_day = (dt.date(y + (m // 12), (m % 12) + 1, 1) - dt.timedelta(days=1)).day
        periods.append(f"{y}{m:02d}{last_day:02d}")
        q += 1
        if q > 4:
            q, y = 1, y + 1

    pro = _pro()
    _log(f"forecast: fetching {len(periods)} report periods")
    frames = []
    for p in periods:
        try:
            df = _call(pro.forecast, period=p)
            if df is not None and not df.empty:
                frames.append(df)
                _log(f"  forecast period {p}: {len(df)} rows")
        except Exception as e:
            print(f"  period {p} failed: {e}")
    if not frames:
        raise RuntimeError("No forecast data")

    merged = pd.concat(frames, ignore_index=True)
    out = os.path.join(OUTPUT_DIR, "forecast_reports.parquet")
    merged.to_parquet(out, index=False, engine="pyarrow")
    _log(f"forecast: saved {out}, rows={len(merged)}")
    return merged


def download_index_component_klines(index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    pro = _pro()
    iw = _call(pro.index_weight, index_code=index_code, trade_date=end_date)
    if iw is None or iw.empty:
        raise RuntimeError(f"No constituents for {index_code}")
    codes = iw["con_code"].drop_duplicates().tolist()
    _log(f"index {index_code}: {len(codes)} constituents, {start_date}~{end_date}")

    frames = []
    for i, c in enumerate(codes, start=1):
        try:
            df = _call(pro.daily, ts_code=c, start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"  {c} failed: {e}")
        if i % 50 == 0 or i == len(codes):
            print(f"  {i}/{len(codes)}", flush=True)

    if not frames:
        raise RuntimeError("No index kline data")
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "index_component_klines.parquet")
    merged.to_parquet(out, index=False, engine="pyarrow")
    _log(f"index {index_code}: saved {out}, rows={len(merged)}, stocks={merged['ts_code'].nunique()}")
    return merged


def fetch_sw_classify(src: str = "SW2021") -> pd.DataFrame:
    """Fetch Shenwan industry classification (pro.index_classify), all levels L1/L2/L3.

    Args:
        src: "SW2021" (default, 31/134/346 levels) or "SW2014" (28/104/227).
    """
    pro = _pro()
    frames = []
    try:
        df = _call(pro.index_classify, src=src)
        if df is not None and not df.empty:
            frames.append(df)
            lvl_counts = df["level"].value_counts().to_dict() if "level" in df.columns else {}
            _log(f"index_classify {src}: {len(df)} rows, levels={lvl_counts}")
    except Exception as e:
        print(f"  index_classify {src} failed: {e}")
    if not frames:
        _log(f"index_classify {src}: empty")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def download_sw_classify(src: str = "SW2021") -> pd.DataFrame:
    """Download Shenwan industry classification to <OUTPUT_DIR>/sw_classify.parquet.

    Single static table — no per-month aggregation, no subdir.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = fetch_sw_classify(src=src)
    if df.empty:
        raise RuntimeError("Failed to fetch sw classify")
    out = os.path.join(OUTPUT_DIR, "sw_classify.parquet")
    df.to_parquet(out, index=False, engine="pyarrow")
    _log(f"sw_classify: saved {out}, rows={len(df)}")
    return df


def _sw_l1_codes(classify_df: pd.DataFrame) -> list[str]:
    """Extract normalized L1 index codes (ensuring .SI suffix) from sw_classify df."""
    if classify_df is None or classify_df.empty or "level" not in classify_df.columns:
        return []
    lvl = classify_df["level"].astype(str)
    l1 = classify_df[lvl.str.upper().eq("L1") | lvl.str.contains("一级", na=False)]
    codes: list[str] = []
    for c in l1["index_code"].astype(str).tolist():
        codes.append(c if c.endswith(".SI") else f"{c}.SI")
    return codes


def fetch_sw_members(l1_codes: list[str]) -> pd.DataFrame:
    """Fetch Shenwan index members (pro.index_member_all) for each L1 industry.

    Fetches ALL membership records (incl. historical in/out via in_date/out_date)
    so the mapping can derive current membership from out_date. is_new is a
    classification-version tag (not a snapshot) and is not used for filtering.
    A single L1's records are well under the 2000-row cap.
    """
    pro = _pro()
    frames = []
    n = len(l1_codes)
    for i, code in enumerate(l1_codes, start=1):
        try:
            df = _call(pro.index_member_all, l1_code=code, is_new='Y')
            if df is not None and not df.empty:
                frames.append(df)
                _log(f"  members l1={code}: {len(df)} rows ({i}/{n})")
            else:
                _log(f"  members l1={code}: empty ({i}/{n})")
        except Exception as e:
            print(f"  index_member_all l1_code={code} failed: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def download_sw_members(
    src: str = "SW2021",
    classify_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Download Shenwan index members (per L1 industry) to <OUTPUT_DIR>/sw_members.parquet.

    Iterates every L1 industry from sw_classify and fetches ALL members via
    pro.index_member_all(l1_code=...) — including historical in/out records.
    Current membership is derived downstream from out_date (is_new is ignored).
    """
    if classify_df is None:
        classify_df = fetch_sw_classify(src=src)
    l1_codes = _sw_l1_codes(classify_df)
    if not l1_codes:
        raise RuntimeError("No L1 industry codes found in sw_classify")
    _log(f"sw_members: fetching {len(l1_codes)} L1 industries (all records)")
    df = fetch_sw_members(l1_codes)
    if df.empty:
        raise RuntimeError("Failed to fetch sw members")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "sw_members.parquet")
    df.to_parquet(out, index=False, engine="pyarrow")
    _log(f"sw_members: saved {out}, rows={len(df)}, stocks={df['ts_code'].nunique()}")
    return df


def build_industry_mapping(
    classify_df: pd.DataFrame,
    members_df: pd.DataFrame,
    l3_min_stocks: int = 10,
) -> pd.DataFrame:
    """Build stock -> effective SW industry index mapping (essential fields only).

    Each stock's L3 industry is used if it has a published index (is_pub='1')
    AND covers more than l3_min_stocks constituent stocks; otherwise falls back
    to L2, then L1 (L1 always has indices). This filters out tiny L3s whose
    industry-index features would be too noisy for quantitative models.

    `is_new` is a classification-version tag (not a snapshot) and is ignored;
    current membership = rows where out_date is null.

    Output columns — core join fields plus common auxiliary fields for audit:
      stock_code (SZ000001, -> stock dataset), index_code (SI801010, -> industry dataset),
      effective_level (L3/L2/L1), effective_industry_name,
      ts_code (raw 000001.SZ), name (stock name),
      l1_code/l1_name, l2_code/l2_name, l3_code/l3_name (original 3-level, SI801010),
      in_date, out_date
    """
    classify = classify_df
    members = members_df

    # current constituents = out_date is null (is_new is a version tag, ignored);
    # one row per stock
    cur = members[members["out_date"].isna()].copy()
    cur = cur.drop_duplicates(subset=["ts_code"]).reset_index(drop=True)

    # published index set + name lookup + L3 stock-count gate (raw .SI codes)
    pub = classify[classify["is_pub"].astype(str) == "1"]
    pub_codes = set(pub["index_code"].astype(str))
    name_map = dict(zip(classify["index_code"].astype(str), classify["industry_name"]))
    # count stocks per L3 for the >l3_min_stocks gate (tiny L3s fall back to L2)
    l3_counts = cur.groupby("l3_code").size().to_dict()

    # L3(>threshold) -> L2 -> L1 fallback
    def _fallback(l3, l2, l1):
        if pd.notna(l3) and str(l3) in pub_codes and l3_counts.get(str(l3), 0) >= l3_min_stocks:
            return str(l3), "L3"
        if pd.notna(l2) and str(l2) in pub_codes:
            return str(l2), "L2"
        if pd.notna(l1) and str(l1) in pub_codes:
            return str(l1), "L1"
        return None, None

    eff = cur.apply(
        lambda r: _fallback(r.get("l3_code"), r.get("l2_code"), r.get("l1_code")),
        axis=1, result_type="expand",
    )

    def _si(x):
        return _to_code(x) if pd.notna(x) else x

    # core join fields + common auxiliary fields for debugging/audit
    cur["stock_code"] = cur["ts_code"].apply(_to_code)                  # SZ000001
    cur["index_code"] = eff[0].apply(_si)                               # SI801010
    cur["effective_level"] = eff[1]
    cur["effective_industry_name"] = eff[0].map(name_map)
    for col in ("l1_code", "l2_code", "l3_code"):                       # normalize to SI801010
        if col in cur.columns:
            cur[col] = cur[col].apply(_si)

    out_cols = [
        "stock_code", "index_code", "effective_level", "effective_industry_name",
        "ts_code", "name",
        "l1_code", "l1_name", "l2_code", "l2_name", "l3_code", "l3_name",
        "in_date", "out_date",
    ]
    out = cur[[c for c in out_cols if c in cur.columns]].copy()

    lvl = out["effective_level"].value_counts().to_dict()
    _log(f"sw_mapping: {len(out)} stocks, effective_level={lvl}, "
         f"None={int(out['index_code'].isna().sum())}")
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────

def _upload(args):
    """Upload to ModelScope, forwarding all relevant args."""
    from ms import upload_to_modelscope
    kwargs = {"repo_id": args.repo_id}
    if getattr(args, "local_dir", None):
        kwargs["local_dir"] = args.local_dir
    if getattr(args, "path_in_repo", None):
        kwargs["path_in_repo"] = args.path_in_repo
    if getattr(args, "allow_patterns", None):
        kwargs["allow_patterns"] = args.allow_patterns
    upload_to_modelscope(**kwargs)


def _upload_klines(repo_id: str, uploads: list[dict]) -> None:
    """Upload each produced kline kind (daily / daily_basic / adj_factor) separately.

    Each spec carries its own local_dir / allow_patterns / path_in_repo, sourced
    from download_all_stocks_klines(), so the three subdirectories are uploaded
    as three independent commits.
    """
    if not uploads:
        print("Nothing to upload (no kline files produced this run)")
        return
    from ms import upload_to_modelscope
    for spec in uploads:
        print(f"[upload] {spec['kind']}: {spec['local_dir']} "
              f"({spec['allow_patterns']}) -> {spec['path_in_repo']}")
        upload_to_modelscope(
            repo_id=repo_id,
            local_dir=spec["local_dir"],
            path_in_repo=spec["path_in_repo"],
            allow_patterns=spec["allow_patterns"],
        )


def _add_upload(p, path_in_repo: str, allow_patterns: str):
    p.add_argument("--upload", action="store_true", help="Upload to ModelScope after processing")
    p.add_argument("--repo-id", default="", help="ModelScope repo (required if --upload)")
    p.add_argument("--local-dir", default=OUTPUT_DIR, help="Local directory to upload")
    p.add_argument("--path-in-repo", default=path_in_repo, help="Target path in repo")
    p.add_argument("--allow-patterns", default=allow_patterns, help="Glob pattern(s) to upload")


def _check_repo(args):
    if getattr(args, "upload", False) and not getattr(args, "repo_id", ""):
        raise SystemExit("--repo-id is required when --upload is set")


def _cmd_basic(args):
    download_stock_basic()
    if args.upload:
        _upload(args)


def _cmd_klines(args):
    uploads = download_all_stocks_klines(
        args.start_date, args.end_date,
        with_basic=not args.no_basic,
        with_adj=not args.no_adj,
        force=args.force,
        workers=args.workers,
        with_st=not args.no_st,
    )
    if args.upload:
        _upload_klines(args.repo_id, uploads)


def _cmd_calendar(args):
    download_calendar_dates(args.start_date, args.end_date)
    if args.upload:
        _upload(args)


def _cmd_forecast(args):
    download_forecast_reports(args.start_date, args.end_date)
    if args.upload:
        _upload(args)


def _cmd_index(args):
    idx_map = {"hs300": "000300.SH", "zz500": "000905.SH"}
    code = idx_map.get(args.index_scope, args.index_scope)
    download_index_component_klines(code, args.start_date, args.end_date)
    if args.upload:
        _upload(args)


def _cmd_sw(args):
    uploads = download_sw_daily(
        args.start_date, args.end_date,
        force=args.force,
        workers=args.workers,
    )
    if args.upload:
        _upload_klines(args.repo_id, uploads)


def _cmd_sw_classify(args):
    classify_df = download_sw_classify(src=args.src)
    if not args.no_members:
        members_df = download_sw_members(src=args.src, classify_df=classify_df)
        # stock -> effective industry index (L3->L2->L1 fallback for unpublished L3)
        mapping = build_industry_mapping(classify_df, members_df)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        mapping.to_parquet(os.path.join(OUTPUT_DIR, "sw_industry_mapping.parquet"),
                           index=False, engine="pyarrow")
        _log("sw_classify: saved sw_industry_mapping.parquet")
        args.allow_patterns = ["sw_classify.parquet", "sw_members.parquet",
                               "sw_industry_mapping.parquet"]
    if args.upload:
        _upload(args)


def _cmd_dataset(args):
    """Build the three standardized datasets: stock (per-month) + industry index
    (per-month) + stock->industry mapping (single file)."""
    kline_dir = args.kline_dir
    s_ym = int(str(args.start_date).replace("-", "")[:6]) if args.start_date else 0
    e_ym = int(str(args.end_date).replace("-", "")[:6]) if args.end_date else 999999

    # month set = daily months ∩ sw_daily months
    daily_months = {f.stem.split("-")[-1] for f in (Path(kline_dir) / "daily").glob("daily-*.parquet")}
    sw_months = {f.stem.split("-")[-1] for f in (Path(kline_dir) / "sw_daily").glob("sw_daily-*.parquet")}
    months = sorted(m for m in (daily_months & sw_months) if s_ym <= int(m) <= e_ym)
    if not months:
        raise FileNotFoundError(f"No overlapping daily/sw_daily months in {kline_dir}")
    _log(f"dataset: {len(months)} months ({months[0]}~{months[-1]})")

    out_root = Path(args.output_dir)
    stock_dir = out_root / "stock"
    industry_dir = out_root / "industry"
    stock_dir.mkdir(parents=True, exist_ok=True)
    industry_dir.mkdir(parents=True, exist_ok=True)

    # 1. stock datasets (per month) — streaming, memory holds ~1 month
    n_stock = 0
    for ym, df in build_stock_dataset(kline_dir, months, basic_path=args.basic_path, adjust=args.adjust,
                                      include_mkts=args.include_mkts):
        df.to_parquet(stock_dir / f"stock-{ym}.parquet", index=False, engine="pyarrow")
        n_stock += 1
    _log(f"dataset: wrote {n_stock} stock month files")

    # 2. industry index datasets (per month) — streaming
    n_ind = 0
    for ym, df in build_industry_dataset(kline_dir, months):
        df.to_parquet(industry_dir / f"industry-{ym}.parquet", index=False, engine="pyarrow")
        n_ind += 1
    _log(f"dataset: wrote {n_ind} industry month files")

    # 3. mapping (single file) — needs sw_classify + sw_members
    classify_path = os.path.join(kline_dir, "sw_classify.parquet")
    members_path = os.path.join(kline_dir, "sw_members.parquet")
    if Path(classify_path).exists() and Path(members_path).exists():
        cls = pd.read_parquet(classify_path, engine="pyarrow")
        mem = pd.read_parquet(members_path, engine="pyarrow")
        mapping = build_industry_mapping(cls, mem)
        mapping.to_parquet(out_root / "industry_mapping.parquet", index=False, engine="pyarrow")
        _log(f"dataset: wrote industry_mapping.parquet (rows={len(mapping)})")
    else:
        _log("dataset: sw_classify.parquet / sw_members.parquet missing — "
             "run `python ts.py sw-classify` first; skipping mapping")

    if args.upload:
        from ms import upload_to_modelscope
        upload_to_modelscope(
            repo_id=args.repo_id,
            local_dir=str(out_root),
            path_in_repo="",
            allow_patterns=["stock/*.parquet", "industry/*.parquet", "industry_mapping.parquet"],
        )


def main():
    parser = argparse.ArgumentParser(
        description="Tushare A-share pipeline (per-day, per-month, incremental)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Subcommands")
    sub.required = True

    p = sub.add_parser("basic", help="Fetch stock_basic")
    _add_upload(p, "basic", "stock_basic.parquet")
    p.set_defaults(func=_cmd_basic)

    p = sub.add_parser("klines", help="Daily+basic+adj per day, per month (incremental)")
    p.add_argument("--start-date", default="20100101")
    p.add_argument("--end-date", default=_today())
    p.add_argument("--no-basic", action="store_true", help="Skip daily_basic (valuation)")
    p.add_argument("--no-adj", action="store_true", help="Skip adj_factor")
    p.add_argument("--no-st", action="store_true", help="Skip ST stock list (pro.stock_st)")
    p.add_argument("--force", action="store_true", help="Re-fetch all days (ignore cache)")
    p.add_argument("--workers", type=int, default=1, help="Parallel month workers")
    p.add_argument("--upload", action="store_true",
                   help="Upload daily/daily_basic/adj_factor/st subdirs to ModelScope")
    p.add_argument("--repo-id", default="", help="ModelScope repo (required if --upload)")
    p.set_defaults(func=_cmd_klines)

    p = sub.add_parser("calendar", help="Download trade calendar")
    p.add_argument("--start-date", default="19901219")
    p.add_argument("--end-date", default=_next_year_end())
    _add_upload(p, "calendar", "calendar_dates.parquet")
    p.set_defaults(func=_cmd_calendar)

    p = sub.add_parser("forecast", help="Download forecast reports")
    p.add_argument("--start-date", default="20200101")
    p.add_argument("--end-date", default=_today())
    _add_upload(p, "forecast", "forecast_reports.parquet")
    p.set_defaults(func=_cmd_forecast)

    p = sub.add_parser("index", help="Download index component daily klines")
    p.add_argument("--index-scope", default="hs300", help="hs300 / zz500 / 000300.SH")
    p.add_argument("--start-date", default="20200101")
    p.add_argument("--end-date", default=_today())
    _add_upload(p, "index", "index_component_klines.parquet")
    p.set_defaults(func=_cmd_index)

    p = sub.add_parser("sw", help="Shenwan industry-index daily (pro.sw_daily) per day, per month")
    p.add_argument("--start-date", default="20100101")
    p.add_argument("--end-date", default=_today())
    p.add_argument("--force", action="store_true", help="Re-fetch all days (ignore cache)")
    p.add_argument("--workers", type=int, default=1, help="Parallel day workers within a month")
    p.add_argument("--upload", action="store_true", help="Upload sw_daily subdir to ModelScope")
    p.add_argument("--repo-id", default="", help="ModelScope repo (required if --upload)")
    p.set_defaults(func=_cmd_sw)

    p = sub.add_parser("sw-classify", help="Shenwan classification + members + stock->industry mapping")
    p.add_argument("--src", default="SW2021", choices=["SW2021", "SW2014"],
                   help="Classification version (SWS2021 default, or SWS2014)")
    p.add_argument("--no-members", action="store_true",
                   help="Skip index members (pro.index_member_all per L1)")
    _add_upload(p, "sw_classify", "sw_classify.parquet")
    p.set_defaults(func=_cmd_sw_classify)

    p = sub.add_parser("dataset", help="Build standardized stock + industry + mapping datasets")
    p.add_argument("--kline-dir", default=OUTPUT_DIR,
                   help="Input root dir (contains daily/ daily_basic/ adj_factor/ stock_st/ sw_daily/ + sw_*.parquet)")
    p.add_argument("--output-dir", default=os.path.join(OUTPUT_DIR, "dataset"),
                   help="Output root for the three datasets (stock/ industry/ + industry_mapping.parquet)")
    p.add_argument("--basic-path", default="output/stock_basic.parquet")
    p.add_argument("--adjust", default="none", choices=["none", "fore", "back"],
                   help="Adjustment for stock dataset: none / fore / back")
    p.add_argument("--include-mkts", default="SH,SZ",
                   help="Markets to include (comma-sep, default SH,SZ; omit BJ for new-third-board)")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--upload", action="store_true", help="Upload the three datasets to ModelScope")
    p.add_argument("--repo-id", default="", help="ModelScope repo (required if --upload)")
    p.set_defaults(func=_cmd_dataset)

    p = sub.add_parser("upload", help="Upload output to ModelScope")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--local-dir", default=OUTPUT_DIR)
    p.add_argument("--path-in-repo", default="stock-day")
    p.add_argument("--allow-patterns", default="*.parquet")
    p.set_defaults(func=_upload)

    args = parser.parse_args()
    _check_repo(args)
    args.func(args)


if __name__ == "__main__":
    main()
