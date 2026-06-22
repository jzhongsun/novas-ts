"""
Tencent Securities tick-by-tick trade data pipeline.

Fetches intraday tick-level trade data (���笔成交明细) for A-share stocks
from the Tencent Securities API (stock.gtimg.cn). Data is real-time only
(no historical archive) — each run captures the current trading day.

Design:
  - Single-stock or batch fetch via ThreadPoolExecutor.
  - Each worker thread creates its own httpx.Client (the client is NOT
    thread-safe, so sharing across threads would corrupt connections).
  - Auto-pagination: fetches consecutive pages until empty or fewer than
    PAGE_SIZE_HINT rows (Tencent returns ~30-50 rows per page).
  - Per-stock parquet files for crash safety + optional periodic combined saves.
  - Code conversion utility: tushare ts_code (000001.SZ) -> Tencent (sz000001).

Subcommands: ticks, upload

Usage:
    python tx.py ticks --codes sh600519,sz000001
    python tx.py ticks --from-stock-basic ./output/stock_basic.parquet --max-workers 5
    python tx.py ticks --from-stock-basic ./output/stock_basic.parquet --upload --repo-id myorg/tick-data

Setup:
    pip install httpx pandas pyarrow modelscope
    export MODELSCOPE_ACCESS_TOKEN=...   # https://modelscope.cn/my/myaccesstoken
"""

from __future__ import annotations

import argparse
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd

try:
    import pyarrow  # noqa: F401
except ImportError:
    raise ImportError("pip install pyarrow")


OUTPUT_DIR = "./output/tick"
MAX_RETRIES = 5
PAGE_SIZE_HINT = 20  # pages with fewer rows are treated as the last page


# ── helpers ──────────────────────────────────────────────────────────────────

def _today() -> str:
    return date.today().strftime("%Y%m%d")


def _log(msg: str) -> None:
    """Timestamped one-line log to stdout (flushed)."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def _fmt_duration(seconds: float) -> str:
    """Human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


# ── trading-day gate ──────────────────────────────────────────────────────────

def is_trading_day(
    calendar_path: str = "output/calendar_dates.parquet",
    *,
    target_date: str | None = None,
) -> bool:
    """Check whether *target_date* (default today, YYYYMMDD) is a trading day.

    Reads the same ``calendar_dates.parquet`` produced by ``ts.py calendar``
    (tushare ``pro.trade_cal``).  Returns False when the file is missing or
    the date is absent / marked non-open — the caller should abort the fetch.

    Use ``--force`` on the CLI to skip this gate (e.g. for testing on
    weekends or during holidays).
    """
    path = Path(calendar_path)
    if not path.exists():
        _log(f"is_trading_day: {calendar_path} not found — assuming non-trading; "
             f"run 'python ts.py calendar' first or use --force")
        return False

    try:
        cal = pd.read_parquet(path, engine="pyarrow")
    except Exception as e:
        _log(f"is_trading_day: failed to read {calendar_path}: {e}")
        return False

    if "cal_date" not in cal.columns or "is_open" not in cal.columns:
        _log("is_trading_day: calendar missing expected columns (cal_date, is_open)")
        return False

    check_date = target_date or _today()
    row = cal[cal["cal_date"].astype(str) == check_date]
    if row.empty:
        _log(f"is_trading_day: {check_date} not in calendar → non-trading")
        return False

    is_open = int(row["is_open"].iloc[0])
    if is_open != 1:
        _log(f"is_trading_day: {check_date} is_open={is_open} → non-trading")
        return False

    return True


# ── code conversion ─────────────────────────────────────────────────────────

def ts_to_tx(ts_code: str) -> Optional[str]:
    """Convert tushare ts_code (000001.SZ) to Tencent code (sz000001).

    Returns None for non-A-share markets (HK, BJ, etc.) so callers can
    filter them out with ``dropna()``.
    """
    if not isinstance(ts_code, str) or "." not in ts_code:
        return None
    code, exchange = ts_code.split(".", 1)
    exchange = exchange.lower()
    if exchange not in ("sz", "sh"):
        return None
    return f"{exchange}{code.zfill(6)}"


def codes_from_stock_basic(path: str) -> list[str]:
    """Load stock_basic.parquet and return A-share-only Tencent codes."""
    df = pd.read_parquet(path, engine="pyarrow")
    codes = df["ts_code"].apply(ts_to_tx).dropna().tolist()
    _log(f"codes_from_stock_basic: {len(codes)} A-share stocks from {path}")
    return codes


# ── core fetcher ────────────────────────────────────────────────────────────

class TxTickFetcher:
    """Fetch tick-by-tick trade data for one stock from Tencent Securities.

    **Not thread-safe** — ``httpx.Client`` is not safe to share across
    threads.  Each worker thread should create its own instance.
    """

    BASE_URL = "https://stock.gtimg.cn/data/index.php"

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout, headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            ),
        })

    def close(self) -> None:
        """Release the underlying httpx connection pool."""
        self.client.close()

    # ── response parsing ─────────────────────────────────────────────────

    @staticmethod
    def _parse_response(resp_text: str) -> list[dict]:
        """Parse Tencent's non-standard JS response into tick records.

        Response format::

            v_detail_data_sh600519=[3,"seq/time/price/change/vol/amt/dir|..."]
        """
        # empty / suspended / non-trading
        if "=\\[\\]" in resp_text or '=""' in resp_text:
            return []

        match = re.search(r"=\[(.+?)\]", resp_text)
        if not match:
            return []

        content = match.group(1)
        parts = content.split(",", 1)
        if len(parts) < 2:
            return []

        data_str = parts[1].strip('"')
        if not data_str:
            return []

        rows: list[dict] = []
        for item in data_str.split("|"):
            fields = item.split("/")
            if len(fields) < 7:
                continue
            try:
                rows.append({
                    "seq":       int(fields[0]),   # trade sequence number
                    "time":      fields[1],        # HH:MM:SS
                    "price":     float(fields[2]), # trade price
                    "change":    float(fields[3]), # price change vs prev tick
                    "volume":    int(fields[4]),   # shares (A: lots×100; HK: shares)
                    "amount":    int(fields[5]),   # amount in yuan
                    "direction": fields[6],        # B=buy, S=sell, M=neutral
                })
            except (ValueError, IndexError):
                continue
        return rows

    # ── page fetch with retry ─────────────────────────────────────────────

    def _fetch_page(self, code: str, page: int) -> list[dict]:
        """Fetch one page of tick data with exponential-backoff retry."""
        params = {"appn": "detail", "action": "data", "c": code, "p": page}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.client.get(self.BASE_URL, params=params)
                resp.raise_for_status()
                return self._parse_response(resp.text)
            except (httpx.HTTPError, httpx.TimeoutException, ConnectionError) as e:
                if attempt < MAX_RETRIES:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    print(f"  [{code}] page {page} retry {attempt}/{MAX_RETRIES} "
                          f"in {wait:.1f}s: {e}", flush=True)
                    time.sleep(wait)
                else:
                    raise
            except Exception:
                # non-retryable (parse error, etc.) — surface immediately
                raise

        # unreachable; satisfy type-checker
        return []

    # ── main fetch loop ───────────────────────────────────────────────────

    def fetch_today(self, code: str, max_pages: int = 500) -> pd.DataFrame:
        """Fetch all tick data for *code* on the current trading day.

        Automatically paginates until an empty page or the last partial page.
        """
        all_ticks: list[dict] = []
        t0 = time.monotonic()

        for page in range(max_pages):
            try:
                page_ticks = self._fetch_page(code, page)
            except Exception as e:
                _log(f"[{code}] page {page} failed after {MAX_RETRIES} retries: {e}")
                break

            if not page_ticks:
                if page == 0:
                    _log(f"[{code}] no data (page 0 empty — "
                         f"check code, trading hours, suspension)")
                break

            all_ticks.extend(page_ticks)

            # last page is typically partial (<20 rows)
            if len(page_ticks) < PAGE_SIZE_HINT:
                break

        if not all_ticks:
            return pd.DataFrame()

        df = pd.DataFrame(all_ticks)
        df = df.drop_duplicates(subset=["seq", "time"]) \
               .sort_values("seq") \
               .reset_index(drop=True)

        # add trade date + compact dtypes
        df["date"] = _today()
        df["price"] = df["price"].astype("float32")
        df["change"] = df["change"].astype("float32")
        df["volume"] = df["volume"].astype("int32")
        df["amount"] = df["amount"].astype("int32")
        df["direction"] = df["direction"].astype("category")

        elapsed = time.monotonic() - t0
        _log(f"[{code}] {len(df)} ticks in {_fmt_duration(elapsed)} "
             f"({page + 1} pages)")
        return df


# ── batch download ──────────────────────────────────────────────────────────

def batch_fetch(
    codes: list[str],
    output_dir: str = OUTPUT_DIR,
    max_workers: int = 5,
    max_pages: int = 500,
    save_interval: int = 0,
    per_stock: bool = False,
) -> pd.DataFrame:
    """Download tick data for multiple stocks concurrently.

    Results are accumulated in memory and written as a single combined
    file on completion.  Both per-stock files and periodic intermediate
    saves are opt-in.

    Args:
        codes: Tencent-format stock codes, e.g. ``['sz000001', 'sh600519']``.
        output_dir: Directory for output parquet files.
        max_workers: Max concurrent download threads (3–8 recommended).
        max_pages: Max pages per stock (safety limit; normal day is 10–30).
        save_interval: If >0, save intermediate combined result every N stocks (opt-in).
        per_stock: If True, also write a per-stock parquet file (opt-in).

    Returns:
        Combined DataFrame of all tick records (columns: date, code, seq, time,
        price, change, volume, amount, direction).
    """
    if not codes:
        _log("batch_fetch: empty code list — nothing to do")
        return pd.DataFrame()

    today = _today()
    out_path = Path(output_dir) / today[:6]   # nest under YYYYMM/
    out_path.mkdir(parents=True, exist_ok=True)

    def _fetch_one(tx_code: str) -> pd.DataFrame:
        """Fetch one stock — each thread owns its TxTickFetcher + client."""
        fetcher = TxTickFetcher()
        try:
            df = fetcher.fetch_today(tx_code, max_pages=max_pages)
            if not df.empty:
                df["code"] = tx_code
                if per_stock:
                    stock_file = out_path / f"tick_{tx_code}_{today}.parquet"
                    df.to_parquet(stock_file, index=False, engine="pyarrow")
            return df
        except Exception as e:
            _log(f"[{tx_code}] unexpected error: {e}")
            return pd.DataFrame()
        finally:
            fetcher.close()

    results: list[pd.DataFrame] = []
    n = len(codes)
    done = 0

    _log(f"batch_fetch: {n} stocks, {max_workers} workers, "
         f"output → {out_path}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, c): c for c in codes}

        for future in as_completed(futures):
            code = futures[future]
            done += 1
            try:
                df = future.result()
                if not df.empty:
                    results.append(df)
            except Exception as e:
                _log(f"[{code}] future exception: {e}")

            if done % 10 == 0 or done == n:
                _log(f"batch_fetch: {done}/{n} stocks done "
                     f"({sum(len(r) for r in results)} ticks so far)")

            # periodic combined save (crash recovery)
            if save_interval and len(results) > 0 and len(results) % save_interval == 0:
                mid = pd.concat(results, ignore_index=True)
                mid_file = out_path / f"tick_{today}_partial_{len(results)}.parquet"
                mid.to_parquet(mid_file, index=False, engine="pyarrow")
                _log(f"  saved partial: {mid_file} ({len(mid)} rows)")

    if not results:
        _log("batch_fetch: no data — check market is open and codes are valid")
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    combined = combined.drop_duplicates(subset=["code", "seq", "time"]) \
                       .reset_index(drop=True)

    all_file = out_path / f"tick_{today}.parquet"
    combined.to_parquet(all_file, index=False, engine="pyarrow")
    _log(f"batch_fetch: saved {all_file} — {len(combined)} rows, "
         f"{combined['code'].nunique()} stocks")
    return combined


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cmd_ticks(args: argparse.Namespace) -> None:
    """Handler for the ``ticks`` subcommand."""
    # trading-day gate (skipped with --force)
    if args.force:
        _log("ticks: --force set — skipping trading-day check")
    elif not is_trading_day(calendar_path=args.calendar_path):
        _log("ticks: today is not a trading day — aborting "
             "(use --force to override)")
        return

    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    else:
        if not Path(args.from_stock_basic).exists():
            raise SystemExit(
                f"{args.from_stock_basic} not found — "
                f"run 'python ts.py basic' first, or use --codes")
        codes = codes_from_stock_basic(args.from_stock_basic)
        if args.limit:
            codes = codes[:args.limit]
            _log(f"ticks: limited to first {args.limit} stocks")

    df = batch_fetch(
        codes=codes,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        max_pages=args.max_pages,
        save_interval=args.save_interval,
        per_stock=args.per_stock,
    )

    if df.empty:
        _log("fetch: no data fetched — "
             "verify codes, trading hours, and that stocks are not suspended")
    else:
        _log(f"fetch: complete — {len(df)} ticks, "
             f"{df['code'].nunique()} stocks")

    if args.upload:
        if not args.repo_id:
            raise SystemExit("--repo-id is required when --upload is set")
        from ms import upload_to_modelscope
        today = _today()
        upload_to_modelscope(
            repo_id=args.repo_id,
            local_dir=f"{args.output_dir}/{today[:6]}",
            path_in_repo=f"tick/{today[:6]}",
            allow_patterns=f"tick_{today}*.parquet",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tencent Securities tick-by-tick trade data fetcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    p = sub.add_parser("ticks", help="Fetch today's tick-by-tick trade data")
    p.add_argument(
        "--codes", default=None,
        help="Comma-separated Tencent-format codes, e.g. sh600519,sz000001")
    p.add_argument(
        "--from-stock-basic", default="data/stock_basic.parquet",
        help="Path to stock_basic.parquet (fetches all A-share stocks by default)")
    p.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of stocks when using --from-stock-basic")
    p.add_argument(
        "--max-workers", type=int, default=5,
        help="Max concurrent download threads (default 5; recommend 3–8)")
    p.add_argument(
        "--max-pages", type=int, default=500,
        help="Max pages per stock, safety limit (default 500; normal day ~10–30)")
    p.add_argument(
        "--save-interval", type=int, default=0,
        help="Save intermediate combined file every N stocks (0 = disabled)")
    p.add_argument(
        "--per-stock", action="store_true",
        help="Also write a per-stock parquet file for crash safety")
    p.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help=f"Output directory for parquet files (default: {OUTPUT_DIR})")
    p.add_argument(
        "--calendar-path", default="data/calendar_dates.parquet",
        help="Path to calendar_dates.parquet for trading-day check")
    p.add_argument(
        "--force", action="store_true",
        help="Skip trading-day check (fetch even on non-trading days)")
    p.add_argument(
        "--upload", action="store_true",
        help="Upload output to ModelScope after fetching")
    p.add_argument(
        "--repo-id", default="",
        help="ModelScope repo (required if --upload)")
    p.set_defaults(func=_cmd_ticks)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
