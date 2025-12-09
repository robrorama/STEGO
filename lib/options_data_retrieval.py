"""
options_data_retrieval.py

Canonical options data loader/downloader with disk caching.

Goals:
- Only download option chains if they are not already cached on disk
- Always load from local disk when possible
- Provide a simple, stable API for other scripts (plotters, analyzers)
- Mirror the structure & style of the stock data_retrieval module

Environment variables:
    BASE_DATA_PATH   (default: /dev/shm/data)
    BASE_CACHE_PATH  (default: /dev/shm/cache)

Filesystem layout (default):
    {BASE_DATA_PATH}/options/{source}/{ticker}/{expiration}.parquet

Where:
    source      – e.g. "yfinance"
    ticker      – e.g. "SPY"
    expiration  – ISO date string "YYYY-MM-DD"

Public API:
    load_or_download_option_chain(ticker, expiration, source="yfinance", force_refresh=False)
    list_cached_option_expirations(ticker, source="yfinance")
    load_all_cached_option_chains(ticker, source="yfinance")
    get_available_remote_expirations(ticker, source="yfinance")   # remote listing (e.g. Yahoo)
"""

import os
import pathlib
from typing import List, Optional, Dict, Union

import pandas as pd

# Optional yfinance import (only required when downloading)
try:
    import yfinance as yf  # type: ignore
    _HAS_YFINANCE = True
except Exception:
    _HAS_YFINANCE = False

# ---------------------------------------------------------------------------
# Base paths & helpers
# ---------------------------------------------------------------------------

BASE_DATA_PATH = os.environ.get("BASE_DATA_PATH", "/dev/shm/data")
BASE_CACHE_PATH = os.environ.get("BASE_CACHE_PATH", "/dev/shm/cache")


def _ensure_dir(path: Union[str, pathlib.Path]) -> None:
    """Create directory if it does not exist."""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def _options_root_dir(source: str = "yfinance") -> pathlib.Path:
    """
    Root directory for options data.

    Default: {BASE_DATA_PATH}/options/{source}
    """
    return pathlib.Path(BASE_DATA_PATH) / "options" / source


def _ticker_dir(ticker: str, source: str = "yfinance") -> pathlib.Path:
    """
    Directory for a given ticker's options under a given source.

    Example: /dev/shm/data/options/yfinance/SPY
    """
    return _options_root_dir(source) / ticker.upper()


def _expiration_filename(expiration: Union[str, pd.Timestamp]) -> str:
    """
    Normalize expiration into a canonical filename "YYYY-MM-DD.parquet".
    """
    if isinstance(expiration, pd.Timestamp):
        exp = expiration.normalize().strftime("%Y-%m-%d")
    else:
        exp = str(expiration)
    return f"{exp}.parquet"


def _option_chain_path(
    ticker: str,
    expiration: Union[str, pd.Timestamp],
    source: str = "yfinance",
) -> pathlib.Path:
    """
    Full path to the cached option chain file for (ticker, expiration, source).
    """
    return _ticker_dir(ticker, source) / _expiration_filename(expiration)


# ---------------------------------------------------------------------------
# Core download/load functions
# ---------------------------------------------------------------------------

def _download_option_chain_yfinance(
    ticker: str,
    expiration: Union[str, pd.Timestamp],
) -> pd.DataFrame:
    """
    Download the option chain for a single (ticker, expiration) from yfinance.

    Returns a normalized DataFrame that:
        - includes both calls and puts
        - has a 'type' column: 'call' / 'put'
        - has an 'expiration' column (Timestamp)
    """
    if not _HAS_YFINANCE:
        raise ImportError(
            "yfinance is not installed. Install it with `pip install yfinance` "
            "or modify options_data_retrieval.py to use your own data source."
        )

    # Normalize expiration to yfinance's string format
    if isinstance(expiration, pd.Timestamp):
        exp_str = expiration.normalize().strftime("%Y-%m-%d")
    else:
        exp_str = str(expiration)

    tkr = yf.Ticker(ticker)
    try:
        chain = tkr.option_chain(exp_str)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download option chain for {ticker} @ {exp_str} from yfinance: {e}"
        ) from e

    calls = chain.calls.copy()
    puts = chain.puts.copy()

    # Tag type and expiration
    calls["type"] = "call"
    puts["type"] = "put"

    # Add expiration column (Timestamp for consistency)
    exp_ts = pd.to_datetime(exp_str).normalize()
    calls["expiration"] = exp_ts
    puts["expiration"] = exp_ts

    df = pd.concat([calls, puts], ignore_index=True)
    # Standardize column order a bit (type, expiration, then rest)
    cols = ["type", "expiration"] + [c for c in df.columns if c not in ("type", "expiration")]
    df = df[cols]

    return df


def load_or_download_option_chain(
    ticker: str,
    expiration: Union[str, pd.Timestamp],
    source: str = "yfinance",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load a single (ticker, expiration) option chain from local cache, or download and cache it.

    Parameters
    ----------
    ticker : str
        Underlying symbol (e.g. "SPY").
    expiration : str or pd.Timestamp
        Expiration date. Example: "2025-01-17" or a Timestamp.
    source : str, default "yfinance"
        Data source namespace, used only in the path structure.
    force_refresh : bool, default False
        If True, always download fresh data and overwrite the cache.

    Returns
    -------
    df : pd.DataFrame
        Option chain with columns from the source plus:
            - 'type'       : "call" or "put"
            - 'expiration' : Timestamp
    """
    path = _option_chain_path(ticker, expiration, source)
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)

    # Need to download
    _ensure_dir(path.parent)

    if source == "yfinance":
        df = _download_option_chain_yfinance(ticker, expiration)
    else:
        raise ValueError(f"Unsupported options data source: {source!r}")

    df.to_parquet(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Cached listing / multi-expiration utilities
# ---------------------------------------------------------------------------

def list_cached_option_expirations(
    ticker: str,
    source: str = "yfinance",
) -> List[pd.Timestamp]:
    """
    List all cached expiration dates for a ticker under the given source.

    Returns a list of normalized Timestamps sorted ascending.
    """
    tdir = _ticker_dir(ticker, source)
    if not tdir.exists():
        return []

    exps: List[pd.Timestamp] = []
    for f in tdir.glob("*.parquet"):
        # Filename is "YYYY-MM-DD.parquet"
        stem = f.stem
        try:
            exps.append(pd.to_datetime(stem).normalize())
        except Exception:
            # Ignore files that don't match the date pattern
            continue

    exps = sorted(set(exps))
    return exps


def load_all_cached_option_chains(
    ticker: str,
    source: str = "yfinance",
    expirations: Optional[List[Union[str, pd.Timestamp]]] = None,
) -> pd.DataFrame:
    """
    Load a concatenated DataFrame of all cached option chains for a ticker.

    Parameters
    ----------
    ticker : str
        Underlying symbol.
    source : str, default "yfinance"
        Source namespace.
    expirations : list-like, optional
        If provided, load only this subset of expirations (must already be cached).
        If None, all cached expirations are loaded.

    Returns
    -------
    df : pd.DataFrame
        Concatenation of per-expiration cached chains. Empty DataFrame if none are available.
    """
    if expirations is None:
        expirations = list_cached_option_expirations(ticker, source)
    if not expirations:
        return pd.DataFrame()

    frames = []
    for exp in expirations:
        path = _option_chain_path(ticker, exp, source)
        if path.exists():
            frames.append(pd.read_parquet(path))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Remote listing helpers (e.g. to know what CAN be downloaded)
# ---------------------------------------------------------------------------

def _remote_expirations_yfinance(ticker: str) -> List[pd.Timestamp]:
    """
    List available option expirations for a ticker from yfinance (remote).

    Returns a list of normalized Timestamps.
    """
    if not _HAS_YFINANCE:
        raise ImportError(
            "yfinance is not installed. Install it with `pip install yfinance` "
            "or modify options_data_retrieval.py to use your own data source."
        )

    tkr = yf.Ticker(ticker)
    try:
        exps = tkr.options  # list of "YYYY-MM-DD" strings
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch remote expirations for {ticker} from yfinance: {e}"
        ) from e

    out: List[pd.Timestamp] = []
    for s in exps:
        try:
            out.append(pd.to_datetime(s).normalize())
        except Exception:
            continue
    return sorted(set(out))


def get_available_remote_expirations(
    ticker: str,
    source: str = "yfinance",
) -> List[pd.Timestamp]:
    """
    Get the list of expirations that are available from the remote data source.

    This does NOT download any chains; it's just a listing.

    Parameters
    ----------
    ticker : str
        Underlying symbol.
    source : str, default "yfinance"
        Data source. Currently only "yfinance" is implemented.

    Returns
    -------
    exps : list of Timestamps
        Sorted list of available expirations.
    """
    if source == "yfinance":
        return _remote_expirations_yfinance(ticker)
    else:
        raise ValueError(f"Unsupported options data source: {source!r}")


# ---------------------------------------------------------------------------
# Convenience: ensure a set of expirations are cached
# ---------------------------------------------------------------------------

def ensure_option_chains_cached(
    ticker: str,
    expirations: Optional[List[Union[str, pd.Timestamp]]] = None,
    source: str = "yfinance",
    force_refresh: bool = False,
) -> Dict[pd.Timestamp, pathlib.Path]:
    """
    Ensure that a set of expirations for a ticker are cached to disk.

    If expirations is None, this function will:
        - Query remote expirations
        - Download & cache all of them (unless already present, unless force_refresh=True)

    Returns a dict mapping each expiration Timestamp to its cached file path.
    """
    if expirations is None:
        expirations = get_available_remote_expirations(ticker, source)

    result: Dict[pd.Timestamp, pathlib.Path] = {}
    for exp in expirations:
        df = load_or_download_option_chain(
            ticker=ticker,
            expiration=exp,
            source=source,
            force_refresh=force_refresh,
        )
        # Path is deterministic
        path = _option_chain_path(ticker, exp, source)
        result[pd.to_datetime(df["expiration"].iloc[0]).normalize()] = path

    return result

