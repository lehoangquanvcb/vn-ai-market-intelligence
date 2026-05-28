from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_price_series(symbol: str, start="2021-01-01", end=None, seed=7):
    """
    Synthetic data is used ONLY for stock-level demo data.
    It is NOT used for VNINDEX.
    """
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32) + seed)

    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    dates = pd.bdate_range(start=start, end=end)

    drift = 0.00025 + rng.normal(0, 0.00008)
    vol = 0.012 + rng.random() * 0.008

    returns = rng.normal(drift, vol, len(dates))
    cycle = 0.004 * np.sin(np.linspace(0, 18, len(dates)))
    returns += cycle / 20

    price0 = rng.uniform(15, 120)
    close = price0 * np.exp(np.cumsum(returns))
    volume = rng.lognormal(mean=14, sigma=0.35, size=len(dates)).astype(int)

    return pd.DataFrame(
        {
            "date": dates,
            "ticker": symbol.upper(),
            "close": close,
            "volume": volume,
            "data_source": "synthetic_stock_demo",
        }
    )


def _get_vnstock_quote_class():
    """
    Compatible with vnstock 4.x and fallback imports.
    Official migration examples recommend Quote instead of Vnstock().stock(...).
    """
    try:
        from vnstock.api.quote import Quote
        return Quote
    except Exception:
        pass

    try:
        from vnstock import Quote
        return Quote
    except Exception as e:
        raise ImportError(
            "Cannot import vnstock Quote API. Please install vnstock>=4.0.4."
        ) from e


def _normalize_price_df(df: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ValueError("Empty dataframe returned from vnstock.")

    df = df.copy()

    # Flatten MultiIndex columns if any.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    # Normalize date column.
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})

    if "tradingdate" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"tradingdate": "date"})

    if "trading_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"trading_date": "date"})

    if "date" not in df.columns:
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        if "index" in df.columns:
            df = df.rename(columns={"index": "date"})

    if "date" not in df.columns:
        raise KeyError(f"No date column. Returned columns: {list(df.columns)}")

    # Normalize close column.
    close_candidates = [
        "close",
        "close_price",
        "match_price",
        "last_price",
        "value",
    ]

    close_col = None
    for c in close_candidates:
        if c in df.columns:
            close_col = c
            break

    if close_col is None:
        raise KeyError(f"No close column. Returned columns: {list(df.columns)}")

    if close_col != "close":
        df = df.rename(columns={close_col: "close"})

    # Normalize volume column.
    if "volume" not in df.columns:
        volume_candidates = ["total_volume", "match_volume", "vol"]
        volume_col = None
        for c in volume_candidates:
            if c in df.columns:
                volume_col = c
                break

        if volume_col:
            df = df.rename(columns={volume_col: "volume"})
        else:
            df["volume"] = np.nan

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df["ticker"] = "VNINDEX"
    df["data_source"] = f"vnstock_quote_{source}_{symbol}"

    df = df[["date", "ticker", "close", "volume", "data_source"]].copy()
    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < 500:
        raise ValueError(f"VNINDEX history too short: {len(df)} rows.")

    return df


def _quote_history(Quote, symbol: str, source: str) -> pd.DataFrame:
    q = Quote(symbol=symbol, source=source)

    start = "2018-01-01"
    end = pd.Timestamp.today().strftime("%Y-%m-%d")

    errors = []

    # Try both interval formats because vnstock versions differ.
    for interval in ["1D", "d", "D"]:
        try:
            df = q.history(start=start, end=end, interval=interval)
            return _normalize_price_df(df, symbol=symbol, source=source)
        except Exception as e:
            errors.append(f"interval={interval}: {e}")

    raise RuntimeError(" | ".join(errors))


def load_vnindex() -> pd.DataFrame:
    """
    Load REAL VNINDEX data using the new vnstock Quote API.

    This function intentionally does NOT fall back to synthetic VNINDEX data,
    because synthetic VNINDEX makes the dashboard numerically misleading.
    """

    Quote = _get_vnstock_quote_class()

    # According to vnstock version notes, VNINDEX/HNXINDEX/UPCOMINDEX are standard index symbols.
    attempts = [
        ("VNINDEX", "VCI"),
        ("VNINDEX", "KBS"),
        ("VNINDEX", "MSN"),
        ("VNINDEX", "FMP"),
    ]

    results = []
    errors = []

    for symbol, source in attempts:
        try:
            df = _quote_history(Quote, symbol=symbol, source=source)

            latest_date = pd.to_datetime(df["date"].max())
            latest_close = float(df.iloc[-1]["close"])

            print(
                f"VNINDEX loaded from {source}: "
                f"latest_date={latest_date.date()}, close={latest_close:,.2f}"
            )

            results.append(df)

        except Exception as e:
            errors.append(f"{symbol}-{source}: {e}")
            print(f"VNINDEX LOAD ERROR {symbol}-{source}: {e}")

    if not results:
        raise RuntimeError(
            "Cannot load REAL VNINDEX data from vnstock Quote API. "
            "Do not use synthetic VNINDEX. Errors: " + " || ".join(errors)
        )

    # Pick the source with the latest available date.
    best = max(results, key=lambda x: x["date"].max())

    latest = pd.to_datetime(best["date"].max())
    if latest < pd.Timestamp.today().normalize() - pd.Timedelta(days=10):
        raise RuntimeError(
            f"VNINDEX data is stale. Latest available date: {latest.date()}. "
            "Please check vnstock source/API status."
        )

    return best


def load_stock_prices(tickers):
    """
    Lightweight stock loader for Streamlit Cloud.
    Stock-level data is still synthetic for demo stability.
    VNINDEX is real-only via load_vnindex().
    """

    frames = []
    tickers = tickers[:8]

    for t in tickers:
        frames.append(
            _synthetic_price_series(
                t,
                start="2021-01-01",
                end=pd.Timestamp.today().strftime("%Y-%m-%d"),
            )
        )

    return pd.concat(frames, ignore_index=True)


def load_sector_mapping():
    return pd.read_csv(ROOT / "data" / "sector_mapping.csv")


def load_macro_assumptions():
    df = pd.read_csv(ROOT / "data" / "macro_assumptions.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def make_synthetic_fundamentals(tickers):
    rng = np.random.default_rng(42)
    rows = []

    for t in tickers:
        rows.append(
            {
                "ticker": t,
                "roe_pct": rng.uniform(6, 28),
                "eps_growth_pct": rng.uniform(-15, 45),
                "revenue_growth_pct": rng.uniform(-10, 35),
                "debt_to_equity": rng.uniform(0.1, 2.5),
                "pe": rng.uniform(6, 28),
                "pb": rng.uniform(0.7, 4.5),
                "market_cap_bn_vnd": rng.uniform(5000, 500000),
                "foreign_flow_20d_bn_vnd": rng.normal(0, 250),
            }
        )

    return pd.DataFrame(rows)
