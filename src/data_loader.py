from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_price_series(symbol: str, start="2018-01-01", periods=1650, seed=7):
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32) + seed)
    dates = pd.bdate_range(start=start, periods=periods)

    drift = 0.00025 + rng.normal(0, 0.00008)
    vol = 0.012 + rng.random() * 0.008
    returns = rng.normal(drift, vol, len(dates))
    cycle = 0.004 * np.sin(np.linspace(0, 18, len(dates)))
    returns += cycle / 20

    price0 = 1000 if symbol.upper() in ["VNINDEX", "VN30"] else rng.uniform(15, 120)
    close = price0 * np.exp(np.cumsum(returns))
    volume = rng.lognormal(mean=14, sigma=0.35, size=len(dates)).astype(int)

    return pd.DataFrame({
        "date": dates,
        "ticker": symbol.upper(),
        "close": close,
        "volume": volume,
    })


def load_vnindex():
    """
    Load live VNINDEX daily history from vnstock when available.
    If the live call fails, fall back to synthetic data.
    """
    try:
        from vnstock import Vnstock

        stock = Vnstock().stock(symbol="VNINDEX", source="VCI")

        df = stock.quote.history(
            start="2018-01-01",
            end=pd.Timestamp.today().strftime("%Y-%m-%d"),
            interval="1D",
        )

        if "time" in df.columns:
            df = df.rename(columns={"time": "date"})

        if "date" not in df.columns:
            df = df.reset_index().rename(columns={"index": "date"})

        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = "VNINDEX"

        if "volume" not in df.columns:
            df["volume"] = np.nan

        df = df[["date", "ticker", "close", "volume"]].copy()
        df = df.dropna(subset=["date", "close"])
        df = df.sort_values("date").reset_index(drop=True)

        if len(df) > 500:
            return df

    except Exception as e:
        print("VNINDEX LOAD ERROR:", e)

    return _synthetic_price_series("VNINDEX", start="2018-01-01", periods=1650)


def load_stock_prices(tickers):
    """
    Lightweight stock loader for Streamlit Cloud.
    To avoid timeout from many live API calls, stock-level data is synthetic.
    VNINDEX remains live through load_vnindex().
    """
    frames = []
    tickers = tickers[:8]

    for t in tickers:
        frames.append(
            _synthetic_price_series(t, start="2021-01-01", periods=900)
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
        rows.append({
            "ticker": t,
            "roe_pct": rng.uniform(6, 28),
            "eps_growth_pct": rng.uniform(-15, 45),
            "revenue_growth_pct": rng.uniform(-10, 35),
            "debt_to_equity": rng.uniform(0.1, 2.5),
            "pe": rng.uniform(6, 28),
            "pb": rng.uniform(0.7, 4.5),
            "market_cap_bn_vnd": rng.uniform(5000, 500000),
            "foreign_flow_20d_bn_vnd": rng.normal(0, 250),
        })

    return pd.DataFrame(rows)
