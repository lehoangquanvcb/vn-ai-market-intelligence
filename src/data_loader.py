from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_price_series(symbol: str, start="2018-01-01", end=None, seed=7):

    rng = np.random.default_rng(abs(hash(symbol)) % (2**32) + seed)

    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    dates = pd.bdate_range(start=start, end=end)

    drift = 0.00025 + rng.normal(0, 0.00008)
    vol = 0.012 + rng.random() * 0.008

    returns = rng.normal(drift, vol, len(dates))

    cycle = 0.004 * np.sin(np.linspace(0, 18, len(dates)))
    returns += cycle / 20

    price0 = 1000 if symbol.upper() in ["VNINDEX", "VNI", "VN30"] else rng.uniform(15, 120)

    close = price0 * np.exp(np.cumsum(returns))

    volume = rng.lognormal(
        mean=14,
        sigma=0.35,
        size=len(dates)
    ).astype(int)

    return pd.DataFrame({
        "date": dates,
        "ticker": symbol.upper(),
        "close": close,
        "volume": volume,
    })


def load_vnindex():

    import yfinance as yf

    try:

        df = yf.download(
            "^VNINDEX.VN",
            start="2018-01-01",
            end=(pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
        )

        if df is None or len(df) == 0:
            raise ValueError("No VNINDEX data")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df = df.reset_index()

        df = df.rename(columns={
            "Date": "date",
            "Close": "close",
            "Volume": "volume",
        })

        df["date"] = pd.to_datetime(df["date"])

        df["ticker"] = "VNINDEX"

        df = df[
            ["date", "ticker", "close", "volume"]
        ].copy()

        df = df.dropna(subset=["close"])

        df = df.sort_values("date")

        print("LIVE VNINDEX DATA:")
        print(df.tail())

        return df

    except Exception as e:

        print("VNINDEX LOAD ERROR:", e)

        staleness_end = pd.Timestamp.today().strftime("%Y-%m-%d")

        print("Fallback synthetic data used.")

        return _synthetic_price_series(
            "VNINDEX",
            start="2018-01-01",
            end=staleness_end,
        )


def load_stock_prices(tickers):

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

    df = pd.read_csv(
        ROOT / "data" / "macro_assumptions.csv"
    )

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
