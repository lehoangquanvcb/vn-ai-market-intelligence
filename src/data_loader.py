from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_price_series(symbol: str, start="2018-01-01", end=None, periods=None, seed=7):
    """
    Fallback synthetic data. If end is supplied, series always runs to latest business day.
    """
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32) + seed)

    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    if periods is None:
        dates = pd.bdate_range(start=start, end=end)
    else:
        dates = pd.bdate_range(start=start, periods=periods)

    drift = 0.00025 + rng.normal(0, 0.00008)
    vol = 0.012 + rng.random() * 0.008

    returns = rng.normal(drift, vol, len(dates))
    cycle = 0.004 * np.sin(np.linspace(0, 18, len(dates)))
    returns += cycle / 20

    price0 = 1000 if symbol.upper() in ["VNINDEX", "VNI", "VN30"] else rng.uniform(15, 120)
    close = price0 * np.exp(np.cumsum(returns))
    volume = rng.lognormal(mean=14, sigma=0.35, size=len(dates)).astype(int)

    return pd.DataFrame({
        "date": dates,
        "ticker": symbol.upper(),
        "close": close,
        "volume": volume,
        "data_source": "synthetic_fallback",
    })


def _normalize_price_df(df, ticker="VNINDEX", source_name="unknown"):
    df = df.copy()

    if "time" in df.columns:
        df = df.rename(columns={"time": "date"})

    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})

    if "Close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"Close": "close"})

    if "Volume" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"Volume": "volume"})

    if "date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = ticker

    if "volume" not in df.columns:
        df["volume"] = np.nan

    df = df[["date", "ticker", "close", "volume"]].copy()
    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date").reset_index(drop=True)
    df["data_source"] = source_name

    return df


def _load_vnindex_from_vnstock():
    from vnstock import Vnstock

    attempts = [
        ("VNINDEX", "VCI"),
        ("VNI", "VCI"),
        ("VNINDEX", "TCBS"),
        ("VNI", "TCBS"),
    ]

    best_df = None

    for symbol, source in attempts:
        try:
            stock = Vnstock().stock(symbol=symbol, source=source)

            df = stock.quote.history(
                start="2018-01-01",
                end=pd.Timestamp.today().strftime("%Y-%m-%d"),
                interval="1D",
            )

            df = _normalize_price_df(
                df,
                ticker="VNINDEX",
                source_name=f"vnstock_{source}_{symbol}",
            )

            if len(df) > 500:
                if best_df is None or df["date"].max() > best_df["date"].max():
                    best_df = df

        except Exception as e:
            print(f"VNSTOCK LOAD ERROR {symbol}-{source}:", e)

    return best_df


def _load_vnindex_from_yfinance():
    try:
        import yfinance as yf

        df = yf.download(
            "^VNINDEX",
            start="2018-01-01",
            end=(pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
        )

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df = df.reset_index()
        df = _normalize_price_df(
            df,
            ticker="VNINDEX",
            source_name="yfinance_^VNINDEX",
        )

        if len(df) > 500:
            return df

    except Exception as e:
        print("YFINANCE LOAD ERROR:", e)

    return None


def load_vnindex():
    """
    Priority:
    1) vnstock live
    2) yfinance live fallback
    3) synthetic fallback that extends to today's business day
    """
    live_dfs = []

    try:
        df_vnstock = _load_vnindex_from_vnstock()
        if df_vnstock is not None:
            live_dfs.append(df_vnstock)
    except Exception as e:
        print("VNSTOCK GLOBAL ERROR:", e)

    try:
        df_yf = _load_vnindex_from_yfinance()
        if df_yf is not None:
            live_dfs.append(df_yf)
    except Exception as e:
        print("YFINANCE GLOBAL ERROR:", e)

    if live_dfs:
        best_df = max(live_dfs, key=lambda x: x["date"].max())
        latest = pd.to_datetime(best_df["date"].max())

        # Accept live data if it is not stale by more than 10 calendar days.
        if latest >= pd.Timestamp.today().normalize() - pd.Timedelta(days=10):
            return best_df

        print(f"LIVE DATA STALE. Latest date: {latest.date()}")

    return _synthetic_price_series(
        "VNINDEX",
        start="2018-01-01",
        end=pd.Timestamp.today().strftime("%Y-%m-%d"),
    )


def load_stock_prices(tickers):
    """
    Lightweight stock loader for Streamlit Cloud.
    Stock-level data remains synthetic to avoid slow multi-ticker live calls.
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
