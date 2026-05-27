import numpy as np
import pandas as pd
from .utils import safe_pct_change

def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("date").copy()
    d["ret_1d"] = safe_pct_change(d["close"], 1)
    d["ret_5d"] = safe_pct_change(d["close"], 5)
    d["ret_20d"] = safe_pct_change(d["close"], 20)
    d["ret_60d"] = safe_pct_change(d["close"], 60)
    d["ma20"] = d["close"].rolling(20).mean()
    d["ma50"] = d["close"].rolling(50).mean()
    d["ma200"] = d["close"].rolling(200).mean()
    d["trend_ma20"] = d["close"] / d["ma20"] - 1
    d["trend_ma50"] = d["close"] / d["ma50"] - 1
    d["trend_ma200"] = d["close"] / d["ma200"] - 1
    d["vol_20d"] = d["ret_1d"].rolling(20).std() * np.sqrt(252)
    d["liquidity_20d"] = d["volume"].rolling(20).mean()
    d["liquidity_chg_20d"] = d["liquidity_20d"] / d["liquidity_20d"].rolling(60).mean() - 1
    d["target_up_20d"] = (d["close"].shift(-20) / d["close"] - 1 > 0.03).astype(int)
    return d.replace([np.inf, -np.inf], np.nan)

def latest_stock_features(price_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, g in price_df.groupby("ticker"):
        g = g.sort_values("date").copy()
        g["ret_20d"] = safe_pct_change(g["close"], 20)
        g["ret_60d"] = safe_pct_change(g["close"], 60)
        g["ma50"] = g["close"].rolling(50).mean()
        g["ma200"] = g["close"].rolling(200).mean()
        g["vol_20d"] = safe_pct_change(g["close"]).rolling(20).std() * np.sqrt(252)
        g["liquidity_20d"] = g["volume"].rolling(20).mean()
        last = g.iloc[-1]
        rows.append({
            "ticker": ticker,
            "date": last["date"],
            "close": last["close"],
            "momentum_20d": last["ret_20d"],
            "momentum_60d": last["ret_60d"],
            "above_ma50": float(last["close"] > last["ma50"]) if pd.notna(last["ma50"]) else 0,
            "above_ma200": float(last["close"] > last["ma200"]) if pd.notna(last["ma200"]) else 0,
            "vol_20d": last["vol_20d"],
            "liquidity_20d": last["liquidity_20d"],
        })
    return pd.DataFrame(rows)
