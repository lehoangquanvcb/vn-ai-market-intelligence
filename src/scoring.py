import pandas as pd
import numpy as np
from .utils import normalize_0_100

def score_stocks(stock_features, fundamentals, sector_map, vnindex_ret20=0.0):
    df = stock_features.merge(fundamentals, on="ticker", how="left").merge(sector_map, on="ticker", how="left")
    df["relative_strength"] = df["momentum_20d"] - vnindex_ret20

    df["momentum_score"] = normalize_0_100(df["momentum_20d"] + 0.5 * df["momentum_60d"] + df["above_ma50"] * 0.05 + df["above_ma200"] * 0.05)
    df["growth_score"] = normalize_0_100(df["eps_growth_pct"] + 0.5 * df["revenue_growth_pct"] + df["roe_pct"])
    df["valuation_score"] = 100 - normalize_0_100(df["pe"] + df["pb"] * 4)
    df["financial_strength_score"] = normalize_0_100(df["roe_pct"] - df["debt_to_equity"] * 5)
    df["liquidity_score"] = normalize_0_100(np.log1p(df["liquidity_20d"].fillna(0)))
    df["flow_score"] = normalize_0_100(df["foreign_flow_20d_bn_vnd"])
    df["risk_score"] = 100 - normalize_0_100(df["vol_20d"].fillna(df["vol_20d"].median()))

    df["stock_score"] = (
        0.20 * df["momentum_score"] +
        0.20 * df["growth_score"] +
        0.15 * df["valuation_score"] +
        0.15 * df["financial_strength_score"] +
        0.10 * df["liquidity_score"] +
        0.10 * df["flow_score"] +
        0.10 * df["risk_score"]
    ).round(1)

    df["signal"] = pd.cut(
        df["stock_score"],
        bins=[-1, 45, 60, 75, 101],
        labels=["Avoid", "Watch", "Accumulate", "Top pick"]
    ).astype(str)
    return df.sort_values("stock_score", ascending=False)

def score_sectors(scored_stocks):
    g = scored_stocks.groupby("sector", dropna=False).agg(
        stock_score=("stock_score", "mean"),
        relative_strength=("relative_strength", "mean"),
        momentum_20d=("momentum_20d", "mean"),
        roe_pct=("roe_pct", "mean"),
        eps_growth_pct=("eps_growth_pct", "mean"),
        foreign_flow_20d_bn_vnd=("foreign_flow_20d_bn_vnd", "sum"),
        ticker_count=("ticker", "count")
    ).reset_index()
    g["sector_score"] = (
        0.25 * normalize_0_100(g["relative_strength"]) +
        0.20 * normalize_0_100(g["eps_growth_pct"]) +
        0.20 * normalize_0_100(g["stock_score"]) +
        0.15 * normalize_0_100(g["foreign_flow_20d_bn_vnd"]) +
        0.10 * normalize_0_100(g["roe_pct"]) +
        0.10 * normalize_0_100(g["momentum_20d"])
    ).round(1)
    return g.sort_values("sector_score", ascending=False)
