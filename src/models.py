import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "ret_5d", "ret_20d", "ret_60d",
    "trend_ma20", "trend_ma50", "trend_ma200",
    "vol_20d", "liquidity_chg_20d"
]

def train_market_classifier(market_features: pd.DataFrame):
    d = market_features.dropna(subset=FEATURE_COLS + ["target_up_20d"]).copy()
    d = d.iloc[:-20] if len(d) > 250 else d
    X = d[FEATURE_COLS]
    y = d["target_up_20d"].astype(int)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    if len(d) > 300 and y.nunique() > 1:
        split = int(len(d) * 0.8)
        model.fit(X.iloc[:split], y.iloc[:split])
        pred = model.predict(X.iloc[split:])
        proba = model.predict_proba(X.iloc[split:])[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y.iloc[split:], pred)),
            "auc": float(roc_auc_score(y.iloc[split:], proba)) if y.iloc[split:].nunique() > 1 else None
        }
        model.fit(X, y)
    else:
        model.fit(X, y)
        metrics = {"accuracy": None, "auc": None}

    return model, metrics

def predict_market(model, market_features: pd.DataFrame):
    latest = market_features.sort_values("date").iloc[-1]
    X_latest = latest[FEATURE_COLS].to_frame().T
    prob_up = float(model.predict_proba(X_latest)[0, 1])
    close = float(latest["close"])
    vol = float(latest.get("vol_20d", 0.2) or 0.2)
    # simple expected range for 20 trading days
    sigma_20 = vol / np.sqrt(252) * np.sqrt(20)
    lower = close * (1 - 1.15 * sigma_20)
    upper = close * (1 + 1.15 * sigma_20)
    if prob_up >= 0.62:
        regime = "Risk-on mạnh"
    elif prob_up >= 0.53:
        regime = "Risk-on yếu"
    elif prob_up >= 0.45:
        regime = "Neutral"
    else:
        regime = "Risk-off"
    return {
        "date": latest["date"],
        "vnindex": close,
        "prob_up_20d": prob_up,
        "expected_lower": lower,
        "expected_upper": upper,
        "regime": regime
    }
