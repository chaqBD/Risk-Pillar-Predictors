"""Risk pillar model engine. Runs in the browser via Pyodide (also works in normal Python).

Each month is predicted from that country's recent history (lags + rolling stats),
models are trained on everything before the test window and scored on the test window.
"""
import io
import json
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

FEATURES = ["lag1", "lag2", "lag3", "lag6", "lag12", "mean3", "mean6", "mean12", "std6", "month"]

MODELS = {
    "linear":   lambda: LinearRegression(),
    "ridge":    lambda: Ridge(alpha=1.0),
    "lasso":    lambda: Lasso(alpha=0.01, max_iter=5000),
    "enet":     lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
    "huber":    lambda: HuberRegressor(max_iter=5000),
    "knn":      lambda: KNeighborsRegressor(n_neighbors=10),
    "tree":     lambda: DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42),
    "rf":       lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42),
    "et":       lambda: ExtraTreesRegressor(n_estimators=100, min_samples_leaf=5, random_state=42),
    "gbr":      lambda: HistGradientBoostingRegressor(random_state=42),
}

STATE = {}


def load_csv(text):
    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip() for c in df.columns]
    if len(df.columns) < 3:
        raise ValueError("Expected at least 3 columns: Country, Period, and one or more risk pillars.")
    lower = {c.lower(): c for c in df.columns}
    ent = next((lower[k] for k in ("country", "entity", "name") if k in lower), df.columns[0])
    per = next((lower[k] for k in ("period", "date", "month") if k in lower), df.columns[1])
    df = df.rename(columns={ent: "Country", per: "Period"})

    df["Period"] = pd.to_datetime(df["Period"].astype(str), errors="coerce")
    dropped = int(df["Period"].isna().sum())
    df = df.dropna(subset=["Period"])
    if df.empty:
        raise ValueError(f"Couldn't read any dates in the '{per}' column. Use a format like 2025-01.")
    df["Period"] = df["Period"].dt.to_period("M").dt.to_timestamp()

    pillars = [c for c in df.columns if c not in ("Country", "Period") and pd.api.types.is_numeric_dtype(df[c])]
    if not pillars:
        raise ValueError("No numeric risk-pillar columns found.")
    df = df.groupby(["Country", "Period"], as_index=False)[pillars].mean().sort_values(["Country", "Period"])

    STATE.update(df=df, pillars=pillars, cache={})
    return json.dumps({
        "rows": int(len(df)), "countries": int(df["Country"].nunique()),
        "start": df["Period"].min().strftime("%Y-%m"), "end": df["Period"].max().strftime("%Y-%m"),
        "months": int(df["Period"].nunique()), "pillars": pillars, "droppedRows": dropped,
        "countryList": sorted(df["Country"].astype(str).unique().tolist()),
    })


def _features(pillar):
    df = STATE["df"]
    d = df[["Country", "Period"]].copy()
    d["target"] = df.groupby("Country")[pillar].transform(lambda x: x.ffill().bfill())
    g = d.groupby("Country")["target"]
    for k in (1, 2, 3, 6, 12):
        d[f"lag{k}"] = g.shift(k)
    prev = g.shift(1).groupby(d["Country"])
    for w in (3, 6, 12):
        d[f"mean{w}"] = prev.transform(lambda x: x.rolling(w).mean())
    d["std6"] = prev.transform(lambda x: x.rolling(6).std(ddof=0))
    d["month"] = d["Period"].dt.month
    return d.dropna(subset=FEATURES + ["target"]).reset_index(drop=True)


def _pipeline(key):
    prep = ColumnTransformer([
        ("country", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Country"]),
        ("num", StandardScaler(), FEATURES),
    ])
    return make_pipeline(prep, MODELS[key]())


def _split(pillar, test_months):
    ck = ("data", pillar, test_months)
    if ck not in STATE["cache"]:
        d = _features(pillar)
        cutoff = d["Period"].max() - pd.DateOffset(months=test_months - 1)
        train, test = d[d["Period"] < cutoff], d[d["Period"] >= cutoff]
        if len(train) < 50 or test.empty:
            raise ValueError("Not enough history: each country needs at least 13 months before the test window.")
        STATE["cache"][ck] = (train, test)
    return STATE["cache"][ck]


def run_model(key, pillar, test_months):
    test_months = int(test_months)
    train, test = _split(pillar, test_months)

    t0 = time.time()
    model = _pipeline(key).fit(train, train["target"])
    pred = model.predict(test)
    if train["target"].min() >= 0:
        pred = np.clip(pred, 0, None)
    secs = time.time() - t0

    y, naive = test["target"].to_numpy(), test["lag1"].to_numpy()
    denom = np.abs(y) + np.abs(pred)
    smape = np.where(denom == 0, 0.0, 2 * np.abs(y - pred) / np.where(denom == 0, 1, denom)).mean() * 100
    mae = mean_absolute_error(y, pred)
    naive_mae = mean_absolute_error(y, naive)

    out = test[["Country", "Period"]].copy()
    out["actual"], out["pred"] = y, pred
    out["Period"] = out["Period"].dt.strftime("%Y-%m")
    avg = out.groupby("Period")[["actual", "pred"]].mean().reset_index()
    by_country = {c: {"actual": g["actual"].round(4).tolist(), "pred": g["pred"].round(4).tolist(),
                      "periods": g["Period"].tolist()} for c, g in out.groupby("Country")}

    return json.dumps({
        "key": key, "pillar": pillar, "testMonths": test_months,
        "metrics": {
            "MAE": mae,
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": float(r2_score(y, pred)),
            "sMAPE": float(smape),
            "MASE": float(mae / naive_mae) if naive_mae > 0 else None,
        },
        "naiveMAE": naive_mae, "trainRows": int(len(train)), "testRows": int(len(test)),
        "testStart": out["Period"].min(), "testEnd": out["Period"].max(), "seconds": secs,
        "series": {"periods": avg["Period"].tolist(), "actual": avg["actual"].round(4).tolist(),
                   "pred": avg["pred"].round(4).tolist()},
        "byCountry": by_country,
    })
