"""Can anything known at a window end predict whether a coin's NEXT 8h funding window is above
the default? Logistic regression first, random forest second, both validated on time-ordered
folds (never shuffled: a panel row's neighbours in time share the regime).

    python scripts/predict_above_default.py [--start 2023-01-01] [--end 2026-10-23]

Panel: one row per (hedgeable coin, 8h window end T) where the coin is listed. Features use only
information known at T. Target: funding in (T, T+8h] > default (0.01% per window).

Features
    f_last, f_3, f_9, f_30        trailing mean funding per window over 1/3/9/90 windows (bps)
    above_now                     last window above default
    spell_len                     consecutive windows above default ending at T
    vol_rank                      point-in-time rank by trailing 30-day perp dollar volume (1 = most liquid)
    log_adv                       log trailing 30-day perp dollar volume
    age_days                      days since first funding settlement IN THE SAMPLE: coins listed before
                                  the start date all look born on it (truncated, not a listing age)
    interval_h                    settlement interval in force (1/4/8h)
    hour                          window end hour (0/8/16 UTC)

Reports per model: AUC, log loss, precision in the top decile of predicted probability, the base
rate, a 10-bin calibration table, and the next-window funding (bps) of the top decile against
a hedged round trip of 4 legs x 7 bps = 28 bps. A no-fit baseline (the last window's funding
as the score) is scored on the same folds; a model is only worth anything above it. Logistic coefficients and forest importances are printed so the
first question — is the top feature leaking? — can be asked.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qr.data.funding import has_spot, interval_hours, load_funding, per_window
from qr.data.prices import load_kline_matrix

DEFAULT = 0.0001
OUT = Path("reports/predict"); OUT.mkdir(parents=True, exist_ok=True)
FEATURES = ["f_last", "f_3", "f_9", "f_30", "above_now", "spell_len", "vol_rank", "log_adv", "age_days", "interval_h", "hour"]


def panel(start: str, end: str) -> pd.DataFrame:
    u = [l.strip() for l in open("data/universe.txt") if l.strip()]
    H = has_spot(u)
    raw = load_funding(H, start, end)
    f8 = per_window(raw)                                                   # index = window end T, value known at T
    dv = load_kline_matrix(list(f8.columns), "PERP", "quote_vol", start, end).reindex(columns=f8.columns)
    tv = dv.fillna(0.0).rolling(24 * 30, min_periods=24 * 7).sum().shift(1).reindex(f8.index)
    rank = tv.where(f8.notna() & (tv > 0)).rank(axis=1, ascending=False, method="first")
    iv = interval_hours(raw).ffill().resample("8h", label="right", closed="right").last().reindex(f8.index).ffill()
    above = (f8 > DEFAULT + 1e-10)
    spell = pd.DataFrame({c: above[c].astype(int).groupby((~above[c]).cumsum()).cumsum() for c in above.columns})
    first = raw.apply(lambda s: s.first_valid_index())
    feats = {
        "f_last": f8 * 1e4,
        "f_3": f8.rolling(3, min_periods=2).mean() * 1e4,
        "f_9": f8.rolling(9, min_periods=5).mean() * 1e4,
        "f_30": f8.rolling(90, min_periods=45).mean() * 1e4,
        "above_now": above.astype(float).where(f8.notna()),
        "spell_len": spell.astype(float).where(f8.notna()),
        "vol_rank": rank,
        "log_adv": np.log(tv / 30).where(tv > 0),
        "interval_h": iv,
    }
    y = above.shift(-1).astype(float).where(f8.shift(-1).notna())              # next window
    df = pd.concat({k: v.stack() for k, v in feats.items()} | {"y": y.stack()}, axis=1).dropna()
    df.index.names = ["ts", "coin"]
    df = df.reset_index()
    df["age_days"] = (df["ts"] - df["coin"].map(first)).dt.total_seconds() / 86400
    df["hour"] = df["ts"].dt.hour
    return df.sort_values(["ts", "coin"]).reset_index(drop=True)


def evaluate(name: str, model, df: pd.DataFrame, n_splits: int = 5, train_cap: int = 400_000, seed: int = 0):
    """Time-ordered folds over distinct window ends; rows of one window end never straddle a split."""
    ts = np.sort(df["ts"].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=1)                 # one window between train and test
    rng = np.random.default_rng(seed)
    preds = pd.Series(np.nan, index=df.index)
    for k, (tr_t, te_t) in enumerate(tscv.split(ts)):
        tr = df.index[df["ts"].isin(ts[tr_t])]; te = df.index[df["ts"].isin(ts[te_t])]
        if len(tr) > train_cap:
            tr = rng.choice(tr, train_cap, replace=False)
        if model is None:                                             # baseline: no fitting, the last window's funding is the score
            preds.loc[te] = df.loc[te, "f_last"].values
            continue
        model.fit(df.loc[tr, FEATURES], df.loc[tr, "y"])
        preds.loc[te] = model.predict_proba(df.loc[te, FEATURES])[:, 1]
    ok = preds.notna()
    y, p = df.loc[ok, "y"].values, preds[ok].values
    top = p >= np.quantile(p, 0.9)
    out = {"model": name, "rows_scored": int(ok.sum()), "base_rate": float(y.mean()), "auc": float(roc_auc_score(y, p)),
           "log_loss": float(log_loss(y, p)) if model is not None else np.nan, "precision_top_decile": float(y[top].mean()),
           "lift_top_decile": float(y[top].mean() / y.mean())}
    calib = pd.DataFrame({"p": p, "y": y}); calib["bin"] = pd.qcut(calib["p"], 10, labels=False, duplicates="drop")
    cal = calib.groupby("bin").agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    return out, cal, preds


def main(a):
    df = panel(a.start, a.end)
    print(f"{len(df):,} coin-windows, {df.coin.nunique()} coins, base rate above default next window: {df.y.mean():.3f}")
    # next-window funding in bps, aligned per coin, for the economic read of the top decile
    df["f_next"] = df.groupby("coin")["f_last"].shift(-1)
    models = {
        "baseline_last_window": None,                                  # what persistence alone gives (study 1)
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=500, n_jobs=-1, random_state=0),
    }
    rows, cals = [], {}
    for name, m in models.items():
        res, cal, preds = evaluate(name, m, df)
        ok = preds.notna(); top = preds[ok] >= preds[ok].quantile(0.9)
        # like-for-like with the persistence rule: precision of the model's k most confident rows,
        # k = number of rows the rule "last window above default" fires on (ties in f_last make the
        # baseline's top-decile figure meaningless: most windows sit exactly at the default)
        k = int(df.loc[ok, "above_now"].sum())
        topk = preds[ok].rank(method="first", ascending=False) <= k
        res["rule_coverage_k"] = k
        res["precision_at_rule_coverage"] = float(df.loc[ok, "y"][topk].mean()) if m is not None \
            else float(df.loc[ok, "y"][df.loc[ok, "above_now"] == 1].mean())
        res["next_funding_bps_at_rule_coverage"] = float(df.loc[ok, "f_next"][topk].mean()) if m is not None \
            else float(df.loc[ok, "f_next"][df.loc[ok, "above_now"] == 1].mean())
        pd.DataFrame({"ts": df.loc[ok, "ts"], "coin": df.loc[ok, "coin"], "y": df.loc[ok, "y"], "p": preds[ok]}) \
            .to_parquet(OUT / f"oof_{name}.parquet")
        res["top_decile_next_funding_bps"] = float(df.loc[ok, "f_next"][top].mean())
        res["all_next_funding_bps"] = float(df.loc[ok, "f_next"].mean())
        rows.append(res); cals[name] = cal
        print(f"\n== {name} ==\n" + pd.Series(res).to_string()); print(cal.round(3).to_string())
        if m is None:
            continue
        fit_idx = df.sample(min(len(df), 400_000), random_state=0).index      # full-sample refit, only for the coefficient table
        m.fit(df.loc[fit_idx, FEATURES], df.loc[fit_idx, "y"])
        if name == "logistic":
            coef = pd.Series(m[-1].coef_[0], index=FEATURES).sort_values(key=abs, ascending=False)
            print("standardised coefficients:\n" + coef.round(3).to_string()); coef.to_csv(OUT / "logistic_coef.csv")
        else:
            imp = pd.Series(m.feature_importances_, index=FEATURES).sort_values(ascending=False)
            print("importances:\n" + imp.round(3).to_string()); imp.to_csv(OUT / "rf_importance.csv")
    summary = pd.DataFrame(rows); summary.to_csv(OUT / "summary.csv", index=False)
    for k, c in cals.items():
        c.to_csv(OUT / f"calibration_{k}.csv")
    print("\n" + summary.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01"); ap.add_argument("--end", default="2026-10-23")
    main(ap.parse_args())
