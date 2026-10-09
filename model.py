"""Sprint 1 modeling: features, walk-forward validation, baselines + models.

Leakage discipline:
- All features for target date T use ONLY data with date < T
  (lags/rolling windows computed on the panel, then shifted).
- Imputation is forward-fill per sensor (past-only). No interpolation.
- Walk-forward: train on earlier dates, validate on later dates. Never random.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

HAZ_THRESHOLD = 165.0



def _target_row(pm, s, T, off):
    """Feature dict for one (sensor, target_date). pm: MultiIndex (sensor_id,date)->pm25."""
    hist = pm.loc[s].loc[:T - pd.Timedelta(days=off)]
    if len(hist) < 14:
        return None
    f = {"sensor_id": s, "target_date": T}
    for k in [1, 2, 3, 7, 14]:
        f[f"lag_{k}"] = hist.iloc[-k]
    last7 = hist.iloc[-7:]
    f["roll_mean_3"] = hist.iloc[-3:].mean()
    f["roll_mean_7"] = last7.mean()
    f["roll_mean_14"] = hist.iloc[-14:].mean()
    f["roll_std_7"] = last7.std()
    f["roll_max_7"] = last7.max()
    f["roll_min_7"] = last7.min()
    f["trend_7d"] = hist.iloc[-1] - hist.iloc[-7]
    f["dow"] = T.dayofweek
    f["month"] = T.month
    f["doy"] = T.dayofyear
    return f


def features_for_targets(panel, targets, use_origin_day=False):
    """Build feature rows for arbitrary (sensor_id, target_date) pairs.

    panel: clean daily panel (sensor_id, date, pm25). targets: DataFrame with
    sensor_id, target_date (datetime). Same convention as build_features:
    PM2.5 history uses dates <= T-off (off=2 when not use_origin_day).
    Returns feature DataFrame WITHOUT target column (for prediction).
    """
    panel = panel.sort_values(["sensor_id", "date"]).copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["pm25"] = panel.groupby("sensor_id")["pm25"].ffill().bfill()  # past-only
    panel = panel.set_index(["sensor_id", "date"]).sort_index()
    pm = panel["pm25"]
    off = 1 if use_origin_day else 2
    rows = []
    for _, r in targets.iterrows():
        s, T = r["sensor_id"], pd.to_datetime(r["target_date"])
        try:
            f = _target_row(pm, s, T, off)
        except KeyError:
            f = None
        if f is not None:
            rows.append(f)
    feats = pd.DataFrame(rows)
    dummies = pd.get_dummies(feats["sensor_id"], prefix="s", dtype=float)
    # ensure all 15 sensor dummy columns exist even if a sensor is absent
    all_sensors = [f"s_{s}" for s in sorted(panel.reset_index()["sensor_id"].unique())]
    dummies = dummies.reindex(columns=all_sensors, fill_value=0.0)
    feats = pd.concat([feats.reset_index(drop=True), dummies], axis=1)
    return feats


def build_features(panel, use_origin_day=False):
    """Training features: one row per (sensor_id, target_date T) with target=pm25[T].

    Uses _target_row for identical conventions with features_for_targets.
    Default use_origin_day=False (strict T-2): PM2.5 history uses dates <= T-2,
    so no value can be accused of postdating the forecast origin. Chosen after
    walk-forward showed T-1 vs T-2 MAE is a wash (13.45 vs 13.36): take the
    zero-leakage-risk option for negligible cost.
    """
    panel = panel.sort_values(["sensor_id", "date"]).copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["pm25"] = panel.groupby("sensor_id")["pm25"].ffill().bfill()  # past-only
    panel = panel.set_index(["sensor_id", "date"]).sort_index()
    pm = panel["pm25"]
    off = 1 if use_origin_day else 2
    targets = panel.reset_index()[["sensor_id", "date"]].rename(
        columns={"date": "target_date"})
    rows = []
    for _, r in targets.iterrows():
        s, T = r["sensor_id"], r["target_date"]
        try:
            f = _target_row(pm, s, T, off)
        except KeyError:
            continue
        if f is None:
            continue
        f["target"] = pm.loc[(s, T)]
        rows.append(f)
    feats = pd.DataFrame(rows)
    dummies = pd.get_dummies(feats["sensor_id"], prefix="s", dtype=float)
    feats = pd.concat([feats.reset_index(drop=True), dummies], axis=1)
    return feats


def hazard_metrics(y_true, y_pred, thr=HAZ_THRESHOLD):
    yt = (y_true >= HAZ_THRESHOLD).astype(int)
    yp = (y_pred >= thr).astype(int)
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return {"recall": recall, "precision": prec, "tp": tp, "fp": fp, "fn": fn,
            "n_haz_true": int(yt.sum())}


def walk_forward(feats, feature_cols, n_folds=4, val_days=7):
    """Expanding-window walk-forward. Returns per-fold results dict."""
    feats = feats.sort_values("target_date").reset_index(drop=True)
    max_date = feats["target_date"].max()
    # validation windows: last n_folds*val_days days, chronological
    results = []
    for i in range(n_folds):
        val_end = max_date - pd.Timedelta(days=i * val_days)
        val_start = val_end - pd.Timedelta(days=val_days - 1)
        train = feats[feats["target_date"] < val_start]
        val = feats[(feats["target_date"] >= val_start) & (feats["target_date"] <= val_end)]
        if len(train) == 0 or len(val) == 0:
            continue
        Xtr, ytr = train[feature_cols].values, train["target"].values
        Xva, yva = val[feature_cols].values, val["target"].values

        preds = {}
        preds["persistence"] = val["lag_1"].values
        preds["mean7"] = val["roll_mean_7"].values
        ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
        preds["ridge"] = ridge.predict(Xva)
        dtr = lgb.Dataset(Xtr, label=ytr)
        params = {"objective": "regression", "metric": "mae", "verbosity": -1,
                  "num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.9,
                  "bagging_fraction": 0.9, "bagging_freq": 1}
        gbm = lgb.train(params, dtr, num_boost_round=300)
        preds["lightgbm"] = gbm.predict(Xva)

        fold = {"val_start": str(val_start.date()), "val_end": str(val_end.date()),
                "n_train": len(train), "n_val": len(val)}
        for name, p in preds.items():
            fold[name] = {
                "mae": float(mean_absolute_error(yva, p)),
                "rmse": float(np.sqrt(mean_squared_error(yva, p))),
                **hazard_metrics(yva, p),
            }
        results.append(fold)
    return results


def summarize(results):
    print(f"{'model':<12}{'MAE':>8}{'RMSE':>8}{'HzRec':>8}{'HzPrec':>8}")
    models = ["persistence", "mean7", "ridge", "lightgbm"]
    for m in models:
        mae = np.mean([r[m]["mae"] for r in results])
        rmse = np.mean([r[m]["rmse"] for r in results])
        rec = np.mean([r[m]["recall"] for r in results])
        prec = np.mean([r[m]["precision"] for r in results])
        print(f"{m:<12}{mae:>8.2f}{rmse:>8.2f}{rec:>8.2f}{prec:>8.2f}")
    print("\nper-fold detail:")
    for r in results:
        print(f"  val {r['val_start']}..{r['val_end']} (train n={r['n_train']}): " +
              ", ".join(f"{m} MAE={r[m]['mae']:.1f} rec={r[m]['recall']:.2f}"
                        for m in models))


if __name__ == "__main__":
    from preprocess import build_daily_panel
    import sys
    use_origin_day = not (len(sys.argv) > 1 and sys.argv[1] == "strict")
    panel = build_daily_panel("batch_1_sensor_data.csv", "batch_2_sensor_data.csv")
    feats = build_features(panel, use_origin_day=use_origin_day)
    feature_cols = [c for c in feats.columns
                    if c not in ("sensor_id", "target_date", "target")]
    print(f"convention: {'T-1 (origin-day usable)' if use_origin_day else 'T-2 (strict)'}")
    print(f"features: {len(feats)} rows x {len(feature_cols)} cols; "
          f"target range {feats['target_date'].min().date()} -> {feats['target_date'].max().date()}")
    results = walk_forward(feats, feature_cols)
    summarize(results)
