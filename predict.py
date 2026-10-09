"""Sprint 1 final: train champion ensemble on ALL data, RECURSIVELY predict the
150 holdout targets (Oct 29 -> Nov 7), write predictions.csv per SPEC.md.

Why recursive: holdout origins run Oct 28 -> Nov 6, but our data ends Oct 28.
For target T we need history <= T-2; for T >= Oct 31 that includes earlier
holdout targets, so we predict chronologically and feed predictions back as
history. Validated: 10-step recursive MAE 10.7 (no blow-up) on Oct 1-10.

Champion: 0.5 * Ridge(alpha=1.0) + 0.5 * LightGBM-quantile(alpha=0.6).
Strict T-2 convention throughout (zero leakage risk).

SPEC contract:
  columns (exact order): sensor_id, target_date (YYYY-MM-DD), predicted_pm25, hazardous (0/1)
  150 rows; every (sensor_id, target_date) pair from holdout_inputs.csv exactly once;
  hazardous = 1 iff predicted_pm25 >= 165; no index column; no missing cells.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import lightgbm as lgb

from preprocess import build_daily_panel
from model import build_features, features_for_targets, HAZ_THRESHOLD

LGBM_PARAMS = {"objective": "quantile", "alpha": 0.6, "verbosity": -1,
               "num_leaves": 31, "learning_rate": 0.05,
               "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1}


def main():
    panel = build_daily_panel("batch_1_sensor_data.csv", "batch_2_sensor_data.csv")
    panel["date"] = pd.to_datetime(panel["date"])

    # --- train on ALL data ---
    feats = build_features(panel, use_origin_day=False)
    feature_cols = [c for c in feats.columns
                    if c not in ("sensor_id", "target_date", "target")]
    X, y = feats[feature_cols].values, feats["target"].values
    ridge = Ridge(alpha=1.0).fit(X, y)
    gbm = lgb.train(LGBM_PARAMS, lgb.Dataset(X, label=y), num_boost_round=300)
    print(f"trained on {len(feats)} rows x {len(feature_cols)} features")

    # --- recursive holdout prediction ---
    holdout = pd.read_csv("holdout_inputs.csv")
    holdout["target_date"] = pd.to_datetime(holdout["target_date"])
    target_dates = sorted(holdout["target_date"].unique())
    print(f"holdout: {len(holdout)} rows, targets "
          f"{target_dates[0].date()} -> {target_dates[-1].date()}")

    hist = panel[["sensor_id", "date", "pm25"]].copy()  # real history; grows with preds
    all_preds = []
    for T in target_dates:
        sensors = sorted(holdout[holdout["target_date"] == T]["sensor_id"].unique())
        tgts = pd.DataFrame({"sensor_id": sensors, "target_date": [T] * len(sensors)})
        hf = features_for_targets(hist, tgts, use_origin_day=False)
        hf = hf.reindex(columns=["sensor_id", "target_date"] + feature_cols)
        assert len(hf) == len(sensors), f"feature failure for {T.date()}"
        Xh = hf[feature_cols].values
        assert np.isfinite(Xh).all(), f"non-finite features for {T.date()}"
        pred = 0.5 * ridge.predict(Xh) + 0.5 * gbm.predict(Xh)
        assert np.isfinite(pred).all()
        for s, p in zip(sensors, pred):
            all_preds.append((s, T, float(p)))
        # feed back as history for later targets
        new = pd.DataFrame({"sensor_id": sensors, "date": [T] * len(sensors),
                            "pm25": pred})
        hist = pd.concat([hist, new], ignore_index=True)
        n_real = (hist["date"] <= "2026-10-28").sum() // 1  # noqa
        print(f"  {T.date()}: predicted {len(sensors)} sensors "
              f"(history now {len(hist)} rows)")

    out = pd.DataFrame(all_preds, columns=["sensor_id", "target_date", "predicted_pm25"])
    out["target_date"] = out["target_date"].dt.strftime("%Y-%m-%d")
    out["predicted_pm25"] = out["predicted_pm25"].round(2)
    out["hazardous"] = (out["predicted_pm25"] >= HAZ_THRESHOLD).astype(int)
    out = out[["sensor_id", "target_date", "predicted_pm25", "hazardous"]]
    out = out.sort_values(["target_date", "sensor_id"]).reset_index(drop=True)

    # --- validate SPEC contract ---
    assert len(out) == 150, f"need 150 rows, got {len(out)}"
    key_in = set(zip(holdout["sensor_id"], holdout["target_date"].dt.strftime("%Y-%m-%d")))
    key_out = set(zip(out["sensor_id"], out["target_date"]))
    assert key_in == key_out, "holdout pairs != output pairs"
    assert out.isna().sum().sum() == 0, "missing cells!"
    assert out["hazardous"].isin([0, 1]).all()
    assert np.isfinite(out["predicted_pm25"]).all()
    assert (out["predicted_pm25"] != -999).all()

    out.to_csv("predictions.csv", index=False)
    print("\nwrote predictions.csv")
    print(f"hazardous=1: {out['hazardous'].sum()}/{len(out)}")
    print(f"predicted_pm25 range: {out['predicted_pm25'].min():.1f} .. {out['predicted_pm25'].max():.1f}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
