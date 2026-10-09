"""Sprint 1 preprocessing: standardize, clean, and merge the two sensor networks.

Pipeline order (per SPEC.md / handbook):
1. Parse timestamps, align to Pakistan local time (Asia/Karachi)
2. Convert AQI -> PM2.5 (US EPA breakpoints, inverted) BEFORE merging
3. -999 (any float form) -> NaN (missing/invalid, never averaged)
4. Flat-line faults: >=3 consecutive identical daily values -> NaN
5. Merge the two networks per (sensor_id, date): mean of available networks

Assumptions (validated against the data):
- batch_1 timestamps are UTC ("2026-06-30 19:00:00" UTC == "2026-07-01" PKT,
  which is exactly batch_2's first date; the S07 stuck interval also lines up
  across networks only after this conversion).
- batch_2 values are US EPA AQI (integers); cross-check: batch_1 S07 stuck at
  PM2.5=71.8 while batch_2 S07 stuck at AQI=159 on the same PKT dates, which
  matches the EPA breakpoint math.
"""

import numpy as np
import pandas as pd

# US EPA PM2.5 breakpoints: (AQI_low, AQI_high, PM_low, PM_high)
_EPA_PM25 = [
    (0, 50, 0.0, 12.0),
    (51, 100, 12.1, 35.4),
    (101, 150, 35.5, 55.4),
    (151, 200, 55.5, 150.4),
    (201, 300, 150.5, 250.4),
    (301, 400, 250.5, 350.4),
    (401, 500, 350.5, 500.4),
]


def aqi_to_pm25(aqi):
    """Invert the US EPA AQI formula for PM2.5. Vectorized via numpy."""
    aqi = np.asarray(aqi, dtype=float)
    pm = np.full_like(aqi, np.nan)
    for i_lo, i_hi, c_lo, c_hi in _EPA_PM25:
        m = (aqi >= i_lo) & (aqi <= i_hi)
        pm[m] = (aqi[m] - i_lo) / (i_hi - i_lo) * (c_hi - c_lo) + c_lo
    return pm


def load_batch1(path):
    """Batch 1: PM2.5 ug/m3, timestamps in UTC -> convert to PKT dates."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Karachi")
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["pm25"] = pd.to_numeric(df["reading_value"], errors="coerce")
    df.loc[df["pm25"] == -999, "pm25"] = np.nan  # -999 / -999.0 sentinel
    df["network"] = "batch_1"
    return df[["sensor_id", "date", "pm25", "network"]]


def load_batch2(path):
    """Batch 2: US EPA AQI integers, date-only (already PKT) -> convert to PM2.5."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)
    aqi = pd.to_numeric(df["reading_value"], errors="coerce")
    aqi = aqi.where(aqi != -999, np.nan)  # sentinel
    df["pm25"] = aqi_to_pm25(aqi)
    df["network"] = "batch_2"
    return df[["sensor_id", "date", "pm25", "network"]]


def flag_flatlines(df, min_run=3):
    """Mark runs of >=min_run identical consecutive daily values as faults (NaN).

    Operates per (sensor_id, network) on date-sorted values. Returns a copy
    with an added boolean column 'flat_fault'.
    """
    df = df.sort_values(["sensor_id", "network", "date"]).copy()
    df["flat_fault"] = False
    for (_, _), g in df.groupby(["sensor_id", "network"]):
        v = g["pm25"].to_numpy()
        # run-length encoding of identical consecutive values (NaN breaks runs)
        is_same = np.concatenate([[False], (v[1:] == v[:-1]) & ~np.isnan(v[1:]) & ~np.isnan(v[:-1])])
        # for each position, length of the run it belongs to
        run_id = np.cumsum(~is_same)
        run_len = np.bincount(run_id)[run_id]
        df.loc[g.index, "flat_fault"] = run_len >= min_run
    df.loc[df["flat_fault"], "pm25"] = np.nan
    return df


def build_daily_panel(b1_path, b2_path):
    """Full cleaning pipeline -> one daily PM2.5 panel per sensor.

    Returns DataFrame: sensor_id, date, pm25 (network mean), n_networks,
    plus pm25_batch_1 / pm25_batch_2 for traceability.
    """
    b1 = flag_flatlines(load_batch1(b1_path))
    b2 = flag_flatlines(load_batch2(b2_path))
    both = pd.concat([b1, b2], ignore_index=True)
    piv = both.pivot_table(index=["sensor_id", "date"], columns="network",
                           values="pm25", aggfunc="mean", dropna=False)
    piv.columns = [f"pm25_{c}" for c in piv.columns]
    piv = piv.reset_index()
    piv["pm25"] = piv[[c for c in piv.columns if c.startswith("pm25_")]].mean(axis=1, skipna=True)
    piv["n_networks"] = piv[[c for c in piv.columns if c.startswith("pm25_")]].notna().sum(axis=1)
    return piv.sort_values(["sensor_id", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    panel = build_daily_panel("batch_1_sensor_data.csv", "batch_2_sensor_data.csv")
    print("panel shape:", panel.shape)
    print("date range:", panel["date"].min(), "->", panel["date"].max())
    print("sensors:", panel["sensor_id"].nunique())
    print("rows with 0 networks (both missing):", (panel["n_networks"] == 0).sum())
    print("rows with 1 network:", (panel["n_networks"] == 1).sum())
    print("rows with 2 networks:", (panel["n_networks"] == 2).sum())
    print(panel.head(8).to_string())
    # sanity: S07 stuck interval should be NaN now
    s07 = panel[(panel["sensor_id"] == "S07")]
    print("\nS07 2026-08-24..28:")
    print(s07[s07["date"].between("2026-08-24", "2026-08-28")][["date", "pm25_batch_1", "pm25_batch_2", "pm25"]].to_string(index=False))
