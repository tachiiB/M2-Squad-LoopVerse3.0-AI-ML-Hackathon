"""Sprint 2 forecast tool: forecast(location, target_date) per SPEC.md contract.

Uses the team's predictions.csv (Sprint 1 output). Validates location against
the 15 covered Lahore areas (and sensor IDs), and target_date against the
supported holdout window. Never guesses: out-of-scope -> structured unavailable.

SPEC contract:
{
  "location": str,
  "target_date": "YYYY-MM-DD",
  "pm25": float | null,
  "hazardous": bool | null,
  "status": "ok" | "unavailable",
  "source": "team_model" | "reference_fallback"
}
"""

import pandas as pd

# Covered areas (from sensor_metadata.csv) + sensor IDs (from holdout_inputs.csv)
AREAS = [
    "Gulberg", "DHA", "Johar Town", "Model Town", "Wagah",
    "Thokar Niaz Baig", "Allama Iqbal Town", "Shadman", "Cantt",
    "Faisal Town", "Township", "Walled City", "Raiwind Road",
    "Bahria Town", "Airport Road",
]
SENSORS = [f"S{i:02d}" for i in range(1, 16)]

_PREDICTIONS = None
_SENSOR_TO_AREA = None


def _load():
    global _PREDICTIONS, _SENSOR_TO_AREA
    if _PREDICTIONS is None:
        _PREDICTIONS = pd.read_csv("predictions.csv")
        _PREDICTIONS["target_date"] = _PREDICTIONS["target_date"].astype(str)
        meta = pd.read_csv("sensor_metadata.csv")
        _SENSOR_TO_AREA = dict(zip(meta["sensor_id"], meta["area"]))
    return _PREDICTIONS, _SENSOR_TO_AREA


def _normalize_location(location):
    """Return (kind, canonical) where kind in {'area','sensor'} or (None, None)."""
    if not isinstance(location, str):
        return None, None
    loc = location.strip()
    # exact sensor ID
    if loc.upper() in SENSORS:
        return "sensor", loc.upper()
    # area: case-insensitive match
    low = loc.lower()
    for a in AREAS:
        if a.lower() == low:
            return "area", a
    return None, None


def forecast(location, target_date, source="team_model"):
    """Structured forecast lookup. Never guesses outside coverage."""
    preds, s2a = _load()
    kind, canon = _normalize_location(location)
    tdate = str(target_date).strip()

    def unavailable(reason):
        return {"location": str(location), "target_date": tdate,
                "pm25": None, "hazardous": None,
                "status": "unavailable", "source": source,
                "reason": reason}

    if kind is None:
        return unavailable(
            f"'{location}' is not a covered Lahore area or sensor. "
            f"Covered areas: {', '.join(AREAS)}.")
    supported = set(preds["target_date"].unique())
    if tdate not in supported:
        return unavailable(
            f"Date {tdate} is outside the supported forecast window "
            f"({min(supported)} to {max(supported)}).")

    if kind == "sensor":
        row = preds[(preds["sensor_id"] == canon) & (preds["target_date"] == tdate)]
        area = s2a.get(canon)
    else:
        sensors = [s for s, a in s2a.items() if a == canon]
        row = preds[(preds["sensor_id"].isin(sensors)) & (preds["target_date"] == tdate)]
        area = canon
    if row.empty:
        return unavailable("No forecast available for this location/date combination.")
    pm25 = float(row["pm25" if "pm25" in row else "predicted_pm25"].mean())
    haz = bool((row["hazardous"] == 1).any())
    return {"location": canon, "area": area, "target_date": tdate,
            "pm25": round(pm25, 1), "hazardous": haz,
            "status": "ok", "source": source}


if __name__ == "__main__":
    import json
    tests = [
        ("Gulberg", "2026-10-29"),      # covered area + date
        ("S01", "2026-11-07"),          # sensor ID + last date
        ("Karachi", "2026-10-29"),      # unsupported city -> unavailable
        ("Gulberg", "2026-12-01"),      # unsupported date -> unavailable
        ("gulberg", "2026-10-29"),      # case-insensitive area
        ("", "2026-10-29"),             # empty -> unavailable
    ]
    for loc, d in tests:
        r = forecast(loc, d)
        print(f"forecast({loc!r}, {d}) -> status={r['status']}, "
              f"pm25={r['pm25']}, hazardous={r['hazardous']}")
    # show one full object
    print("\nfull object:")
    print(json.dumps(forecast("DHA", "2026-11-01"), indent=2))
