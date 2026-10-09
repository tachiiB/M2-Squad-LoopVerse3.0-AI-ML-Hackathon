# Recommendation — Lahore Smog Intelligence Challenge

## What we built and why we recommend it
A two-part system that is simple, tested, and honest:

1. **Forecast (Sprint 1):** An ensemble (Ridge + LightGBM-quantile) on lag,
   rolling, trend, and seasonal features, validated with chronological
   walk-forward (MAE 12.92). Every preprocessing decision is documented and
   every feature is provably leak-free (strict T-2 convention).
2. **Advisory assistant (Sprint 2):** Retrieval + templates with **no LLM in
   the generation path**. This is a deliberate safety choice: it makes prompt
   injection structurally impossible, citations exact, and every answer
   reproducible. 19/19 hardening tests pass.

We recommend this system for deployment as a *decision-support* tool (not a
sole authority): it gives calibrated next-day PM2.5 numbers, applies the
official hazardous rule (≥165 µg/m³) deterministically, and grounds every
piece of advice in dated, authoritative documents with current guidance
preferred over superseded versions.

## Disclosure
- The **reference forecast was not used**. All predictions are from the team
  model (`source: "team_model"`).
- No hidden labels were accessed; no data beyond the provided package was used.

## Limitations (read before trusting the system)
1. **Hazardous spikes are not predictable from the available data.**
   Hazardous days are 3% of history and typically arrive without warning in
   the sensor lags (median prior-day PM2.5: 103.6). Our alarm recall is
   therefore low. We chose an honest forecast over tuning the alarm to
   *look* sensitive. For operational use, pair this system with meteorological
   inputs (inversion/wind forecasts), which the provided weather file cannot
   supply at the needed granularity.
2. **Recursive horizon decay.** Targets after Oct 30 are predicted using
   earlier predictions as history. Validated stable over 10 steps
   (MAE 10.7), but treat Nov 6–7 as lower-confidence.
3. **Coverage is 15 Lahore areas, Oct 29–Nov 7.** The assistant refuses
   everything else rather than guessing — this is a feature, not a gap.

## If we had more time
- Ingest live meteorological forecasts (not just observed weather) as
  exogenous features; this is the most likely route to hazardous recall.
- Calibrate the alarm threshold jointly with the forecast once the true
  scoring weights are known.
- Expand Roman Urdu template coverage from the released question set.
