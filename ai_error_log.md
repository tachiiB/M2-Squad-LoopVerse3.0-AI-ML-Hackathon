# AI Error Log — Lahore Smog Intelligence Challenge

Three genuine mistakes made by the AI assistant during this build, how they were
caught, and what changed. (Requirement: SPEC.md §6, "AI tools are allowed.
Responsibility stays with your team.")

---

## 1. Wrong shell-escape advice for the Kaggle console

- **Context and wrong output:** While verifying Kaggle GPU access, the assistant
  told the user to run `nvidia-smi` *without* the `!` prefix in the Kaggle
  console, assuming it was a raw terminal. It is actually an IPython console,
  so the bare command was parsed as Python and raised `NameError: name 'nvidia'
  is not defined`, and the session stopped.
- **Risk:** Wasted setup time and user confusion right before the event; a
  wrong mental model of the execution environment could have led to further
  bad commands during the timed build.
- **Detection:** The user pasted a screenshot showing the `NameError` and
  `Session stopped.` The assistant re-examined the assumption against the
  evidence instead of defending it.
- **Correction and verification:** Owned the mistake, corrected course: set the
  accelerator under Session options, then ran `!nvidia-smi` in a normal code
  cell. Output showed two Tesla T4 GPUs (CUDA 13.0), confirming GPU access.
  Lesson recorded: verify the execution environment (cell vs. console vs.
  terminal) before giving shell advice.

## 2. `pivot_table` silently dropped the faulty-sensor rows

- **Context and wrong output:** The cleaning pipeline used
  `pd.pivot_table(...)` with default arguments to merge the two sensor
  networks. The resulting panel had **1797 rows instead of the expected 1800**
  (15 sensors × 120 days). The 3 missing rows were S07 on 2026-08-25..27 —
  exactly the stuck-sensor interval the pipeline had just flagged as faults.
- **Risk:** Silently dropping rows breaks the regular daily time series the
  lag/rolling features depend on, and hides data-quality events instead of
  documenting them. A submission built on a silently truncated panel would be
  unreproducible and untrustworthy.
- **Detection:** A row-count assertion (`15 × 120 = 1800`) failed. Inspection
  showed `pivot_table`'s default `dropna=True` discards groups whose values
  are all-NaN — precisely the rows where both networks were correctly marked
  as faulty.
- **Correction and verification:** Passed `dropna=False` so fault intervals
  survive as explicit NaNs, then imputed them with past-only forward-fill
  (no future leakage). Re-ran the assertion: 1800 rows, 3 explicit NaNs, and
  the S07 stuck interval is now visible and documented in the panel instead of
  silently gone.

## 3. Sample-weighting for hazardous recall hurt the model and taught nothing

- **Context and wrong output:** Hazardous days (PM2.5 ≥ 165) are only 3% of
  training data and every lag-only model had 0.00 hazardous recall. The
  assistant tried upweighting high-PM2.5 samples 5× in LightGBM. Result: MAE
  worsened (16.37 vs 14.88 unweighted) and recall stayed 0.00 — the model just
  predicted ≥165 on 6 wrong days.
- **Risk:** Chasing a rare-class metric with a blunt reweighting degrades the
  primary forecast (PM2.5 MAE) while creating false alarms — the exact
  "misleading score" trap the handbook warns about. Shipping this would have
  traded real accuracy for the illusion of action.
- **Detection:** Walk-forward validation (chronological, 4 × 7-day folds)
  compared weighted vs. unweighted on identical folds. A follow-up diagnostic
  showed *why* it failed: on hazardous days the median previous-day PM2.5 was
  only 103.6 (38 of 47 had lag_1 < 140) — the spikes are driven by exogenous
  factors (weather), not by anything in the lag features, so no reweighting
  of lag features can learn them.
- **Correction and verification:** Abandoned sample-weighting. Kept the
  honest baseline (quantile LightGBM α=0.6, walk-forward MAE 13.18) and
  recorded the diagnostic conclusion: hazardous recall requires weather
  features (wind/humidity stagnation signals), which were added to the plan
  as the top priority once `weather_history.csv` arrived. Lesson: when a
  metric won't move, diagnose the *information* gap before tuning the *loss*.

---

## 4. Misread the sensor metadata: assumed one network per sensor

- **Context and wrong output:** After seeing `sensor_metadata.csv` map odd
  sensors to batch_1 and even sensors to batch_2, the assistant briefly
  concluded each sensor should use only its "native" batch file, and that the
  other file's readings were lossy conversions.
- **Risk:** Dropping half the measurements per sensor would have discarded
  genuine independent observations and weakened the daily estimates — and the
  S14 stuck-sensor fault (visible in only one network) would have been missed
  or mishandled.
- **Detection:** Direct data check: S14 was stuck at AQI 153 for 3 days in
  batch_2 while batch_1 showed varying values (59.8, 59.6, 58.6) on the same
  dates. A mere conversion copy could not diverge like that — the networks
  are independent instruments. The per-network flat-line detector plus
  averaging was already the correct design.
- **Correction and verification:** Kept the merge-both-networks pipeline;
  documented that the metadata's `batch` column describes file properties
  (timezone/unit), not an exclusive assignment. Lesson: test structural
  assumptions against the data before refactoring a working pipeline.

## 5. Chased a −0.79 weather correlation that was pure confounding

- **Context and wrong output:** Temperature correlated −0.79 with next-day
  PM2.5 at city level. The assistant added 15 weather features expecting a
  breakthrough, especially for hazardous recall. Walk-forward MAE got
  *worse* (Ridge 13.54 → 14.59) and hazardous recall stayed 0.00.
- **Risk:** Shipping 15 noise features would have degraded the submission and
  wasted the limited build window tuning a dead end.
- **Detection:** Ridge coefficients told the story: temp's partial
  coefficient sign-flipped to +0.70 (vs −0.79 raw) — classic multicollinearity
  with the seasonal trend the model already captured via day-of-year/month.
  A weather-only model managed just MAE 19.4 vs 13.7 sensor-only.
- **Correction and verification:** Removed all weather features; champion
  stayed sensor-only (ensemble MAE 12.92). Recorded the finding honestly in
  the README instead of hiding a negative result. Lesson: a strong raw
  correlation is not a causal feature — check the *partial* effect before
  committing.
