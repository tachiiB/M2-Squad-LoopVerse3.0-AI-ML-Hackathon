# Lahore Smog Intelligence Challenge — Team Solution

## What this is
Two-sprint system: (1) next-day PM2.5 forecasting for 15 Lahore sensors,
(2) a grounded RAG advisory assistant that cites official documents and
calls the forecast tool.

## Reproduce everything
```bash
pip install pandas numpy scikit-learn lightgbm
python predict.py        # -> predictions.csv (Sprint 1 output, ~10 s)
python test_sprint2.py   # -> 19/19 hardening tests (Sprint 2)
```

## Sprint 1: forecast
**Pipeline** (`preprocess.py`): UTC→PKT date alignment, US-EPA AQI→PM2.5
inversion, `-999`→NaN, ≥3-day flat-line faults→NaN, per-network merge.

**Model** (`model.py`): 30 features (PM2.5 lags 1/2/3/7/14, rolling
3/7/14-day stats, 7-day trend, day-of-week/month/day-of-year, sensor
one-hots). Strict T-2 convention: features for target T use PM2.5 dates
≤ T−2 only — zero leakage risk by construction.
Champion = 0.5·Ridge(α=1.0) + 0.5·LightGBM-quantile(α=0.6):
walk-forward MAE **12.92** (4×7-day chronological folds).

**Holdout** (`predict.py`): origins run past the data end, so targets from
Oct 31 on use recursive forecasting (predictions fed back as history).
10-step recursive diagnostic MAE 10.7 — no blow-up.
`hazardous = 1 iff predicted PM2.5 ≥ 165` (SPEC rule, deterministic).

**Key findings** (see `ai_error_log.md` for the full story):
- The two sensor batches are *independent* networks (proven by S14:
  batch_2 stuck 3 days while batch_1 varied) — averaged, not picked.
- Weather looked predictive (temp −0.79) but it was pure seasonal
  confounding; day-to-day weather adds nothing once lags+seasonality are in.
- Hazardous spikes (3% of days) are unpredictable from available features —
  median prior-day PM2.5 on hazardous days was only 103.6. We report honest
  forecasts rather than inventing alarms.

## Sprint 2: advisory assistant
**Design choice: no LLM in the generation path.** Answers are assembled from
TF-IDF-retrieved passages + forecast output via templates. Prompt injection
is *structurally* impossible — retrieved text is data, never instructions
(HTML comments stripped at ingest; DOC-07's embedded instruction neutralized).

- `forecast_tool.py` — `forecast(location, target_date)` per SPEC; validates
  against 15 areas/sensors and the Oct 29–Nov 7 window; never guesses.
- `docstore.py` — frontmatter metadata, chunking (58 chunks).
- `vector_db.py` — **VectorDB**: dense vector storage + cosine similarity
  search. Chunks are vectorized (TF-IDF → TruncatedSVD → 57-dim dense
  vectors), stored in a normalized numpy matrix, and queried by embedding
  the question into the same space. Ranked by similarity × currency ×
  authority (current docs outrank superseded: DOC-03 beats DOC-02).
- `ask.py` — `ask(question)` → `{question_id, answer, sources,
  forecast_called}`; routes 8 question categories; Roman Urdu detection +
  normalization; refusals for out-of-scope cities/dates.
- `test_sprint2.py` — 19/19 pass: forecast wiring, citations, refusal,
  old-vs-new, injection resistance, Roman Urdu, schema.

## Files
| File | Purpose |
|---|---|
| `predictions.csv` | Sprint 1 submission (150 rows) |
| `preprocess.py` / `model.py` / `predict.py` | Sprint 1 pipeline |
| `forecast_tool.py` / `docstore.py` / `ask.py` | Sprint 2 assistant |
| `test_sprint2.py` | Sprint 2 hardening tests |
| `ai_error_log.md` | Required AI error log (5 entries) |
| `recommendation.md` | Required recommendation |

## Limitations
- Hazardous-day recall is ~0: spikes are not predictable from the given
  features (see findings above). We chose honest MAE over alarm theater.
- Recursive forecasts degrade with horizon; Nov 6–7 predictions rely partly
  on earlier predictions.
- Reference forecast was **not** used; all numbers are from the team model.
