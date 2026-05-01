# Data & Clustering Changes — Phase 1 & 2 Summary

## What changed and why

### 1. Rebuilt the daily data pipeline from scratch (notebook 02)

The prior team's `train_daily.parquet` had three problems we fixed:

**Net-negative daily sales clipped to 0.**
The prior team dropped `Quantity < 0` at the transaction row level, but returns and sales for the same product family on the same day are aggregated together — so a family with more returns than purchases on a given day still had a negative `total_sales`. We clip these to 0 after aggregation. This affected 14,525 day-product pairs across 1,088 products.

**Winsorization at 99th percentile per product.**
886 products have p99 > 10× their median daily sales — mostly Christmas spikes. Untreated, these dominate training loss and cause systematic underprediction on normal days. Caps are fit on train only and saved to `winsor_caps.parquet`. Raw (unwinsorized) splits are kept separately for MAPE evaluation against true values.

**Holiday and calendar features added.**
The prior team had no holiday encoding at all. We added:
- `days_to_holiday` — signed distance to the nearest UK bank holiday (±30 cap). UK bank holidays have zero transactions in this dataset (shops close), so the holiday day itself doesn't appear in the data. The days *around* holidays drive demand surges, which this feature captures.
- `is_christmas_week` — 1 if ISO week 51 or 52.
- Day-of-week, week-of-year, month cyclical encodings (sin/cos) to prevent ordinal leakage at year boundaries.
- 28-day rolling unit price per product — captures promotional discounting signals.

**Split boundaries match the prior team exactly** (train ≤ 2011-06-08, test > 2011-06-08) so our MAPE numbers are directly comparable. We carved a validation set from the last 15% of training days (≥ 2011-03-15) for hyperparameter tuning.

---

### 2. Improved cluster routing — C1 split (notebook 03)

The prior team routed all 473 C1 ("Seasonal-Intermittent") products to Prophet. But 156 of them have `stl_seasonality_strength < 0.5` on the daily series — meaning they have no meaningful day-of-week seasonality. Applying Prophet to non-seasonal series inflates C1's MAPE and makes Prophet look worse than it is.

We split C1 into:
- **C1a** (317 products, daily STL ≥ 0.5) → Prophet
- **C1b** (156 products, daily STL < 0.5) → LGBM (same treatment as C0)

We used **daily STL** (period=7, day-of-week seasonality) rather than weekly STL because C1's defining seasonality pattern is day-of-week — gift and seasonal products that sell heavily on weekdays. Weekly STL (period=52, annual seasonality) is also unreliable with only ~73 training weeks (1.4 annual cycles).

---

### 3. Re-clustering experiment — kept original (notebook 04)

The prior team's 50-feature set has 7 pairs with r ≥ 0.999 (exact duplicates like `nonzero_rate` ~ `tsb_p`, `adi` ~ `croston_mean_interval`). K-Means double-weights correlated dimensions, which biases cluster geometry.

We re-ran K-Means after dropping the 7 redundant features (43-feature clean set) and compared results:
- **ARI = 0.54** against original assignments — substantial reshuffling, not stable
- **Silhouette dropped by 0.05** — geometry got worse, not better

**Conclusion: keep original assignments.** The redundant features are all intermittency metrics (nonzero_rate, ADI, TSB variants) that are genuinely the strongest clustering signal in this dataset. Removing them hurts more than the double-weighting.

---

## Final model routing

| Cluster | n products | Model |
|---------|-----------|-------|
| C0 Erratic | 515 | LGBM |
| C1a Seasonal (STL ≥ 0.5) | 317 | Prophet |
| C1b Non-Seasonal (STL < 0.5) | 156 | LGBM |
| C2 Dense HiVol | 10 | Prophet + global LGBM |
| C3 Sparse Long-Tail | 1017 | DeepAR |
| C4 Ultra-Sparse | 8 | SWLY rule |

---

### 4. Baseline models — Phase 3 (notebook 05)

Classical baselines fitted on train+val history, evaluated on test. Serves as the comparison table floor in Phase 5.

| Cluster | Baselines | Best WMAPE |
|---------|-----------|-----------|
| C0 | Naive-7, TSB | TSB: 108.0% |
| C1a | Naive-7, iMAPA | iMAPA: 95.8% |
| C1b | Naive-7, iMAPA | iMAPA: 102.9% |
| C2 | Naive-7, AutoETS | AutoETS: 93.9% |
| C3 | Naive-7, TSB, iMAPA | iMAPA: 110.5% |
| C4 | SWLY (364-day offset) | 116.6% |

**Key finding**: iMAPA beats the prior team's production ZINB (126.1% WMAPE) on C1 and their TS-HGB (150.1%) on C3 using no features at all. Our Phase 4 models must beat iMAPA, not just the prior team's numbers.

Metrics reported: WMAPE, ε-MAPE (eps=1.0), MAPE (eps=1e-8), NZ-MAPE (sale days only), zero_rate% (fraction of test days with zero actual sales — context for why MAPE is large).

---

## Final model routing

| Cluster | n products | Model |
|---------|-----------|-------|
| C0 Erratic | 515 | LGBM |
| C1a Seasonal (STL ≥ 0.5) | 317 | Prophet |
| C1b Non-Seasonal (STL < 0.5) | 156 | LGBM |
| C2 Dense HiVol | 10 | Prophet + global LGBM |
| C3 Sparse Long-Tail | 1017 | DeepAR |
| C4 Ultra-Sparse | 8 | SWLY rule |

## Key files

| File | Description |
|------|-------------|
| `data/daily_train_clustered.parquet` | Unwinsorized train — use for evaluation |
| `data/daily_train_clustered_winsorized.parquet` | Winsorized train — use for model training |
| `data/winsor_caps.parquet` | Per-product p99 caps, fit on train only |
| `data/clustering/clusters_final.parquet` | Final cluster assignments with C1 split |
| `data/forecasting/c{0-3}_prediction.parquet` | Prior team's predictions — baseline to beat |
| `data/baselines/c{0,1a,1b,2,3,4}_baselines.parquet` | Phase 3 baseline predictions + actuals |
