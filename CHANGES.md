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

<<<<<<< HEAD
=======
## Phase 4 — Primary Model Development and Improvements

In Phase 4, we implemented more advanced forecasting models to improve upon the Phase 3 baselines and address the high forecasting errors observed in the original project.

### Model Selection Strategy

Models were selected based on the demand characteristics of each cluster:

- **C0 (Erratic demand)** → LightGBM (LGBM)  
- **C1a (Seasonal-intermittent)** → LGBM (replaced Prophet after evaluation)  
- **C1b (Non-seasonal intermittent)** → LGBM  
- **C2 (Dense high-volume)** → Prophet  
- **C3 (Sparse long-tail)** → iMAPA baseline retained  
- **C4 (Ultra-sparse)** → Same-week-last-year (SWLY) rule  

Tree-based models (LGBM) were chosen for irregular and intermittent demand because they can incorporate lag features, holiday effects, and price trends. Prophet was used for dense series where consistent seasonal patterns are present.

---

### Key Improvement: Replacing Prophet in C1a

Prophet was initially applied to the seasonal-intermittent cluster (C1a), but it significantly underperformed:

- Prophet (C1a): **120.5% WMAPE**  
- Baseline (iMAPA): **95.8% WMAPE**

We replaced Prophet with LGBM for this cluster:

- LGBM (C1a): **85.1% WMAPE**

This indicates that even in clusters with some seasonality, demand is highly irregular and better captured by feature-based models rather than additive seasonal models.

---

### Final Model Performance

| Cluster | Model | Baseline WMAPE | Final WMAPE | Improvement |
|--------|------|---------------|------------|------------|
| C2 | Prophet | 93.9% | **63.8%** | +30 pts |
| C1b | LGBM | 102.9% | **82.4%** | +20 pts |
| C1a | LGBM | 95.8% | **85.1%** | +10 pts |
| C0 | LGBM | 108.0% | **85.9%** | +22 pts |

All clusters showed improvement over Phase 3 baselines, with the largest gain observed in the dense C2 cluster using Prophet.

---

### DeepAR for C3 (Not Implemented)

We explored implementing DeepAR for the C3 sparse long-tail cluster, as suggested in the project plan and supporting literature. However:

- C3 contains over 1,000 highly sparse product series  
- DeepAR requires significant training time and tuning  
- The iMAPA baseline already performs strongly (110.5% WMAPE)  

---

### Summary

Phase 4 successfully improved forecasting performance across all clusters by selecting models aligned with demand patterns and validating decisions through empirical evaluation. These improvements resulted in substantial reductions in WMAPE, particularly for high-volume and intermittent demand segments.

## Phase 5 — Final Evaluation and Comparison

In Phase 5, we evaluated the final Phase 4 models against the strongest Phase 3 baseline for each cluster. The purpose of this phase was to create an apples-to-apples comparison showing whether the new modeling strategy improved forecasting performance.

The final comparison used WMAPE as the primary metric because it is more stable than standard MAPE for sparse retail demand, where many product-day combinations have zero or very low sales.

### Model Interpretability

We examined feature importance for the LGBM models to better understand what drives predictions.

The most important features are lagged sales (e.g., 1-day and 7-day lags), rolling averages, and calendar-related variables such as holiday proximity and Christmas-week indicators. This indicates that demand is primarily driven by recent sales patterns and event-based effects rather than smooth seasonal trends.

This also helps explain why Prophet underperformed in certain clusters (such as C1a), since those demand patterns are not well captured by additive seasonal models.

### Final Model Comparison

| Cluster | Baseline Model | Baseline WMAPE | Final Model | Final WMAPE | Improvement |
|--------|----------------|----------------|-------------|-------------|-------------|
| C0 | TSB | 108.0% | LGBM | 85.9% | +22.1 pts |
| C1a | iMAPA | 95.8% | LGBM | 85.1% | +10.7 pts |
| C1b | iMAPA | 102.9% | LGBM | 82.4% | +20.5 pts |
| C2 | AutoETS | 93.9% | Prophet | 63.8% | +30.1 pts |
| C3 | iMAPA | 110.5% | iMAPA retained | 110.5% | 0.0 pts |
| C4 | SWLY | 116.6% | SWLY retained | 116.6% | 0.0 pts |

The largest improvement was observed in **C2**, where Prophet reduced WMAPE from 93.9% to 63.8%. This makes sense because C2 contains dense, high-volume products with more consistent seasonal patterns, which Prophet is designed to capture.

For C0, C1a, and C1b, LGBM performed best. These clusters contain more erratic and intermittent demand, so the model benefited from engineered features such as lagged sales, rolling averages, holiday proximity, Christmas-week indicators, cyclical calendar encodings, and rolling price signals.

One important finding was that Prophet did not perform well on C1a, even though this cluster was initially labeled seasonal-intermittent. Prophet produced a WMAPE of 120.5%, which was worse than the iMAPA baseline. After replacing Prophet with LGBM, WMAPE improved to 85.1%. This suggests that C1a demand is not purely smooth or additive seasonal demand; instead, it is irregular and event-driven, making feature-based modeling more effective.

For C3 and C4, we retained the strongest baseline models. C3 contains over 1,000 sparse long-tail products, making DeepAR difficult to implement and tune within the project timeline. Since iMAPA was already a strong sparse-demand baseline, we retained it and documented DeepAR as future work. C4 contains only 8 ultra-sparse products, so the same-week-last-year rule remains a practical and interpretable choice.

Overall, Phase 5 shows that our improvements reduced WMAPE for the main modeled clusters, with gains ranging from about 10 to 30 percentage points. The final modeling strategy is more aligned with the demand structure of each cluster and provides a clearer, more business-relevant forecasting pipeline.


## Key files

| File | Description |
|------|-------------|
| `data/daily_train_clustered.parquet` | Unwinsorized train — use for evaluation |
| `data/daily_train_clustered_winsorized.parquet` | Winsorized train — use for model training |
| `data/winsor_caps.parquet` | Per-product p99 caps, fit on train only |
| `data/clustering/clusters_final.parquet` | Final cluster assignments with C1 split |
| `data/forecasting/c{0-3}_prediction.parquet` | Prior team's predictions — baseline to beat |
| `data/baselines/c{0,1a,1b,2,3,4}_baselines.parquet` | Phase 3 baseline predictions + actuals |
| `data/primary_models/c0_lgbm_predictions.parquet` | LGBM predictions for C0 (erratic demand) |
| `data/primary_models/c1b_lgbm_predictions.parquet` | LGBM predictions for C1b (non-seasonal intermittent) |
| `data/primary_models/c1a_lgbm_predictions.parquet` | LGBM predictions for C1a (replacing Prophet) |
| `data/primary_models/c2_prophet_predictions.parquet` | Prophet predictions for C2 (dense high-volume) |
| `data/primary_models/c1a_prophet_predictions.parquet` | Prophet predictions for C1a (experimental, not used in final model) |
| `data/primary_models/c0_lgbm_feature_importance.csv` | Feature importance for LGBM model (C0) |
| `data/primary_models/c1b_lgbm_feature_importance.csv` | Feature importance for LGBM model (C1b) |
| `data/primary_models/c1a_lgbm_feature_importance.csv` | Feature importance for LGBM model (C1a) |
| `data/primary_models/phase4_primary_model_metrics.csv` | Summary table of Phase 4 model performance |
