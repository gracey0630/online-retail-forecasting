# Project 2: Online Retail Forecasting — Plan v3 (AI-Assisted Scope)

> This document extends plan-v2.md. Sections 1–8 from v2 remain the baseline commitment.
> Everything in **Section 9** is *exploratory* — ideas to pursue if time permits, not requirements.
> With Claude Code handling boilerplate and debugging, the team can realistically attempt 2–3 of these.

---

## 9. OPTIONAL EXTENSIONS (AI-ASSISTED)

The core observation: with Claude Code, the bottleneck shifts from *writing code* to *running experiments and interpreting results*. Tasks that would take 2–3 days manually (library setup, data formatting, eval loops) take a few hours. This opens up scope that would otherwise be unrealistic for a one-month course project.

The items below are ordered by **confidence that they improve the final result**. Pick what fits.

---

### Idea A — Recompute clustering features at weekly granularity

**What**: The prior team computed all 50 clustering features (ACF lags 1/7/14/30, STL, ADI, CV²) on the *daily* time series. Since the project now targets weekly forecasts, the features should reflect weekly structure: ACF at lags 1/4/8/13/26 weeks, STL with period=52, ADI counted in weeks not days.

**Why it matters**: A product with strong day-of-week seasonality (e.g. ships only on Thursdays) appears "seasonal" in daily clustering but flat at weekly granularity — it gets the wrong model assigned. Recomputing at weekly granularity fixes this alignment.

**Also**: The 50-feature set has 8 pairs with correlation >0.90 (exact duplicates like `nonzero_rate` ~ `tsb_p`, `adi` ~ `croston_mean_interval`, `mean_daily_sales` ~ `total_sales`). K-Means double-weights correlated dimensions. Dropping these 8 before scaling gives cleaner cluster geometry and is easy to justify in the report.

**Expected effort with Claude Code**: 1 day (feature recomputation notebook + re-run K-Means).

**Risk**: New cluster assignments may shift some products into different clusters, requiring model re-routing. Run in a separate notebook first and compare assignment overlap before committing.

---

### Idea B — Split C1 into C1a (truly seasonal) and C1b (misclassified)

**What**: 156 of 473 C1 products have `stl_seasonality_strength < 0.5` on the daily series. C1 is labeled "Seasonal-Intermittent" but a third of it isn't actually seasonal — these products have higher `nonzero_rate` (0.175 vs 0.112) and higher mean sales, making them more like C0.

**Why it matters**: Prophet performs well only on genuinely seasonal series. Applying Prophet to low-STL C1 products will drag down C1's MAPE and make Prophet look worse than it is. Splitting them out allows correct model routing:
- **C1a** (STL ≥ 0.5): Prophet / seasonal model
- **C1b** (STL < 0.5): LGBM or N-BEATS, same treatment as C0

**Expected effort with Claude Code**: half a day. Just re-check STL on the weekly series (STL strength may change at weekly granularity), add a flag column, update model routing.

---

### Idea C — Add TFT back as primary model for C0 and C1a

**What**: Plan v2 recommended N-BEATS over TFT due to the short series length (~73 weekly training points) and library complexity. With Claude Code handling PyTorch Forecasting boilerplate and data formatting, TFT becomes viable again.

**Why**: TFT is what the professor explicitly asked for, and it has the strongest published evidence on retail transaction data (Lim et al. 2019). With static covariates (cluster_id, product_family) and time-varying regressors (holiday flags, cyclical week encoding), TFT can learn cross-series patterns that per-SKU models miss.

**Practical notes**:
- Use `hidden_size=32`, `attention_head_size=2`, max 50 epochs on CPU — feasible training time.
- Feed all C0 + C1a series jointly as a global model. The cluster_id static covariate lets the model condition on demand regime.
- If TFT underperforms Prophet/LGBM on the val set, keep it as an ablation in the report — it still counts as "we tried deep learning and compared it."

**Expected effort with Claude Code**: 1 day setup + overnight training run.

---

### Idea D — Ensemble / stacking on the validation set

**What**: Once Prophet, TFT (or N-BEATS), LGBM, and DeepAR are all producing forecasts, a simple linear stacking layer trained on the validation set takes ~2 hours. Fit a per-cluster weighted average where weights are learned by minimizing MAPE on val predictions.

**Why**: Competition forecasting (M5, Kaggle retail) consistently shows that ensembles beat any single model by 5–15% on MAPE, even when the individual models are similar in quality. This is especially true for intermittent demand where model errors are uncorrelated.

**Simplest version**: `forecast_final = w1 * prophet + w2 * tft + w3 * lgbm` where weights are fit by scipy.optimize or a Ridge regression on val residuals. No extra training, just a 10-line combiner.

**Expected effort with Claude Code**: 2–3 hours once all individual forecasts exist.

---

### Idea E — Prediction intervals for all models

**What**: Report 80% and 95% prediction intervals alongside point forecasts. The prior team reported only point MAPE — adding intervals is a meaningful statistical contribution they missed entirely.

**How**:
- Prophet: built-in `uncertainty_samples` parameter, free.
- TFT: use quantile loss (`[0.1, 0.5, 0.9]`) — one line change in PyTorch Forecasting.
- DeepAR: NegBinomial head outputs a distribution — sample from it.
- LGBM: use `objective='quantile'` and train two models at q=0.1 and q=0.9.

**Metric to report**: empirical coverage — "80% of actual values fell within our 80% interval." If coverage is close to nominal, the intervals are well-calibrated. This is easy to compute and looks strong in the report.

**Expected effort with Claude Code**: 1 day across all models.

---

### Idea F — Hyperparameter tuning with Optuna

**What**: A lightweight Optuna search over key hyperparameters for Prophet (changepoint_prior_scale, seasonality_prior_scale) and LGBM (num_leaves, learning_rate, min_child_samples). TFT has its own validation-based early stopping, so it's less critical there.

**Why**: The prior team used default or hand-tuned hyperparameters. A systematic search, even over a small grid (20–50 trials), is easy to justify as "optimization" in the rubric and often yields 5–10% MAPE improvement.

**Expected effort with Claude Code**: half a day per model. Optuna's API is simple; Claude handles the trial setup.

---

## 10. REVISED EFFORT ESTIMATE WITH AI TOOLING

| Task | v2 estimate (manual) | v3 estimate (with Claude Code) |
|------|---------------------|-------------------------------|
| Data wrangling / feature engineering | 1–2 days | 2–3 hours |
| Library setup + debugging (TFT, GluonTS) | 2–3 days | 3–5 hours |
| Model training scaffold + eval loop | 1 day | 1–2 hours |
| Evaluation notebook + comparison table | 1 day | 1–2 hours |
| Streamlit agent updates | 1–2 days | half a day |

**Recommended v3 scope for 3 people / 1 month**:
- **Committed (from v2)**: weekly pipeline, holiday features, Prophet, N-BEATS or TFT, DeepAR, SHAP, C4 SWLY, evaluation notebook, agent update
- **Add from v3**: Idea B (C1 split) + Idea D (ensemble) + Idea E (prediction intervals) — these three have the best effort-to-impact ratio
- **If time allows**: Idea A (weekly re-clustering) and Idea C (TFT back in) — these are the most academically ambitious and would clearly distinguish the project

**What the presentation story becomes**:
> "We improved the prior team's work along four dimensions: (1) weekly granularity reducing baseline MAPE by ~X%; (2) corrected C1 cluster purity; (3) three new model families with ensembling; (4) prediction intervals the prior team lacked entirely."

---

## 11. WHAT NOT TO OVER-ENGINEER

Even with AI tooling, some things are not worth the time:

- **Re-tuning K** (number of clusters): K=5 profiles are interpretable and stable. Don't touch it.
- **Adding external data** (weather, promotions, Google Trends): interesting but requires data sourcing and alignment — not worth it in a course context.
- **Online learning / streaming updates**: architecturally interesting, zero payoff for a static dataset evaluation.
- **Switching from Streamlit to a different frontend**: the Streamlit app is solid. Update routing, don't rewrite.
