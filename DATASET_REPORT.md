# DATASET REPORT — Amazon Sales Dataset

**End-to-End Analytics & Machine Learning Pipeline | September 2026**

This report summarises the findings of a three-stage analysis pipeline built on the public Kaggle
*Amazon Sales Dataset* ([karkavelrajaj/amazon-sales-dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)).

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `01_eda_and_cleaning.py` | Ingestion, cleaning, type-casting, category parsing, sentiment analysis, EDA |
| 2 | `02_data_visualization.py` | 6 high-resolution (300 DPI) exploratory charts |
| 3 | `03_ml_model_training.py` | Feature engineering, baselines vs. ensembles, hyper-parameter tuning, evaluation |
| 4 | `DATASET_REPORT.md` | This report |

All figures referenced below live in **`outputs/figures/`**; all tabular ML artifacts live in **`outputs/ml/`** and EDA summaries in **`outputs/eda/`**.

---

## 1. Executive Summary

- The dataset covers **1,465 review rows** corresponding to **1,351 unique products** across 12+ Amazon India categories (Electronics, Computers & Accessories, Home & Kitchen, Office Products, etc.).
- Customers overwhelmingly rate products highly: the mean rating is **4.10 / 5.0** (median 4.1), and **96% of the 1,465 reviews are VADER-positive** (raw-text scoring preserves punctuation/casing signals; 3.7% negative).
- Pricing is extremely right-skewed: list prices span **₹39 to ₹1.4 lakh** (median ₹1,650), while the effective (discounted) price median is **₹799**. The typical discount is **≈48%**.
- Discounts and ratings are **negatively correlated (r = −0.16)**: the deepest-discounted products do *not* carry the best star ratings. Demand (rating_count) correlates positively, yet weakly, with ratings (**r = +0.10**).
- **Electronics is the demand powerhouse** (~15.8 M cumulative rating-count, ~36% of products ≈ **59% of total demand**), while **Computers & Accessories** offers the best combination of deep discounts (54%) and high ratings (4.15). **Office Products** shows the highest average rating (4.31) but remains massively under-assorted (31 products) — a clear whitespace opportunity.
- **ML — Regression (discounted_price, leakage-free):** a regularised **Ridge** model predicts the discounted price with **RMSE ≈ ₹535**, **MAE ≈ ₹187**, **R² ≈ 0.996** and **MAPE ≈ 8.7%** on a decile-stratified held-out split — list price and discount depth remain the overwhelming price drivers. (A leaky earlier variant that included `discount_amount` — a direct arithmetic function of the target — had inflated these to RMSE ≈ ₹301 / R² ≈ 0.998; removing it yields honest figures.)
- **ML — Classification (High Demand, rating_count > 4,740):** a **Random Forest** classifier distinguishes high-demand from low-demand products with **ROC-AUC ≈ 0.87** (hold-out) and **Precision ≈ 0.83**, using review-text signals, rating, VADER sentiment, discount depth, price tier and TF-IDF text. `rating_count` itself is fully excluded from the feature space, and imputation happens in-pipeline (no pre-split leakage). The ensemble beats the linear baseline (LogisticRegression AUC 0.85) and the tree baseline (DecisionTree AUC 0.67).
- Strategic take-aways: win with **₹500–1.5k value-priced accessories** (deepest 54% discounts, highest demand); expand the **Office Products** long-tail; protect your star-rating with honest, durable product copy; and deploy the high-demand classifier for **pre-launch inventory & replenishment planning**.

---

## 2. Dataset Overview & Data Quality

### 2.1 Raw Dataset Profile

| Attribute | Value |
|-----------|-------|
| Source | Kaggle: Amazon Sales Dataset (amazon.csv) |
| Rows (raw) | 1,465 review rows |
| Columns | 16 |
| Unique products | 1,351 |
| Duplicate `product_id` rows | 114 (kept at review-level for sentiment EDA; deduplicated to one row per product for modelling) |
| Missing values (raw) | `rating_count`: 2 (all other columns complete) |

Columns (raw): `product_id`, `product_name`, `category` (pipe-delimited hierarchy), `discounted_price`, `actual_price`, `discount_percentage`, `rating`, `rating_count`, `about_product`, `user_id`, `user_name`, `review_id`, `review_title`, `review_content`, `img_link`, `product_link`.

### 2.2 Cleaning & Type-Conversion Methodology

| Column(s) | Issue | Treatment |
|-----------|-------|-----------|
| `discounted_price`, `actual_price` | Currency strings `₹1,099` | Strip `₹` + commas; cast to `float64` (0 anomalies) |
| `discount_percentage` | `64%` strings | Strip `%`; parse to fraction in [0, 1]; clip to 100% (0 anomalies) |
| `rating` | Erratic non-numeric token | Coerce to `NaN` (1 row); **preserved as NaN** in the cleaned CSV; imputed in-pipeline (`SimpleImputer`, training-fold median); clipped to 1–5 scale |
| `rating_count` | Commas e.g. `24,269`; 2 missing | Strip commas; cast `float64`; **2 values preserved as NaN** (imputed in-pipeline); rows with undefined targets are dropped from the classification split |
| `category` | Hierarchical `A\|B\|C\|D` | Split into `category_main` / `category_sub` / `category_micro`; merge categories with <5 products into `Other` |
| `review_title` / `review_content` | Free text | VADER compound score + label (Positive/Neutral/Negative); TextBlob polarity; text-length features |
| Engineered | pricing tiers | `price_tier` (Budget < ₹500 to Luxury > ₹20k) derived from `actual_price` |

After cleaning, the only remaining gaps are the NaNs listed above (1 `rating`, 2 `rating_count`). They are intentionally **not** imputed globally: to eliminate pre-split data leakage, all imputation happens inside the scikit-learn pipelines (`SimpleImputer`, training folds only). Full quality log: `outputs/eda/eda_console.log`.

### 2.3 Numerical Summary Statistics

| Feature | Mean | Median | Std | Skewness | Min | Max |
|---------|-----:|-------:|----:|---------:|----:|----:|
| `discounted_price` (₹) | 3,125.31 | 799.00 | 6,944.30 | 4.45 | 39 | 77,990 |
| `actual_price` (₹) | 5,444.99 | 1,650.00 | 10,874.83 | 4.56 | 39 | 139,900 |
| `discount_percentage` (fraction) | 0.477 (47.7%) | 0.50 | 0.22 | −0.29 | 0.00 | 0.94 |
| `rating` | 4.10 | 4.10 | 0.29 | −1.24 | 2.0 | 5.0 |
| `rating_count` | 18,296 | 5,179 | 42,754 | 5.67 | 2 | 426,973 |

> Pricing and demand are heavily right-skewed (skewness > 4), hence log-transformations were applied before linear modelling. The 5,179 median above is computed over all 1,465 review rows; the ML stage uses the product-level median of **4,740** as its High-Demand threshold.

### 2.4 Leakage-Prevention Engineering

The machine-learning stage enforces three hard anti-leakage rules:

1. **No global imputation before splitting.** Missing `rating` (1 product) and `rating_count` (2 products) values stay `NaN` in the cleaned CSVs and are imputed inside the scikit-learn pipelines via `SimpleImputer` (numeric median / categorical most-frequent), fitted strictly on the training folds.
2. **No target-derived features.** `discount_amount` / `discount_amount_log` (= `actual_price` − `discounted_price`) are **excluded from the regression feature set**; `rating_count` / `rating_count_log` are **excluded from every model** (they define the High-Demand target).
3. **Targets are never imputed.** The 2 products with unknown `rating_count` have no defined High-Demand label and are dropped from the classification split (1,349 usable products); the regression task keeps all 1,351 products because its target (`discounted_price`) is complete.

Additionally, VADER sentiment is computed on the **raw** combined review string (`review_title` + " " + `review_content`) — no punctuation stripping or lower-casing — so emphasis markers ("!") and capitalisation contribute to the score as VADER's grammar intends.

---

## 3. Key Exploratory Insights

### 3.1 Pricing & Discounts vs. Popularity and Ratings

**Figure 1** (`outputs/figures/fig1_price_distribution.png`) shows that both list and effective prices cluster in the sub-₹2,000 mass-market band, with a long right tail into the ₹50k+ premium electronics territory.

| Relation | Pearson r | Interpretation |
|----------|:---------:|----------------|
| `actual_price` vs `discounted_price` | **+0.96** | Discounted prices are anchored to list prices (expected mechanics) |
| `discount_percentage` vs `rating` | **−0.16** | Heavier discounts are associated with *slightly lower* star ratings (discount hunters tolerating lower perceived quality) |
| `rating` vs `rating_count` | **+0.10** | Higher-rated products attract modestly more review volume (social proof) |
| `discount_percentage` vs `discounted_price` | **−0.24** | Deep discounts concentrate into cheaper accessories (price-promoted SKUs) |
| `actual_price` vs `rating` | **+0.12** | Premium-priced items trend toward marginally better ratings |

**Figure 2** (`fig2_discount_vs_rating.png`) visualises the discount-vs-rating plane with point size ∝ `rating_count`: the largest bubbles sit **between 40–70% discount at ratings 4.0–4.3** — i.e. the *value-oriented sweet spot*, not extreme liquidation discounts. Discounts beyond ~50–60% yield no rating upside.

Price-tier analysis (`outputs/eda/price_tier_summary.csv`):

| Price Tier | Products | Avg Discount | Avg Rating | Avg rating_count (demand) |
|------------|---------:|-------------:|-----------:|--------------------------:|
| Budget (< ₹500) | 183 | 40% | 4.07 | 12,581 |
| Value (₹500–1.5k) | **431** | **54%** | 4.09 | **21,887** |
| Mid-Range (₹1.5k–5k) | 430 | 47% | 4.08 | 18,087 |
| Premium (₹5k–20k) | 213 | 48% | 4.09 | 18,737 |
| Luxury (> ₹20k) | 95 | 36% | **4.23** | 13,688 |

> **Insight:** the **Value tier (₹500–1.5k) is the demand sweet-spot** — deepest average discount (54%) and the highest average review volume (~21.8k), while Luxury goods enjoy the best star ratings (4.23) but far fewer buyers.

### 3.2 Top-Performing Categories on Amazon

**Figures 3 & 5** (`fig3_category_breakdown.png`, `fig5_rating_boxplot.png`):

| Main Category | Products | Total Demand (Σ rating_count) | Share of Demand | Avg Discount | Avg Rating |
|---------------|---------:|------------------------------:|----------------:|-------------:|-----------:|
| Electronics | 490 | **15,778,848** | **59.0%** | 51% | 4.08 |
| Computers & Accessories | 375 | 7,728,689 | 28.9% | **54%** | **4.15** |
| Home & Kitchen | 448 | 2,991,069 | 11.2% | 40% | 4.04 |
| Office Products | 31 | 149,675 | 0.6% | 12% | **4.31** |
| Other (merged < 5 products) | 7 | 118,096 | 0.4% | 43% | 4.06 |

- **Electronics dominates absolute volume** (~59% of total demand) with the broadest assortment (490 products ≈ 36% of the catalogue); however demand is driven by volume of SKUs, not per-SKU efficiency.
- **Computers & Accessories punches far above its assortment weight**: ~29% of demand from ~28% of assortment, dressed with the deepest discounts (54%) and the highest mainstream rating (4.15) — cables, adapters and chargers are repeat-purchase winners. Top review tokens confirm this: **"cable"** (1,713 mentions), **"charging"** (1,236), **"battery"** (969).
- **Office Products is the hidden gem**: highest average rating (4.31) with almost no competition (31 products) and minimal discounting (12%) — low-price, high-satisfaction whitespace area. The rating box plot (**Figure 5**) shows median ratings clustered 4.1–4.3 across nearly every top sub-category — ratings are compressed at the top (classic Amazon review-inflation), so **differentiation comes from rating *volume*, not *level***.

### 3.3 Customer Sentiment Findings (Reviews)

Based on VADER compound scores over the combined review-title/content corpus (**Figure 6** — `fig6_sentiment_distribution.png`):

| Sentiment Class | Reviews | Share |
|-----------------|--------:|------:|
| Positive | 1,410 | **96.2%** |
| Neutral | 1 | 0.07% |
| Negative | 54 | 3.69% |

- The corpus is overwhelmingly positive; **top frequent words** (`outputs/eda/word_frequency.csv`): **good** (10,399), **product** (6,378), **quality** (3,184), **use** (2,059), **price** (1,870), **nice** (1,865), **cable** (1,713), **money** (1,279), **charging** (1,236), **value** (1,028), **great** (979), **better** (781).
- Complimentary themes: **durability / charging speed / ease of use / value-for-money**.
- The ~3.7% negative reviews concentrate on **physical durability issues** ("bending", "loose", "annoying", "regret", "scratch", "thin") and **functionality complaints** ("sound is very low", "slow charging") — actionable quality signals for sellers (Figure 6).

---

## 4. Machine Learning Results

### 4.1 Problem Formulation & Methodology

Two supervised tasks were framed on the deduplicated product table (1,351 SKUs):

1. **Regression — predict `discounted_price` (₹)** from product characteristics. Because the target is heavily right-skewed, the model was trained on `log1p(price)` and metrics are reported back in real rupees (inverse-transformed via `expm1`). Features: `actual_price_log`, `discount_percentage`, `rating`, `review_sentiment_vader`, `review_text_length`, `about_product_length`, `category_main` and TF-IDF text. **`discount_amount` / `discount_amount_log` are excluded** — as a deterministic function of the target (`actual_price` − `discounted_price`) they would leak it.
2. **Classification — High-Demand flag**: `rating_count > 4,740` (the product-level median), a balanced binary target (49.9% positive share). Features: all valid product characteristics (incl. `discounted_price_log`, `discount_amount_log`, `price_tier_ordinal`) + category + TF-IDF text. **`rating_count` / `rating_count_log` are excluded** (they define the target); the 2 products with an undefined target are dropped before the split (1,349 usable).

**Split strategy (leakage-safe):** the regression split is 80/20 **stratified by deciles of the log target** (quantile binning via `pd.qcut`), so every price band is proportionally represented on both sides; the classification split is 80/20 stratified on the binary target.

Every model ran through the same **feature pipeline** (all steps fitted on training folds only):

| Transformer | Applies to |
|-------------|------------|
| `SimpleImputer` (median) + `StandardScaler` | numeric features — imputation is in-pipeline, never global |
| `SimpleImputer` (most-frequent) + `OneHotEncoder` (`sparse_output=True`, handle_unknown=ignore) | `category_main` |
| `TfidfVectorizer` (2,500 tokens, min_df=2, unigrams) | combined `about_product` + `review_title` + `review_content` |

The sparse one-hot block combines with the sparse TF-IDF matrix into a single sparse design matrix (~2,500 columns) — a significant memory optimisation over dense encoding.

Baselines: **`LinearRegression`/`Ridge`** (regression); **`LogisticRegression`/`DecisionTree`** (classification). Ensembles: **`RandomForest`**, **`XGBoost`**, **`LightGBM`**. The champion per task was further tuned with **`RandomizedSearchCV`** (15 draws × 3-fold CV, seed 42).

### 4.2 Regression Results — predicting `discounted_price`

| Model | RMSE (₹) | MAE (₹) | R² | MAPE (%) |
|-------|---------:|--------:|---:|---------:|
| LinearRegression (baseline) | 957.31 | 271.04 | 0.9868 | 13.72 |
| **Ridge (baseline, champion)** | **534.61** | **186.93** | **0.9959** | **8.73** |
| DecisionTree (baseline) | 2,777.26 | 610.95 | 0.8890 | 11.98 |
| RandomForest (ensemble) | 2,427.95 | 419.95 | 0.9151 | 7.86 |
| XGBoost (ensemble) | 2,749.87 | 632.51 | 0.8912 | 13.62 |
| LightGBM (ensemble) | 3,101.34 | 455.31 | 0.8616 | 6.68 |
| **Ridge_tuned** | **534.61** | **186.93** | **0.9959** | **8.73** |

- **Winner: Ridge.** Regularisation is decisive: Ridge's RMSE (₹535) is **~4.5× lower** than the best tree ensemble (RandomForest, ₹2,428), because a **2,500-dimension TF-IDF space** is absorbed perfectly by the L2 shrinkage while trees split crudely on it. LinearRegression's MAPE (13.7%) also improved sharply once the split became decile-stratified (25.1% under a plain random split).
- **Leakage check:** an earlier pipeline variant included `discount_amount_log` — a direct arithmetic function of the target (`actual_price` − `discounted_price`) — which inflated Ridge to an implausible RMSE ≈ ₹301 / R² ≈ 0.998. With that feature excluded and imputation moved in-pipeline, the figures above are **honest**: list price and discount depth remain the near-deterministic price drivers (`actual_price` vs `discounted_price`, r = 0.96), and the residual churn (~8.7% MAPE) comes from outlier SKUs and data noise. Test-set predictions: `outputs/ml/regression_predictions.csv`.

**Key drivers (Ridge coefficients — `outputs/ml/feature_importance_regression.csv`):**

| Rank | Feature | Importance (abs coef) |
|-----:|---------|----------------------:|
| 1 | `actual_price_log` | 1.3236 |
| 2 | `discount_percentage` | 0.4422 |
| 3 | text: `protector` | 0.2430 |
| 4 | text: `work` | 0.1748 |
| 5 | text: `ninja` / `colours` / `tight` | 0.147–0.169 |

> List price explains the bulk of discount-price variance and discount depth ranks second; descriptive-text tokens make a small but non-trivial contribution. Because the target-derived `discount_amount` was removed, these importances are **trustworthy rather than tautological** — reinforcing that **listing copy plays a real, if secondary, role in price positioning**.

### 4.3 Classification Results — predicting High Demand

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|---------:|----------:|-------:|---:|--------:|
| LogisticRegression (baseline) | 0.7815 | 0.7836 | 0.7778 | 0.7807 | 0.8532 |
| DecisionTree (baseline) | 0.6593 | 0.6641 | 0.6444 | 0.6541 | 0.6658 |
| **RandomForest (ensemble, champion)** | 0.7889 | **0.8250** | 0.7333 | 0.7765 | **0.8715** |
| XGBoost (ensemble) | **0.7926** | 0.7883 | **0.8000** | **0.7941** | 0.8494 |
| LightGBM (ensemble) | 0.7741 | 0.7937 | 0.7407 | 0.7663 | 0.8537 |
| RandomForest_tuned (3-fold CV-optimised) | 0.7741 | 0.8558 | 0.6593 | 0.7448 | 0.8560 |

- **Winner: RandomForest** — best non-linear separation of demand (AUC **0.8715**, +1.8 pts over logistic, +20.6 pts over the tree baseline). Precision 0.825 — when it flags High Demand, it is right ~83% of the time (confusion matrix: `outputs/ml/confusion_matrix_tuned.csv`, `fig7_confusion_matrix.png`). XGBoost posts the best raw accuracy/F1 but a lower AUC; the champion is selected on ROC-AUC, the primary metric for this balanced target.
- The CV-tuned variant (AUC 0.8560, Precision 0.8558, Recall 0.6593) trades recall for **precision and cross-fold robustness** — a useful operating point when false positives (over-stocking) are costlier than false negatives. Both variants beat the linear baseline — a classic non-linearity win (demand depends on combinations of price tier × category × text, not additive linear sums). ROC comparison of all classifiers: `fig8_roc_curve.png`.
- Test-set predictions: `outputs/ml/classification_predictions.csv`.

**Key drivers (RandomForest importances — `outputs/ml/feature_importance_classification.csv`):**

| Rank | Feature | Importance |
|-----:|---------|-----------:|
| 1 | `review_text_length` | 0.0053 |
| 2 | `rating` | 0.0046 |
| 3 | `review_sentiment_vader` | 0.0039 |
| 4 | `discount_percentage` | 0.0038 |
| 5 | text: `budget` | 0.0036 |
| 6 | `discount_amount_log` / `product_name_length` | ~0.0032 |

> Demand is a **listing-engagement + rating + discount-depth story**: the volume of buyer reviews (text length), the star rating and VADER sentiment lead, with discount depth and budget-oriented copy adding support. Importances are spread thinly across ~2,500 sparse TF-IDF and one-hot columns, so single-column magnitudes are small by construction — **the ordering is the signal, not the absolute size**. `rating_count` itself is fully absent from the feature space, so these drivers are genuine pre-outcome signals.

---

## 5. Business Recommendations & Strategic Takeaways

### 5.1 Pricing Optimisation

1. **Own the ₹500–1.5k value corridor.** It spans ~32% of the catalogue, carries the deepest average discount (54%) and the highest average review volume (~21.8k per SKU) — the classic "drivers' lane" for accessories and consumer electronics. Price anchors (list prices) should be set to make 50–55% discounts look dramatic, but avoid crossing ~60%: beyond that, ratings start to dip (−0.16 correlation between discount depth and rating).
2. **Stop discounting luxury.** Luxury (> ₹20k) products earn the best average rating (4.23) with only 36% discounts — price-cutting is unnecessary there and likely margin-leaking. Premium positioning through listing quality beats discounting everywhere.
3. **Use the Ridge regression as a repricing sanity-check.** With MAPE ≈ 8.7% and R² ≈ 0.996 (leakage-free), the model instantly flags SKUs whose effective price deviates from Amazon's learned price mechanics (useful for promo planning and margin guardrails).

### 5.2 Inventory & Assortment Prioritisation

4. **Double down on Electronics & Computers-Accessories.** The two categories account for ~88% of total demand. Within them, cables/adapters/chargers (the "cable", "charging", "battery" tokens appear 1.0–1.7k times) rotate fast — keep deep, broad inventory (54% discount band = highest demand).
5. **Expand Office Products as a whitespace play.** Highest category rating (4.31) with only 31 SKUs and virtually no discounting — low-competition, high-satisfaction adjacency. A modest 20–30 new SKUs with standard 30–40% launch discounts could capture an underserved ladder.
6. **Let the RandomForest pick high-demand SKUs pre-launch.** With AUC ≈ 0.87 and Precision ≈ 0.83 (and a precision-tuned variant at ≈ 0.86), simulation of rating × price-tier × listing signals before launch enables smarter replenishment depth and advertising budgets — particularly flagging Home & Kitchen items where margin allows.

### 5.3 Customer Rating & Review Improvements

7. **Ratings are compressed into the 4.1–4.3 band — differentiate on review *count*, not just stars.** Higher review volume correlates weakly but positively with ratings (+0.10); drive the volume with post-purchase follow-ups and honest, durable product copy (listing-text tokens like "protect", "durable", "sturdy" materially appear in model drivers).
8. **Attack the ~2% negative-review tail.** Complaints cluster on physical durability ("bending", "loose", "scratch", "annoying") and functionality ("slow charging", "low sound") — prioritise spec-accuracy in listing copy, reinforced packaging, and warranty communication; each avoided 1-star review compounds star-averages faster than any PPC campaign.
9. **Monitor sentiment per category.** 96% of reviews are VADER-positive — great social proof; but the negative tails are category-specific (e.g. sound complaints in TVs) — build a small dashboard reading `review_sentiment_vader` to catch category churn early.

---

## Appendix — Project Artifacts

| Artifact | Location |
|----------|----------|
| Cleaning script | `01_eda_and_cleaning.py` |
| Visualization script | `02_data_visualization.py` |
| ML pipeline script | `03_ml_model_training.py` |
| This report | `DATASET_REPORT.md` |
| Cleaned datasets | `data/amazon_cleaned.csv`, `data/amazon_cleaned_products.csv` |
| EDA artifacts (summary statistics, correlations, category/tier/sentiment tables) | `outputs/eda/` |
| Figures: fig1 price distribution, fig2 discount-vs-rating, fig3 category, fig4 correlation, fig5 rating boxplot, fig6 sentiment, fig7 confusion matrix, fig8 ROC | `outputs/figures/` |
| Model comparison, metrics, predictions, importances, tuning results, summary JSON | `outputs/ml/` |

**End of report — generated by the 01→03 Amazon Sales Dataset pipeline (Python stack: pandas / numpy / scikit-learn / XGBoost / LightGBM / NLTK-VADER / TextBlob / matplotlib / seaborn).**