# -*- coding: utf-8 -*-
"""
================================================================================
03_ml_model_training.py
================================================================================
Module 3 of the "Amazon Sales Dataset" end-to-end analytics pipeline.

This module frames and answers TWO supervised machine-learning tasks on the
deduplicated product-level dataset produced by module #1:

  TASK 1 - REGRESSION
      Predict ``discounted_price`` (in Indian Rupees) from product
      characteristics (list price, discount depth, rating, review sentiment,
      category, and TF-IDF vectors of the product description / reviews).

  TASK 2 - CLASSIFICATION
      Predict whether a product is a HIGH-DEMAND item, defined as
      ``rating_count`` strictly above the cross-product median.

Methodology:
  1. Feature engineering (log-transforms, price tiers, combined text, ...).
  2. Leakage-safe 80/20 splits: the regression split is stratified by
     quantile bins (deciles) of the log target; the classification split is
     stratified on the binary target, and rows with an undefined target are
     dropped beforehand (targets are never imputed).
  3. Scikit-learn pipelines combining ``SimpleImputer`` + ``StandardScaler``
     (numeric), ``SimpleImputer`` + sparse ``OneHotEncoder`` (categorical)
     and ``TfidfVectorizer`` (text). ALL preprocessing is fitted on the
     training folds only, so no aggregate statistic computed on the full
     dataset can leak into the held-out data.
  4. Baseline models (Linear/Ridge/Decision Tree + Logistic Regression),
     then ensemble models (Random Forest, XGBoost, LightGBM).
  5. ``RandomizedSearchCV`` hyper-parameter tuning on the best model per task.
  6. Evaluation (RMSE/MAE/R2 and Accuracy/Precision/Recall/F1/ROC-AUC) with
     feature-importance extraction and persisted artifacts.

Leakage rules enforced:
  * ``rating_count`` / ``rating_count_log`` are NEVER model features - they
    define the High-Demand classification target.
  * ``discount_amount`` / ``discount_amount_log`` are NEVER regression
    features - ``discounted_price = actual_price - discount_amount`` makes
    them a deterministic function of the regression target.
  * Missing-value imputation lives inside the pipelines (``SimpleImputer``),
    never in the upstream cleaning stage.

Outputs (written to ``outputs/ml/`` and ``outputs/figures/``):
   * model_comparison.csv            - head-to-head model matrix
   * regression_metrics.csv          - RMSE / MAE / R2 per regression model
   * classification_metrics.csv      - acc / prec / rec / f1 / auc per model
   * regression_predictions.csv      - y_true vs y_pred on the test split
   * classification_predictions.csv  - y_true vs y_pred_proba on the test split
   * feature_importance_{task}.csv   - top drivers per task
   * tuning_results_{task}.csv       - RandomizedSearchCV details
   * pipeline_summary.json           - datasets, splits, thresholds
   * fig7_confusion_matrix.png       - confusion matrix for the tuned model
   * fig8_roc_curve.png              - ROC curves for every classifier

Input : ``data/amazon_cleaned_products.csv`` (from module #1)

Usage:
    python 03_ml_model_training.py
================================================================================
"""

# ------------------------------------------------------------------------------
# Standard library imports
# ------------------------------------------------------------------------------
import json
import os
import warnings

# ------------------------------------------------------------------------------
# Third-party numeric / dataframe imports
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

# ------------------------------------------------------------------------------
# Scikit-learn model selection & preprocessing
# ------------------------------------------------------------------------------
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------------------------------------------------------
# Scikit-learn regression models
# ------------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------------------------------------------------
# Scikit-learn classification models
# ------------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------------------------
# Scikit-learn evaluation metrics
# ------------------------------------------------------------------------------
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ------------------------------------------------------------------------------
# Plotting (Agg for headless rendering of the confusion matrix / ROC charts)
# ------------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# Gradient boosting libraries
# ------------------------------------------------------------------------------
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

warnings.filterwarnings("ignore")  # keep logs clean


# ------------------------------------------------------------------------------
# Path configuration
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ML_DIR = os.path.join(BASE_DIR, "outputs", "ml")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")

PRODUCTS_PATH = os.path.join(DATA_DIR, "amazon_cleaned_products.csv")


# ------------------------------------------------------------------------------
# Global configuration
# ------------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 3
N_ITER_SEARCH = 15          # RandomizedSearchCV iterations (runtime trade-off)
N_JOBS = -1
TFIDF_MAX_FEATURES = 2500   # cap the text feature space
TFIDF_MIN_DF = 2            # drop words seen in fewer than 2 documents
TFIDF_NGRAM_RANGE = (1, 1)  # unigrams keep the model fast and stable

# Numeric features. Two hard leakage rules are enforced here:
#   * ``rating_count`` / ``rating_count_log`` are EXCLUDED from every task -
#     they define the High-Demand classification target.
#   * ``discount_amount`` / ``discount_amount_log`` are EXCLUDED from the
#     regression task - ``discounted_price = actual_price - discount_amount``
#     makes them a deterministic function of the regression target.
BASE_NUMERIC_FEATURES = [
    "actual_price_log",
    "discount_percentage",
    "rating",
    "review_sentiment_vader",
    "review_text_length",
    "about_product_length",
]
REGRESSION_NUMERIC_FEATURES = list(BASE_NUMERIC_FEATURES)
# Classification (target = High Demand, derived from rating_count) may safely
# use the remaining valid product characteristics, including pricing features.
CLASSIFICATION_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + [
    "discount_amount_log",
    "discounted_price_log",
    "product_name_length",
    "price_tier_ordinal",
]
CATEGORICAL_FEATURES = ["category_main"]
TEXT_FEATURE = "combined_text"

# High-Rating definition used in the report narrative (>= 4.2 stars).
HIGH_RATING_THRESHOLD = 4.2


# ------------------------------------------------------------------------------
# 1. Helpers
# ------------------------------------------------------------------------------
def ensure_directories():
    """Create the output directories for ML artifacts and figures."""
    os.makedirs(ML_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def log(msg=""):
    """Small console logger used by the module."""
    print(msg)


# ------------------------------------------------------------------------------
# 2. Data loading & feature engineering
# ------------------------------------------------------------------------------
def load_products():
    """
    Load the deduplicated product-level dataset produced by module #1.

    Raises ``FileNotFoundError`` when the cleaning script has not been run.
    """
    if not os.path.exists(PRODUCTS_PATH):
        raise FileNotFoundError(
            "data/amazon_cleaned_products.csv not found. Please run module #1 "
            "(01_eda_and_cleaning.py) before training models."
        )
    df = pd.read_csv(PRODUCTS_PATH, encoding="utf-8-sig")
    return df


def engineer_features(df):
    """
    Create the modelling-grade feature set from the cleaned product table.

    Engineered columns:
      * ``actual_price_log`` / ``discounted_price_log`` - log1p of list prices
        to tame the heavy right-skew of the price distributions.
      * ``discount_amount`` - the absolute rupee discount (list minus sale).
      * ``discount_amount_log`` - log1p of the rupee discount.
      * ``combined_text`` - concatenated product description + review title +
        review content, used as the TF-IDF corpus.

    Returns a copy of ``df`` with the new columns, plus a ``high_demand``
    binary target (``rating_count`` > median).
    """
    out = df.copy()

    if "discount_amount" not in out.columns:
        out["discount_amount"] = (out["actual_price"]
                                  - out["discounted_price"]).clip(lower=0.0)

    out["actual_price_log"] = np.log1p(out["actual_price"])
    out["discounted_price_log"] = np.log1p(out["discounted_price"])
    out["discount_amount_log"] = np.log1p(out["discount_amount"])
    out["rating_count_log"] = np.log1p(out["rating_count"])

    out["combined_text"] = (
        out["about_product"].fillna("").astype(str)
        + " "
        + out["review_title"].fillna("").astype(str)
        + " "
        + out["review_content"].fillna("").astype(str)
    )

    # Median is computed with pandas, which skips NaN rating counts.
    demand_median = float(out["rating_count"].median())
    # NaN-safe target: products with an unknown rating_count get NaN instead
    # of being silently mapped to the negative class by the NaN > x -> False
    # comparison. Those rows are dropped from the classification split later
    # (targets are never imputed).
    out["high_demand"] = np.where(
        out["rating_count"].isna(),
        np.nan,
        (out["rating_count"] > demand_median).astype(float),
    )
    out["high_rating"] = (out["rating"] >= HIGH_RATING_THRESHOLD).astype(int)

    # Ordinal encoding of the price tier (0 = Budget ... 4 = Luxury).
    tier_order = {
        "Budget (<500)": 0,
        "Value (500-1.5k)": 1,
        "Mid-Range (1.5k-5k)": 2,
        "Premium (5k-20k)": 3,
        "Luxury (>20k)": 4,
    }
    out["price_tier_ordinal"] = out["price_tier"].map(tier_order).fillna(0).astype(int)

    log(f"   Demand median used for High-Demand target : {demand_median:,.0f}")
    pos_rate = out["high_demand"].mean() * 100
    log(f"   High-Demand positive-class share           : {pos_rate:.1f}%")
    return out


def make_feature_columns(task):
    """
    Pick the numeric feature list for a given task (leakage-safe).

    * Regression on ``discounted_price`` uses the base characteristics ONLY:
      ``discount_amount_log`` is a deterministic function of the target and
      ``discounted_price_log`` IS the target - both are excluded.
    * Classification on ``high_demand`` may use every valid product feature:
      only ``rating_count`` / ``rating_count_log`` (the target's raw source)
      are excluded, and they are excluded from both tasks.
    """
    if task == "regression":
        return REGRESSION_NUMERIC_FEATURES
    return CLASSIFICATION_NUMERIC_FEATURES


# ------------------------------------------------------------------------------
# 3. Preprocessing pipeline
# ------------------------------------------------------------------------------
def build_preprocessor(numeric_features):
    """
    Build a ``ColumnTransformer`` that:

      * imputes missing numeric values with the **training-fold median**
        (``SimpleImputer(strategy="median")``) - imputation is part of the
        pipeline so it never sees the held-out data (no leakage);
      * standardises numeric features (``StandardScaler``);
      * imputes missing categorical values with the most frequent value and
        one-hot encodes the main category (``OneHotEncoder``) so new/unseen
        categories fall into an ignored bucket;
      * vectorises the combined review/description text (``TfidfVectorizer``
        with a capped vocabulary).

    The one-hot encoder emits SPARSE output so the sparse TF-IDF block and
    the tiny categorical block combine into one sparse design matrix - a
    decisive memory saving over densifying ~2,500 TF-IDF columns.

    The transformers are chained inside every model pipeline so imputation,
    scaling and encoding are always fitted on the training fold only.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore",
                                     sparse_output=True)),
        ]
    )
    text_transformer = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                min_df=TFIDF_MIN_DF,
                ngram_range=TFIDF_NGRAM_RANGE,
                sublinear_tf=True,
            ))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("text", text_transformer, TEXT_FEATURE),
        ]
    )
    return preprocessor


# ------------------------------------------------------------------------------
# 4. Dataset preparation & splitting
# ------------------------------------------------------------------------------
def prepare_task_data(df, task, preprocessor):
    """
    Build X/y for a task and return a leakage-safe train/test split.

    Regression uses ``log1p(discounted_price)`` as the target (deeply skewed
    on the raw scale). A plain random split could skew the price distribution
    between train and test, so the split is **stratified by quantile bins
    (deciles) of the log target** - every price band is proportionally
    represented on both sides. The inverse transform (``expm1``) is applied
    before metric computation.

    Classification uses the ``high_demand`` binary flag. Rows whose
    ``rating_count`` (and therefore the target) is missing are dropped
    BEFORE the split - targets are never imputed - and the split is
    stratified on the target to keep the class balance.
    """
    numeric_features = make_feature_columns(task)

    if task == "regression":
        frame = df
        y = np.log1p(frame["discounted_price"].to_numpy(dtype="float64"))
        # Continuous-target stratification: decile bins of the log target.
        strata = pd.qcut(y, q=10, labels=False, duplicates="drop")
        stratify = np.asarray(strata)
    else:
        # Drop rows with an undefined target (NaN rating_count) BEFORE the
        # split; never impute a target variable.
        frame = df[df["high_demand"].notna()].reset_index(drop=True)
        dropped = len(df) - len(frame)
        if dropped:
            log(f"   Dropped {dropped} product(s) with missing rating_count "
                f"(undefined High-Demand target).")
        y = frame["high_demand"].astype(int).to_numpy()
        stratify = y

    X = frame[numeric_features + CATEGORICAL_FEATURES + [TEXT_FEATURE]].copy()
    # Normalise the text column so NaN never reaches the vectoriser.
    X[TEXT_FEATURE] = X[TEXT_FEATURE].fillna("").astype(str)

    # Explicit boundary check: every numeric feature must be present.
    expected = set(numeric_features)
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"Missing engineered features: {sorted(missing)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED,
        stratify=stratify,
    )
    log(f"   Split {task}: train={len(X_train):,} test={len(X_test):,}")
    return X_train, X_test, y_train, y_test, numeric_features


# ------------------------------------------------------------------------------
# 5. Model registry
# ------------------------------------------------------------------------------
def get_models(task, preprocessor):
    """
    Return the {name: pipeline} dictionary for a given task.

    Every pipeline shares the same ``preprocessor`` ColumnTransformer so the
    feature preparation is identical across all models and comparisons stay
    fair. Baselines (linear / ridge / trees) are deliberately simple; the
    ensembles use moderate capacities suited to the ~1,350-row dataset.
    """
    if task == "regression":
        models = {
            "LinearRegression": Pipeline([
                ("prep", preprocessor),
                ("model", LinearRegression()),
            ]),
            "Ridge": Pipeline([
                ("prep", preprocessor),
                ("model", Ridge(alpha=1.0)),
            ]),
            "DecisionTree": Pipeline([
                ("prep", preprocessor),
                ("model", DecisionTreeRegressor(
                    max_depth=12, random_state=RANDOM_SEED)),
            ]),
            "RandomForest": Pipeline([
                ("prep", preprocessor),
                ("model", RandomForestRegressor(
                    n_estimators=300, max_depth=None, min_samples_leaf=2,
                    n_jobs=N_JOBS, random_state=RANDOM_SEED)),
            ]),
            "XGBoost": Pipeline([
                ("prep", preprocessor),
                ("model", XGBRegressor(
                    n_estimators=300, learning_rate=0.08, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=RANDOM_SEED, n_jobs=N_JOBS, verbosity=0)),
            ]),
            "LightGBM": Pipeline([
                ("prep", preprocessor),
                ("model", LGBMRegressor(
                    n_estimators=300, learning_rate=0.08, num_leaves=31,
                    random_state=RANDOM_SEED, n_jobs=N_JOBS, verbosity=-1)),
            ]),
        }
    else:
        # classification: LogisticRegression is the linear baseline here.
        models = {
            "LogisticRegression": Pipeline([
                ("prep", preprocessor),
                ("model", LogisticRegression(max_iter=2000, C=1.0,
                                             random_state=RANDOM_SEED)),
            ]),
            "DecisionTree": Pipeline([
                ("prep", preprocessor),
                ("model", DecisionTreeClassifier(
                    max_depth=10, class_weight="balanced",
                    random_state=RANDOM_SEED)),
            ]),
            "RandomForest": Pipeline([
                ("prep", preprocessor),
                ("model", RandomForestClassifier(
                    n_estimators=300, max_depth=None, min_samples_leaf=2,
                    class_weight="balanced", n_jobs=N_JOBS,
                    random_state=RANDOM_SEED)),
            ]),
            "XGBoost": Pipeline([
                ("prep", preprocessor),
                ("model", XGBClassifier(
                    n_estimators=300, learning_rate=0.08, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=RANDOM_SEED, n_jobs=N_JOBS, verbosity=0)),
            ]),
            "LightGBM": Pipeline([
                ("prep", preprocessor),
                ("model", LGBMClassifier(
                    n_estimators=300, learning_rate=0.08, num_leaves=31,
                    random_state=RANDOM_SEED, n_jobs=N_JOBS, verbosity=-1)),
            ]),
        }
    return models


# ------------------------------------------------------------------------------
# 6. Evaluation helpers
# ------------------------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    """
    Compute RMSE / MAE / R2 / MAPE on prediction errors.

    Division-by-zero is guarded by flooring the MAPE denominator.
    """
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    denom = np.maximum(np.abs(y_true), 1e-9)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 4),
        "MAPE_%": round(mape, 2),
    }


def classification_metrics(y_true, y_pred, y_proba=None):
    """
    Compute Accuracy / Precision / Recall / F1 / ROC-AUC.

    ``y_proba`` is required for the AUC; when it is missing, the AUC is NaN.
    ``zero_division=0`` guards against degenerate empty prediction labels.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    auc = np.nan
    if y_proba is not None:
        try:
            auc = float(roc_auc_score(y_true, y_proba))
        except ValueError:  # single-class test batch edge case
            auc = np.nan

    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "ROC_AUC": round(auc, 4),
    }


# ------------------------------------------------------------------------------
# 7. Baseline model comparison (single stratified train/test split)
# ------------------------------------------------------------------------------
def run_comparison(X_train, X_test, y_train, y_test, task, preprocessor):
    """
    Fit every registered model and return a comparison table + fitted dict.

    Returns
    -------
    (pandas.DataFrame, dict)
        Comparison table with one row per model, and a mapping
        ``name -> (fitted_pipeline, predictions, [proba])`` used later.
    """
    models = get_models(task, preprocessor)
    rows = []
    fitted = {}

    for name, pipe in models.items():
        try:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            if task == "regression":
                # Invert the log1p target transform back to rupees.
                y_test_orig = np.expm1(y_test)
                pred_orig = np.expm1(pred)
                metrics = regression_metrics(y_test_orig, pred_orig)
                fitted[name] = (pipe, pred_orig)
            else:
                y_proba = pipe.predict_proba(X_test)[:, 1]
                metrics = classification_metrics(y_test, pred, y_proba)
                fitted[name] = (pipe, pred, y_proba)

            rows.append({"model": name, **metrics})
            log(f"   [{task}] {name:<20} {metrics}")
        except Exception as err:  # keep the comparison going on one failure
            log(f"   [{task}] {name} FAILED: {err}")
            rows.append({"model": name, "error": str(err)})

    comparison = pd.DataFrame(rows)
    return comparison, fitted


# ------------------------------------------------------------------------------
# 8. Hyper-parameter tuning (RandomizedSearchCV on the best model per task)
# ------------------------------------------------------------------------------
def get_param_grids(task):
    """
    Per-model search spaces for RandomizedSearchCV.

    Values are either crisp lists (discrete choices) or continuous scipy
    distributions (sampled ``n_iter`` times). Spaces are deliberately small to
    keep the search tractable with 3-fold CV on ~1,080 training rows.
    """
    if task == "regression":
        return {
            "LinearRegression": {"model__fit_intercept": [True, False]},
            "Ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "DecisionTree": {
                "model__max_depth": randint(4, 30),
                "model__min_samples_leaf": randint(1, 10),
            },
            "RandomForest": {
                "model__n_estimators": randint(150, 500),
                "model__max_depth": randint(6, 30),
                "model__min_samples_leaf": randint(1, 6),
                "model__max_features": ["sqrt", "log2"],
            },
            "XGBoost": {
                "model__n_estimators": randint(150, 500),
                "model__learning_rate": uniform(0.02, 0.18),
                "model__max_depth": randint(3, 10),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.6, 0.4),
            },
            "LightGBM": {
                "model__n_estimators": randint(150, 500),
                "model__learning_rate": uniform(0.02, 0.18),
                "model__num_leaves": randint(15, 80),
                "model__subsample": uniform(0.7, 0.3),
                "model__colsample_bytree": uniform(0.7, 0.3),
            },
        }
    return {
        "LogisticRegression": {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "DecisionTree": {
            "model__max_depth": randint(4, 25),
            "model__min_samples_leaf": randint(1, 8),
        },
        "RandomForest": {
            "model__n_estimators": randint(150, 500),
            "model__max_depth": randint(6, 30),
            "model__min_samples_leaf": randint(1, 6),
            "model__max_features": ["sqrt", "log2"],
        },
        "XGBoost": {
            "model__n_estimators": randint(150, 500),
            "model__learning_rate": uniform(0.02, 0.18),
            "model__max_depth": randint(3, 10),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
        },
        "LightGBM": {
            "model__n_estimators": randint(150, 500),
            "model__learning_rate": uniform(0.02, 0.18),
            "model__num_leaves": randint(15, 80),
            "model__subsample": uniform(0.7, 0.3),
            "model__colsample_bytree": uniform(0.7, 0.3),
        },
    }


def tune_best_model(name, base_pipe, X_train, y_train, task):
    """
    Run ``RandomizedSearchCV`` over the best comparison model.

    Returns the fitted best estimator, its score, and the search result table.
    """
    grids = get_param_grids(task)
    if name not in grids:
        log(f"   No search space for {name}; skipping tuning.")
        return base_pipe, None, None

    scoring = ("neg_root_mean_squared_error" if task == "regression"
               else "roc_auc")
    search = RandomizedSearchCV(
        estimator=base_pipe,
        param_distributions=grids[name],
        n_iter=N_ITER_SEARCH,
        cv=CV_FOLDS,
        scoring=scoring,
        n_jobs=N_JOBS,
        random_state=RANDOM_SEED,
        verbose=0,
    )
    log(f"   Tuning {name} with {N_ITER_SEARCH} random draws x {CV_FOLDS}-fold CV ...")
    search.fit(X_train, y_train)

    results = pd.DataFrame(search.cv_results_)
    log(f"   Best {scoring} (CV) = {search.best_score_:.4f}")
    return search.best_estimator_, search.best_score_, results


# ------------------------------------------------------------------------------
# 9. Feature-importance extraction
# ------------------------------------------------------------------------------
def feature_names_from_ct(ct, numeric_features):
    """Recover the full transformed feature-name vector from a fitted CT."""
    names = []
    for name, transformer, cols in ct.transformers_:
        if name == "num":
            try:
                names.extend(list(transformer.feature_names_in_))
            except AttributeError:
                names.extend(numeric_features)
        elif name == "cat":
            onehot = transformer.named_steps["onehot"]
            token_cols = cols if not hasattr(cols, "tolist") else cols
            names.extend(onehot.get_feature_names_out(input_features=list(token_cols)))
        elif name == "text":
            tfidf = transformer.named_steps["tfidf"]
            names.extend(tfidf.get_feature_names_out())
    return names


def extract_feature_importances(pipe, task, numeric_features):
    """
    Pull importance scores out of the fitted best pipeline.

    Tree ensembles expose ``feature_importances_``; linear models fall back to
    the absolute magnitude of their coefficients.
    """
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    names = feature_names_from_ct(prep, numeric_features)

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype="float64")
    elif hasattr(model, "coef_"):
        coef = np.atleast_2d(model.coef_)
        importance = np.abs(coef[0])
    else:
        importance = np.zeros(len(names))

    importance = np.asarray(importance).flatten()
    k = min(len(names), len(importance))
    table = pd.DataFrame({
        "feature": names[:k],
        "importance": importance[:k],
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return table


# ------------------------------------------------------------------------------
# 10. Per-task runner
# ------------------------------------------------------------------------------
def run_task(task, df):
    """
    Execute the complete modelling flow for a single task and persist outputs.

    Returns a summary dictionary with the chosen model and its test metrics.
    """
    log("\n" + "=" * 80)
    log(f"TASK: {task.upper()}")
    log("=" * 80)

    numeric_features = make_feature_columns(task)
    preprocessor = build_preprocessor(numeric_features)

    X_train, X_test, y_train, y_test, num_feats = prepare_task_data(
        df, task, preprocessor)

    # 10.1 - Fit & compare all baselines/ensembles.
    comparison, fitted = run_comparison(
        X_train, X_test, y_train, y_test, task, preprocessor)
    comparison.to_csv(os.path.join(ML_DIR, f"{task}_model_comparison.csv"),
                      index=False)

    # 10.2 - Select the champion model by the task's primary metric.
    primary = "R2" if task == "regression" else "ROC_AUC"
    valid = comparison.dropna(subset=[primary]) if primary in comparison else comparison
    best_name = valid.sort_values(primary, ascending=False).iloc[0]["model"]
    log(f"\n   Champion on {primary}: {best_name}")

    # 10.3 - Hyper-parameter tuning on the champion.
    best_pipe, best_cv, search_results = tune_best_model(
        best_name, fitted[best_name][0], X_train, y_train, task)

    if search_results is not None:
        search_results.to_csv(
            os.path.join(ML_DIR, f"tuning_results_{task}.csv"), index=False)

    # 10.4 - Final evaluation of the tuned model on the held-out test set.
    if task == "regression":
        y_test_orig = np.expm1(y_test)
        pred_orig = np.expm1(best_pipe.predict(X_test))
        tuned_metrics = regression_metrics(y_test_orig, pred_orig)
        predictions = pd.DataFrame({
            "y_true": y_test_orig,
            "y_pred": pred_orig,
            "error": y_test_orig - pred_orig,
        })
    else:
        pred = best_pipe.predict(X_test)
        proba = best_pipe.predict_proba(X_test)[:, 1]
        tuned_metrics = classification_metrics(y_test, pred, proba)
        predictions = pd.DataFrame({
            "y_true": y_test,
            "y_pred": pred,
            "y_pred_proba": proba,
        })
    log(f"   Tuned {task} metrics: {tuned_metrics}")

    predictions.to_csv(os.path.join(ML_DIR, f"{task}_predictions.csv"), index=False)

    # 10.5 - Feature importances of the tuned pipeline.
    importance = extract_feature_importances(best_pipe, task, num_feats)
    importance.to_csv(os.path.join(ML_DIR, f"feature_importance_{task}.csv"),
                      index=False)
    log("\n   Top-10 feature importances:")
    log(importance.head(10).to_string(index=False))

    # 10.6 - Append the tuned champion to the model comparison table.
    tuned_row = {"model": f"{best_name}_tuned", **tuned_metrics}
    final_table = comparison.copy()
    final_table = pd.concat([final_table, pd.DataFrame([tuned_row])],
                            ignore_index=True)
    final_table.to_csv(os.path.join(ML_DIR, f"{task}_metrics.csv"), index=False)

    summary = {
        "task": task,
        "champion_baseline": best_name,
        "cv_score": best_cv,
        "tuned_metrics": tuned_metrics,
        "test_rows": int(len(X_test)),
    }
    return summary, final_table, fitted, y_test


# ------------------------------------------------------------------------------
# 11. Classification charts (confusion matrix + ROC curves)
# ------------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred):
    """Persist the confusion matrix heatmap of the tuned classifier."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Low Demand", "High Demand"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Confusion Matrix - Tuned High-Demand Model")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig7_confusion_matrix.png"), dpi=300)
    plt.close(fig)
    # Also persist the raw matrix as a small CSV for the report.
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        os.path.join(ML_DIR, "confusion_matrix_tuned.csv"))


def plot_roc_curves(fitted, y_test):
    """Overlay the ROC curve of every fitted classifier on one figure."""
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, (pipe, _pred, proba) in fitted.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        try:
            auc = float(roc_auc_score(y_test, proba))
        except ValueError:
            auc = float("nan")
        ax.plot(fpr, tpr, lw=1.7, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6,
            label="Random chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - High-Demand Classification")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig8_roc_curve.png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------------------
# 12. Main entry point
# ------------------------------------------------------------------------------
def main():
    """Run both tasks, persist artifacts and print the final comparison."""
    ensure_directories()
    np.random.seed(RANDOM_SEED)

    log("Loading product-level dataset ...")
    df = load_products()
    log(f"   {df.shape[0]} products loaded.")

    df = engineer_features(df)
    df.to_csv(os.path.join(ML_DIR, "products_feature_engineered.csv"),
              index=False)

    summaries = []
    tables = []
    for task in ("regression", "classification"):
        summary, final_table, fitted, y_test = run_task(task, df)
        summaries.append(summary)
        final_table["task"] = task
        tables.append(final_table)

        if task == "classification":
            # The tuning step returned the tuned pipeline predictions inside
            # ``final_table``; regenerate the predictions for the plot here.
            # (predictions were already persisted in ``run_task``)
            preds = pd.read_csv(os.path.join(ML_DIR, "classification_predictions.csv"))
            plot_confusion_matrix(preds["y_true"], preds["y_pred"])
            plot_roc_curves(fitted, y_test)

    # Merged head-to-head model comparison across both tasks.
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(os.path.join(ML_DIR, "model_comparison.csv"), index=False)

    summary_payload = {
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "n_iter_search": N_ITER_SEARCH,
        "results": summaries,
    }
    with open(os.path.join(ML_DIR, "pipeline_summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, default=str)

    log("\n" + "=" * 80)
    log("FINAL MODEL COMPARISON")
    log("=" * 80)
    log(combined.to_string(index=False))
    log(f"\nAll ML artifacts saved under: {ML_DIR}")


if __name__ == "__main__":
    main()