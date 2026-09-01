# -*- coding: utf-8 -*-
"""
================================================================================
01_eda_and_cleaning.py
================================================================================
Module 1 of the "Amazon Sales Dataset" end-to-end analytics pipeline.

This module performs

  1. DATA INGESTION
       - Loads the raw Kaggle dataset (``amazon.csv``).

  2. DATA CLEANING & TYPE CONVERSION
       - Strips currency symbols (``₹``) and thousands separators (``,``) from
         ``discounted_price`` / ``actual_price`` and casts them to ``float64``.
       - Parses ``discount_percentage`` (strips ``%``, handles "No Discount").
       - Cleans ``rating`` (non-numeric anomalies -> NaN). Missing values are
         preserved in the cleaned CSVs; imputation happens later inside the
         scikit-learn pipelines (``SimpleImputer``, fitted on training folds
         only) to guarantee leakage-free modelling.
       - Cleans ``rating_count`` (strips commas; missing values preserved as
         NaN for the same reason).
       - Splits the hierarchical ``category`` field
         (``Computers&Accessories|Accessories&Peripherals|...``) into
         ``category_main``, ``category_sub`` and ``category_micro``.

  3. EXPLORATORY DATA ANALYSIS (EDA)
       - Summary statistics (mean, median, std, skew, min, max) for all numeric
         features.
       - Rating / discount / pricing-tier distribution analysis.
       - Pearson correlation analysis among the pricing, discount and rating
         features.
       - Text analysis on ``review_title`` + ``review_content``:
         word frequency and sentiment polarity (VADER + TextBlob).

  4. PERSISTENCE
       - Writes the fully cleaned dataset(s) to ``data/`` as CSV.
       - Writes all EDA summary artifacts to ``outputs/eda/`` as CSV/TXT.

Outputs consumed by:
   * 02_data_visualization.py   (reads ``data/amazon_cleaned.csv``)
   * 03_ml_model_training.py    (reads ``data/amazon_cleaned_products.csv``)
   * DATASET_REPORT.md          (uses the artifacts in ``outputs/eda/``)

Usage:
    python 01_eda_and_cleaning.py
================================================================================
"""

# ------------------------------------------------------------------------------
# Standard library imports
# ------------------------------------------------------------------------------
import os
import re
import warnings

# ------------------------------------------------------------------------------
# Third-party imports
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import skew

# ------------------------------------------------------------------------------
# Natural Language Processing imports (NLTK / TextBlob)
# ------------------------------------------------------------------------------
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer

import nltk  # noqa: E402  module-level access to nltk.data / nltk.download

warnings.filterwarnings("ignore")  # silence noisy dependency warnings

# TextBlob is optional: it is used as a secondary sentiment source, so the
# pipeline keeps running even when the extra library is unavailable.
try:
    from textblob import TextBlob as _TextBlob
    HAS_TEXTBLOB = True
except Exception:  # pragma: no cover - defensive fallback
    HAS_TEXTBLOB = False


# ------------------------------------------------------------------------------
# Path configuration
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
EDA_DIR = os.path.join(OUTPUTS_DIR, "eda")

RAW_DATA_PATH = os.path.join(BASE_DIR, "amazon.csv")
CLEAN_ALL_PATH = os.path.join(DATA_DIR, "amazon_cleaned.csv")
CLEAN_PRODUCTS_PATH = os.path.join(DATA_DIR, "amazon_cleaned_products.csv")
LOG_PATH = os.path.join(EDA_DIR, "eda_console.log")

# ------------------------------------------------------------------------------
# Global constants
# ------------------------------------------------------------------------------
NUMERIC_COLS = [
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating",
    "rating_count",
]

PRICE_TIER_BINS = [-np.inf, 500, 1500, 5000, 20000, np.inf]
PRICE_TIER_LABELS = [
    "Budget (<500)",
    "Value (500-1.5k)",
    "Mid-Range (1.5k-5k)",
    "Premium (5k-20k)",
    "Luxury (>20k)",
]

# VADER compound-score cut-offs used to bucket reviews into sentiment classes.
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

RANDOM_SEED = 42


# ------------------------------------------------------------------------------
# 1. Small logging helper
# ------------------------------------------------------------------------------
class Tee:
    """Write every log line both to the console and to a persistent log file."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._handle = open(path, "w", encoding="utf-8")

    def log(self, message="", nl=True):
        """Print ``message`` to stdout and append it to the log file."""
        text = f"{message}\n" if nl else str(message)
        print(message)
        self._handle.write(text)
        self._handle.flush()

    def close(self):
        """Flush and close the underlying file handle."""
        self._handle.close()


# ------------------------------------------------------------------------------
# 2. Directory bootstrap
# ------------------------------------------------------------------------------
def ensure_directories():
    """Create all required output directories if they do not exist yet."""
    for directory in (DATA_DIR, EDA_DIR):
        os.makedirs(directory, exist_ok=True)


# ------------------------------------------------------------------------------
# 3. Data ingestion
# ------------------------------------------------------------------------------
def load_raw_data(path):
    """
    Load the raw Amazon Sales dataset from ``path``.

    Parameters
    ----------
    path : str
        Absolute path to the ``amazon.csv`` file.

    Returns
    -------
    pandas.DataFrame
        Raw dataframe. Raises ``FileNotFoundError`` when the file is missing
        so the failure is explicit and easy to debug.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw dataset not found at: {path}. Please download amazon.csv "
            "from the Kaggle Amazon Sales Dataset page and place it next to "
            "this script."
        )
    return pd.read_csv(path, encoding="utf-8-sig")


# ------------------------------------------------------------------------------
# 4. Type casting & cleaning utilities
# ------------------------------------------------------------------------------
def clean_currency_column(series, column_name):
    """
    Convert a Rupee-formatted string column (e.g. ``"₹1,099"``) to float64.

    The ``₹`` symbol, thousands separators ``","`` and stray whitespace are
    removed before ``pd.to_numeric`` performs the cast. Any unexpected token
    (e.g. ``"NaN"``) is coerced to ``NaN`` and reported so it can be imputed
    later in the pipeline.

    Parameters
    ----------
    series : pandas.Series
        The raw monetary string series.
    column_name : str
        Display name used in the anomaly log.

    Returns
    -------
    (pandas.Series, int)
        Clean float64 series and the number of coerced (anomalous) values.
    """
    cleaned = (
        series.astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    numeric = pd.to_numeric(cleaned, errors="coerce").astype("float64")
    n_anomalies = int(numeric.isna().sum() - series.isna().sum())
    return numeric, n_anomalies


def clean_discount_percentage(series):
    """
    Parse the ``discount_percentage`` column into a float fraction in [0, 1].

    Handles the three formats observed in the wild for this column:

      * ``"64%"``         -> 0.64
      * ``"No Discount"`` -> 0.0
      * ``"NaN"``         -> NaN (imputed afterwards)

    Parameters
    ----------
    series : pandas.Series
        Raw discount percentage strings.

    Returns
    -------
    (pandas.Series, int)
        Float series of the discount fraction and number of coerced values.
    """
    raw = series.astype(str).str.strip()
    # Map explicit "No Discount" tokens to a literal 0% discount. NaN stays NaN.
    cleaned = raw.str.replace("No Discount", "0", case=False)
    extracted = cleaned.str.extract(r"(\d+(?:\.\d+)?)")[0]
    numeric = pd.to_numeric(extracted, errors="coerce") / 100.0
    numeric = numeric.clip(lower=0.0, upper=1.0)  # discount never exceeds 100%
    n_anomalies = int(numeric.isna().sum() - series.isna().sum())
    return numeric.astype("float64"), n_anomalies


def clean_rating_column(series):
    """
    Clean the ``rating`` column and report any non-numeric anomalies.

    A handful of rows in the public dataset carry erratic tokens (e.g. ``"|"``,
    ``"4.2 |"``). Those are coerced to ``NaN`` by ``pd.to_numeric`` and left
    as NaN in the cleaned output - the machine-learning stage imputes them
    inside its pipelines (``SimpleImputer``) so that no aggregate statistics
    computed on the full dataset leak into the training folds.

    Parameters
    ----------
    series : pandas.Series
        Raw rating strings.

    Returns
    -------
    (pandas.Series, int)
        Float rating series clipped to the valid 1-5 range and anomaly count.
    """
    raw = series.astype(str).str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")
    n_anomalies = int(numeric.isna().sum() - series.isna().sum())
    # Ratings are officially bound to the 1..5 scale; hard-clip defensively.
    numeric = numeric.clip(lower=1.0, upper=5.0).astype("float64")
    return numeric, n_anomalies


def clean_rating_count(series):
    """
    Strip thousands separators from ``rating_count`` and cast to float64.

    Parameters
    ----------
    series : pandas.Series
        Raw rating-count strings such as ``"24,269"``.

    Returns
    -------
    (pandas.Series, int)
        Float series and the number of coerced (anomalous) values.
    """
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    numeric = pd.to_numeric(cleaned, errors="coerce").astype("float64")
    n_anomalies = int(numeric.isna().sum() - series.isna().sum())
    return numeric, n_anomalies


# ------------------------------------------------------------------------------
# 5. Category parsing
# ------------------------------------------------------------------------------
def parse_category_columns(df):
    """
    Split the hierarchical ``category`` field into its top three levels.

    Example
    -------
    "Computers&Accessories|Accessories&Peripherals|Cables&Accessories|Cables"
        -> category_main  = "Computers&Accessories"
        -> category_sub   = "Accessories&Peripherals"
        -> category_micro = "Cables"

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataframe carrying the ``category`` column.

    Returns
    -------
    (pandas.DataFrame, int)
        Copy of the dataframe with the three new columns plus the number of
        rows where the category string was empty / split failed.
    """
    out = df.copy()
    parts = out["category"].astype(str).str.split("|")

    out["category_main"] = parts.str[0].str.strip()
    # Sub-category: second level when available, otherwise the main category.
    out["category_sub"] = parts.str[1].str.strip().where(
        parts.str.len() >= 2, out["category_main"]
    )
    # Micro-category: the deepest level of the hierarchy.
    out["category_micro"] = parts.apply(
        lambda p: p[-1].strip() if p else np.nan
    )

    # Replace empties with a safe token so downstream one-hot encoding works.
    for col in ("category_main", "category_sub", "category_micro"):
        out[col] = out[col].replace("", "Unknown").fillna("Unknown")

    n_parse_failures = int(out["category_main"].eq("Unknown").sum())
    return out, n_parse_failures


def aggregate_small_categories(df, column="category_main", threshold=5,
                               other_label="Other"):
    """
    Re-label small categories (fewer than ``threshold`` products) as "Other".

    Rare categories carry almost no statistical signal and destabilise
    one-hot-encoded feature spaces during modelling; rolling them into a
    single bucket keeps the pipeline robust.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe carrying ``column``.
    column : str
        Categorical column to normalise.
    threshold : int
        Minimum product count a category must reach to keep its own label.
    other_label : str
        Label given to the merged small categories.

    Returns
    -------
    pandas.DataFrame
        Modified copy of the dataframe (safe aggregation semantics).
    """
    out = df.copy()
    counts = out[column].value_counts()
    small = set(counts[counts < threshold].index)
    if small:
        out[column] = out[column].where(~out[column].isin(small), other_label)
    return out


# ------------------------------------------------------------------------------
# 6. Pricing-tier engineering
# ------------------------------------------------------------------------------
def add_price_tier(df, price_column="actual_price"):
    """
    Bucket ``actual_price`` into ordered, business-friendly price tiers.

    Tiers are defined on the list (MRP) price because that is the reference
    price customers compare the discounted price against.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataframe.
    price_column : str
        Column holding the list price.

    Returns
    -------
    pandas.DataFrame
        Copy of the dataframe with a new ordered categorical ``price_tier``.
    """
    out = df.copy()
    out["price_tier"] = pd.cut(
        out[price_column],
        bins=PRICE_TIER_BINS,
        labels=PRICE_TIER_LABELS,
        right=False,
    )
    missing_tier = out["price_tier"].isna()
    if missing_tier.any():
        out.loc[missing_tier, "price_tier"] = PRICE_TIER_LABELS[0]
    out["price_tier"] = out["price_tier"].astype("category")
    return out


# ------------------------------------------------------------------------------
# 7. Text cleaning & sentiment analysis
# ------------------------------------------------------------------------------
_TEXT_CLEAN_RE = re.compile(r"[^a-zA-Z\s]")


def normalise_text(text):
    """Lowercase and strip any non-alpha characters for safe tokenisation."""
    if not isinstance(text, str):
        return ""
    text = _TEXT_CLEAN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def vader_sentiment(text, analyzer):
    """
    Return the VADER compound sentiment score in [-1, +1] for ``text``.

    VADER integrity rule: the RAW string is scored directly. Punctuation
    (e.g. "!" emphasis), capitalisation and emoticons are integral parts of
    the VADER grammar, so the text must NOT be lower-cased, stripped of
    punctuation or otherwise normalised before scoring.

    A missing/empty text produces 0.0 (neutral) so the numeric column always
    stays dense.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return float(analyzer.polarity_scores(text)["compound"])


def textblob_polarity(text):
    """
    Return the TextBlob polarity in [-1, +1] for ``text`` (optional feature).

    Returns ``NaN`` when TextBlob is unavailable or raises, keeping the column
    nullable instead of crashing the pipeline.
    """
    if not HAS_TEXTBLOB:
        return np.nan
    if not isinstance(text, str) or not text.strip():
        return np.nan
    try:
        return float(_TextBlob(text).sentiment.polarity)
    except Exception:
        return np.nan


def analyse_word_frequency(text_series, top_n=30):
    """
    Count word occurrences across a corpus of review text.

    Stop-words and tokens shorter than two characters are discarded. The result
    is returned as a 2-column dataframe (word, count) sorted descending.

    Parameters
    ----------
    text_series : pandas.Series
        Review titles / contents.
    top_n : int
        Number of most frequent words to keep.

    Returns
    -------
    pandas.DataFrame
        ``word`` / ``count`` dataframe ordered by frequency.
    """
    tokenizer = RegexpTokenizer(r"[a-z]{2,}")
    word_bag = []
    for text in text_series.astype(str):
        word_bag.extend(tokenizer.tokenize(normalise_text(text).lower()))
    try:
        stop = set(stopwords.words("english"))
    except Exception:  # pragma: no cover - corpus download fallback
        stop = set()
    counts = pd.Series(word_bag).value_counts()
    counts = counts[~counts.index.isin(stop)]
    return counts.head(top_n).rename_axis("word").rename("count").reset_index()


def sentiment_label(compound):
    """Bucket a VADER compound score into Positive / Neutral / Negative."""
    if compound >= POSITIVE_THRESHOLD:
        return "Positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


# ------------------------------------------------------------------------------
# 8. Exploratory data analysis helpers
# ------------------------------------------------------------------------------
def summary_statistics(df, numeric_columns):
    """
    Compute a full summary-statistics table for every numeric feature.

    Statistics included: count, mean, median (50%), standard deviation,
    skewness, minimum and maximum.
    """
    rows = {}
    for col in numeric_columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        rows[col] = {
            "count": int(s.count()),
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std(ddof=1)), 4),
            "skewness": round(float(skew(s, bias=False)), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
        }
    return pd.DataFrame.from_dict(rows, orient="index").reset_index().rename(
        columns={"index": "feature"}
    )


def correlation_matrix(df, numeric_columns):
    """Pearson correlation matrix of the pricing / discount / rating columns."""
    corr = df[numeric_columns].corr(method="pearson")
    return corr.round(4)


def price_tier_summary(df):
    """Aggregate average rating, discount and volume per pricing tier."""
    agg = df.groupby("price_tier", observed=True).agg(
        product_count=("product_id", "nunique"),
        avg_actual_price=("actual_price", "mean"),
        avg_discounted_price=("discounted_price", "mean"),
        avg_discount=("discount_percentage", "mean"),
        avg_rating=("rating", "mean"),
        avg_rating_count=("rating_count", "mean"),
    ).round(2)
    return agg.reset_index()


def category_summary(df):
    """Per main-category aggregates used for business-level comparisons."""
    agg = df.groupby("category_main", observed=True).agg(
        product_count=("product_id", "nunique"),
        total_sales_volume=("rating_count", "sum"),
        avg_discounted_price=("discounted_price", "mean"),
        avg_discount=("discount_percentage", "mean"),
        avg_rating=("rating", "mean"),
    ).round(2)
    return agg.sort_values(
        ["total_sales_volume", "product_count"], ascending=False
    ).reset_index()


def sentiment_summary(df, sentiment_col="review_sentiment_class"):
    """Produce the Positive/Neutral/Negative count and share table."""
    counts = df[sentiment_col].value_counts()
    table = pd.DataFrame({
        "count": counts.fillna(0),
        "percentage": (counts.fillna(0) / counts.sum() * 100).round(2),
    })
    order = ["Positive", "Neutral", "Negative"]
    return table.reindex(order).fillna({"count": 0.0, "percentage": 0.0}
                                       ).reset_index().rename(
                                           columns={"index": "sentiment"})


# ------------------------------------------------------------------------------
# 9. Core cleaning orchestration
# ------------------------------------------------------------------------------
def clean_dataframe(df, log):
    """Run every cleaning / type-casting step and log its outcome."""
    log.log("\n[STEP 1] Cleaning monetary columns ...")
    df["discounted_price"], n1 = clean_currency_column(df["discounted_price"],
                                                       "discounted_price")
    df["actual_price"], n2 = clean_currency_column(df["actual_price"],
                                                   "actual_price")
    log.log(f"   - discounted_price: {n1} anomalous values coerced to NaN.")
    log.log(f"   - actual_price:     {n2} anomalous values coerced to NaN.")

    log.log("\n[STEP 2] Parsing discount percentages ...")
    df["discount_percentage"], n3 = clean_discount_percentage(
        df["discount_percentage"])
    log.log(f"   - {n3} anomalous discount values coerced to NaN (imputed later).")

    log.log("\n[STEP 3] Cleaning rating column ...")
    df["rating"], n4 = clean_rating_column(df["rating"])
    log.log(f"   - {n4} erratic rating tokens coerced to NaN.")

    log.log("\n[STEP 4] Cleaning rating_count column ...")
    df["rating_count"], n5 = clean_rating_count(df["rating_count"])
    log.log(f"   - {n5} rating-count values coerced to NaN.")

    log.log("\n[STEP 5] Parsing hierarchical category field ...")
    df, n6 = parse_category_columns(df)
    log.log(f"   - Categories split into main / sub / micro ({n6} empty rows).")
    # Keep the original, un-merged main category for richer EDA plots.
    df["category_main_raw"] = df["category_main"].copy()
    df = aggregate_small_categories(df, threshold=5, other_label="Other")
    log.log("   - Categories with fewer than 5 products merged into 'Other'.")

    log.log("\n[STEP 6] Missing-value accounting (imputation deferred to ML "
            "pipelines) ...")
    modelling_cols = ["actual_price", "discounted_price",
                      "discount_percentage", "rating", "rating_count"]
    remaining = df[modelling_cols].isna().sum()
    if remaining.sum() == 0:
        log.log("   - No missing values remain in the modelling columns.")
    else:
        for col, n in remaining[remaining > 0].items():
            log.log(f"   - {col}: {n} missing value(s) preserved as NaN and "
                    f"imputed in-pipeline (SimpleImputer, training folds only).")

    log.log("\n[STEP 7] Engineering price tiers ...")
    df = add_price_tier(df)
    log.log(f"   - Added price_tier with {len(PRICE_TIER_LABELS)} tiers.")

    # Discount in % is friendlier for reporting than the 0-1 fraction.
    df["discount_percentage_display"] = (df["discount_percentage"] * 100).round(1)
    return df


# ------------------------------------------------------------------------------
# 10. Sentiment & text enrichment
# ------------------------------------------------------------------------------
def enrich_text_features(df, log):
    """Compute review text sentiment and length features for every row."""
    log.log("\n[STEP 8] Running sentiment analysis (VADER + TextBlob) ...")
    try:
        nltk.data.path.append(
            os.path.join(os.environ.get("APPDATA", ""), "nltk_data"))
        analyzer = SentimentIntensityAnalyzer()
    except Exception as err:  # pragma: no cover - lexicon download fallback
        log.log(f"   ! VADER lexicon unavailable ({err}); installing ...")
        nltk.download("vader_lexicon")
        analyzer = SentimentIntensityAnalyzer()

    # Raw combined review string (title + content). VADER requires the raw
    # text WITH punctuation and casing intact - no cleaning is applied here.
    review_text = (df["review_title"].fillna("")
                   + " " + df["review_content"].fillna(""))

    df["review_sentiment_vader"] = review_text.apply(
        lambda t: vader_sentiment(t, analyzer))
    df["review_sentiment_class"] = df["review_sentiment_vader"].apply(
        sentiment_label)
    if HAS_TEXTBLOB:
        df["review_polarity_textblob"] = review_text.apply(textblob_polarity)
    else:
        df["review_polarity_textblob"] = np.nan

    # Text-length features used later for machine learning.
    df["review_text_length"] = review_text.str.len()
    df["about_product_length"] = (df["about_product"].fillna("")
                                  .astype(str).str.len())
    df["product_name_length"] = (df["product_name"].fillna("")
                                 .astype(str).str.len())

    log.log("   - VADER compound scores computed for every review row.")
    log.log("   - TextBlob polarity computed where available.")
    return df


# ------------------------------------------------------------------------------
# 11. EDA runner
# ------------------------------------------------------------------------------
def run_eda(df, log):
    """Produce and persist every structured EDA artifact."""
    log.log("\n" + "=" * 80)
    log.log("EXPLORATORY DATA ANALYSIS")
    log.log("=" * 80)

    # 11.1 Summary statistics
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    stats = summary_statistics(df, num_cols)
    stats.to_csv(os.path.join(EDA_DIR, "summary_statistics.csv"), index=False)
    log.log("\n[EDA-1] Summary statistics (saved: summary_statistics.csv)")
    log.log(stats.to_string(index=False))

    # 11.2 Correlation matrix
    corr = correlation_matrix(df, num_cols)
    corr.to_csv(os.path.join(EDA_DIR, "correlation_matrix.csv"))
    log.log("\n[EDA-2] Correlation matrix (saved: correlation_matrix.csv)")
    log.log(corr.to_string())

    # 11.3 Category summary
    cat = category_summary(df)
    cat.to_csv(os.path.join(EDA_DIR, "category_summary.csv"), index=False)
    log.log("\n[EDA-3] Category summary (saved: category_summary.csv)")
    log.log(cat.to_string(index=False))

    # 11.4 Price-tier summary
    tier = price_tier_summary(df)
    tier.to_csv(os.path.join(EDA_DIR, "price_tier_summary.csv"), index=False)
    log.log("\n[EDA-4] Price-tier summary (saved: price_tier_summary.csv)")
    log.log(tier.to_string(index=False))

    # 11.5 Word frequency
    words = analyse_word_frequency(
        df["review_title"].fillna("") + " " + df["review_content"].fillna(""),
        top_n=30,
    )
    words.to_csv(os.path.join(EDA_DIR, "word_frequency.csv"), index=False)
    log.log("\n[EDA-5] Top-30 review words (saved: word_frequency.csv)")
    log.log(words.to_string(index=False))

    # 11.6 Sentiment summary
    senti = sentiment_summary(df)
    senti.to_csv(os.path.join(EDA_DIR, "sentiment_summary.csv"), index=False)
    log.log("\n[EDA-6] Sentiment distribution (saved: sentiment_summary.csv)")
    log.log(senti.to_string(index=False))

    # 11.7 Rating distribution quantiles
    rating_q = df["rating"].describe(percentiles=[0.25, 0.5, 0.75, 0.9])
    rating_q.to_csv(os.path.join(EDA_DIR, "rating_distribution.csv"))
    log.log("\n[EDA-7] Rating distribution (saved: rating_distribution.csv)")
    log.log(rating_q.to_string())


# ------------------------------------------------------------------------------
# 12. Data-quality report
# ------------------------------------------------------------------------------
def initial_data_quality_report(raw_df, log):
    """Report the shape, dtypes and missing-value profile of the raw input."""
    log.log("=" * 80)
    log.log("DATA INGESTION & DATA-QUALITY REPORT (raw)")
    log.log("=" * 80)
    log.log(f"   Total records           : {raw_df.shape[0]}")
    log.log(f"   Columns                 : {raw_df.shape[1]}")
    log.log(f"   Duplicate product ids   : {int(raw_df['product_id'].duplicated().sum())}")
    log.log(f"   Unique products         : {int(raw_df['product_id'].nunique())}")

    missing = raw_df.isna().sum()
    missing = missing[missing > 0] if missing.any() else pd.Series(dtype=int)
    log.log("\n   Missing values (raw):")
    if missing.empty:
        log.log("      - none")
    else:
        log.log(missing.to_string())

    log.log("\n   Dtypes (raw):")
    log.log(raw_df.dtypes.to_string())

    # Persist a snapshot of the raw missing-value profile.
    missing_report = raw_df.isna().sum().rename("missing_count").reset_index()
    missing_report["missing_pct"] = (raw_df.isna().mean() * 100).round(2).to_numpy()
    missing_report.columns = ["column", "missing_count", "missing_pct"]
    missing_report.to_csv(
        os.path.join(EDA_DIR, "missing_values_initial.csv"), index=False)


# ------------------------------------------------------------------------------
# 13. Main entry point
# ------------------------------------------------------------------------------
def main():
    """Execute the full ingest -> clean -> EDA -> persist workflow."""
    ensure_directories()
    log = Tee(LOG_PATH)

    log.log("=" * 80)
    log.log("01_eda_and_cleaning.py - Amazon Sales Dataset pipeline")
    log.log("=" * 80)

    # Ingest.
    raw = load_raw_data(RAW_DATA_PATH)
    initial_data_quality_report(raw, log)

    # Clean.
    df = raw.copy()
    df = clean_dataframe(df, log)
    df = enrich_text_features(df, log)

    # EDA.
    run_eda(df, log)

    # Persist full cleaned dataset (every review row).
    df.to_csv(CLEAN_ALL_PATH, index=False)
    log.log(f"\n[SAVE] Full cleaned dataset  -> {CLEAN_ALL_PATH}")

    # Persist the product-level (deduplicated) dataset for modelling & plots.
    products = df.drop_duplicates(subset="product_id", keep="first").copy()
    products.to_csv(CLEAN_PRODUCTS_PATH, index=False)
    log.log(f"[SAVE] Product-level dataset -> {CLEAN_PRODUCTS_PATH}")
    log.log(
        f"\nFinished. Cleaned {df.shape[0]} review rows covering "
        f"{products.shape[0]} unique products."
    )
    log.close()


if __name__ == "__main__":
    main()