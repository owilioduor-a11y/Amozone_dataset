# -*- coding: utf-8 -*-
"""
================================================================================
02_data_visualization.py
================================================================================
Module 2 of the "Amazon Sales Dataset" end-to-end analytics pipeline.

This module loads the cleaned dataset produced by ``01_eda_and_cleaning.py``
and generates a set of high-resolution (300 DPI) static charts that together
summarise the pricing, discount, category and sentiment story of the dataset:

  1. Price Distribution Plot   - histogram + KDE of actual vs discounted price.
  2. Discount vs Rating Scatter- rating vs discount with size ~ rating_count.
  3. Category Breakdown        - top-10 main categories by volume / product count.
  4. Correlation Heatmap       - Pearson correlations of numeric features.
  5. Rating vs Sales Box Plot  - rating distributions across top categories.
  6. Sentiment Distribution    - Positive / Neutral / Negative review shares.

Use ``matplotlib.use("Agg")`` so charts can be rendered on a headless
backend. Every figure is saved to ``outputs/figures/`` as a 300-DPI PNG.

Input : ``data/amazon_cleaned.csv``   (from script #1)
Output: ``outputs/figures/``          (six PNG charts)

Usage:
    python 02_data_visualization.py
================================================================================
"""

# ------------------------------------------------------------------------------
# Standard library imports
# ------------------------------------------------------------------------------
import os
from matplotlib.ticker import FuncFormatter

# ------------------------------------------------------------------------------
# Third-party imports
# ------------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")  # headless backend (no interactive display needed)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ------------------------------------------------------------------------------
# Path configuration
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")

CLEAN_ALL_PATH = os.path.join(DATA_DIR, "amazon_cleaned.csv")
CLEAN_PRODUCTS_PATH = os.path.join(DATA_DIR, "amazon_cleaned_products.csv")

RANDOM_SEED = 42

# Shared matplotlib style: clean white grids and a consistent colour palette.
PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def ensure_directories():
    """Create the output directory for figures if it does not exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data():
    """
    Load the cleaned datasets produced by module #1.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        ``df_all`` (every review row) and ``df_products`` (one row per unique
        product). Raises ``FileNotFoundError`` when the cleaning script has
        not been run yet.
    """
    if not os.path.exists(CLEAN_ALL_PATH):
        raise FileNotFoundError(
            "data/amazon_cleaned.csv not found. Please run module #1 "
            "(01_eda_and_cleaning.py) before generating visualisations."
        )
    df_all = pd.read_csv(CLEAN_ALL_PATH, encoding="utf-8-sig")
    df_products = pd.read_csv(CLEAN_PRODUCTS_PATH, encoding="utf-8-sig")
    return df_all, df_products


def rupiah_formatter(value, _pos):
    """Format chart tick labels like ``₹12,400`` (Indian Rupee)."""
    if value >= 1_000_000:
        return f"₹{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"₹{value / 1000:,.0f}k"
    return f"₹{value:,.0f}"


def fig1_price_distribution(df_products):
    """
    Histogram + KDE of ``actual_price`` vs ``discounted_price``.

    Prices are heavily right-skewed (median ~₹800, maximum ~₹1.4L), so panel A
    uses log-spaced histogram bins while panel B shows KDE curves estimated on
    the log-transformed prices. Both panels therefore stay readable.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: log-scaled histograms of the raw rupee prices.
    bins = np.logspace(np.log10(30), np.log10(150_000), 55)
    axes[0].hist(df_products["actual_price"], bins=bins, alpha=0.5,
                 color="#c44e52", label="Actual (list) price")
    axes[0].hist(df_products["discounted_price"], bins=bins, alpha=0.5,
                 color="#55a868", label="Discounted price")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Price (₹, log scale)")
    axes[0].set_ylabel("Number of products")
    axes[0].xaxis.set_major_formatter(FuncFormatter(rupiah_formatter))
    axes[0].set_title("Histogram (log scale)")
    axes[0].legend()

    # Panel B: KDE over log1p-transformed prices (readable overlap).
    sns.kdeplot(np.log1p(df_products["actual_price"]), ax=axes[1],
                color="#c44e52", label="Actual price", warn_singular=False)
    sns.kdeplot(np.log1p(df_products["discounted_price"]), ax=axes[1],
                color="#55a868", label="Discounted price", warn_singular=False)
    axes[1].set_xlabel("log(1 + price in ₹)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("KDE (log-transformed)")
    axes[1].legend()

    fig.suptitle("Distribution of Actual vs Discounted Prices", y=1.02,
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_price_distribution.png"))
    plt.close(fig)


def fig2_discount_vs_rating(df_products):
    """
    Scatter of ``discount_percentage`` vs ``rating`` with point size
    proportional to ``rating_count`` (sales demand).

    A trend line and the Pearson correlation are overlaid to summarise the
    observed relationship between deep discounts and star ratings.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    x = df_products["discount_percentage"] * 100
    y = df_products["rating"]
    max_count = df_products["rating_count"].max()
    sizes = (10 + 120 * df_products["rating_count"] / max_count).clip(upper=200)

    points = ax.scatter(x, y, s=sizes, alpha=0.4,
                        c=df_products["rating_count"], cmap="viridis")

    # Overlay a linear trend line. NaN-safe: rows with a missing rating are
    # preserved in the cleaned data, and np.polyfit / np.corrcoef cannot
    # handle NaN - fit only on complete (x, y) pairs.
    complete = x.notna() & y.notna()
    slope, intercept = np.polyfit(x[complete], y[complete], 1)
    xs = np.linspace(x[complete].min(), x[complete].max(), 100)
    ax.plot(xs, slope * xs + intercept, "--", color="black", lw=1.2,
            label="Linear trend")

    corr = np.corrcoef(x[complete], y[complete])[0, 1]
    ax.set_xlabel("Discount percentage (%)")
    ax.set_ylabel("Product rating (1-5)")
    ax.set_title("Discount vs Rating (point size ~ customer demand)")
    fig.colorbar(points, ax=ax, label="rating_count (demand)")
    ax.annotate(
        f"Pearson correlation: {corr:.2f}",
        xy=(0.02, 0.98), xycoords="axes fraction", fontsize=10,
    )
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_discount_vs_rating.png"))
    plt.close(fig)


def fig3_category_breakdown(df_products):
    """
    Top-10 main product categories by total sales volume and product count.

    Two horizontal panels make both business metrics readable on a single
    figure (volume spans multiple orders of magnitude, so it is log-scaled).
    """
    # category_main_raw holds the un-merged main category; fall back to the
    # first category segment when the column is absent (defensive).
    if "category_main_raw" in df_products.columns:
        cat_col = "category_main_raw"
    else:
        cat_col = "category_main"

    agg = (
        df_products.groupby(cat_col)
        .agg(product_count=("product_id", "count"),
             total_sales_volume=("rating_count", "sum"))
        .sort_values("total_sales_volume", ascending=False)
        .head(10)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    # Panel 1: total sales volume (log scale).
    sns.barplot(x=np.log10(agg["total_sales_volume"]), y=agg.index,
                ax=axes[0], palette="crest", hue=agg.index,
                legend=False)
    axes[0].set_title("Total sales volume (log10)")
    axes[0].set_xlabel("log10(total rating_count)")
    axes[0].set_ylabel("")

    # Panel 2: number of products.
    sns.barplot(x=agg["product_count"], y=agg.index,
                ax=axes[1], palette="flare", hue=agg.index,
                legend=False)
    axes[1].set_title("Product count")
    axes[1].set_xlabel("Number of products")

    fig.suptitle("Top-10 Main Product Categories: Demand vs Assortment",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_category_breakdown.png"))
    plt.close(fig)


def fig4_correlation_heatmap(df_products):
    """
    Annotated Pearson correlation heatmap of the pricing / discount / rating
    numeric features.
    """
    cols = ["discounted_price", "actual_price", "discount_percentage",
            "rating", "rating_count"]
    corr = df_products[cols].corr(method="pearson").round(2)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1,
                cbar_kws={"label": "Pearson correlation"}, ax=ax)
    ax.set_title("Correlation Heatmap of Numeric Features")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_correlation_heatmap.png"))
    plt.close(fig)


def fig5_rating_boxplot(df_products):
    """
    Box plot of ``rating`` across the top (sub-) product categories by volume.

    Restricting to the largest categories avoids degenerate one-product boxes
    and keeps the comparison focused on the categories that matter most.
    """
    top_cats = (
        df_products.groupby("category_sub")
        .agg(volume=("rating_count", "sum"))
        .sort_values("volume", ascending=False)
        .head(8)
        .index.tolist()
    )
    data = df_products[df_products["category_sub"].isin(top_cats)]
    medians = (
        data.groupby("category_sub")["rating"].median()
        .reindex(top_cats).sort_values(ascending=False)
    )
    order = medians.index.tolist()

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(x="category_sub", y="rating", data=data, order=order,
                hue="category_sub", hue_order=order, palette="Set2",
                legend=False, fliersize=3, ax=ax)
    ax.set_title("Rating Distribution Across Top Product Categories")
    ax.set_xlabel("Sub-category")
    ax.set_ylabel("Rating (1-5)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig5_rating_boxplot.png"))
    plt.close(fig)


def fig6_sentiment_distribution(df_all):
    """
    Bar + donut chart of the review sentiment distribution.

    Uses every review row (each user review is a distinct sentiment sample).
    """
    counts = df_all["review_sentiment_class"].value_counts()
    order = ["Positive", "Neutral", "Negative"]
    counts = counts.reindex(order).fillna(0)
    colors = ["#55a868", "#c4b445", "#c44e52"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                             gridspec_kw={"width_ratios": [1, 1.4]})

    # Left panel: bar plot with percentage labels.
    ticks = np.arange(len(order))
    pct = counts / counts.sum() * 100
    axes[0].bar(ticks, counts, color=colors)
    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels(order)
    for tick, cnt in enumerate(counts):
        axes[0].text(tick, cnt, f"{pct.iloc[tick]:.1f}%",
                     ha="center", va="bottom")
    axes[0].set_title("Sentiment counts")
    axes[0].set_ylabel("Number of reviews")

    # Right panel: donut chart.
    axes[1].pie(counts, labels=order, colors=colors, autopct="%1.1f%%",
                startangle=90, wedgeprops={"width": 0.4})
    axes[1].set_title("Sentiment share")
    axes[1].axis("equal")

    fig.suptitle("Customer Review Sentiment Distribution (VADER)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig6_sentiment_distribution.png"))
    plt.close(fig)


# ------------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------------
def main():
    """Generate and save all six visualisations at 300 DPI."""
    ensure_directories()
    plt.rcParams.update(PLOT_STYLE)
    np.random.seed(RANDOM_SEED)

    df_all, df_products = load_data()

    fig1_price_distribution(df_products)
    fig2_discount_vs_rating(df_products)
    fig3_category_breakdown(df_products)
    fig4_correlation_heatmap(df_products)
    fig5_rating_boxplot(df_products)
    fig6_sentiment_distribution(df_all)

    print(f"All figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()