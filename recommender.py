"""
recommender.py
──────────────
Core logic for the Hybrid AutoML Algorithm Recommender.

Three-layer hybrid approach:
  1. Rule-based meta-feature scoring
  2. Similarity matching against reference dataset profiles
  3. Smart weighted fusion of both signals
"""

import pandas as pd
import numpy as np


# ── Reference dataset profiles (simulates FAISS similarity search) ──────────
# Each profile represents a "known" dataset with the best algorithm for it.
REFERENCE_PROFILES = [
    # (n_samples, n_features, missing_pct, imbalance_ratio, n_numeric_ratio, best_algo)
    {"tag": "small_clean_cls",     "n_s": 200,    "n_f": 8,   "miss": 0,  "imb": 1.0, "num_r": 1.0, "algo": "Logistic Regression"},
    {"tag": "medium_balanced_cls", "n_s": 5000,   "n_f": 15,  "miss": 2,  "imb": 1.2, "num_r": 0.8, "algo": "Random Forest"},
    {"tag": "large_cls",           "n_s": 50000,  "n_f": 20,  "miss": 5,  "imb": 1.5, "num_r": 0.7, "algo": "Gradient Boosting (XGBoost)"},
    {"tag": "imbalanced_cls",      "n_s": 10000,  "n_f": 12,  "miss": 3,  "imb": 10,  "num_r": 0.9, "algo": "Gradient Boosting (XGBoost)"},
    {"tag": "high_dim_cls",        "n_s": 3000,   "n_f": 100, "miss": 1,  "imb": 1.1, "num_r": 1.0, "algo": "SVM (RBF Kernel)"},
    {"tag": "tiny_cls",            "n_s": 100,    "n_f": 5,   "miss": 0,  "imb": 1.0, "num_r": 0.6, "algo": "K-Nearest Neighbors"},
    {"tag": "cat_heavy_cls",       "n_s": 8000,   "n_f": 20,  "miss": 4,  "imb": 1.3, "num_r": 0.2, "algo": "Gradient Boosting (XGBoost)"},
    {"tag": "small_reg",           "n_s": 300,    "n_f": 6,   "miss": 0,  "imb": 1.0, "num_r": 1.0, "algo": "Linear Regression"},
    {"tag": "medium_reg",          "n_s": 6000,   "n_f": 18,  "miss": 3,  "imb": 1.0, "num_r": 0.9, "algo": "Random Forest Regressor"},
    {"tag": "large_reg",           "n_s": 80000,  "n_f": 25,  "miss": 6,  "imb": 1.0, "num_r": 0.8, "algo": "Gradient Boosting (XGBoost)"},
    {"tag": "noisy_reg",           "n_s": 2000,   "n_f": 10,  "miss": 15, "imb": 1.0, "num_r": 0.7, "algo": "Random Forest Regressor"},
    {"tag": "linear_reg",          "n_s": 1000,   "n_f": 8,   "miss": 0,  "imb": 1.0, "num_r": 1.0, "algo": "Ridge Regression"},
]


# ── ALGORITHM METADATA ────────────────────────────────────────────────────────
ALGO_META = {
    # Classification
    "Logistic Regression": {
        "type": "classification",
        "strengths": ["linearly separable data", "small datasets", "interpretable"],
        "weaknesses": ["non-linear patterns", "many features"],
        "reason": "Works great for clean, small, linearly separable datasets with numeric features.",
    },
    "Random Forest": {
        "type": "both",
        "strengths": ["handles missing values", "non-linear", "robust to outliers"],
        "weaknesses": ["large datasets (slow)", "memory heavy"],
        "reason": "Excellent all-rounder — handles mixed data types, missing values, and non-linear patterns.",
    },
    "Gradient Boosting (XGBoost)": {
        "type": "both",
        "strengths": ["imbalanced data", "large datasets", "wins Kaggle"],
        "weaknesses": ["many hyperparameters", "slow to train"],
        "reason": "State-of-the-art for structured data. Best for imbalanced or large datasets with complex patterns.",
    },
    "SVM (RBF Kernel)": {
        "type": "both",
        "strengths": ["high-dimensional", "small-medium datasets", "robust margin"],
        "weaknesses": ["slow on large data", "needs scaling"],
        "reason": "Powerful for high-dimensional feature spaces with clear class separation.",
    },
    "K-Nearest Neighbors": {
        "type": "both",
        "strengths": ["very small datasets", "no training phase", "simple"],
        "weaknesses": ["slow prediction", "high dimensions", "sensitive to noise"],
        "reason": "Simple and effective for tiny, clean datasets where patterns are local.",
    },
    "Decision Tree": {
        "type": "both",
        "strengths": ["interpretable", "categorical data", "fast"],
        "weaknesses": ["overfitting", "unstable"],
        "reason": "Highly interpretable — great when explainability matters more than peak accuracy.",
    },
    # Regression
    "Linear Regression": {
        "type": "regression",
        "strengths": ["linear relationship", "small datasets", "fast"],
        "weaknesses": ["non-linear patterns", "outliers"],
        "reason": "Perfect baseline for regression when the target has a linear relationship with features.",
    },
    "Ridge Regression": {
        "type": "regression",
        "strengths": ["multicollinearity", "many features", "regularization"],
        "weaknesses": ["non-linear patterns"],
        "reason": "Linear regression with L2 regularization — ideal when features are correlated.",
    },
    "Random Forest Regressor": {
        "type": "regression",
        "strengths": ["non-linear", "robust", "handles missing"],
        "weaknesses": ["memory", "extrapolation"],
        "reason": "Robust non-linear regression that handles noisy data and missing values well.",
    },
    "Gradient Boosting (XGBoost)": {
        "type": "both",
        "strengths": ["large datasets", "complex patterns", "state-of-the-art"],
        "weaknesses": ["hyperparameter tuning", "training time"],
        "reason": "Top performer for large, complex regression problems with many features.",
    },
}


# ── STEP 1: ANALYZE DATASET ───────────────────────────────────────────────────
def analyze_dataset(df: pd.DataFrame, target_col: str, task_type: str) -> dict:
    """Extract meta-features from the dataset."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    n_samples    = len(df)
    n_features   = X.shape[1]
    n_numeric    = X.select_dtypes(include=np.number).shape[1]
    n_categorical= X.select_dtypes(include="object").shape[1]
    missing_pct  = (X.isnull().sum().sum() / (X.shape[0] * X.shape[1])) * 100
    num_ratio    = n_numeric / n_features if n_features > 0 else 1.0

    # Imbalance ratio (classification only)
    imbalance_ratio = 1.0
    if task_type == "Classification":
        vc = y.value_counts()
        imbalance_ratio = round(vc.max() / vc.min(), 2) if vc.min() > 0 else 1.0

    # Feature-to-sample ratio
    feature_sample_ratio = round(n_features / n_samples, 4)

    # Numeric correlations
    num_df = X.select_dtypes(include=np.number)
    avg_corr = 0.0
    if num_df.shape[1] > 1:
        corr_mat = num_df.corr().abs()
        upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
        avg_corr = round(upper.stack().mean(), 4)

    # Generate human-readable insights
    insights = []
    if n_samples < 500:
        insights.append(f"Small dataset ({n_samples} rows) — prefer simple, low-variance models.")
    elif n_samples > 50000:
        insights.append(f"Large dataset ({n_samples:,} rows) — tree-based boosting models will shine.")
    else:
        insights.append(f"Medium dataset ({n_samples:,} rows) — Random Forest or boosting are solid choices.")

    if missing_pct > 10:
        insights.append(f"High missing data ({missing_pct:.1f}%) — tree-based models handle this better than linear models.")
    elif missing_pct > 0:
        insights.append(f"Some missing values ({missing_pct:.1f}%) — imputation recommended before linear models.")

    if task_type == "Classification" and imbalance_ratio > 3:
        insights.append(f"Imbalanced classes (ratio {imbalance_ratio}:1) — use class_weight='balanced' or SMOTE.")

    if n_features > 50:
        insights.append(f"High dimensionality ({n_features} features) — SVM or regularized models may help.")

    if avg_corr > 0.6:
        insights.append(f"High feature correlations (avg={avg_corr}) — consider Ridge/Lasso or feature selection.")

    if num_ratio < 0.3:
        insights.append(f"Mostly categorical features — Gradient Boosting handles encoding internally.")

    if feature_sample_ratio > 0.1:
        insights.append("High feature-to-sample ratio — regularized models reduce overfitting risk.")

    return {
        "n_samples"           : n_samples,
        "n_features"          : n_features,
        "n_numeric"           : n_numeric,
        "n_categorical"       : n_categorical,
        "missing_pct"         : round(missing_pct, 2),
        "imbalance_ratio"     : imbalance_ratio,
        "num_ratio"           : round(num_ratio, 3),
        "avg_correlation"     : avg_corr,
        "feature_sample_ratio": feature_sample_ratio,
        "insights"            : insights,
    }


# ── STEP 2: SIMILARITY SEARCH ────────────────────────────────────────────────
def _dataset_similarity(stats: dict, ref: dict) -> float:
    """Compute similarity score between current dataset and a reference profile (0–1)."""
    def norm_diff(a, b, scale):
        return 1 - min(abs(a - b) / scale, 1)

    s1 = norm_diff(np.log1p(stats["n_samples"]),  np.log1p(ref["n_s"]),  5)
    s2 = norm_diff(np.log1p(stats["n_features"]), np.log1p(ref["n_f"]),  4)
    s3 = norm_diff(stats["missing_pct"],           ref["miss"],           30)
    s4 = norm_diff(min(stats["imbalance_ratio"], 20), min(ref["imb"], 20), 20)
    s5 = norm_diff(stats["num_ratio"],             ref["num_r"],           1)

    return round((s1 * 0.35 + s2 * 0.25 + s3 * 0.15 + s4 * 0.15 + s5 * 0.10), 4)


def _similarity_votes(stats: dict, task_type: str) -> dict:
    """Weighted votes from top-3 similar reference datasets."""
    scores = []
    for ref in REFERENCE_PROFILES:
        sim = _dataset_similarity(stats, ref)
        scores.append((sim, ref["algo"]))

    scores.sort(key=lambda x: -x[0])
    top3 = scores[:3]

    votes = {}
    for sim, algo in top3:
        votes[algo] = votes.get(algo, 0) + sim
    return votes


# ── STEP 3: META-LEARNING RULE ENGINE ────────────────────────────────────────
def _meta_learning_scores(stats: dict, task_type: str) -> dict:
    """
    Rule-based meta-learning: assigns scores to each algorithm
    based on dataset characteristics.
    Returns dict {algo_name: score (0-100)}.
    """
    s   = stats
    is_cls = task_type == "Classification"

    scores = {}

    if is_cls:
        # Logistic Regression
        lr = 50
        if s["n_samples"] < 1000:  lr += 15
        if s["missing_pct"] < 2:   lr += 10
        if s["num_ratio"] > 0.8:   lr += 10
        if s["imbalance_ratio"] < 2: lr += 10
        if s["n_features"] > 50:   lr -= 15
        if s["avg_correlation"] > 0.5: lr += 5
        scores["Logistic Regression"] = max(lr, 5)

        # Random Forest
        rf = 60
        if 500 <= s["n_samples"] <= 50000: rf += 15
        if s["missing_pct"] > 5:           rf += 10
        if s["n_categorical"] > 0:         rf += 5
        if s["imbalance_ratio"] < 5:       rf += 5
        if s["n_samples"] > 100000:        rf -= 10
        scores["Random Forest"] = max(rf, 5)

        # Gradient Boosting
        gb = 65
        if s["n_samples"] > 5000:          gb += 15
        if s["imbalance_ratio"] > 3:       gb += 15
        if s["missing_pct"] > 5:           gb += 5
        if s["n_categorical"] > 3:         gb += 5
        if s["n_samples"] < 300:           gb -= 20
        scores["Gradient Boosting (XGBoost)"] = max(gb, 5)

        # SVM
        svm = 45
        if s["n_features"] > 30:           svm += 15
        if s["n_samples"] < 10000:         svm += 10
        if s["missing_pct"] < 1:           svm += 10
        if s["n_samples"] > 50000:         svm -= 20
        scores["SVM (RBF Kernel)"] = max(svm, 5)

        # KNN
        knn = 30
        if s["n_samples"] < 500:           knn += 20
        if s["n_features"] < 10:           knn += 10
        if s["missing_pct"] < 1:           knn += 10
        if s["n_samples"] > 10000:         knn -= 25
        if s["n_features"] > 20:           knn -= 15
        scores["K-Nearest Neighbors"] = max(knn, 5)

        # Decision Tree
        dt = 40
        if s["n_samples"] < 2000:          dt += 10
        if s["n_categorical"] > 3:         dt += 10
        if s["missing_pct"] < 5:           dt += 5
        scores["Decision Tree"] = max(dt, 5)

    else:  # Regression
        # Linear Regression
        lr = 50
        if s["n_samples"] < 1000:          lr += 15
        if s["avg_correlation"] < 0.4:     lr += 10
        if s["num_ratio"] > 0.9:           lr += 10
        if s["missing_pct"] < 2:           lr += 10
        scores["Linear Regression"] = max(lr, 5)

        # Ridge
        ridge = 48
        if s["avg_correlation"] > 0.5:     ridge += 20
        if s["n_features"] > 20:           ridge += 10
        if s["missing_pct"] < 3:           ridge += 5
        scores["Ridge Regression"] = max(ridge, 5)

        # RF Regressor
        rf = 60
        if 500 <= s["n_samples"] <= 50000: rf += 15
        if s["missing_pct"] > 5:           rf += 10
        if s["n_categorical"] > 0:         rf += 5
        scores["Random Forest Regressor"] = max(rf, 5)

        # XGBoost
        gb = 65
        if s["n_samples"] > 5000:          gb += 15
        if s["missing_pct"] > 5:           gb += 5
        if s["n_categorical"] > 3:         gb += 5
        if s["n_samples"] < 300:           gb -= 20
        scores["Gradient Boosting (XGBoost)"] = max(gb, 5)

    return scores


# ── STEP 4: HYBRID FUSION ─────────────────────────────────────────────────────
def recommend_algorithm(stats: dict, task_type: str) -> list:
    """
    Fuse meta-learning scores + similarity votes into final ranked list.
    Returns list of dicts sorted by confidence (desc).
    """
    meta_scores = _meta_learning_scores(stats, task_type)
    sim_votes   = _similarity_votes(stats, task_type)

    # Normalize meta scores to 0-100
    max_meta = max(meta_scores.values()) if meta_scores else 1
    meta_norm = {k: (v / max_meta) * 100 for k, v in meta_scores.items()}

    # Normalize similarity votes to 0-100
    max_sim = max(sim_votes.values()) if sim_votes else 1
    sim_norm = {k: (v / max_sim) * 100 for k, v in sim_votes.items()}

    # Weighted fusion: 60% meta-learning + 40% similarity
    all_algos = set(meta_norm.keys()) | set(sim_norm.keys())
    fused = {}
    for algo in all_algos:
        m = meta_norm.get(algo, 0)
        si = sim_norm.get(algo, 0)
        fused[algo] = round(m * 0.60 + si * 0.40, 1)

    # Sort descending
    ranked = sorted(fused.items(), key=lambda x: -x[1])

    # Scale so top score = ~92 (realistic, not 100%)
    top_raw = ranked[0][1] if ranked else 1
    results = []
    for algo, raw_score in ranked:
        conf = round((raw_score / top_raw) * 92, 1)
        meta = ALGO_META.get(algo, {})
        results.append({
            "algorithm" : algo,
            "confidence": conf,
            "reason"    : meta.get("reason", ""),
            "strengths" : meta.get("strengths", []),
            "weaknesses": meta.get("weaknesses", []),
        })

    return results
