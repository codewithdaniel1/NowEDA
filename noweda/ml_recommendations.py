"""
ML algorithm recommendations based on data characteristics.

Every algorithm always appears — with a rating, data-specific reasoning,
and concrete warnings. No hard-gating behind thresholds.
"""

import numpy as np
import pandas as pd


# ─── Helpers ────────────────────────────────────────────────────────────────

def _stars(score):
    """Convert 1-5 int to star string."""
    full = int(round(score))
    return "★" * full + "☆" * (5 - full)


def _profile(df, stats, schema, scores, results):
    """Build a rich data profile dict used by all recommenders."""
    missing = results.get("missing", {})
    outliers = results.get("outliers", {})
    corr = results.get("correlation", {})

    num_cols = [c for c in df.columns if df[c].dtype.kind in ("i", "u", "f")]
    cat_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "category"]
    n_rows, n_cols = len(df), len(df.columns)

    # Skewness: count of highly skewed numeric cols
    n_skewed = sum(
        1 for c in num_cols
        if c in stats and abs(stats[c].get("skewness", 0)) > 1
    )

    # Multicollinearity: any correlation pair > 0.85
    has_high_corr = False
    for c1 in corr:
        for c2, v in corr[c1].items():
            if c1 != c2 and abs(v) > 0.85:
                has_high_corr = True
                break

    # Missing: max pct across all columns
    max_missing_pct = max(missing.values()) if missing else 0

    # Cardinality: any cat column with > 20 unique values
    high_card_cats = [c for c in cat_cols if df[c].nunique() > 20]

    # Outlier density
    total_outliers = sum(outliers.values()) if outliers else 0
    outlier_heavy = total_outliers > (n_rows * 0.05)

    # Class imbalance detection: check all categorical columns
    imbalanced_cols = {}
    for c in cat_cols:
        value_counts = df[c].value_counts(normalize=True)
        if len(value_counts) > 1:
            max_freq = value_counts.iloc[0]
            # Imbalanced if dominant class is >70% or <30% of data
            if max_freq > 0.70 or max_freq < 0.30:
                imbalanced_cols[c] = max_freq

    has_imbalance = len(imbalanced_cols) > 0

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_numeric": len(num_cols),
        "n_categorical": len(cat_cols),
        "numeric_cols": num_cols,
        "cat_cols": cat_cols,
        "small":  n_rows < 1_000,
        "medium": 1_000 <= n_rows < 100_000,
        "large":  n_rows >= 100_000,
        "wide":   n_cols > 50,
        "mostly_numeric": len(num_cols) >= len(cat_cols),
        "mostly_categorical": len(cat_cols) > len(num_cols),
        "mixed": len(num_cols) >= 2 and len(cat_cols) >= 1,
        "has_high_corr": has_high_corr,
        "n_skewed": n_skewed,
        "max_missing_pct": max_missing_pct,
        "high_missing": max_missing_pct > 0.20,
        "high_card_cats": high_card_cats,
        "outlier_heavy": outlier_heavy,
        "has_imbalance": has_imbalance,
        "imbalanced_cols": imbalanced_cols,
        "dq": scores.get("data_quality", 0),
        "model_readiness": scores.get("model_readiness", 0),
        "risk": scores.get("risk", 0),
    }


# ─── Supervised Learning Recommendations ────────────────────────────────────

def supervised_recommendations(p):
    """Return a list of supervised algorithm assessments."""

    recs = []

    # ── Linear / Logistic Regression ──────────────────────────────────────
    score = 3
    reasons_for  = ["Always a strong interpretable baseline"]
    reasons_against = []
    warnings = []

    if p["mostly_numeric"]:
        score += 1
        reasons_for.append("Works well with your predominantly numeric features")
    if not p["has_high_corr"]:
        score += 0.5
        reasons_for.append("Low multicollinearity — linear models behave well")
    else:
        score -= 1
        reasons_against.append("High multicollinearity detected (VIF > 5) — use Ridge/Lasso variant")
        warnings.append("Use Ridge or Lasso to penalise correlated features")
    if p["n_skewed"] > 0:
        score -= 0.5
        reasons_against.append(f"{p['n_skewed']} skewed column(s) — log-transform before fitting")
    if p["mostly_categorical"]:
        score -= 1
        reasons_against.append("Many categorical features require careful encoding before fitting")
    if p["high_missing"]:
        score -= 0.5
        reasons_against.append("Missing data (>20%) — impute before fitting")
        warnings.append("Impute or use SimpleImputer; linear models fail on NaN")
    if p["has_imbalance"]:
        score -= 0.5
        reasons_against.append("Class imbalance detected — use class_weight='balanced' or SMOTE")
        warnings.append("Set class_weight='balanced' in LogisticRegression() or resample with SMOTE")

    recs.append({
        "name": "Linear / Logistic Regression",
        "score": min(max(round(score * 2) / 2, 1), 5),
        "good_for": reasons_for,
        "avoid_because": reasons_against,
        "warnings": warnings,
        "note": "Best starting baseline. Highly interpretable. Use Ridge/Lasso if correlated features present."
    })

    # ── Random Forest ─────────────────────────────────────────────────────
    score = 4
    reasons_for  = ["Handles mixed data types (numeric + categorical) natively"]
    reasons_against = []
    warnings = []

    reasons_for.append("No scaling/normalisation required")
    reasons_for.append("Built-in feature importance for feature selection")
    if p["outlier_heavy"]:
        score += 0.5
        reasons_for.append("Robust to outliers — tree splits are not affected by extreme values")
    if p["high_missing"]:
        score -= 0.5
        reasons_against.append("Needs missing values imputed first (sklearn RF doesn't handle NaN)")
        warnings.append("Use SimpleImputer or IterativeImputer before fitting")
    if p["large"]:
        score -= 0.5
        reasons_against.append("Can be slow on very large datasets — consider LightGBM instead")
    if p["small"]:
        score -= 0.5
        reasons_against.append("Small dataset — overfitting risk; tune n_estimators and max_depth")
        warnings.append("Use cross-validation on small datasets to avoid overfitting")

    recs.append({
        "name": "Random Forest",
        "score": min(max(round(score * 2) / 2, 1), 5),
        "good_for": reasons_for,
        "avoid_because": reasons_against,
        "warnings": warnings,
        "note": "Excellent all-rounder. Often the second model you should try after the linear baseline."
    })

    # ── Gradient Boosting (XGBoost / LightGBM / CatBoost) ─────────────────
    score = 4
    reasons_for  = ["State-of-the-art on tabular data competitions"]
    reasons_against = []
    warnings = []

    if p["mixed"]:
        score += 0.5
        reasons_for.append("Handles mixed numeric and categorical features well")
    if p["large"]:
        score += 0.5
        reasons_for.append("LightGBM and XGBoost scale efficiently to millions of rows")
    if p["has_high_corr"]:
        score += 0.5
        reasons_for.append("Tree-based — not hurt by multicollinearity")
    if p["small"]:
        score -= 1
        reasons_against.append("High risk of overfitting on small datasets — tune carefully")
        warnings.append("Use early stopping and tune max_depth, learning_rate, n_estimators")
    if p["high_missing"]:
        reasons_for.append("XGBoost handles sparse/missing values natively")

    recs.append({
        "name": "Gradient Boosting (XGBoost / LightGBM / CatBoost)",
        "score": min(max(round(score * 2) / 2, 1), 5),
        "good_for": reasons_for,
        "avoid_because": reasons_against,
        "warnings": warnings,
        "note": "Top performer on tabular data. CatBoost is best for datasets dominated by categoricals."
    })

    # ── Support Vector Machine ────────────────────────────────────────────
    score = 3
    reasons_for  = ["Effective in high-dimensional spaces"]
    reasons_against = []
    warnings = []

    if p["mostly_numeric"] and not p["has_high_corr"]:
        score += 0.5
        reasons_for.append("Works well with numeric, low-correlation features")
    if p["small"] or p["medium"]:
        score += 0.5
        reasons_for.append("Good performance on small to medium-sized datasets")
    if p["large"]:
        score -= 2
        reasons_against.append("Scales poorly beyond ~100k rows — training becomes prohibitively slow")
    if p["mostly_categorical"]:
        score -= 1
        reasons_against.append("Categorical features must be numerically encoded first")
    if p["high_missing"]:
        score -= 1
        reasons_against.append("Cannot handle NaN — imputation required")
        warnings.append("Scale features with StandardScaler before fitting; SVM is distance-based")
    else:
        warnings.append("Always scale features with StandardScaler before fitting SVM")

    recs.append({
        "name": "Support Vector Machine (SVM)",
        "score": min(max(round(score * 2) / 2, 1), 5),
        "good_for": reasons_for,
        "avoid_because": reasons_against,
        "warnings": warnings,
        "note": "RBF kernel is good default. Use LinearSVC for speed on larger datasets."
    })

    # ── K-Nearest Neighbours ──────────────────────────────────────────────
    score = 2
    reasons_for  = ["Zero training time — stores the dataset directly"]
    reasons_against = []
    warnings = []

    if p["small"]:
        score += 1
        reasons_for.append("Works well on small datasets")
    if p["mostly_numeric"]:
        score += 0.5
        reasons_for.append("Distance metrics work best on numeric features")
    if p["large"]:
        score -= 2
        reasons_against.append("Prediction is O(n) per query — unusably slow on large data")
    if p["wide"]:
        score -= 1.5
        reasons_against.append("Curse of dimensionality: distance becomes meaningless in many dimensions")
    if p["has_high_corr"]:
        score -= 0.5
        reasons_against.append("Correlated features distort distance calculations")
    if p["mostly_categorical"]:
        score -= 1
        reasons_against.append("Euclidean distance on one-hot encoded data is unreliable")
        warnings.append("Scale all features — KNN is highly sensitive to feature magnitudes")

    recs.append({
        "name": "K-Nearest Neighbors (KNN)",
        "score": min(max(round(score * 2) / 2, 1), 5),
        "good_for": reasons_for,
        "avoid_because": reasons_against,
        "warnings": warnings,
        "note": "Use as a quick sanity-check baseline, not a production model on large datasets."
    })

    # ── Neural Network / MLP ──────────────────────────────────────────────
    score = 3
    reasons_for  = ["Learns complex non-linear feature interactions automatically"]
    reasons_against = []
    warnings = []

    if p["large"]:
        score += 1.5
        reasons_for.append("Shines on large datasets where patterns are complex")
    if p["mostly_numeric"]:
        score += 0.5
        reasons_for.append("Works well with numeric inputs after scaling")
    if p["small"]:
        score -= 2
        reasons_against.append("Severely prone to overfitting on small datasets without heavy regularisation")
        warnings.append("If using NN on small data, use dropout, early stopping, and weight decay")
    if p["high_missing"]:
        score -= 1
        reasons_against.append("Requires complete data — impute or use embedding layers for sparse inputs")
    if p["mostly_categorical"]:
        score -= 0.5
        reasons_against.append("Categorical features need embedding layers for best results")
    if not p["large"]:
        reasons_against.append("Tree-based models typically outperform NNs on tabular data this size")

    recs.append({
        "name": "Neural Network (MLP / Deep Learning)",
        "score": min(max(round(score * 2) / 2, 1), 5),
        "good_for": reasons_for,
        "avoid_because": reasons_against,
        "warnings": warnings,
        "note": "Only worth the complexity when n_rows > 50k and tree models have been exhausted."
    })

    # ── Naive Bayes ───────────────────────────────────────────────────────
    score = 2
    reasons_for  = ["Extremely fast, lightweight, works with tiny datasets"]
    reasons_against = []
    warnings = []

    if p["mostly_categorical"]:
        score += 1.5
        reasons_for.append("CategoricalNB natively handles categorical features without encoding")
    if p["high_missing"]:
        score += 0.5
        reasons_for.append("Can work with missing values under certain NB variants")
    if p["has_high_corr"]:
        score -= 1.5
        reasons_against.append("Assumes feature independence — violated by your correlated features")
    if p["mostly_numeric"]:
        score -= 0.5
        reasons_against.append("GaussianNB assumes normal distribution — check distributions first")
    if p["n_skewed"] > 0:
        score -= 0.5
        reasons_against.append(f"{p['n_skewed']} skewed feature(s) violate the Gaussian NB assumption")

    recs.append({
        "name": "Naive Bayes",
        "score": min(max(round(score * 2) / 2, 1), 5),
        "good_for": reasons_for,
        "avoid_because": reasons_against,
        "warnings": warnings,
        "note": "Use as a fast baseline for text or categorical-heavy data. Rarely the best final model."
    })

    # Sort by score descending
    return sorted(recs, key=lambda x: -x["score"])


# ─── Unsupervised Learning Recommendations ──────────────────────────────────

def unsupervised_recommendations(p):
    """Return a list of unsupervised algorithm assessments."""

    recs = []

    # ── K-Means ───────────────────────────────────────────────────────────
    score = 4
    reasons_for  = ["Simple, fast, and interpretable — great starting point"]
    reasons_against = []
    warnings = []

    if p["mostly_numeric"]:
        score += 0.5
        reasons_for.append("Euclidean distance works well with your numeric features")
    if p["large"]:
        score += 0.5
        reasons_for.append("Mini-batch K-Means scales to millions of rows")
    if p["mostly_categorical"]:
        score -= 1.5
        reasons_against.append("Euclidean distance is meaningless on categorical data — use K-Modes instead")
    if p["outlier_heavy"]:
        score -= 1
        reasons_against.append("Outliers skew cluster centroids — consider K-Medoids or remove outliers first")
        warnings.append("Run DBSCAN first to detect/remove outliers before K-Means")
    if p["wide"]:
        score -= 0.5
        reasons_against.append("High dimensionality reduces cluster separation — run PCA first")

    recs.append({
        "name": "K-Means Clustering",
        "score": min(max(round(score * 2) / 2, 1), 5),
        "good_for": reasons_for,
        "avoid_because": reasons_against,
        "warnings": warnings,
        "note": "Use the elbow method or silhouette score to select optimal k."
    })

    # ── DBSCAN ────────────────────────────────────────────────────────────
    score = 3
    reasons_for  = ["Discovers clusters of arbitrary shape — not just spherical"]
    reasons_against = []
    warnings = []

    if p["outlier_heavy"]:
        score += 1
        reasons_for.append("Explicitly labels outliers as noise — great for anomaly detection")
    if p["small"] or p["medium"]:
        score += 0.5
        reasons_for.append("Runs well on small to medium datasets")
    if p["large"]:
        score -= 1
        reasons_against.append("Slow on large datasets (O(n log n) with KD-tree, worse naively)")
    if p["wide"]:
        score -= 1.5
        reasons_against.append("Distance-based — suffers severely in high dimensions")
    if p["mostly_categorical"]:
        score -= 1
        reasons_against.append("Requires a meaningful distance metric — tricky for categorical data")
        warnings.append("Tune eps carefully using a k-distance plot before running")

    recs.append({
        "name": "DBSCAN",
        "score": min(max(round(score * 2) / 2, 1), 5),
        "good_for": reasons_for,
        "avoid_because": reasons_against,
        "warnings": warnings,
        "note": "Ideal when cluster count is unknown and outlier detection is valuable."
    })

    # ── Hierarchical Clustering ────────────────────────────────────────────
    score = 3
    reasons_for  = ["Dendrogram shows cluster structure at every level of granularity"]
    reasons_against = []
    warnings = []

    if p["small"]:
        score += 1
        reasons_for.append("Perfect for small datasets — no scalability concerns")
    if p["medium"] or p["large"]:
        score -= 1.5
        reasons_against.append("O(n² log n) complexity — prohibitively slow beyond ~10k rows")
    if p["wide"]:
        score -= 0.5
        reasons_against.append("High dimensionality makes linkage distances unreliable")

    recs.append({
        "name": "Hierarchical Clustering (Agglomerative)",
        "score": min(max(round(score * 2) / 2, 1), 5),
        "good_for": reasons_for,
        "avoid_because": reasons_against,
        "warnings": warnings,
        "note": "Use Ward linkage as default. Best for small datasets where interpretability matters."
    })

    # ── PCA ───────────────────────────────────────────────────────────────
    score = 3
    reasons_for  = ["Reduces noise, speeds up downstream ML, and prevents overfitting"]
    reasons_against = []
    warnings = []

    if p["wide"]:
        score += 2
        reasons_for.append("Essential preprocessing for high-dimensional data")
    if p["has_high_corr"]:
        score += 1
        reasons_for.append("Correlated features detected — PCA will compress them into orthogonal components")
    if p["mostly_categorical"]:
        score -= 2
        reasons_against.append("PCA requires numeric input — use MCA (Multiple Correspondence Analysis) instead")
    if not p["wide"] and p["n_numeric"] <= 5:
        score -= 1
        reasons_against.append("Low-dimensional data — dimensionality reduction unlikely to help")
        warnings.append("Use PCA for preprocessing before SVM, KNN, or NN — not before tree models")

    recs.append({
        "name": "PCA (Principal Component Analysis)",
        "score": min(max(round(score * 2) / 2, 1), 5),
        "good_for": reasons_for,
        "avoid_because": reasons_against,
        "warnings": warnings,
        "note": "Use as preprocessing, not as a standalone model. Keep components explaining 95% variance."
    })

    # ── t-SNE / UMAP ──────────────────────────────────────────────────────
    score = 3
    reasons_for  = ["Best tool for 2D/3D visualisation of high-dimensional data"]
    reasons_against = []
    warnings = []

    if p["wide"]:
        score += 1
        reasons_for.append("Uncovers non-linear structure invisible to PCA")
    if p["large"]:
        score -= 1.5
        reasons_against.append("t-SNE is O(n²) — very slow beyond 50k rows; use UMAP instead")
        warnings.append("Use UMAP for large datasets — much faster than t-SNE with similar quality")
    else:
        reasons_for.append("Dataset size is suitable for t-SNE runtime")

    reasons_against.append("Results are not deterministic and cannot be used for inference — visualisation only")

    recs.append({
        "name": "t-SNE / UMAP (Dimensionality Reduction)",
        "score": min(max(round(score * 2) / 2, 1), 5),
        "good_for": reasons_for,
        "avoid_because": reasons_against,
        "warnings": warnings,
        "note": "Use UMAP in production pipelines; t-SNE is for exploration only."
    })

    # ── Isolation Forest (Anomaly Detection) ─────────────────────────────
    score = 3
    reasons_for  = ["Detects anomalies/outliers without needing labels"]
    reasons_against = []
    warnings = []

    if p["outlier_heavy"]:
        score += 1.5
        reasons_for.append("Outliers detected in your data — Isolation Forest will isolate them efficiently")
    if p["large"]:
        score += 0.5
        reasons_for.append("Scales well to large datasets — sub-sampling makes it fast")
    if p["mostly_categorical"]:
        score -= 1
        reasons_against.append("Works best with numeric features; encode categoricals carefully")

    recs.append({
        "name": "Isolation Forest (Anomaly Detection)",
        "score": min(max(round(score * 2) / 2, 1), 5),
        "good_for": reasons_for,
        "avoid_because": reasons_against,
        "warnings": warnings,
        "note": "Excellent first step in any pipeline — identify and remove or flag anomalies before modelling."
    })

    # Sort by score descending
    return sorted(recs, key=lambda x: -x["score"])


# ─── Preprocessing Pipeline ─────────────────────────────────────────────────

def preprocessing_pipeline(p, df):
    """Return ordered list of concrete preprocessing steps for this dataset."""
    steps = []

    # Step 1: Drop IDs
    id_cols = [c for c, info in {} .items()]  # placeholder
    schema_roles = {}
    id_cols = [c for c in df.columns if "id" in c.lower() and df[c].nunique() == len(df)]
    if id_cols:
        steps.append(f"1. DROP identifier columns: {id_cols} — not useful as ML features")

    # Step 2: Handle missing data
    if p["high_missing"]:
        steps.append("2. HANDLE MISSING DATA:")
        steps.append("   • Columns with >50% missing → consider dropping entirely")
        steps.append("   • Numeric columns         → median imputation (SimpleImputer)")
        steps.append("   • Code: from sklearn.impute import SimpleImputer")
        steps.append("           imputer = SimpleImputer(strategy='median')")
        steps.append("           X_imputed = imputer.fit_transform(X[numeric_cols])")
        steps.append("   • Categorical columns      → mode imputation or 'Unknown' category")
        steps.append("   • Code: X[cat_cols] = X[cat_cols].fillna('Unknown')")
        steps.append("   • Advanced option          → IterativeImputer (MICE) for correlated features")
    elif p["max_missing_pct"] > 0:
        steps.append("2. HANDLE MISSING DATA: Impute with median (numeric) / mode (categorical)")
        steps.append("   • Code: X = X.fillna(X.median(numeric_only=True))")
        steps.append("           X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])")

    # Step 3: Encode categoricals
    if p["n_categorical"] > 0:
        if p["high_card_cats"]:
            steps.append(f"3. ENCODE CATEGORICALS:")
            steps.append(f"   • Low cardinality (<10 unique)  → One-Hot Encoding (pd.get_dummies)")
            steps.append(f"   • Code: X = pd.get_dummies(X, columns=[low_card_cols], drop_first=True)")
            steps.append(f"   • High cardinality {[c[:15] for c in p['high_card_cats'][:3]]} → Target Encoding or Frequency Encoding")
            steps.append(f"   • Code: from category_encoders import TargetEncoder")
            steps.append(f"           encoder = TargetEncoder()")
            steps.append(f"           X[high_card_cols] = encoder.fit_transform(X[high_card_cols], y)")
        else:
            steps.append("3. ENCODE CATEGORICALS: One-Hot Encoding (pd.get_dummies or sklearn OneHotEncoder)")
            steps.append("   • Code: X = pd.get_dummies(X, drop_first=True)")

    # Step 4: Scale
    if p["mostly_numeric"] or p["n_numeric"] > 0:
        if p["n_skewed"] > 0:
            steps.append(f"4. TRANSFORM THEN SCALE:")
            steps.append(f"   • {p['n_skewed']} skewed column(s) → apply np.log1p() first")
            steps.append(f"   • Code: import numpy as np; X_skewed = np.log1p(X[skewed_cols])")
            steps.append(f"   • Then scale all numeric features with StandardScaler")
            steps.append(f"   • Code: from sklearn.preprocessing import StandardScaler")
            steps.append(f"           scaler = StandardScaler()")
            steps.append(f"           X_scaled = scaler.fit_transform(X)")
            steps.append(f"   • Note: Tree models (RF, XGBoost) do NOT need scaling")
        else:
            steps.append("4. SCALE NUMERIC FEATURES: StandardScaler (zero mean, unit variance)")
            steps.append("   • Code: from sklearn.preprocessing import StandardScaler")
            steps.append("           scaler = StandardScaler()")
            steps.append("           X_scaled = scaler.fit_transform(X)")
            steps.append("   • Note: Tree models (RF, XGBoost) do NOT need scaling")

    # Step 5: Handle multicollinearity
    if p["has_high_corr"]:
        steps.append("5. ADDRESS MULTICOLLINEARITY:")
        steps.append("   • Drop one from each highly correlated pair (|r| > 0.85)")
        steps.append("   • Or use PCA to compress correlated features into components")
        steps.append("   • Or use Ridge/Lasso regression which handles it via regularisation")

    # Step 6: Handle outliers
    if p["outlier_heavy"]:
        steps.append("6. HANDLE OUTLIERS:")
        steps.append("   • Use IQR clipping: df.clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)")
        steps.append("   • Or use RobustScaler instead of StandardScaler")
        steps.append("   • Or run Isolation Forest first to remove anomaly rows")

    # Step 7: Feature selection
    if p["wide"]:
        steps.append("7. FEATURE SELECTION (high-dimensional data):")
        steps.append("   • Run PCA or SelectKBest to reduce dimensionality")
        steps.append("   • Or use a Random Forest to rank feature importances first")

    # Step 8: Split
    steps.append(f"{'8' if p['wide'] else '7'}. SPLIT: train_test_split(X, y, test_size=0.2, random_state=42)")
    steps.append("   • Use stratify=y for classification with imbalanced classes")
    steps.append("   • Use TimeSeriesSplit if data has temporal ordering")

    return steps


# ─── Formatter ──────────────────────────────────────────────────────────────

def format_recommendations(supervised, unsupervised, pipeline, profile,
                            _BOLD, _CYAN, _GREEN, _YELLOW, _RED, _RESET):
    """Print the full recommendation block to stdout."""
    W = 70

    def rule(title):
        print(f"\n  {_BOLD}{title}{_RESET}")
        print(f"  {'─'*62}")

    # ── Data Warnings ───────────────────────────────────────────────────────
    rule("⚠ Important Data Characteristics")
    if profile.get("has_imbalance"):
        print(f"\n  {_YELLOW}Class Imbalance Detected:{_RESET}")
        for col, freq in profile["imbalanced_cols"].items():
            print(f"    • {col}: Dominant class is {freq:.0%}")
        print(f"    → Use stratified train/test split")
        print(f"    → Consider SMOTE, class_weight='balanced', or threshold tuning")
    if profile.get("n_skewed", 0) > 0:
        print(f"\n  {_YELLOW}Skewed Features Detected ({profile['n_skewed']}):{_RESET}")
        print(f"    → Apply log/sqrt transformation before modeling")
    if profile.get("high_missing"):
        print(f"\n  {_YELLOW}High Missingness (>20%):{_RESET}")
        print(f"    → Use imputation strategy: median for numeric, mode for categorical")

    # ── Supervised ──────────────────────────────────────────────────────
    rule("Supervised Learning (Classification & Regression)")
    for rec in supervised:
        stars = _stars(rec["score"])
        color = _GREEN if rec["score"] >= 4 else _YELLOW if rec["score"] >= 3 else _RED
        print(f"\n  {_BOLD}{rec['name']}{_RESET}")
        print(f"  Rating : {color}{stars} ({rec['score']:.1f}/5){_RESET}")

        if rec["good_for"]:
            print(f"  {_GREEN}✓ Why it fits:{_RESET}")
            for r in rec["good_for"]:
                print(f"      · {r}")

        if rec["avoid_because"]:
            print(f"  {_RED}✗ Caution:{_RESET}")
            for r in rec["avoid_because"]:
                print(f"      · {r}")

        if rec["warnings"]:
            print(f"  {_YELLOW}⚠ Before fitting:{_RESET}")
            for w in rec["warnings"]:
                print(f"      → {w}")

        print(f"  {_CYAN}ℹ  {rec['note']}{_RESET}")

    # ── Unsupervised ────────────────────────────────────────────────────
    rule("Unsupervised Learning (Clustering, Reduction & Anomaly Detection)")
    for rec in unsupervised:
        stars = _stars(rec["score"])
        color = _GREEN if rec["score"] >= 4 else _YELLOW if rec["score"] >= 3 else _RED
        print(f"\n  {_BOLD}{rec['name']}{_RESET}")
        print(f"  Rating : {color}{stars} ({rec['score']:.1f}/5){_RESET}")

        if rec["good_for"]:
            print(f"  {_GREEN}✓ Why it fits:{_RESET}")
            for r in rec["good_for"]:
                print(f"      · {r}")

        if rec["avoid_because"]:
            print(f"  {_RED}✗ Caution:{_RESET}")
            for r in rec["avoid_because"]:
                print(f"      · {r}")

        if rec["warnings"]:
            print(f"  {_YELLOW}⚠ Before fitting:{_RESET}")
            for w in rec["warnings"]:
                print(f"      → {w}")

        print(f"  {_CYAN}ℹ  {rec['note']}{_RESET}")

    # ── Correlation Insights ────────────────────────────────────────────
    if profile.get("has_high_corr"):
        rule("Multicollinearity: High Correlations Detected (|r| > 0.85)")
        print(f"  {_YELLOW}⚠ Problem:{_RESET}")
        print(f"    Highly correlated features can:")
        print(f"    • Inflate model coefficients in linear models")
        print(f"    • Make feature importance hard to interpret")
        print(f"    • Cause overfitting and instability")
        print(f"  {_GREEN}Solutions:{_RESET}")
        print(f"    1. Drop one from each correlated pair")
        print(f"    2. Combine into PCA components (for linear models)")
        print(f"    3. Use Ridge/Lasso regression (automatic handling)")
        print(f"    4. Tree models (RF, XGBoost) are naturally robust to multicollinearity")

    # ── Preprocessing Pipeline ──────────────────────────────────────────
    rule("Recommended Preprocessing Pipeline (in order)")
    for step in pipeline:
        print(f"  {step}")
