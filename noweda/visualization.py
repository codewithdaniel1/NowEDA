"""Ranked, ML-aware visual diagnostics used by ``DataFrame.eda.vizall``.

Most analysis is descriptive.  Small, disposable diagnostic estimators may be
fitted to draw regression, classification-boundary, and clustering overlays.
They are visual baselines, not persisted models or final model selection.
"""

from __future__ import annotations

import math
import inspect
import re
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from noweda.dtypes import is_textual
from noweda.ml_tasks import _infer_supervised_type


class VizResult(dict):
    """Dictionary-like result with a concise notebook representation."""

    def __repr__(self) -> str:
        return (
            "NowEDA VizResult(target={!r}, problem_type={!r}, plots={}, "
            "selected_features={})"
        ).format(
            self.get("target"),
            self.get("problem_type"),
            self.get("plots_generated", 0),
            len(self.get("selected_features", [])),
        )


def _column_position(df: pd.DataFrame, column: Any) -> int:
    if column not in df.columns:
        raise ValueError("Target column not found: {!r}".format(column))
    location = df.columns.get_loc(column)
    if not isinstance(location, (int, np.integer)):
        raise ValueError("NowEDA requires unique column labels for visualization.")
    return int(location)


def _label(value: Any) -> str:
    return str(value)


def _name_tokens(value: Any) -> set:
    return {
        token for token in re.split(r"[^a-z0-9]+", _label(value).lower())
        if len(token) >= 3 and token not in {"target", "label", "score", "value"}
    }


def _is_identifier(column: Any, series: pd.Series, role: Optional[str]) -> bool:
    name = _label(column).strip().lower()
    if role == "id_candidate":
        return True
    if name in {"id", "index", "idx"} or name.endswith("_id") or name.startswith("id_"):
        return True
    sensitive_names = {
        "email", "email_address", "phone", "phone_number", "ssn",
        "credit_card", "payment_card", "account_number",
    }
    if name in sensitive_names:
        return True
    return bool(is_textual(series) and len(series) and series.nunique(dropna=True) / len(series) > 0.95)


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    pairs = pd.DataFrame({"category": categories, "value": values})
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pairs) < 3 or pairs["category"].nunique() < 2:
        return 0.0
    numeric = pd.to_numeric(pairs["value"], errors="coerce")
    pairs = pairs.loc[numeric.notna()].copy()
    pairs["value"] = numeric[numeric.notna()].astype(float)
    if len(pairs) < 3:
        return 0.0
    total = float(((pairs["value"] - pairs["value"].mean()) ** 2).sum())
    if total <= 0:
        return 0.0
    grouped = pairs.groupby("category", observed=True)["value"]
    between = sum(len(group) * (float(group.mean()) - float(pairs["value"].mean())) ** 2
                  for _, group in grouped)
    return float(np.clip(math.sqrt(between / total), 0, 1))


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    pairs = pd.DataFrame({"left": left, "right": right})
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pairs) < 3 or pairs["left"].nunique() < 2 or pairs["right"].nunique() < 2:
        return 0.0
    value = pairs["left"].rank(method="average").corr(
        pairs["right"].rank(method="average")
    )
    return 0.0 if pd.isna(value) else float(abs(value))


def _entropy_score(series: pd.Series) -> float:
    counts = series.value_counts(normalize=True, dropna=True)
    if len(counts) < 2:
        return 0.0
    entropy = float(-(counts * np.log(counts)).sum())
    return float(np.clip(entropy / math.log(len(counts)), 0, 1))


def _association(feature: pd.Series, target: pd.Series, problem_type: str) -> float:
    feature_numeric = pd.api.types.is_numeric_dtype(feature.dtype)
    if problem_type == "regression":
        if feature_numeric:
            return _rank_correlation(feature, target)
        return _correlation_ratio(feature, target)
    if feature_numeric:
        return _correlation_ratio(target, feature)
    try:
        from noweda import ml_utils
        value = ml_utils.cramers_v(feature, target)
        return 0.0 if pd.isna(value) else float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _numeric_shape(feature: pd.Series, target: pd.Series, problem_type: str) -> Optional[str]:
    pairs = pd.DataFrame({"feature": feature, "target": target})
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pairs) < 30 or pairs["feature"].nunique() < 5:
        return None
    try:
        bins = pd.qcut(
            pairs["feature"], q=min(8, pairs["feature"].nunique()), duplicates="drop"
        )
    except (TypeError, ValueError):
        return None
    if bins.nunique() < 4:
        return None

    if problem_type == "regression":
        pearson = pairs["feature"].corr(pairs["target"])
        linear_strength = 0.0 if pd.isna(pearson) else float(abs(pearson))
        binned_strength = _correlation_ratio(bins, pairs["target"])
        nonlinear_gain = binned_strength ** 2 - linear_strength ** 2
        if linear_strength >= 0.50 and nonlinear_gain < 0.10:
            return "linear"
        if binned_strength >= 0.40 and nonlinear_gain >= 0.10:
            return "nonlinear"
        return None

    if pairs["target"].nunique() != 2:
        return None
    event = pairs["target"].value_counts().idxmin()
    working = pairs.assign(_bin=bins, _event=(pairs["target"] == event).astype(float))
    grouped = working.groupby("_bin", observed=True)
    x_values = grouped["feature"].mean().to_numpy(dtype=float)
    rates = grouped["_event"].mean().to_numpy(dtype=float)
    # Rare binary outcomes can carry a useful curved signal even when the
    # absolute event-rate spread is modest.  A 10-point spread is large enough
    # to flag for visual validation without treating small bin noise as shape.
    if len(rates) < 4 or float(np.nanmax(rates) - np.nanmin(rates)) < 0.10:
        return None
    logits = np.log(np.clip(rates, 0.01, 0.99) / (1 - np.clip(rates, 0.01, 0.99)))
    coefficients = np.polyfit(x_values, logits, 1)
    fitted = np.poly1d(coefficients)(x_values)
    total = float(((logits - logits.mean()) ** 2).sum())
    r_squared = 0.0 if total <= 0 else 1 - float(((logits - fitted) ** 2).sum()) / total
    return "linear" if r_squared >= 0.75 else "nonlinear"


def build_visual_diagnostics(
    df: pd.DataFrame,
    report: dict,
    *,
    target: Any = None,
) -> dict:
    """Return ranked evidence for visualization and ML guidance."""
    if not df.columns.is_unique:
        raise ValueError("NowEDA requires unique column labels for visualization.")

    results = report.get("results", {})
    schema = results.get("schema", {})
    pii_columns = set(results.get("pii", {}))
    target_series = None
    problem_type = None
    target_reason = None
    target_position = None

    if target is not None:
        target_position = _column_position(df, target)
        target_series = df.iloc[:, target_position]
        if pd.api.types.is_numeric_dtype(target_series.dtype):
            target_series = pd.to_numeric(target_series, errors="coerce")
            target_series = target_series.replace([np.inf, -np.inf], np.nan)
        if target_series.dropna().empty:
            raise ValueError("Target column {!r} has no observed values.".format(target))
        problem_type, target_reason = _infer_supervised_type(target_series)

    numeric = []
    categorical = []
    datetime = []
    excluded = []
    associations = []

    for position, column in enumerate(df.columns):
        if target_position is not None and position == target_position:
            continue
        series = df.iloc[:, position]
        role = schema.get(column, {}).get("role")
        if column in pii_columns or _is_identifier(column, series, role):
            excluded.append(column)
            continue
        if role == "datetime" or pd.api.types.is_datetime64_any_dtype(series.dtype):
            datetime.append(column)
            continue
        if pd.api.types.is_numeric_dtype(series.dtype):
            numeric.append(column)
            kind = "numeric"
        elif is_textual(series) and series.nunique(dropna=True) <= 30:
            categorical.append(column)
            kind = "categorical"
        else:
            continue

        if target_series is None:
            score = (
                min(float(series.nunique(dropna=True)) / max(len(series), 1), 1.0)
                if kind == "numeric" else _entropy_score(series)
            )
        else:
            score = _association(series, target_series, problem_type)
        associations.append({"column": column, "kind": kind, "score": round(score, 4)})

    associations.sort(key=lambda item: (-item["score"], _label(item["column"])))
    ranked_numeric = [item["column"] for item in associations if item["kind"] == "numeric"]
    ranked_categorical = [item["column"] for item in associations if item["kind"] == "categorical"]
    observations = []
    model_signals = []
    shapes = {}

    target_summary = None
    if target_series is not None:
        target_observed = target_series
        if pd.api.types.is_numeric_dtype(target_series.dtype):
            target_observed = pd.to_numeric(target_series, errors="coerce")
            target_observed = target_observed.replace([np.inf, -np.inf], np.nan)
        target_summary = {
            "observed": int(target_observed.notna().sum()),
            "missing": int(target_observed.isna().sum()),
            "missing_rate": float(target_observed.isna().mean()),
            "inference_reason": target_reason,
        }
        if problem_type == "classification":
            counts = target_observed.value_counts(dropna=True)
            counts = counts[counts > 0]
            target_summary["class_counts"] = {
                _label(label): int(count) for label, count in counts.items()
            }
            target_summary["classes"] = int(len(counts))
            if len(counts) >= 2 and int(counts.max()) > 0:
                minority_share = float(counts.min() / counts.sum())
                target_summary["minority_share"] = minority_share
                if minority_share < 0.20:
                    observations.append(
                        "Class imbalance is visible; prefer stratified validation and imbalance-aware metrics."
                    )
        else:
            target_summary.update({
                "mean": float(target_observed.mean()),
                "median": float(target_observed.median()),
                "skewness": float(target_observed.skew()) if len(target_observed.dropna()) >= 3 else None,
            })

        if associations:
            top = associations[0]
            observations.append(
                "Strongest displayed univariate association: {} ({:.2f}).".format(
                    _label(top["column"]), top["score"]
                )
            )
        observations.append(
            "Use these charts to choose model families to validate; they do not identify a winning model."
        )

        target_tokens = _name_tokens(target)
        leakage_candidates = [
            item["column"] for item in associations
            if item["score"] >= 0.9995
            or (
                item["score"] >= 0.90
                and bool(target_tokens.intersection(_name_tokens(item["column"])))
            )
        ]
        if leakage_candidates:
            model_signals.append(
                "Possible target leakage: {} has a near-deterministic univariate association; verify when it becomes available."
                .format(", ".join(_label(column) for column in leakage_candidates[:3]))
            )
        for column in ranked_numeric[:10]:
            score = next(
                item["score"] for item in associations if item["column"] == column
            )
            if column in leakage_candidates:
                continue
            shape = _numeric_shape(df[column], target_series, problem_type)
            if shape:
                shapes[column] = shape
        linear_features = [column for column, shape in shapes.items() if shape == "linear"]
        nonlinear_features = [column for column, shape in shapes.items() if shape == "nonlinear"]
        if linear_features:
            family = "linear regression" if problem_type == "regression" else "logistic regression or a linear SVM"
            model_signals.append(
                "Approximately linear target structure appears in {}; {} is a reasonable baseline to validate."
                .format(", ".join(_label(column) for column in linear_features[:3]), family)
            )
        if nonlinear_features:
            model_signals.append(
                "Curved or threshold-like target structure appears in {}; compare tree-based or nonlinear-kernel models."
                .format(", ".join(_label(column) for column in nonlinear_features[:3]))
            )
        if associations and associations[0]["score"] < 0.15:
            model_signals.append(
                "Univariate target separation is limited; interactions or additional features may matter more than any single chart."
            )
        if problem_type == "classification":
            model_signals.append(
                "Choose between linear and nonlinear classifiers with stratified validation; visual separation alone is not decisive."
            )
        else:
            model_signals.append(
                "Compare linear and nonlinear regressors with validation; residual behavior and predictive metrics decide the final model."
            )
    else:
        leakage_candidates = []

    numeric_scales = {}
    for column in numeric:
        values = pd.to_numeric(df[column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) >= 2:
            numeric_scales[column] = float(values.std())
    finite_scales = [value for value in numeric_scales.values() if value > 0 and np.isfinite(value)]
    if len(finite_scales) >= 2 and max(finite_scales) / min(finite_scales) >= 100:
        observations.append(
            "Feature scales differ substantially; scale-sensitive models need preprocessing."
        )

    return {
        "target": target,
        "problem_type": problem_type,
        "target_summary": target_summary,
        "associations": associations,
        "numeric_features": ranked_numeric,
        "categorical_features": ranked_categorical,
        "datetime_features": datetime,
        "excluded_features": excluded,
        "leakage_candidates": leakage_candidates,
        "feature_shapes": shapes,
        "numeric_scales": numeric_scales,
        "observations": observations,
        "model_signals": model_signals,
        "diagnostic_models": [],
    }


class _PlotBudget:
    def __init__(self, maximum: int):
        self.maximum = maximum
        self.used = 0

    @property
    def remaining(self) -> int:
        return self.maximum - self.used

    def take(self, requested: int) -> int:
        amount = max(0, min(int(requested), self.remaining))
        self.used += amount
        return amount


def _grid(plt, count: int, *, width: float = 5.5, height: float = 4.0):
    columns = min(3, count)
    rows = int(math.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows), squeeze=False)
    return fig, axes, columns, rows


def _finish_grid(plt, fig, axes, columns: int, rows: int, count: int, title: str):
    for index in range(count, columns * rows):
        axes[index // columns][index % columns].set_visible(False)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.show()


def _sklearn_tools():
    """Load optional estimator tools only when model overlays are applicable."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, r2_score, silhouette_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError:
        return None
    return {
        "KMeans": KMeans,
        "PCA": PCA,
        "LinearRegression": LinearRegression,
        "LogisticRegression": LogisticRegression,
        "balanced_accuracy_score": balanced_accuracy_score,
        "r2_score": r2_score,
        "silhouette_score": silhouette_score,
        "train_test_split": train_test_split,
        "make_pipeline": make_pipeline,
        "StandardScaler": StandardScaler,
        "SVC": SVC,
    }


def _bounded_classification_sample(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Return a deterministic bounded sample while retaining small classes."""
    frame = frame.reset_index(drop=True)
    if len(frame) <= limit:
        return frame
    rng = np.random.RandomState(42)
    selected = []
    for indices in frame.groupby("target", sort=False).groups.values():
        positions = np.asarray(list(indices), dtype=int)
        amount = min(
            len(positions),
            max(10, int(round(limit * len(positions) / len(frame)))),
        )
        selected.extend(rng.choice(positions, size=amount, replace=False).tolist())
    if len(selected) > limit:
        selected = rng.choice(np.asarray(selected), size=limit, replace=False).tolist()
    elif len(selected) < limit:
        remaining = np.setdiff1d(np.arange(len(frame)), np.asarray(selected), assume_unique=False)
        extra = rng.choice(remaining, size=min(limit - len(selected), len(remaining)), replace=False)
        selected.extend(extra.tolist())
    return frame.iloc[sorted(selected)].reset_index(drop=True)


def _model_numeric_features(diagnostics: dict) -> list:
    blocked = set(diagnostics.get("leakage_candidates", []))
    return [
        column for column in diagnostics.get("numeric_features", [])
        if column not in blocked
    ]


def _boundary_features(diagnostics: dict, model_numeric: list) -> list:
    """Prefer a detected nonlinear feature in the two-dimensional projection."""
    nonlinear = [
        column for column, shape in diagnostics.get("feature_shapes", {}).items()
        if shape == "nonlinear" and column in model_numeric
    ]
    if not nonlinear:
        return model_numeric[:2]
    if len(nonlinear) >= 2:
        return nonlinear[:2]
    curved = nonlinear[0]
    companion = next((column for column in model_numeric if column != curved), None)
    return [companion, curved] if companion is not None else [curved]


def _regression_diagnostic(plt, df, target, feature, tools):
    pairs = pd.DataFrame({"feature": df[feature], "target": df[target]})
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pairs) < 40 or pairs["feature"].nunique() < 5:
        return None
    if len(pairs) > 3_000:
        pairs = pairs.sample(n=3_000, random_state=42)
    train, validation = tools["train_test_split"](
        pairs, test_size=0.25, random_state=42
    )
    model = tools["LinearRegression"]()
    model.fit(train[["feature"]], train["target"])
    predicted = model.predict(validation[["feature"]])
    score = float(tools["r2_score"](validation["target"], predicted))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    display = pairs.sample(n=min(1_200, len(pairs)), random_state=42)
    axes[0].scatter(display["feature"], display["target"], alpha=0.32, s=14, color="#4C72B0")
    x_values = np.linspace(float(pairs["feature"].min()), float(pairs["feature"].max()), 150)
    axes[0].plot(x_values, model.predict(pd.DataFrame({"feature": x_values})),
                 color="#E24A33", linewidth=2.5, label="linear regression")
    axes[0].set_xlabel(_label(feature))
    axes[0].set_ylabel(_label(target))
    axes[0].set_title("Observed relationship and fitted line")
    axes[0].legend()
    residuals = validation["target"].to_numpy() - predicted
    axes[1].scatter(predicted, residuals, alpha=0.38, s=16, color="#55A868")
    axes[1].axhline(0, color="#E24A33", linewidth=2)
    axes[1].set_xlabel("Predicted {}".format(_label(target)))
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Validation residuals")
    fig.suptitle(
        "Linear regression diagnostic · {} · validation R² {:.2f}".format(
            _label(feature), score
        ),
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()
    record = {
        "model": "linear_regression",
        "features": [feature],
        "rows": int(len(pairs)),
        "validation_metric": "r2",
        "validation_score": round(score, 4),
        "scope": "diagnostic train/validation fit",
    }
    return fig, record


def _logistic_diagnostic(plt, df, target, feature, tools):
    pairs = pd.DataFrame({"feature": df[feature], "target": df[target]})
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
    if pairs["target"].nunique() != 2 or pairs["feature"].nunique() < 5:
        return None
    pairs = _bounded_classification_sample(pairs, 3_000)
    codes, labels = pd.factorize(pairs["target"], sort=False)
    if np.bincount(codes).min() < 5:
        return None
    train_x, valid_x, train_y, valid_y = tools["train_test_split"](
        pairs[["feature"]], codes, test_size=0.25, random_state=42, stratify=codes
    )
    model = tools["make_pipeline"](
        tools["StandardScaler"](), tools["LogisticRegression"](max_iter=500)
    )
    model.fit(train_x, train_y)
    score = float(tools["balanced_accuracy_score"](valid_y, model.predict(valid_x)))
    lower, upper = pairs["feature"].quantile([0.01, 0.99])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        return None
    x_values = np.linspace(float(lower), float(upper), 250)
    probabilities = model.predict_proba(pd.DataFrame({"feature": x_values}))[:, 1]

    fig, ax = plt.subplots(figsize=(9, 5))
    rng = np.random.RandomState(42)
    display = pairs.sample(n=min(1_200, len(pairs)), random_state=42)
    display_codes = pd.Categorical(display["target"], categories=labels).codes
    ax.scatter(display["feature"], display_codes + rng.normal(0, 0.018, len(display)),
               alpha=0.24, s=13, color="#4C72B0", label="observed classes")
    ax.plot(x_values, probabilities, color="#F5A623", linewidth=3, label="logistic probability")
    try:
        bins = pd.qcut(pairs["feature"], q=min(10, pairs["feature"].nunique()), duplicates="drop")
        working = pairs.assign(_event=(pairs["target"] == labels[1]).astype(float), _bin=bins)
        grouped = working.groupby("_bin", observed=True)
        ax.plot(grouped["feature"].mean(), grouped["_event"].mean(), color="#55A868",
                marker="o", linewidth=2, label="binned observed rate")
    except (TypeError, ValueError):
        pass
    ax.set_xlabel(_label(feature))
    ax.set_ylabel("Probability of {}".format(_label(labels[1])))
    ax.set_ylim(-0.08, 1.08)
    ax.set_title(
        "Logistic diagnostic · validation balanced accuracy {:.2f}\n"
        "One-feature visual baseline; final models may use more features".format(score)
    )
    ax.legend()
    fig.tight_layout()
    plt.show()
    record = {
        "model": "logistic_regression",
        "features": [feature],
        "rows": int(len(pairs)),
        "validation_metric": "balanced_accuracy",
        "validation_score": round(score, 4),
        "scope": "one-feature diagnostic train/validation fit",
    }
    return fig, record


def _svm_diagnostic(plt, df, target, features, tools):
    left, right = features[:2]
    pairs = pd.DataFrame({"left": df[left], "right": df[right], "target": df[target]})
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
    classes = pairs["target"].nunique()
    if classes < 2 or classes > 6 or pairs["left"].nunique() < 4 or pairs["right"].nunique() < 4:
        return None
    pairs = _bounded_classification_sample(pairs, 2_000)
    codes, labels = pd.factorize(pairs["target"], sort=False)
    if np.bincount(codes).min() < 5:
        return None
    train_x, valid_x, train_y, valid_y = tools["train_test_split"](
        pairs[["left", "right"]], codes, test_size=0.25,
        random_state=42, stratify=codes,
    )
    lower_left, upper_left = pairs["left"].quantile([0.01, 0.99])
    lower_right, upper_right = pairs["right"].quantile([0.01, 0.99])
    if lower_left == upper_left or lower_right == upper_right:
        return None
    x_grid, y_grid = np.meshgrid(
        np.linspace(float(lower_left), float(upper_left), 90),
        np.linspace(float(lower_right), float(upper_right), 90),
    )
    grid = pd.DataFrame({"left": x_grid.ravel(), "right": y_grid.ravel()})
    models = (
        ("linear_svm", "Linear SVM", tools["SVC"](kernel="linear", class_weight="balanced")),
        ("rbf_svm", "Nonlinear RBF-SVM", tools["SVC"](kernel="rbf", class_weight="balanced")),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    records = []
    # Keep extreme values in the fitted diagnostic, but zoom the visual panel
    # to the central 98% so a handful of showcase outliers cannot flatten the
    # decision surface and hide its shape.
    display = pairs.loc[
        pairs["left"].between(lower_left, upper_left)
        & pairs["right"].between(lower_right, upper_right)
    ]
    display = display.sample(n=min(800, len(display)), random_state=42)
    display_codes = pd.Categorical(display["target"], categories=labels).codes
    for ax, (key, title, estimator) in zip(axes, models):
        model = tools["make_pipeline"](tools["StandardScaler"](), estimator)
        model.fit(train_x, train_y)
        score = float(tools["balanced_accuracy_score"](valid_y, model.predict(valid_x)))
        surface = model.predict(grid).reshape(x_grid.shape)
        ax.contourf(x_grid, y_grid, surface, levels=np.arange(classes + 1) - 0.5,
                    cmap="Pastel1", alpha=0.72)
        for code, label in enumerate(labels):
            class_rows = display.iloc[np.flatnonzero(display_codes == code)]
            ax.scatter(
                class_rows["left"], class_rows["right"],
                color=plt.cm.tab10(code % 10),
                alpha=0.82,
                s=30 if code else 20,
                edgecolors="white",
                linewidths=0.35,
                label=_label(label),
                zorder=3 + code,
            )
        ax.set_xlabel(_label(left))
        ax.set_ylabel(_label(right))
        ax.set_xlim(float(lower_left), float(upper_left))
        ax.set_ylim(float(lower_right), float(upper_right))
        ax.set_title("{} · balanced accuracy {:.2f}".format(title, score))
        ax.legend(fontsize=8, title=_label(target))
        records.append({
            "model": key,
            "features": [left, right],
            "rows": int(len(pairs)),
            "validation_metric": "balanced_accuracy",
            "validation_score": round(score, 4),
            "scope": "two-feature diagnostic train/validation fit",
        })
    fig.suptitle(
        "Two-feature decision boundaries · diagnostic projection only",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()
    return fig, records


def _clustering_diagnostic(plt, df, diagnostics, tools):
    candidates = [
        column for column in diagnostics.get("numeric_features", [])
        if df[column].nunique(dropna=True) >= 10
        and not any(token in _label(column).lower() for token in ("fraud", "churn", "status"))
    ]
    if len(candidates) < 2:
        return None
    candidate_frame = df[candidates[:12]].replace([np.inf, -np.inf], np.nan)
    correlations = candidate_frame.corr().abs().fillna(0)
    connectivity = (correlations.sum() - 1).sort_values(ascending=False)
    selected = []
    for column in connectivity.index:
        if all(correlations.loc[column, existing] < 0.96 for existing in selected):
            selected.append(column)
        if len(selected) == 6:
            break
    if len(selected) < 2:
        return None
    working = candidate_frame[selected].copy()
    working = working.fillna(working.median(numeric_only=True)).dropna(axis=1)
    if working.shape[1] < 2 or len(working) < 50:
        return None
    if len(working) > 3_000:
        working = working.sample(n=3_000, random_state=42)
    scaled = tools["StandardScaler"]().fit_transform(working)
    projection = tools["PCA"](n_components=2, random_state=42).fit_transform(scaled)
    scores = {}
    fitted = {}
    upper_k = min(6, len(working) - 1)
    for clusters in range(2, upper_k + 1):
        model = tools["KMeans"](n_clusters=clusters, n_init=10, random_state=42)
        labels = model.fit_predict(scaled)
        score = float(tools["silhouette_score"](
            scaled, labels, sample_size=min(1_000, len(working)), random_state=42
        ))
        scores[clusters] = score
        fitted[clusters] = labels
    if not scores:
        return None
    best_k = max(scores, key=scores.get)
    labels = fitted[best_k]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(projection[:, 0], projection[:, 1], c=labels, cmap="tab10",
                    alpha=0.55, s=16)
    axes[0].set_xlabel("PCA component 1")
    axes[0].set_ylabel("PCA component 2")
    axes[0].set_title("K-Means preview · k={}".format(best_k))
    axes[1].plot(list(scores), list(scores.values()), marker="o", color="#4C72B0", linewidth=2)
    axes[1].axvline(best_k, color="#E24A33", linestyle="--", label="best displayed k")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Cluster separation across k")
    axes[1].legend()
    fig.suptitle(
        "Exploratory clustering diagnostic · scaled sample · validate business meaning",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()
    record = {
        "model": "kmeans_pca_preview",
        "features": list(working.columns),
        "rows": int(len(working)),
        "best_k": int(best_k),
        "validation_metric": "silhouette",
        "validation_score": round(scores[best_k], 4),
        "scores_by_k": {int(key): round(value, 4) for key, value in scores.items()},
        "scope": "exploratory unsupervised sample",
    }
    return fig, record


def render_vizall(
    df: pd.DataFrame,
    report: dict,
    *,
    target: Any = None,
    max_plots: int = 15,
) -> VizResult:
    """Render a ranked set of visual diagnostics and return their evidence."""
    if isinstance(max_plots, bool) or not isinstance(max_plots, int) or max_plots <= 0:
        raise ValueError("max_plots must be a positive integer")
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for vizall(). Install it with: pip install matplotlib"
        ) from None

    diagnostics = build_visual_diagnostics(df, report, target=target)
    diagnostic_tools = _sklearn_tools()
    budget = _PlotBudget(max_plots)
    figures = []
    plot_titles = []
    selected = []
    style = "dark_background" if "dark_background" in plt.style.available else "default"
    plt.style.use(style)

    def register(fig, panels: int, title: str, features: Iterable[Any] = ()):
        figures.append(fig)
        plot_titles.append(title)
        for feature in features:
            if feature not in selected:
                selected.append(feature)

    target_series = df.iloc[:, _column_position(df, target)] if target is not None else None
    problem_type = diagnostics["problem_type"]
    numeric = diagnostics["numeric_features"]
    categorical = diagnostics["categorical_features"]
    datetime = diagnostics["datetime_features"]

    # Target overview is always the first supervised diagnostic.
    if target_series is not None and budget.take(1):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        observed = target_series.dropna()
        if pd.api.types.is_numeric_dtype(observed.dtype):
            observed = observed.replace([np.inf, -np.inf], np.nan).dropna()
        if problem_type == "classification":
            counts = observed.value_counts()
            counts = counts[counts > 0]
            bars = ax.bar(range(len(counts)), counts.values, color="#4C72B0")
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels([_label(value)[:24] for value in counts.index], rotation=30, ha="right")
            ax.set_ylabel("Rows")
            ax.set_title("Target Class Distribution: {}".format(_label(target)))
            for bar, value in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2, value, str(int(value)), ha="center", va="bottom")
            title = "Target class distribution"
        else:
            ax.hist(observed, bins=30, color="#4C72B0", alpha=0.8, edgecolor="white")
            ax.axvline(observed.median(), color="#DD8452", linewidth=2, label="median")
            ax.set_xlabel(_label(target))
            ax.set_ylabel("Rows")
            ax.set_title("Regression Target Distribution: {}".format(_label(target)))
            ax.legend()
            title = "Regression target distribution"
        fig.tight_layout()
        plt.show()
        register(fig, 1, title)

    # The strongest target relationships receive a reserved part of the budget.
    if target_series is not None and numeric and budget.remaining:
        count = budget.take(min(3, len(numeric)))
        columns_to_plot = numeric[:count]
        fig, axes, grid_columns, grid_rows = _grid(plt, count)
        for index, column in enumerate(columns_to_plot):
            ax = axes[index // grid_columns][index % grid_columns]
            pairs = pd.DataFrame({"feature": df[column], "target": target_series})
            pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
            if problem_type == "classification":
                level_counts = pairs["target"].value_counts()
                levels = list(level_counts[level_counts > 0].head(8).index)
                values = [pairs.loc[pairs["target"] == level, "feature"].to_numpy() for level in levels]
                tick_labels = [_label(level)[:18] for level in levels]
                label_parameter = (
                    "tick_labels" if "tick_labels" in inspect.signature(ax.boxplot).parameters
                    else "labels"
                )
                ax.boxplot(values, showfliers=False, **{label_parameter: tick_labels})
                ax.set_xlabel(_label(target))
                ax.set_ylabel(_label(column))
            else:
                if len(pairs) > 2000:
                    pairs = pairs.sample(n=2000, random_state=42)
                ax.scatter(pairs["feature"], pairs["target"], alpha=0.35, s=14, color="#4C72B0")
                if pairs["feature"].nunique() >= 2:
                    coefficients = np.polyfit(pairs["feature"].astype(float), pairs["target"].astype(float), 1)
                    x_values = np.linspace(float(pairs["feature"].min()), float(pairs["feature"].max()), 100)
                    ax.plot(x_values, np.poly1d(coefficients)(x_values), color="#DD8452", linewidth=2,
                            label="linear trend")
                    try:
                        bins = pd.qcut(pairs["feature"], q=min(10, pairs["feature"].nunique()), duplicates="drop")
                        grouped = pairs.assign(_bin=bins).groupby("_bin", observed=True)
                        x_means = grouped["feature"].mean()
                        y_means = grouped["target"].mean()
                        ax.plot(x_means, y_means, color="#55A868", marker="o", linewidth=2,
                                label="binned mean")
                    except (TypeError, ValueError):
                        pass
                    ax.legend(fontsize=8)
                ax.set_xlabel(_label(column))
                ax.set_ylabel(_label(target))
            score = next(item["score"] for item in diagnostics["associations"] if item["column"] == column)
            ax.set_title("{} (association {:.2f})".format(_label(column), score))
        title = "Top numeric target relationships"
        _finish_grid(plt, fig, axes, grid_columns, grid_rows, count, title)
        register(fig, count, title, columns_to_plot)

    if target_series is not None and categorical and budget.remaining:
        count = budget.take(min(2, len(categorical)))
        columns_to_plot = categorical[:count]
        fig, axes, grid_columns, grid_rows = _grid(plt, count)
        for index, column in enumerate(columns_to_plot):
            ax = axes[index // grid_columns][index % grid_columns]
            pairs = pd.DataFrame({"feature": df[column], "target": target_series})
            pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
            top_levels = list(pairs["feature"].value_counts().head(12).index)
            pairs = pairs[pairs["feature"].isin(top_levels)]
            if problem_type == "classification":
                table = pd.crosstab(pairs["feature"], pairs["target"], normalize="index")
                table.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", legend=True)
                ax.set_ylabel("Class share")
                ax.legend(title=_label(target), fontsize=7)
            else:
                means = pairs.groupby("feature", observed=True)["target"].mean().sort_values(ascending=False)
                ax.bar(range(len(means)), means.values, color="#4C72B0")
                ax.set_xticks(range(len(means)))
                ax.set_xticklabels([_label(value)[:18] for value in means.index], rotation=45, ha="right")
                ax.set_ylabel("Mean {}".format(_label(target)))
            ax.set_title(_label(column))
            ax.tick_params(axis="x", rotation=45)
        title = "Top categorical target relationships"
        _finish_grid(plt, fig, axes, grid_columns, grid_rows, count, title)
        register(fig, count, title, columns_to_plot)

    # Small diagnostic estimators make model shape visible. They use bounded,
    # deterministic samples and are deliberately labeled as visual baselines.
    model_numeric = _model_numeric_features(diagnostics)
    if diagnostic_tools is None and ((target_series is not None and model_numeric) or target is None):
        diagnostics["observations"].append(
            "Fitted model overlays require scikit-learn; install noweda[viz] or noweda[ml]."
        )
    elif target_series is not None and problem_type == "regression":
        if budget.remaining >= 2 and model_numeric:
            rendered = _regression_diagnostic(
                plt, df, target, model_numeric[0], diagnostic_tools
            )
            if rendered is not None:
                fig, record = rendered
                budget.take(2)
                diagnostics["diagnostic_models"].append(record)
                register(fig, 2, "Linear regression fit and residuals", model_numeric[:1])
    elif target_series is not None and problem_type == "classification":
        classes = target_series.nunique(dropna=True)
        if classes == 2 and budget.remaining >= 1 and model_numeric:
            rendered = _logistic_diagnostic(
                plt, df, target, model_numeric[0], diagnostic_tools
            )
            if rendered is not None:
                fig, record = rendered
                budget.take(1)
                diagnostics["diagnostic_models"].append(record)
                register(fig, 1, "Logistic probability curve", model_numeric[:1])
        boundary_features = _boundary_features(diagnostics, model_numeric)
        if budget.remaining >= 2 and len(boundary_features) >= 2:
            rendered = _svm_diagnostic(
                plt, df, target, boundary_features, diagnostic_tools
            )
            if rendered is not None:
                fig, records = rendered
                budget.take(2)
                diagnostics["diagnostic_models"].extend(records)
                register(fig, 2, "Linear and nonlinear SVM boundaries", boundary_features)
    elif target_series is None and budget.remaining >= 2 and diagnostic_tools is not None:
        rendered = _clustering_diagnostic(plt, df, diagnostics, diagnostic_tools)
        if rendered is not None:
            fig, record = rendered
            budget.take(2)
            diagnostics["diagnostic_models"].append(record)
            register(fig, 2, "PCA and K-Means cluster diagnostic", record["features"])

    results = report.get("results", {})
    missing_columns = [column for column in numeric + categorical + datetime if df[column].isna().any()]
    if missing_columns and budget.take(1):
        rates = pd.Series({column: float(df[column].isna().mean() * 100) for column in missing_columns})
        rates = rates.sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(max(7, len(rates) * 0.55), 4.5))
        ax.bar(range(len(rates)), rates.values, color="#DD8452")
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([_label(value)[:20] for value in rates.index], rotation=45, ha="right")
        ax.set_ylabel("Missing %")
        ax.set_ylim(0, 100)
        ax.set_title("Missing Values by Feature")
        fig.tight_layout()
        plt.show()
        register(fig, 1, "Missing values", rates.index)

    if len(numeric) >= 2 and budget.take(1):
        corr_columns = numeric[: min(10, len(numeric))]
        corr_frame = pd.concat(
            [df[column].rename(index) for index, column in enumerate(corr_columns)], axis=1
        )
        corr = corr_frame.corr()
        fig_size = max(6, min(len(corr_columns) * 1.0, 12))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
        cmap = matplotlib.colormaps.get_cmap("coolwarm") if hasattr(matplotlib, "colormaps") else plt.cm.coolwarm
        image = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        fig.colorbar(image, ax=ax, shrink=0.8)
        ax.set_xticks(range(len(corr_columns)))
        ax.set_yticks(range(len(corr_columns)))
        ax.set_xticklabels([_label(value) for value in corr_columns], rotation=45, ha="right")
        ax.set_yticklabels([_label(value) for value in corr_columns])
        for row in range(len(corr_columns)):
            for column_index in range(len(corr_columns)):
                value = corr.iloc[row, column_index]
                ax.text(column_index, row, "{:.2f}".format(value), ha="center", va="center", fontsize=7,
                        color="white" if abs(value) > 0.5 else "black")
        ax.set_title("Correlation Heatmap (ranked numeric features)")
        fig.tight_layout()
        plt.show()
        register(fig, 1, "Correlation heatmap", corr_columns)

    if numeric and budget.remaining:
        count = budget.take(min(3 if target is None else 2, len(numeric)))
        columns_to_plot = numeric[:count]
        fig, axes, grid_columns, grid_rows = _grid(plt, count)
        for index, column in enumerate(columns_to_plot):
            ax = axes[index // grid_columns][index % grid_columns]
            values = pd.to_numeric(df[column], errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan).dropna()
            ax.hist(values, bins=30, color="#4C72B0", alpha=0.75, edgecolor="white")
            ax.set_title(_label(column))
            ax.set_ylabel("Rows")
        title = "Selected numeric distributions"
        _finish_grid(plt, fig, axes, grid_columns, grid_rows, count, title)
        register(fig, count, title, columns_to_plot)

    if categorical and budget.remaining:
        count = budget.take(min(3 if target is None else 2, len(categorical)))
        columns_to_plot = categorical[:count]
        fig, axes, grid_columns, grid_rows = _grid(plt, count)
        for index, column in enumerate(columns_to_plot):
            ax = axes[index // grid_columns][index % grid_columns]
            counts = df[column].value_counts(dropna=False).head(15)
            ax.bar(range(len(counts)), counts.values, color="#4C72B0")
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels([_label(value)[:18] for value in counts.index], rotation=45, ha="right")
            ax.set_title(_label(column))
            ax.set_ylabel("Rows")
        title = "Selected categorical distributions"
        _finish_grid(plt, fig, axes, grid_columns, grid_rows, count, title)
        register(fig, count, title, columns_to_plot)

    outliers = results.get("outliers", {})
    outlier_rates = {
        column: float(count) / len(df) * 100
        for column, count in outliers.items()
        if column in numeric and isinstance(count, (int, float, np.integer, np.floating)) and count > 0
    }
    if outlier_rates and budget.take(1):
        rates = pd.Series(outlier_rates).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(8, max(4, len(rates) * 0.35)))
        ax.barh([_label(value) for value in rates.index], rates.values, color="#C44E52")
        ax.invert_yaxis()
        ax.set_xlabel("Outlier rows (%)")
        ax.set_title("Outlier Prevalence")
        fig.tight_layout()
        plt.show()
        register(fig, 1, "Outlier prevalence", rates.index)

    scales = {column: value for column, value in diagnostics["numeric_scales"].items()
              if value > 0 and np.isfinite(value)}
    if len(scales) >= 2 and budget.take(1):
        scale_series = pd.Series(scales).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(8, max(4, len(scale_series) * 0.35)))
        ax.barh([_label(value) for value in scale_series.index], scale_series.values, color="#8172B2")
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("Standard deviation (log scale)")
        ax.set_title("Feature Scale Comparison")
        fig.tight_layout()
        plt.show()
        register(fig, 1, "Feature scale comparison", scale_series.index)

    if len(numeric) >= 2 and budget.remaining:
        pair_columns = numeric[: min(12, len(numeric))]
        pair_frame = pd.concat(
            [df[column].rename(index) for index, column in enumerate(pair_columns)], axis=1
        )
        corr = pair_frame.corr().abs()
        pairs = []
        for left_index in range(len(pair_columns)):
            for right_index in range(left_index + 1, len(pair_columns)):
                value = corr.iloc[left_index, right_index]
                if pd.notna(value):
                    pairs.append((pair_columns[left_index], pair_columns[right_index], float(value)))
        pairs.sort(key=lambda item: -item[2])
        count = budget.take(min(2, len(pairs)))
        if count:
            fig, axes, grid_columns, grid_rows = _grid(plt, count)
            used_features = []
            for index, (left, right, value) in enumerate(pairs[:count]):
                ax = axes[index // grid_columns][index % grid_columns]
                points = pd.DataFrame({
                    "left": df[left].reset_index(drop=True),
                    "right": df[right].reset_index(drop=True),
                }).replace([np.inf, -np.inf], np.nan).dropna()
                if len(points) > 2000:
                    points = points.sample(n=2000, random_state=42)
                ax.scatter(points["left"], points["right"], alpha=0.35, s=14, color="#4C72B0")
                ax.set_xlabel(_label(left))
                ax.set_ylabel(_label(right))
                ax.set_title("|r| = {:.2f}".format(value))
                used_features.extend([left, right])
            title = "Top Numeric Feature Relationships"
            _finish_grid(plt, fig, axes, grid_columns, grid_rows, count, title)
            register(fig, count, title, used_features)

    if len(categorical) >= 2 and budget.take(1):
        columns = categorical[: min(10, len(categorical))]
        matrix = np.full((len(columns), len(columns)), np.nan)
        for left_index, left in enumerate(columns):
            matrix[left_index, left_index] = 1.0
            for right_index in range(left_index + 1, len(columns)):
                try:
                    from noweda import ml_utils
                    value = ml_utils.cramers_v(df[left], df[columns[right_index]])
                    matrix[left_index, right_index] = matrix[right_index, left_index] = value
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = matplotlib.colormaps.get_cmap("YlOrRd") if hasattr(matplotlib, "colormaps") else plt.cm.YlOrRd
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        fig.colorbar(image, ax=ax, label="Cramér's V")
        ax.set_xticks(range(len(columns)))
        ax.set_yticks(range(len(columns)))
        ax.set_xticklabels([_label(value) for value in columns], rotation=45, ha="right")
        ax.set_yticklabels([_label(value) for value in columns])
        for row in range(len(columns)):
            for column_index in range(len(columns)):
                value = matrix[row, column_index]
                ax.text(column_index, row, "N/A" if np.isnan(value) else "{:.2f}".format(value),
                        ha="center", va="center", fontsize=7,
                        color="white" if pd.notna(value) and value >= 0.5 else "black")
        ax.set_title("Categorical Association (Cramér's V)")
        fig.tight_layout()
        plt.show()
        register(fig, 1, "Categorical association", columns)

    if datetime and numeric and budget.take(1):
        date_column = datetime[0]
        value_column = target if target is not None and problem_type == "regression" else numeric[0]
        points = pd.DataFrame({
            "date": pd.to_datetime(df[date_column], errors="coerce"),
            "value": df[value_column],
        }).replace([np.inf, -np.inf], np.nan).dropna().sort_values("date")
        if not points.empty:
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(points["date"], points["value"], color="#4C72B0", linewidth=1)
            ax.set_xlabel(_label(date_column))
            ax.set_ylabel(_label(value_column))
            ax.set_title("{} over {}".format(_label(value_column), _label(date_column)))
            fig.autofmt_xdate()
            fig.tight_layout()
            plt.show()
            register(fig, 1, "Temporal relationship", [date_column, value_column])

    result = VizResult(diagnostics)
    result.update({
        "max_plots": max_plots,
        "plots_generated": budget.used,
        "plot_titles": plot_titles,
        "selected_features": selected,
        "figures": figures,
        "scope": {"rows": int(len(df)), "sample_based": False},
    })
    if not figures:
        print("No visualizations could be generated for this dataset.")
    if diagnostics["model_signals"]:
        print("\nVisual ML signals")
        for signal in diagnostics["model_signals"]:
            print("  - {}".format(signal))
    return result
