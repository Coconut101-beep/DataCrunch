"""Computation layer for the interactive Lending Club regression dashboard.

Every function here is pure Python/pandas and contains no Streamlit calls, so the
same code can be exercised headlessly by ``verify_checks.py``.

Target units: ``int_rate`` is stored as a fraction (0.05 = 5%). Metrics are
returned in fraction units and in percentage points (``*_pp``) so the interface
can label them explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import build_design_matrices, dmatrices
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "int_rate"

# Columns the notebook keeps from the raw export (the first 19).
RAW_KEEP_COLUMNS = 19

# Variables every model draws on. A single complete-case sample over these
# columns keeps all stage comparisons on the same observations.
MODELLING_COLUMNS = [TARGET, "loan_amnt", "term", "dti", "annual_inc", "grade", "issue_d"]

# Categorical source variables whose levels must exist in training data before a
# fitted model can predict a row.
CATEGORICAL_SOURCES = ["term", "grade", "delinq_2yrs", "ym"]

GRADE_ORDER = list("ABCDEFG")


# --------------------------------------------------------------------------- #
# Model specifications (taken verbatim from the assignment notebook)
# --------------------------------------------------------------------------- #

LOAN_POLY = "loan_amnt + I(loan_amnt**2) + I(loan_amnt**3)"

COMPLEX_FORMULA = (
    f"{TARGET} ~ ({LOAN_POLY}) + C(term) + dti + annual_inc + C(grade)"
    f" + C(grade):C(term)"
    f" + ({LOAN_POLY}):C(term)"
    f" + C(grade):C(term):({LOAN_POLY})"
)

BENCHMARK_FORMULA = f"{TARGET} ~ loan_amnt * C(grade) + C(term) + dti + annual_inc"


@dataclass(frozen=True)
class Spec:
    """A named regression specification."""

    key: str
    label: str
    formula: str
    note: str


SPECS: dict[str, Spec] = {
    "model0": Spec(
        "model0",
        "Model 0 — term, income, DTI, grade",
        f"{TARGET} ~ term + annual_inc + dti + grade",
        "The assignment baseline: loan term, annual income, debt-to-income and loan grade.",
    ),
    "model1": Spec(
        "model1",
        "Model 1 — Model 0 plus loan amount",
        f"{TARGET} ~ term + annual_inc + dti + grade + loan_amnt",
        "Model 0 with loan amount added as a main effect.",
    ),
    "model2": Spec(
        "model2",
        "Model 2 — grade x loan amount interaction",
        f"{TARGET} ~ term + annual_inc + dti + grade * loan_amnt",
        "Lets the loan-amount slope differ by grade.",
    ),
    "model3": Spec(
        "model3",
        "Model 3 — adds grade x term and income squared",
        f"{TARGET} ~ term + annual_inc + dti + grade * loan_amnt + grade * term + I(annual_inc**2)",
        "Model 2 plus a grade-by-term interaction and a quadratic income term.",
    ),
    "complex": Spec(
        "complex",
        "Complex model — cubic loan amount x grade x term",
        COMPLEX_FORMULA,
        "The deliberately over-parameterised specification used for the regularisation comparison.",
    ),
    "benchmark": Spec(
        "benchmark",
        "Benchmark (no time information)",
        BENCHMARK_FORMULA,
        "The time-stage benchmark: loan amount interacted with grade, plus term, DTI and income.",
    ),
    "time1": Spec(
        "time1",
        "time1 — linear month trend",
        BENCHMARK_FORMULA + " + t_linear",
        "One common linear trend in months since the first loan in the sample.",
    ),
    "time2": Spec(
        "time2",
        "time2 — linear month trend by grade",
        BENCHMARK_FORMULA + " + t_linear * C(grade)",
        "A linear monthly trend that is allowed to differ across grades.",
    ),
    "time3": Spec(
        "time3",
        "time3 — month-year dummies",
        BENCHMARK_FORMULA + " + C(ym)",
        "A separate free effect for every observed month-year.",
    ),
    "time4": Spec(
        "time4",
        "time4 — month-year dummies by grade",
        BENCHMARK_FORMULA + " + C(ym) * C(grade)",
        "Month-year effects that are allowed to differ across grades.",
    ),
}

STAGE2_SPECS = ["model0", "model1"]
STAGE3_SPECS = ["model0", "model1", "model2", "model3"]
STAGE4_SPECS = ["model0", "model1", "model2", "model3", "complex"]
STAGE5_SPECS = ["model1", "model2", "model3", "complex"]
TIME_SPECS = ["benchmark", "time1", "time2", "time3", "time4"]

# Specifications that cannot produce predictions for months absent from the
# training data, because the month itself carries a free coefficient.
MONTH_DUMMY_SPECS = {"time3", "time4"}


def build_model_specification(key: str) -> Spec:
    """Return the named specification, raising a clear error for unknown keys."""
    if key not in SPECS:
        raise KeyError(f"Unknown specification {key!r}. Available: {sorted(SPECS)}")
    return SPECS[key]


# --------------------------------------------------------------------------- #
# Loading and cleaning
# --------------------------------------------------------------------------- #


def load_raw(source) -> pd.DataFrame:
    """Read the Lending Club export.

    The file carries a one-line prospectus banner above the real header, which is
    why ``skiprows=1`` is required.
    """
    return pd.read_csv(source, skiprows=1, low_memory=False)


def _missing_frame(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame.isna().sum()
    return pd.DataFrame({"variable": counts.index.astype(str), "missing": counts.to_numpy()})


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Turn the raw export into the modelling sample, recording every removal.

    Returns the cleaned frame and a report describing each step, the missing
    values encountered, and the columns dropped.
    """
    steps: list[dict] = []
    n_start = len(raw)
    steps.append({"step": "Raw rows read from file", "rows": n_start, "removed": 0, "reason": "—"})

    df = raw.iloc[:, :RAW_KEEP_COLUMNS].copy()
    trailing = raw.shape[1] - RAW_KEEP_COLUMNS

    before = len(df)
    df = df[df[TARGET].notna()]
    steps.append(
        {
            "step": "Drop rows with no interest rate",
            "rows": len(df),
            "removed": before - len(df),
            "reason": "The target cannot be modelled or evaluated when it is missing.",
        }
    )

    # Explicit parsing: the export stores dates as m/d/Y text and several
    # numeric fields with surrounding whitespace.
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%m/%d/%Y", errors="coerce")
    for column in ["loan_amnt", "dti", "annual_inc", TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["term"] = df["term (months)"].astype("category")
    df["delinq_2yrs"] = df["delinq_2yrs"].astype("category")
    # The notebook's grade order comes from unique(), which follows order of
    # appearance. Setting it explicitly keeps A < B < ... < G.
    df["grade"] = pd.Categorical(df["grade"], categories=GRADE_ORDER, ordered=True)

    dropped_columns = ["emp_title", "installment", "term (months)"]
    df = df.drop(columns=dropped_columns)

    missing_before_complete_case = _missing_frame(df[MODELLING_COLUMNS])

    before = len(df)
    df = df.dropna(subset=MODELLING_COLUMNS)
    steps.append(
        {
            "step": "Keep complete cases on modelling variables",
            "rows": len(df),
            "removed": before - len(df),
            "reason": "Every stage then compares models on exactly the same observations.",
        }
    )

    # Time helpers, created once so later stages never recompute them differently.
    month = df["issue_d"].dt.to_period("M")
    df["t_linear"] = (month.astype(int) - month.astype(int).min()).astype(int)
    df["ym"] = month.astype(str)
    df = df.reset_index(drop=True)

    report = {
        "steps": pd.DataFrame(steps),
        "missing_by_variable": missing_before_complete_case,
        "missing_all_columns": _missing_frame(df),
        "dropped_columns": dropped_columns,
        "trailing_unnamed_columns": int(trailing),
        "n_raw": int(n_start),
        "n_clean": int(len(df)),
        "n_months": int(df["ym"].nunique()),
        "month_range": (df["issue_d"].min(), df["issue_d"].max()),
    }
    return df, report


def data_signature(df: pd.DataFrame) -> str:
    """A short, stable signature used as a cache key for a given sample."""
    return f"{len(df)}x{df.shape[1]}:{pd.util.hash_pandas_object(df[TARGET], index=False).sum()}"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _rmse(actual, predicted) -> float:
    return float(root_mean_squared_error(actual, predicted))


def _formula_categoricals(formula: str) -> list[str]:
    """Categorical source variables that appear in a formula."""
    return [name for name in CATEGORICAL_SOURCES if name in formula]


def support_mask(formula: str, train: pd.DataFrame, candidate: pd.DataFrame) -> pd.Series:
    """Rows of ``candidate`` whose categorical levels all appear in ``train``.

    A fitted model has no coefficient for a level it never saw, so predicting
    such rows would invent an effect. They are flagged instead of being filled.
    """
    mask = pd.Series(True, index=candidate.index)
    for column in _formula_categoricals(formula):
        seen = set(train[column].dropna().unique())
        mask &= candidate[column].isin(seen)
    return mask


def unsupported_levels(formula: str, train: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, list]:
    """Which categorical levels in ``candidate`` are missing from ``train``."""
    missing: dict[str, list] = {}
    for column in _formula_categoricals(formula):
        seen = set(train[column].dropna().unique())
        unseen = sorted({v for v in candidate[column].dropna().unique() if v not in seen}, key=str)
        if unseen:
            missing[column] = unseen
    return missing


def matrix_rank(matrix) -> int:
    """Rank of a design matrix, computed scale-invariantly.

    Columns are normalised first. Specifications here mix 0/1 dummies with terms
    such as ``annual_inc**2`` that are ~13 orders of magnitude larger, and
    ``np.linalg.matrix_rank`` sets its tolerance from the largest singular
    value, so on the raw matrix it discards columns that are genuinely
    independent.
    """
    values = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(values, axis=0)
    norms[norms == 0] = 1.0
    return int(np.linalg.matrix_rank(values / norms))


def design_rank(fit) -> int:
    """Effective number of parameters: the rank of the fitted design matrix."""
    return matrix_rank(fit.model.exog)


def create_split(
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve a common test set. Used by every stage that reports test error."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    return train.copy(), test.copy()


def split_by_month(
    df: pd.DataFrame, holdout_months: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Split on whole months: the earliest months train, the latest months test."""
    months = sorted(df["ym"].unique())
    if holdout_months >= len(months):
        raise ValueError("The hold-out period must leave at least one training month.")
    test_months = months[-holdout_months:]
    train = df[~df["ym"].isin(test_months)].copy()
    test = df[df["ym"].isin(test_months)].copy()
    return train, test, test_months


# --------------------------------------------------------------------------- #
# Stage 2-4: fitting, evaluation, cross-validation
# --------------------------------------------------------------------------- #


def fit_and_evaluate(
    formula: str,
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    *,
    with_coefficients: bool = True,
) -> dict:
    """Fit by OLS on ``train`` and report training and (optional) test metrics.

    Test rows carrying categorical levels absent from the training data are
    excluded and counted rather than silently predicted.
    """
    fit = smf.ols(formula, data=train).fit()

    fitted = np.asarray(fit.fittedvalues, dtype=float)
    actual_train = np.asarray(train[TARGET], dtype=float)
    n_params = int(np.asarray(fit.model.exog).shape[1])
    rank = design_rank(fit)

    result = {
        "formula": formula,
        "n_train": int(len(train)),
        "n_params": n_params,
        "n_effective_params": rank,
        "rank_deficient": rank < n_params,
        "rmse_train": _rmse(actual_train, fitted),
        "r2_train": float(fit.rsquared),
        "adj_r2_train": float(fit.rsquared_adj),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "constant_columns": _constant_design_columns(fit),
        "fit": fit,
        "train_actual": actual_train,
        "train_predicted": fitted,
    }

    if test is not None:
        mask = support_mask(formula, train, test)
        supported = test[mask]
        predicted = fit.predict(supported)
        finite = np.isfinite(np.asarray(predicted, dtype=float))
        supported = supported[finite]
        predicted = predicted[finite]
        actual_test = np.asarray(supported[TARGET], dtype=float)
        result.update(
            {
                "n_test": int(len(test)),
                "n_test_evaluated": int(len(supported)),
                "n_test_unsupported": int(len(test) - len(supported)),
                "unsupported_levels": unsupported_levels(formula, train, test),
                "rmse_test": _rmse(actual_test, predicted) if len(supported) else float("nan"),
                "r2_test": float(r2_score(actual_test, predicted)) if len(supported) else float("nan"),
                "test_actual": actual_test,
                "test_predicted": np.asarray(predicted, dtype=float),
                "test_index": supported.index,
            }
        )

    if with_coefficients:
        result["coefficients"] = coefficient_table(fit)

    for key in [k for k in list(result) if k.startswith("rmse_")]:
        result[f"{key}_pp"] = result[key] * 100.0
    return result


def _constant_design_columns(fit) -> list[str]:
    """Design columns with no variation other than the intercept."""
    exog = np.asarray(fit.model.exog, dtype=float)
    names = list(fit.model.exog_names)
    constant = []
    for position, name in enumerate(names):
        if name == "Intercept":
            continue
        if np.ptp(exog[:, position]) == 0:
            constant.append(name)
    return constant


def coefficient_table(fit) -> pd.DataFrame:
    """Tidy coefficient table with standard errors, t statistics and p values."""
    table = pd.DataFrame(
        {
            "term": fit.params.index,
            "coefficient": fit.params.to_numpy(),
            "std_error": fit.bse.to_numpy(),
            "t": fit.tvalues.to_numpy(),
            "p_value": fit.pvalues.to_numpy(),
        }
    )
    conf = fit.conf_int()
    table["ci_lower"] = conf.iloc[:, 0].to_numpy()
    table["ci_upper"] = conf.iloc[:, 1].to_numpy()
    return table


def run_cross_validation(
    formula: str, train: pd.DataFrame, k: int, seed: int
) -> dict:
    """K-fold cross-validation carried out strictly inside the training set.

    Each fold rebuilds its own design matrix, so no information crosses folds,
    and fold-test rows with unseen categorical levels are excluded and counted.
    """
    folds = KFold(n_splits=k, shuffle=True, random_state=seed)
    rows = []
    for number, (train_index, validate_index) in enumerate(folds.split(train), start=1):
        fold_train = train.iloc[train_index]
        fold_validate = train.iloc[validate_index]
        fold_fit = smf.ols(formula, data=fold_train).fit()

        mask = support_mask(formula, fold_train, fold_validate)
        evaluated = fold_validate[mask]
        predicted = fold_fit.predict(evaluated)
        finite = np.isfinite(np.asarray(predicted, dtype=float))
        evaluated, predicted = evaluated[finite], predicted[finite]

        rows.append(
            {
                "fold": number,
                "n_fit": int(len(fold_train)),
                "n_validate": int(len(fold_validate)),
                "n_evaluated": int(len(evaluated)),
                "n_unsupported": int(len(fold_validate) - len(evaluated)),
                "rmse": _rmse(evaluated[TARGET], predicted) if len(evaluated) else np.nan,
                "r2": float(r2_score(evaluated[TARGET], predicted)) if len(evaluated) else np.nan,
            }
        )

    folds_frame = pd.DataFrame(rows)
    return {
        "k": int(k),
        "seed": int(seed),
        "folds": folds_frame,
        "rmse_mean": float(folds_frame["rmse"].mean()),
        "rmse_std": float(folds_frame["rmse"].std(ddof=1)) if k > 1 else 0.0,
        "rmse_mean_pp": float(folds_frame["rmse"].mean()) * 100.0,
        "n_unsupported": int(folds_frame["n_unsupported"].sum()),
    }


# --------------------------------------------------------------------------- #
# Stage 5: learning curve
# --------------------------------------------------------------------------- #


def build_learning_curve(
    formula: str,
    train_pool: pd.DataFrame,
    test: pd.DataFrame,
    *,
    max_train_size: int,
    n_points: int = 12,
    repeats: int = 3,
    seed: int = 24,
) -> dict:
    """Refit on growing subsamples of the training pool against a fixed test set.

    Samples are drawn only from ``train_pool``; ``test`` never changes, so the
    curve reflects sample size alone.
    """
    max_train_size = int(min(max_train_size, len(train_pool)))
    reference = smf.ols(formula, data=train_pool).fit()
    full_rank = design_rank(reference)

    # Start at two observations per parameter: below that nothing is estimable
    # and the curve is pure noise.
    smallest = max(2 * full_rank, 50)
    sizes = np.unique(
        np.geomspace(min(smallest, max_train_size), max_train_size, num=n_points).astype(int)
    )

    rows = []
    for size in sizes:
        for repeat in range(repeats):
            subsample = train_pool.sample(n=int(size), random_state=seed * 1000 + repeat)
            try:
                sub_fit = smf.ols(formula, data=subsample).fit()
            except Exception:  # pragma: no cover - singular tiny samples
                rows.append(
                    {
                        "sample_size": int(size),
                        "repeat": repeat,
                        "rmse_train": np.nan,
                        "rmse_test": np.nan,
                        "r2_test": np.nan,
                        "effective_params": np.nan,
                        "n_test_evaluated": 0,
                        "estimable": False,
                    }
                )
                continue

            sub_rank = design_rank(sub_fit)
            mask = support_mask(formula, subsample, test)
            evaluated = test[mask]
            predicted = sub_fit.predict(evaluated)
            finite = np.isfinite(np.asarray(predicted, dtype=float))
            evaluated, predicted = evaluated[finite], predicted[finite]

            rows.append(
                {
                    "sample_size": int(size),
                    "repeat": repeat,
                    "rmse_train": _rmse(subsample[TARGET], sub_fit.fittedvalues),
                    "rmse_test": _rmse(evaluated[TARGET], predicted) if len(evaluated) else np.nan,
                    "r2_test": float(r2_score(evaluated[TARGET], predicted)) if len(evaluated) else np.nan,
                    "effective_params": sub_rank,
                    "n_test_evaluated": int(len(evaluated)),
                    # A subsample that cannot reproduce the full design rank, or
                    # that has fewer than two observations per parameter, cannot
                    # support reliable estimation.
                    "estimable": bool(sub_rank == full_rank and size >= 2 * full_rank),
                }
            )

    curve = pd.DataFrame(rows)
    summary = (
        curve.groupby("sample_size", as_index=False)
        .agg(
            rmse_train=("rmse_train", "mean"),
            rmse_test_mean=("rmse_test", "mean"),
            rmse_test_min=("rmse_test", "min"),
            rmse_test_max=("rmse_test", "max"),
            rmse_test_std=("rmse_test", "std"),
            effective_params=("effective_params", "max"),
            estimable=("estimable", "all"),
        )
        .sort_values("sample_size")
        .reset_index(drop=True)
    )
    summary["rmse_test_mean_pp"] = summary["rmse_test_mean"] * 100.0
    summary["rmse_train_pp"] = summary["rmse_train"] * 100.0

    reliable = summary[summary["estimable"]]
    plateau = _approximate_plateau(reliable)

    return {
        "curve": curve,
        "summary": summary,
        "full_rank": int(full_rank),
        "n_test": int(len(test)),
        "n_pool": int(len(train_pool)),
        "repeats": int(repeats),
        "seed": int(seed),
        "plateau_size": plateau,
    }


def _approximate_plateau(summary: pd.DataFrame, tolerance: float = 0.02) -> int | None:
    """First sample size whose held-out RMSE is within ``tolerance`` of the best.

    This is a description of the curve that was computed, not a guaranteed
    minimum sample requirement.
    """
    usable = summary.dropna(subset=["rmse_test_mean"])
    if usable.empty:
        return None
    best = usable["rmse_test_mean"].min()
    within = usable[usable["rmse_test_mean"] <= best * (1 + tolerance)]
    return int(within["sample_size"].iloc[0]) if not within.empty else None


# --------------------------------------------------------------------------- #
# Stage 6: OLS vs LASSO vs Ridge
# --------------------------------------------------------------------------- #

LASSO_ALPHA_RANGE = (1e-5, 1e-1)
RIDGE_ALPHA_RANGE = (1e-2, 1e4)


def _alpha_grid(bounds: tuple[float, float], n: int = 60) -> np.ndarray:
    return np.geomspace(bounds[0], bounds[1], num=n)


def _design_matrices(
    formula: str, train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
    """Build the training design, then encode the test set with the same coding.

    Using the training ``design_info`` for the test set is what keeps predictor
    columns identical across train and test. The notebook calls ``dmatrices``
    separately on each split, which can produce different columns when a small
    training sample misses a category.
    """
    y_train_frame, x_train = dmatrices(formula, train, return_type="dataframe")
    design_info = x_train.design_info

    mask = support_mask(formula, train, test)
    supported_test = test[mask]
    unseen = unsupported_levels(formula, train, test)

    (x_test,) = build_design_matrices([design_info], supported_test, return_type="dataframe")
    y_test = np.asarray(supported_test[TARGET], dtype=float)

    if "Intercept" in x_train.columns:
        x_train = x_train.drop(columns=["Intercept"])
        x_test = x_test.drop(columns=["Intercept"])

    info = {
        "n_test_unsupported": int(len(test) - len(supported_test)),
        "unsupported_levels": unseen,
        "constant_columns": [c for c in x_train.columns if np.ptp(x_train[c].to_numpy()) == 0],
    }
    return x_train, np.asarray(y_train_frame).ravel(), x_test, y_test, supported_test, info


def _cv_alpha_path(
    estimator_factory,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    alphas: np.ndarray,
    k: int,
    seed: int,
) -> pd.DataFrame:
    """Mean CV RMSE for each alpha, standardising inside every fold."""
    folds = list(KFold(n_splits=k, shuffle=True, random_state=seed).split(x_train))
    x_values = x_train.to_numpy()
    records = []
    for alpha in alphas:
        fold_rmses = []
        for fit_index, validate_index in folds:
            pipeline = Pipeline(
                [("scaler", StandardScaler()), ("model", estimator_factory(alpha))]
            )
            pipeline.fit(x_values[fit_index], y_train[fit_index])
            predicted = pipeline.predict(x_values[validate_index])
            fold_rmses.append(_rmse(y_train[validate_index], predicted))
        records.append(
            {"alpha": float(alpha), "cv_rmse": float(np.mean(fold_rmses)),
             "cv_rmse_std": float(np.std(fold_rmses, ddof=1))}
        )
    path = pd.DataFrame(records)
    path["cv_rmse_pp"] = path["cv_rmse"] * 100.0
    return path


def _lasso(alpha: float) -> Lasso:
    # fit_intercept=True leaves the intercept out of the penalty.
    return Lasso(alpha=alpha, max_iter=50_000, tol=1e-7, fit_intercept=True)


def _ridge(alpha: float) -> Ridge:
    return Ridge(alpha=alpha, fit_intercept=True)


def compare_regularisation(
    df: pd.DataFrame,
    *,
    formula: str = COMPLEX_FORMULA,
    train_size: float = 0.008,
    seed: int = 24,
    cv_folds: int = 5,
    cv_seed: int = 4444,
    lasso_alpha: float | None = None,
    ridge_alpha: float | None = None,
    n_alphas: int = 60,
) -> dict:
    """Fit OLS, LASSO and Ridge on identical observations and predictor columns.

    ``lasso_alpha`` / ``ridge_alpha`` override the cross-validated choice when
    given. OLS is fitted on the same standardised design by least squares, which
    keeps the unpenalised baseline numerically stable and comparably scaled.
    """
    train, test = train_test_split(df, train_size=train_size, random_state=seed)
    x_train, y_train, x_test, y_test, _, info = _design_matrices(formula, train, test)

    lasso_alphas = _alpha_grid(LASSO_ALPHA_RANGE, n_alphas)
    ridge_alphas = _alpha_grid(RIDGE_ALPHA_RANGE, n_alphas)
    lasso_path = _cv_alpha_path(_lasso, x_train, y_train, lasso_alphas, cv_folds, cv_seed)
    ridge_path = _cv_alpha_path(_ridge, x_train, y_train, ridge_alphas, cv_folds, cv_seed)

    cv_lasso_alpha = float(lasso_path.loc[lasso_path["cv_rmse"].idxmin(), "alpha"])
    cv_ridge_alpha = float(ridge_path.loc[ridge_path["cv_rmse"].idxmin(), "alpha"])
    used_lasso_alpha = float(lasso_alpha) if lasso_alpha is not None else cv_lasso_alpha
    used_ridge_alpha = float(ridge_alpha) if ridge_alpha is not None else cv_ridge_alpha

    models = {
        "OLS": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "LASSO": Pipeline([("scaler", StandardScaler()), ("model", _lasso(used_lasso_alpha))]),
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", _ridge(used_ridge_alpha))]),
    }

    performance = []
    coefficients = {"feature": list(x_train.columns)}
    zero_counts = []
    constant = set(info["constant_columns"])

    for name, pipeline in models.items():
        pipeline.fit(x_train.to_numpy(), y_train)
        predicted = pipeline.predict(x_test.to_numpy())
        scaler = pipeline.named_steps["scaler"]
        standardised = np.asarray(pipeline.named_steps["model"].coef_, dtype=float).ravel()
        original = standardised / scaler.scale_

        coefficients[name] = original
        coefficients[f"{name}_standardised"] = standardised

        zeroed = [
            feature
            for feature, value in zip(x_train.columns, standardised)
            if value == 0.0 and feature not in constant
        ]
        zero_counts.append(
            {
                "model": name,
                "n_features": int(x_train.shape[1]),
                "n_zero": int(np.sum(standardised == 0.0)),
                "n_nonzero": int(np.sum(standardised != 0.0)),
                "n_zero_from_selection": len(zeroed),
                "n_zero_constant_column": int(len(constant)),
            }
        )
        performance.append(
            {
                "model": name,
                "alpha": {"OLS": np.nan, "LASSO": used_lasso_alpha, "Ridge": used_ridge_alpha}[name],
                "test_rmse": _rmse(y_test, predicted),
                "test_rmse_pp": _rmse(y_test, predicted) * 100.0,
                "test_r2": float(r2_score(y_test, predicted)),
            }
        )

    return {
        "performance": pd.DataFrame(performance),
        "coefficients": pd.DataFrame(coefficients),
        "shrinkage": pd.DataFrame(zero_counts),
        "lasso_path": lasso_path,
        "ridge_path": ridge_path,
        "cv_lasso_alpha": cv_lasso_alpha,
        "cv_ridge_alpha": cv_ridge_alpha,
        "used_lasso_alpha": used_lasso_alpha,
        "used_ridge_alpha": used_ridge_alpha,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_test_evaluated": int(len(y_test)),
        "n_features": int(x_train.shape[1]),
        "train_rank": matrix_rank(x_train.to_numpy()),
        "seed": int(seed),
        "cv_folds": int(cv_folds),
        "cv_seed": int(cv_seed),
        **info,
    }


# --------------------------------------------------------------------------- #
# Stage 7: time information
# --------------------------------------------------------------------------- #


def evaluate_time_models(
    df: pd.DataFrame,
    *,
    spec_keys: Sequence[str] = tuple(TIME_SPECS),
    mode: str = "within",
    test_size: float = 0.2,
    seed: int = 24,
    holdout_months: int = 6,
) -> dict:
    """Evaluate the time specifications under one of two validation goals.

    ``mode="within"`` predicts randomly held-out loans from months the model has
    already seen. ``mode="future"`` trains on earlier whole months and predicts
    later ones, which month-year dummy models cannot do: those months carry free
    coefficients that do not exist until the month is observed.
    """
    if mode == "within":
        train, test = create_split(df, test_size=test_size, seed=seed)
        test_months: list[str] = []
    elif mode == "future":
        train, test, test_months = split_by_month(df, holdout_months)
    else:
        raise ValueError("mode must be 'within' or 'future'")

    rows = []
    monthly_frames = []
    for key in spec_keys:
        spec = build_model_specification(key)

        if mode == "future" and key in MONTH_DUMMY_SPECS:
            rows.append(
                {
                    "model": spec.label,
                    "key": key,
                    "supported": False,
                    "reason": (
                        "Month-year dummies have no coefficient for a month that is not in the "
                        "training data, so this specification cannot forecast future months."
                    ),
                    "rmse_test": np.nan,
                    "rmse_test_pp": np.nan,
                    "r2_test": np.nan,
                    "n_train": int(len(train)),
                    "n_test_evaluated": 0,
                    "n_test_unsupported": int(len(test)),
                    "effective_params": np.nan,
                }
            )
            continue

        evaluation = fit_and_evaluate(spec.formula, train, test, with_coefficients=False)
        rows.append(
            {
                "model": spec.label,
                "key": key,
                "supported": True,
                "reason": "",
                "rmse_train": evaluation["rmse_train"],
                "rmse_train_pp": evaluation["rmse_train_pp"],
                "rmse_test": evaluation["rmse_test"],
                "rmse_test_pp": evaluation["rmse_test_pp"],
                "r2_test": evaluation["r2_test"],
                "n_train": evaluation["n_train"],
                "n_test_evaluated": evaluation["n_test_evaluated"],
                "n_test_unsupported": evaluation["n_test_unsupported"],
                "effective_params": evaluation["n_effective_params"],
            }
        )

        monthly = _monthly_means(evaluation, train, test, spec.label)
        monthly_frames.append(monthly)

    return {
        "results": pd.DataFrame(rows),
        "monthly": pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame(),
        "mode": mode,
        "test_months": test_months,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "seed": int(seed),
        "train_month_range": (train["ym"].min(), train["ym"].max()),
        "test_month_range": (test["ym"].min(), test["ym"].max()) if len(test) else ("—", "—"),
    }


def _monthly_means(evaluation: dict, train: pd.DataFrame, test: pd.DataFrame, label: str) -> pd.DataFrame:
    """Observed and fitted monthly mean interest rates for one specification."""
    fitted_train = pd.DataFrame(
        {
            "ym": train["ym"].to_numpy(),
            "observed": evaluation["train_actual"],
            "fitted": evaluation["train_predicted"],
            "split": "Training",
        }
    )
    frames = [fitted_train]
    if "test_index" in evaluation and len(evaluation["test_index"]):
        frames.append(
            pd.DataFrame(
                {
                    "ym": test.loc[evaluation["test_index"], "ym"].to_numpy(),
                    "observed": evaluation["test_actual"],
                    "fitted": evaluation["test_predicted"],
                    "split": "Held out",
                }
            )
        )
    combined = pd.concat(frames, ignore_index=True)
    monthly = (
        combined.groupby(["ym", "split"], as_index=False)
        .agg(observed_mean=("observed", "mean"), fitted_mean=("fitted", "mean"), n=("observed", "size"))
        .sort_values("ym")
    )
    monthly["model"] = label
    return monthly


# --------------------------------------------------------------------------- #
# Naive benchmark (assignment Q6), used as a reference line in stage 2
# --------------------------------------------------------------------------- #


def decile_benchmark(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Predict each loan by the training mean of its loan-amount decile."""
    deciles, edges = pd.qcut(train["loan_amnt"], q=10, labels=False, retbins=True, duplicates="drop")
    edges = edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    means = train.assign(_decile=deciles).groupby("_decile")[TARGET].mean()

    train_predicted = deciles.map(means)
    test_deciles = pd.cut(test["loan_amnt"], bins=edges, labels=False, include_lowest=True)
    test_predicted = test_deciles.map(means)
    usable = test_predicted.notna()

    return {
        "rmse_train": _rmse(train[TARGET], train_predicted),
        "rmse_test": _rmse(test.loc[usable, TARGET], test_predicted[usable]),
        "rmse_train_pp": _rmse(train[TARGET], train_predicted) * 100.0,
        "rmse_test_pp": _rmse(test.loc[usable, TARGET], test_predicted[usable]) * 100.0,
        "n_bins": int(means.size),
    }
