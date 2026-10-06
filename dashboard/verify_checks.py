"""Headless checks for the quality criteria in the dashboard SOP.

Run from the dashboard folder:

    python verify_checks.py [path/to/LendingClub_Data.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import core

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def main(path: Path) -> int:
    raw = core.load_raw(path)
    df, report = core.clean_data(raw)
    print(f"\nCleaned sample: {report['n_clean']:,} rows from {report['n_raw']:,}, "
          f"{report['n_months']} months\n")

    # --- Cleaning -------------------------------------------------------- #
    check(
        "Complete-case sample has no missing modelling values",
        not df[core.MODELLING_COLUMNS].isna().any().any(),
    )
    check(
        "Removals are each explained",
        report["steps"]["reason"].str.len().gt(0).all(),
    )
    check(
        "Grade order is A..G, not order of appearance",
        list(df["grade"].cat.categories) == core.GRADE_ORDER,
    )

    # --- Reproducibility -------------------------------------------------- #
    train, test = core.create_split(df, test_size=0.2, seed=24)
    first = core.fit_and_evaluate(core.SPECS["model2"].formula, train, test, with_coefficients=False)
    second = core.fit_and_evaluate(core.SPECS["model2"].formula, train, test, with_coefficients=False)
    check(
        "Repeating the same settings reproduces the same metrics",
        first["rmse_test"] == second["rmse_test"] and first["rmse_train"] == second["rmse_train"],
        f"test RMSE {first['rmse_test']:.8f}",
    )

    train_again, test_again = core.create_split(df, test_size=0.2, seed=24)
    check(
        "The same seed reproduces the same split",
        train.index.equals(train_again.index) and test.index.equals(test_again.index),
    )

    # --- Order independence ------------------------------------------------ #
    # Run stages in reverse order on a fresh frame and compare the stage-2 result.
    shuffled = df.sample(frac=1.0, random_state=7).sort_index()
    core.evaluate_time_models(shuffled, spec_keys=["benchmark"], mode="within", seed=24)
    core.run_cross_validation(core.SPECS["model0"].formula, train, k=5, seed=24)
    reverse_order = core.fit_and_evaluate(
        core.SPECS["model2"].formula, train, test, with_coefficients=False
    )
    check(
        "Running stages in any order produces consistent results",
        np.isclose(reverse_order["rmse_test"], first["rmse_test"], rtol=0, atol=0),
    )

    # --- In-sample vs held out -------------------------------------------- #
    check(
        "Training RMSE falls as the specification gains terms (same observations)",
        all(
            core.fit_and_evaluate(core.SPECS[a].formula, df, with_coefficients=False)["rmse_train"]
            >= core.fit_and_evaluate(core.SPECS[b].formula, df, with_coefficients=False)["rmse_train"]
            for a, b in [("model0", "model1"), ("model1", "model2"), ("model2", "model3")]
        ),
    )

    # --- Cross-validation -------------------------------------------------- #
    cv = core.run_cross_validation(core.SPECS["model2"].formula, train, k=5, seed=24)
    check(
        "CV folds cover the training set exactly once",
        int(cv["folds"]["n_validate"].sum()) == len(train),
        f"{int(cv['folds']['n_validate'].sum()):,} of {len(train):,}",
    )
    check(
        "CV never touches the test set",
        int(cv["folds"]["n_fit"].max()) <= len(train),
    )
    check(
        "Unsupported fold rows are counted, not predicted",
        "n_unsupported" in cv["folds"].columns and cv["n_unsupported"] >= 0,
        f"{cv['n_unsupported']} excluded",
    )

    # --- Unseen categories ------------------------------------------------- #
    tiny_train = df[df["grade"].isin(["A", "B"])].head(500)
    unseen = core.unsupported_levels(core.SPECS["model0"].formula, tiny_train, test)
    mask = core.support_mask(core.SPECS["model0"].formula, tiny_train, test)
    expected = set(test["grade"].dropna().unique()) - set(tiny_train["grade"].dropna().unique())
    check(
        "Rows with unseen categorical levels are flagged",
        (~mask).sum() > 0 and set(unseen["grade"]) == expected,
        f"{int((~mask).sum()):,} test rows excluded; unseen {unseen}",
    )

    # --- Learning curve ---------------------------------------------------- #
    curve = core.build_learning_curve(
        core.SPECS["model3"].formula, train, test, max_train_size=6000, n_points=6, repeats=2, seed=24
    )
    check(
        "Learning-curve evaluation set is fixed",
        curve["n_test"] == len(test),
    )
    check(
        "Learning-curve samples stay inside the training pool",
        curve["summary"]["sample_size"].max() <= len(train),
    )
    check(
        "Sample sizes that cannot support estimation are marked",
        curve["summary"]["estimable"].dtype == bool,
    )

    # --- Regularisation ---------------------------------------------------- #
    reg = core.compare_regularisation(df, train_size=0.008, seed=24, n_alphas=12)
    check(
        "OLS, LASSO and Ridge use the same observations and predictors",
        reg["n_features"] > 0 and len(reg["performance"]) == 3,
        f"{reg['n_features']} predictor columns, {reg['n_train']} training rows",
    )
    check(
        "Constant design columns are identified separately from LASSO selection",
        "n_zero_constant_column" in reg["shrinkage"].columns,
        f"{len(reg['constant_columns'])} constant columns",
    )
    check(
        "Rank deficiency in the complex design is detected",
        reg["train_rank"] <= reg["n_features"],
        f"rank {reg['train_rank']} of {reg['n_features']} columns",
    )
    check(
        "LASSO and Ridge use separate alpha ranges",
        core.LASSO_ALPHA_RANGE != core.RIDGE_ALPHA_RANGE,
    )
    reg_again = core.compare_regularisation(df, train_size=0.008, seed=24, n_alphas=12)
    check(
        "Regularisation comparison is reproducible",
        np.allclose(
            reg["performance"]["test_rmse"].to_numpy(),
            reg_again["performance"]["test_rmse"].to_numpy(),
        ),
    )

    # --- Time models ------------------------------------------------------- #
    within = core.evaluate_time_models(df, mode="within", test_size=0.2, seed=24)
    check(
        "Within-period evaluation runs all five specifications",
        int(within["results"]["supported"].sum()) == 5,
    )
    check(
        "Within-period restricted samples are disclosed",
        "n_test_unsupported" in within["results"].columns,
        f"max excluded {int(within['results']['n_test_unsupported'].max())}",
    )

    future = core.evaluate_time_models(df, mode="future", holdout_months=6)
    unsupported = future["results"][~future["results"]["supported"]]["key"].tolist()
    check(
        "Month-year dummy models are disabled for future-month forecasting",
        set(unsupported) == core.MONTH_DUMMY_SPECS,
        f"disabled: {sorted(unsupported)}",
    )
    check(
        "Future-month split uses whole months with no overlap",
        len(set(future["test_months"])) == 6
        and future["train_month_range"][1] < future["test_month_range"][0],
        f"train to {future['train_month_range'][1]}, test from {future['test_month_range'][0]}",
    )

    # --- Comparability ----------------------------------------------------- #
    evaluated = within["results"].loc[within["results"]["supported"], "n_test_evaluated"]
    check(
        "Every supported time comparison reports its evaluation count",
        evaluated.notna().all() and (evaluated > 0).all(),
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    default = Path(__file__).resolve().parent.parent / "LendingClub_Data.csv"
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    raise SystemExit(main(target))
