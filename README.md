# DataCrunch — Interactive Regression Learning Dashboard

An interactive Streamlit dashboard that walks through a full linear-modelling workflow on the
Lending Club loan dataset, built from an AM04 course assignment.

**Cleaning → simple regression → feature engineering → validation → learning curves → OLS, LASSO
and Ridge → time effects**

Every number is calculated from the dataset at run time. Nothing is simulated, and no result is
read back from a notebook's saved output — moving a control refits the model.

## Quick start

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

The app opens at <http://localhost:8501> and picks up `LendingClub_Data.csv` from the repository
root automatically. You can also upload a different CSV from the sidebar.

To run the quality checks without the interface:

```bash
cd dashboard && python verify_checks.py
```

## What's in here

| Path | Contents |
| --- | --- |
| `dashboard/app.py` | The Streamlit interface: seven tabs, controls, charts, explanations. |
| `dashboard/core.py` | All calculation — pandas, statsmodels, scikit-learn. No Streamlit imports, so it can be run or tested headlessly. |
| `dashboard/verify_checks.py` | Headless checks: reproducibility, no test-set leakage, unseen-category handling, rank detection. |
| `dashboard/README.md` | Fuller notes, including where the dashboard deliberately departs from the source notebook. |
| `LendingClub_Data.csv` | The loan dataset (42,538 rows; 37,869 complete cases). |
| `MonthBondYields2024.csv` | Monthly US Treasury yields. |

## The seven stages

1. **Inspect and clean** — how the raw rows become the modelling sample, with every removal listed.
2. **Simple regression** — the baseline models, in-sample only.
3. **Feature engineering** — nested specifications compared on training fit, with the F test.
4. **Test and cross-validate** — a reserved test set, cross-validation inside the training set only,
   and the test set scored exactly once.
5. **Learning curve** — held-out error against training-sample size, against a fixed test set.
6. **OLS, LASSO and Ridge** — an over-parameterised specification on a small sample, where
   regularisation earns its keep.
7. **Time information** — five time specifications under two different validation goals.

## Things the dashboard is careful about

- **Units.** `int_rate` is stored as a fraction (0.12 = 12%), so RMSE is shown in **percentage
  points**: 1.05 pp means predictions are typically 1.05 percentage points off.
- **One sample throughout.** A single complete-case sample is built in Stage 1 and reused, so model
  comparisons never differ because of which rows happened to be available.
- **No leakage.** Cross-validation runs strictly inside the training set, each fold builds its own
  design matrix, and no parameter is ever chosen using the test set.
- **Unseen categories are flagged, not predicted.** A model has no coefficient for a level it never
  saw, so those rows are counted and excluded rather than given an invented effect.
- **Unsupported comparisons are disabled.** Month-year dummy models cannot forecast future months —
  the dashboard says so and switches them off rather than printing a misleading number.

See [`dashboard/README.md`](dashboard/README.md) for the full list of corrections made relative to
the source notebook.
