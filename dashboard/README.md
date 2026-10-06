# Interactive Regression Learning Dashboard

A seven-stage Streamlit dashboard built from the AM04 Lending Club assignment notebook. Every
number is calculated from the Lending Club CSV at run time — nothing is simulated, and no result
is copied from the notebook's saved outputs.

## Setup

```bash
pip install -r requirements.txt
```

Run it from this folder:

```bash
streamlit run app.py
```

The app opens at <http://localhost:8501>. It looks for `LendingClub_Data.csv` in the parent
folder by default; you can also upload a CSV or type a different path in the sidebar.

To re-run the quality checks without the interface:

```bash
python verify_checks.py
```

## Files

| File | What it holds |
| --- | --- |
| `app.py` | The Streamlit interface: seven tabs, controls, charts and explanations. |
| `core.py` | All calculation. Pure pandas/statsmodels/scikit-learn, no Streamlit. |
| `verify_checks.py` | Headless checks for the SOP's quality criteria. |
| `requirements.txt` | Dependencies. |

## The seven stages

1. **Inspect and clean** — how 42,538 raw rows become the 37,869-row modelling sample, with every
   removal listed.
2. **Simple regression** — Models 0 and 1, in-sample only, to establish the baseline.
3. **Feature engineering** — Models 0–3 compared on training fit, with the nested-model F test.
4. **Test and cross-validate** — a reserved test set, cross-validation inside the training set,
   and the test set scored once.
5. **Learning curve** — held-out error against training-sample size, against a fixed test set.
6. **OLS, LASSO and Ridge** — the over-parameterised specification on a small sample, where
   regularisation earns its keep.
7. **Time information** — the five time specifications under two different validation goals.

## Conventions worth knowing

- **Units.** `int_rate` is stored as a fraction (0.12 = 12%). RMSE is displayed in **percentage
  points**, so 1.05 pp means predictions are typically 1.05 percentage points off.
- **One sample.** A single complete-case sample is built in Stage 1 and used everywhere, so model
  comparisons never differ because of which rows happened to be available.
- **Shared split.** The test proportion, random seed and fold count are set in Stage 4 and reused
  by Stages 5 and 7. The sidebar always shows the settings currently in force.
- **Expensive stages wait.** Stages 4–7 calculate only when you press **Update results**. Changing
  a display option never refits a model, and if the controls drift away from the displayed result
  the app says so rather than showing mismatched numbers.
- **Download.** The sidebar exports every metric currently on screen as one CSV.

## Where this departs from the notebook, and why

These are corrections, not stylistic changes. Each one is also explained inside the relevant tab.

- **Grade ordering.** The notebook takes grade categories from `unique()`, which follows order of
  appearance. Here `grade` is an ordered category A < B < … < G.
- **Train/test predictor coding (Stage 6).** The notebook calls `dmatrices` separately on the
  training and test splits. With a 0.8% training sample that can produce different columns on each
  side. The dashboard builds the test design from the *training* `design_info`, so the coding is
  identical, and reports how many test rows had to be excluded for carrying unseen categories.
- **Unseen categorical levels.** Anywhere a fitted model meets a level it never saw, those rows are
  counted and excluded rather than predicted — a model has no coefficient for such a level, so a
  prediction would be an invented effect.
- **Design-matrix rank.** Rank is computed on a column-normalised matrix. Specifications mixing 0/1
  dummies with `annual_inc**2` span ~13 orders of magnitude, and `numpy.linalg.matrix_rank` sets
  its tolerance from the largest singular value, so on the raw matrix it discards columns that are
  genuinely independent.
- **Regularisation alpha grids.** LASSO and Ridge use separate logarithmic ranges rather than the
  notebook's linear grids. Their penalty values are not on a comparable scale.
- **Time models.** The notebook evaluates the time specifications in-sample and with a stratified
  CV that mixes months together. The dashboard separates the two questions explicitly: within-period
  prediction, and future-month forecasting — where the month-year dummy specifications are disabled,
  because a free coefficient per month cannot exist for a month that has not happened.
