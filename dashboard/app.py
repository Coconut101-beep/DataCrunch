"""Interactive regression learning dashboard for the Lending Club assignment.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.stats.anova import anova_lm

import core

st.set_page_config(page_title="Lending Club regression dashboard", layout="wide")

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "LendingClub_Data.csv"
PLOT_TEMPLATE = "plotly_white"
SCATTER_LIMIT = 5000

UNITS_NOTE = (
    "`int_rate` is stored as a fraction (0.12 = 12%), so RMSE is reported in "
    "**percentage points (pp)**: 1.05 pp means predictions are typically about "
    "1.05 percentage points away from the actual rate."
)


# --------------------------------------------------------------------------- #
# Cached computation wrappers
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def cached_raw(file_bytes: bytes | None, path: str) -> pd.DataFrame:
    import io

    return core.load_raw(io.BytesIO(file_bytes) if file_bytes is not None else path)


@st.cache_data(show_spinner=False)
def cached_clean(raw: pd.DataFrame):
    return core.clean_data(raw)


@st.cache_data(show_spinner=False)
def cached_evaluation(df: pd.DataFrame, formula: str, test_size: float | None, seed: int) -> dict:
    if test_size is None:
        train, test = df, None
    else:
        train, test = core.create_split(df, test_size, seed)
    result = core.fit_and_evaluate(formula, train, test)
    result.pop("fit", None)
    return result


@st.cache_data(show_spinner=False)
def cached_anova(df: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    import statsmodels.formula.api as smf

    fits = [smf.ols(core.SPECS[key].formula, data=df).fit() for key in keys]
    table = anova_lm(*fits)
    table.insert(0, "model", [core.SPECS[key].label for key in keys])
    return table.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def cached_cross_validation(df: pd.DataFrame, formula: str, test_size: float, seed: int, k: int) -> dict:
    train, _ = core.create_split(df, test_size, seed)
    return core.run_cross_validation(formula, train, k, seed)


@st.cache_data(show_spinner=False)
def cached_learning_curve(
    df: pd.DataFrame, formula: str, test_size: float, seed: int, max_train_size: int, repeats: int
) -> dict:
    train, test = core.create_split(df, test_size, seed)
    return core.build_learning_curve(
        formula, train, test, max_train_size=max_train_size, n_points=12, repeats=repeats, seed=seed
    )


@st.cache_data(show_spinner=False)
def cached_regularisation(
    df: pd.DataFrame, train_size: float, seed: int, lasso_alpha: float | None, ridge_alpha: float | None
) -> dict:
    return core.compare_regularisation(
        df, train_size=train_size, seed=seed, lasso_alpha=lasso_alpha, ridge_alpha=ridge_alpha
    )


@st.cache_data(show_spinner=False)
def cached_time_models(
    df: pd.DataFrame, mode: str, test_size: float, seed: int, holdout_months: int
) -> dict:
    return core.evaluate_time_models(
        df, mode=mode, test_size=test_size, seed=seed, holdout_months=holdout_months
    )


@st.cache_data(show_spinner=False)
def cached_decile_benchmark(df: pd.DataFrame, test_size: float, seed: int) -> dict:
    train, test = core.create_split(df, test_size, seed)
    return core.decile_benchmark(train, test)


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #


def register(stage: str, rows: list[dict]) -> None:
    """Record the current stage results so they can be downloaded together."""
    st.session_state.setdefault("results", {})[stage] = pd.DataFrame(rows)


def results_csv() -> bytes:
    registry = st.session_state.get("results", {})
    if not registry:
        return b""
    frames = []
    for stage, frame in registry.items():
        copy = frame.copy()
        copy.insert(0, "stage", stage)
        frames.append(copy)
    return pd.concat(frames, ignore_index=True).to_csv(index=False).encode("utf-8")


def stage_gate(stage_key: str, params: dict, compute):
    """Run ``compute`` only when the user asks, and keep the result until they do.

    Returns ``(result, params_used)``. Display-only changes never trigger a
    refit, and when the controls no longer match the stored result the user is
    told to recalculate rather than being shown mismatched numbers.
    """
    state = st.session_state.setdefault(f"gate_{stage_key}", {})
    signature = json.dumps(params, sort_keys=True, default=str)

    if st.button("Update results", key=f"update_{stage_key}", type="primary"):
        with st.spinner("Calculating from the dataset…"):
            state["result"] = compute()
            state["signature"] = signature
            state["params"] = dict(params)

    if "result" not in state:
        st.info("Press **Update results** to calculate with the current settings.")
        return None, params
    if state["signature"] != signature:
        st.warning(
            "The controls have changed since this result was calculated. "
            "Press **Update results** to recalculate.",
            icon="⚠️",
        )
    return state["result"], state["params"]


def settings_caption(params: dict) -> None:
    """Show the settings that produced the numbers on screen."""
    st.caption(" · ".join(f"**{label}:** {value}" for label, value in params.items()))


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    for column, (label, value, help_text) in zip(st.columns(len(items)), items):
        column.metric(label, value, help=help_text)


def sample_for_plot(frame: pd.DataFrame, limit: int = SCATTER_LIMIT) -> tuple[pd.DataFrame, bool]:
    if len(frame) <= limit:
        return frame, False
    return frame.sample(limit, random_state=0), True


def spec_label(key: str) -> str:
    return core.SPECS[key].label


# --------------------------------------------------------------------------- #
# Sidebar: data source and shared state
# --------------------------------------------------------------------------- #

st.sidebar.title("Data and settings")

upload = st.sidebar.file_uploader("Lending Club CSV", type="csv")
local_path = st.sidebar.text_input("…or a local file path", value=str(DEFAULT_DATA_PATH))

if upload is None and not Path(local_path).exists():
    st.title("Interactive regression learning dashboard")
    st.error(
        "No dataset available. Upload the Lending Club CSV in the sidebar, or give a path to a "
        "local copy. The dashboard calculates every number from this file — it has no stored results."
    )
    st.stop()

try:
    raw_data = cached_raw(upload.getvalue() if upload is not None else None, local_path)
    clean, clean_report = cached_clean(raw_data)
except Exception as error:  # pragma: no cover - surfaced in the interface
    st.title("Interactive regression learning dashboard")
    st.error(f"The file could not be read as the Lending Club export: {error}")
    st.stop()

st.session_state.setdefault("test_size", 0.20)
st.session_state.setdefault("seed", 24)
st.session_state.setdefault("cv_folds", 5)
st.session_state.setdefault("selected_spec", "model2")

TEST_SIZE = st.session_state["test_size"]
SEED = st.session_state["seed"]

st.sidebar.markdown("---")
st.sidebar.subheader("Active settings")
st.sidebar.markdown(
    f"""
- **Modelling sample:** {clean_report['n_clean']:,} complete cases
- **Split method:** random hold-out, {TEST_SIZE:.0%} test
- **Random seed:** {SEED}
- **CV folds:** {st.session_state['cv_folds']}
- **Selected specification:** {spec_label(st.session_state['selected_spec'])}
"""
)
st.sidebar.caption("The split proportion, seed and fold count are set in Stage 4 and reused by Stages 5 and 7.")

st.sidebar.markdown("---")
csv_bytes = results_csv()
st.sidebar.download_button(
    "Download current results (CSV)",
    data=csv_bytes if csv_bytes else b"stage,metric,value\n",
    file_name="dashboard_results.csv",
    mime="text/csv",
    disabled=not csv_bytes,
    help="Every metric currently displayed across the stages you have calculated.",
)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.title("Interactive regression learning dashboard")
st.markdown(
    "Cleaning → simple regression → feature engineering → validation → learning curves → "
    "regularisation → time effects, all calculated from the Lending Club file you supplied."
)
st.caption(UNITS_NOTE)

tabs = st.tabs(
    [
        "1 · Inspect and clean",
        "2 · Simple regression",
        "3 · Feature engineering",
        "4 · Test and cross-validate",
        "5 · Learning curve",
        "6 · OLS, LASSO, Ridge",
        "7 · Time information",
    ]
)

# --------------------------------------------------------------------------- #
# Stage 1 — Inspect and clean
# --------------------------------------------------------------------------- #

with tabs[0]:
    st.subheader("How raw observations become the modelling sample")
    st.write(
        "Every later stage uses the same complete-case sample built here, so model comparisons "
        "are never confounded by differences in which rows were available."
    )

    controls = st.columns(2)
    view = controls[0].radio("Preview", ["Cleaned data", "Raw data"], horizontal=True)
    inspect_columns = [
        core.TARGET, "loan_amnt", "dti", "annual_inc", "grade", "term", "delinq_2yrs", "issue_d"
    ]
    variable = controls[1].selectbox("Variable to inspect", inspect_columns, index=0)

    steps = clean_report["steps"]
    metric_row(
        [
            ("Rows before cleaning", f"{clean_report['n_raw']:,}", "Rows in the file below the banner line."),
            ("Rows after cleaning", f"{clean_report['n_clean']:,}", "The common complete-case modelling sample."),
            (
                "Rows removed",
                f"{clean_report['n_raw'] - clean_report['n_clean']:,}",
                "Each removal is listed in the methodology expander.",
            ),
            ("Issue months covered", f"{clean_report['n_months']}", "Distinct month-year values."),
        ]
    )

    series = clean[variable]
    if variable == "issue_d":
        monthly = clean.groupby(clean["issue_d"].dt.to_period("M").astype(str)).size()
        figure = px.bar(
            x=monthly.index, y=monthly.to_numpy(),
            labels={"x": "Issue month", "y": "Loans"}, template=PLOT_TEMPLATE,
        )
    elif isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object:
        counts = series.value_counts().sort_index()
        figure = px.bar(
            x=counts.index.astype(str), y=counts.to_numpy(),
            labels={"x": variable, "y": "Loans"}, template=PLOT_TEMPLATE,
        )
    else:
        figure = px.histogram(clean, x=variable, nbins=40, template=PLOT_TEMPLATE)
    figure.update_layout(
        title=f"Distribution of {variable} in the cleaned sample", height=420, bargap=0.05
    )
    st.plotly_chart(figure, use_container_width=True)

    preview = raw_data.iloc[:, : core.RAW_KEEP_COLUMNS] if view == "Raw data" else clean
    st.dataframe(preview.head(12), use_container_width=True)

    register(
        "1 Cleaning",
        [
            {"model": "—", "metric": "rows_raw", "value": clean_report["n_raw"]},
            {"model": "—", "metric": "rows_clean", "value": clean_report["n_clean"]},
            {"model": "—", "metric": "months", "value": clean_report["n_months"]},
        ],
    )

    with st.expander("Cleaning steps, missing values and methodology"):
        st.markdown("**Every removal, in order**")
        st.dataframe(steps, use_container_width=True, hide_index=True)

        st.markdown("**Missing values among the modelling variables**")
        st.dataframe(clean_report["missing_by_variable"], use_container_width=True, hide_index=True)

        st.markdown(
            f"""
**What this stage does and why**

- The export carries a one-line prospectus banner above the header row, so it is read with
  `skiprows=1`. Only the first {core.RAW_KEEP_COLUMNS} columns hold data; the remaining
  {clean_report['trailing_unnamed_columns']} are empty padding.
- Rows with no `int_rate` are removed: the target cannot be fitted or scored when absent.
- Dates are parsed explicitly with `format="%m/%d/%Y"`, and `loan_amnt`, `dti`, `annual_inc`
  and `int_rate` are coerced to numbers (the file stores some of them with surrounding spaces).
- `grade` is set as an **ordered** category A < B < … < G. The notebook derived this order from
  `unique()`, which follows order of appearance rather than the intended ranking.
- Columns dropped as unused: {", ".join(f"`{c}`" for c in clean_report["dropped_columns"])}.
- A single complete-case sample is formed over
  {", ".join(f"`{c}`" for c in core.MODELLING_COLUMNS)} so that every model in every stage is
  compared on exactly the same observations.
- `t_linear` (months since the first loan) and `ym` (month-year label) are created once here,
  so the time stage cannot define them differently.

The original file is never modified, and no optional cleaning switches are offered: changing
which rows are dropped would silently change every comparison in the dashboard.
"""
        )
        st.code(
            """df = raw.iloc[:, :19]
df = df[df["int_rate"].notna()]
df["issue_d"] = pd.to_datetime(df["issue_d"], format="%m/%d/%Y", errors="coerce")
df["term"] = df["term (months)"].astype("category")
df["grade"] = pd.Categorical(df["grade"], categories=list("ABCDEFG"), ordered=True)
df = df.dropna(subset=MODELLING_COLUMNS)""",
            language="python",
        )

# --------------------------------------------------------------------------- #
# Stage 2 — Simple regression
# --------------------------------------------------------------------------- #

with tabs[1]:
    st.subheader("The baseline, before any complexity is added")
    st.write(
        "These two specifications are fitted on the whole modelling sample, so the metrics "
        "describe how well the model *describes* these loans, not how well it predicts new ones."
    )

    controls = st.columns(2)
    stage2_key = controls[0].radio(
        "Specification", core.STAGE2_SPECS, format_func=spec_label, horizontal=True, key="stage2_spec"
    )
    stage2_colour = controls[1].selectbox(
        "Colour points by", ["grade", "term", "loan_amnt", "dti", "annual_inc"], key="stage2_colour"
    )

    stage2 = cached_evaluation(clean, core.SPECS[stage2_key].formula, None, SEED)

    metric_row(
        [
            ("Training RMSE", f"{stage2['rmse_train_pp']:.3f} pp", "In-sample root mean squared error."),
            ("R-squared", f"{stage2['r2_train']:.4f}", "Share of in-sample variation explained."),
            ("Adjusted R-squared", f"{stage2['adj_r2_train']:.4f}", "Penalised for the number of terms."),
            ("Effective parameters", f"{stage2['n_effective_params']}", "Rank of the design matrix."),
            ("Observations", f"{stage2['n_train']:,}", None),
        ]
    )

    colour_values = clean[stage2_colour]
    is_categorical = isinstance(colour_values.dtype, pd.CategoricalDtype)
    stage2_frame = pd.DataFrame(
        {
            "actual": stage2["train_actual"] * 100,
            "predicted": stage2["train_predicted"] * 100,
            stage2_colour: colour_values.astype(str).to_numpy() if is_categorical else colour_values.to_numpy(),
        }
    )
    plotted, was_sampled = sample_for_plot(stage2_frame)
    stage2_figure = px.scatter(
        plotted,
        x="predicted",
        y="actual",
        color=stage2_colour,
        opacity=0.5,
        template=PLOT_TEMPLATE,
        category_orders=(
            {stage2_colour: [str(level) for level in colour_values.cat.categories]}
            if is_categorical
            else {}
        ),
        labels={"predicted": "Predicted interest rate (%)", "actual": "Actual interest rate (%)"},
    )
    limits = [
        min(plotted["predicted"].min(), plotted["actual"].min()),
        max(plotted["predicted"].max(), plotted["actual"].max()),
    ]
    stage2_figure.add_trace(
        go.Scatter(
            x=limits, y=limits, mode="lines", name="Perfect prediction",
            line=dict(dash="dash", color="#444"),
        )
    )
    stage2_figure.update_layout(
        title=f"Actual versus predicted interest rates — {spec_label(stage2_key)}", height=480
    )
    st.plotly_chart(stage2_figure, use_container_width=True)
    if was_sampled:
        st.caption(
            f"A random {SCATTER_LIMIT:,} of {stage2['n_train']:,} loans are drawn for legibility; "
            "all metrics use every observation."
        )

    settings_caption(
        {
            "Specification": spec_label(stage2_key),
            "Fitted on": f"all {stage2['n_train']:,} loans",
            "Evaluation": "in-sample",
        }
    )

    register(
        f"2 Simple regression",
        [
            {"model": spec_label(stage2_key), "metric": "rmse_train_pp", "value": stage2["rmse_train_pp"]},
            {"model": spec_label(stage2_key), "metric": "r2_train", "value": stage2["r2_train"]},
            {"model": spec_label(stage2_key), "metric": "adj_r2_train", "value": stage2["adj_r2_train"]},
            {"model": spec_label(stage2_key), "metric": "n_effective_params", "value": stage2["n_effective_params"]},
        ],
    )

    with st.expander("Coefficients, methodology and code"):
        st.markdown(f"**Formula** — `{core.SPECS[stage2_key].formula}`")
        st.dataframe(
            stage2["coefficients"].style.format(
                {
                    "coefficient": "{:.3e}",
                    "std_error": "{:.3e}",
                    "t": "{:.2f}",
                    "p_value": "{:.4f}",
                    "ci_lower": "{:.3e}",
                    "ci_upper": "{:.3e}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            """
**Explanatory fit is not predictive performance.** Everything on this tab is measured on the
same loans the model was fitted to, so it can only tell you how well the specification
*describes* this sample. A model can describe a sample almost perfectly and still predict new
loans poorly, which is exactly what Stage 4 tests. R-squared also cannot fall when a term is
added, so it is no guide to whether the extra term is worth having.

Coefficients are in the units of the fraction target: a coefficient of 0.0043 on the 60-month
term means a 60-month loan carries about 0.43 percentage points more interest than a
comparable 36-month loan.
"""
        )
        st.code(
            f'model = smf.ols("{core.SPECS[stage2_key].formula}", data=df_clean).fit()',
            language="python",
        )

# --------------------------------------------------------------------------- #
# Stage 3 — Feature engineering
# --------------------------------------------------------------------------- #

with tabs[2]:
    st.subheader("What happens when interactions and nonlinear terms are added")
    st.write(
        "The four assignment specifications are nested, so each one can only fit these loans at "
        "least as well as the one before it — which is precisely why training fit cannot rank them."
    )

    controls = st.columns(2)
    stage3_keys = controls[0].multiselect(
        "Specifications to compare",
        core.STAGE3_SPECS,
        default=core.STAGE3_SPECS,
        format_func=spec_label,
        key="stage3_keys",
    )
    focus = controls[1].selectbox(
        "Carry forward to Stages 4 and 5",
        core.STAGE4_SPECS,
        index=core.STAGE4_SPECS.index(st.session_state["selected_spec"]),
        format_func=spec_label,
        key="stage3_focus",
    )
    st.session_state["selected_spec"] = focus

    if not stage3_keys:
        st.info("Select at least one specification to compare.")
    else:
        stage3 = {key: cached_evaluation(clean, core.SPECS[key].formula, None, SEED) for key in stage3_keys}
        comparison = pd.DataFrame(
            [
                {
                    "model": spec_label(key),
                    "Training RMSE (pp)": result["rmse_train_pp"],
                    "Adjusted R-squared": result["adj_r2_train"],
                    "Effective parameters": result["n_effective_params"],
                    "AIC": result["aic"],
                    "BIC": result["bic"],
                }
                for key, result in stage3.items()
            ]
        )

        chosen = stage3.get(focus) or cached_evaluation(clean, core.SPECS[focus].formula, None, SEED)
        metric_row(
            [
                ("Training RMSE", f"{chosen['rmse_train_pp']:.3f} pp", f"{spec_label(focus)}, in-sample."),
                ("Adjusted R-squared", f"{chosen['adj_r2_train']:.4f}", "Penalised for added terms."),
                ("Effective parameters", f"{chosen['n_effective_params']}", "Rank of the design matrix."),
                (
                    "Terms in formula",
                    f"{chosen['n_params']}",
                    "Design columns before rank reduction."
                    + (" Rank deficient." if chosen["rank_deficient"] else ""),
                ),
            ]
        )

        stage3_figure = go.Figure()
        stage3_figure.add_trace(
            go.Bar(
                x=comparison["model"],
                y=comparison["Training RMSE (pp)"],
                name="Training RMSE (pp)",
                marker_color="#2E86AB",
                text=[f"{v:.3f}" for v in comparison["Training RMSE (pp)"]],
                textposition="outside",
            )
        )
        stage3_figure.add_trace(
            go.Scatter(
                x=comparison["model"],
                y=comparison["Adjusted R-squared"],
                name="Adjusted R-squared",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color="#D95D39", width=3),
            )
        )
        stage3_figure.update_layout(
            template=PLOT_TEMPLATE,
            title="Training fit across the nested specifications",
            yaxis=dict(title="Training RMSE (pp)"),
            yaxis2=dict(title="Adjusted R-squared", overlaying="y", side="right", showgrid=False),
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(stage3_figure, use_container_width=True)
        st.dataframe(
            comparison.style.format(
                {
                    "Training RMSE (pp)": "{:.4f}",
                    "Adjusted R-squared": "{:.4f}",
                    "AIC": "{:.0f}",
                    "BIC": "{:.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        settings_caption(
            {
                "Fitted on": f"all {chosen['n_train']:,} loans",
                "Evaluation": "in-sample",
                "Carried to Stages 4 and 5": spec_label(focus),
            }
        )

        register(
            "3 Feature engineering",
            [
                {"model": row["model"], "metric": metric, "value": row[metric]}
                for _, row in comparison.iterrows()
                for metric in ["Training RMSE (pp)", "Adjusted R-squared", "Effective parameters"]
            ],
        )

        with st.expander("Formulas, nested-model test and methodology"):
            for key in stage3_keys:
                st.markdown(f"**{spec_label(key)}** — `{core.SPECS[key].formula}`  \n{core.SPECS[key].note}")

            nested = [key for key in core.STAGE3_SPECS if key in stage3_keys]
            if len(nested) > 1:
                st.markdown("**Nested-model comparison (`anova_lm`)**")
                st.dataframe(cached_anova(clean, tuple(nested)), use_container_width=True, hide_index=True)
                st.caption(
                    "The F test says whether the extra terms reduce the residual sum of squares by "
                    "more than their cost in degrees of freedom **on these loans**. It is still an "
                    "in-sample test."
                )

            st.markdown(
                """
**Why this stage cannot pick a model.** Adding features can improve training fit without
improving performance on unseen loans. Each specification here contains the one before it, so
its training RMSE is mathematically guaranteed not to be worse — a lower number is not evidence
of a better model. Adjusted R-squared, AIC and BIC apply a penalty for extra terms, which helps,
but they still only look at the loans used for fitting. The honest comparison is a held-out one,
which is what Stage 4 does with the specification you carry forward from here.
"""
            )

# --------------------------------------------------------------------------- #
# Stage 4 — Held-out testing and cross-validation
# --------------------------------------------------------------------------- #

with tabs[3]:
    selected = st.session_state["selected_spec"]
    st.subheader("Separating training fit, validation and final test performance")
    st.write(
        "A test set is reserved and left untouched; cross-validation happens only inside the "
        "training set; the test set is scored once at the end."
    )
    st.info(f"Specification under test: **{spec_label(selected)}** — chosen in Stage 3.", icon="📌")

    controls = st.columns(3)
    test_size = controls[0].slider("Test-set proportion", 0.10, 0.40, TEST_SIZE, 0.05, key="stage4_test_size")
    folds = controls[1].radio("Cross-validation folds", [5, 10], horizontal=True, key="stage4_folds")
    seed = controls[2].number_input("Random seed", 0, 9999, SEED, 1, key="stage4_seed")

    st.session_state["test_size"] = test_size
    st.session_state["seed"] = int(seed)
    st.session_state["cv_folds"] = folds

    stage4_params = {
        "specification": selected,
        "test_size": float(test_size),
        "folds": int(folds),
        "seed": int(seed),
    }

    def compute_stage4():
        formula = core.SPECS[selected].formula
        return {
            "evaluation": cached_evaluation(clean, formula, float(test_size), int(seed)),
            "cv": cached_cross_validation(clean, formula, float(test_size), int(seed), int(folds)),
            "naive": cached_decile_benchmark(clean, float(test_size), int(seed)),
        }

    stage4, stage4_used = stage_gate("stage4", stage4_params, compute_stage4)

    if stage4 is not None:
        evaluation, cv, naive = stage4["evaluation"], stage4["cv"], stage4["naive"]
        metric_row(
            [
                ("Training RMSE", f"{evaluation['rmse_train_pp']:.3f} pp", "Fit on the training set."),
                (
                    f"{stage4_used['folds']}-fold CV RMSE",
                    f"{cv['rmse_mean_pp']:.3f} pp",
                    "Mean across folds, inside the training set only.",
                ),
                ("Test RMSE", f"{evaluation['rmse_test_pp']:.3f} pp", "The untouched test set, scored once."),
                ("Test R-squared", f"{evaluation['r2_test']:.4f}", "Negative means worse than the test mean."),
                (
                    "Naive decile RMSE",
                    f"{naive['rmse_test_pp']:.3f} pp",
                    "Predicting each loan by its loan-amount decile mean.",
                ),
            ]
        )

        stage4_figure = go.Figure(
            go.Bar(
                x=["Training", f"{stage4_used['folds']}-fold CV", "Test", "Naive decile (test)"],
                y=[
                    evaluation["rmse_train_pp"],
                    cv["rmse_mean_pp"],
                    evaluation["rmse_test_pp"],
                    naive["rmse_test_pp"],
                ],
                marker_color=["#7FB069", "#F2C14E", "#2E86AB", "#B0B0B0"],
                text=[
                    f"{evaluation['rmse_train_pp']:.3f}",
                    f"{cv['rmse_mean_pp']:.3f}",
                    f"{evaluation['rmse_test_pp']:.3f}",
                    f"{naive['rmse_test_pp']:.3f}",
                ],
                textposition="outside",
            )
        )
        stage4_figure.update_layout(
            template=PLOT_TEMPLATE,
            title=f"RMSE by evaluation method — {spec_label(stage4_used['specification'])}",
            yaxis_title="RMSE (percentage points)",
            height=460,
        )
        st.plotly_chart(stage4_figure, use_container_width=True)

        gap = evaluation["rmse_test_pp"] - evaluation["rmse_train_pp"]
        st.markdown(
            f"**Reading this:** the test RMSE is {gap:+.3f} pp away from the training RMSE. "
            + (
                "A gap this small means the specification is not relying on peculiarities of the "
                "training loans."
                if abs(gap) < 0.05
                else "A gap this large is the signature of a specification that has fitted features "
                "of the training loans that do not carry over."
            )
        )

        if evaluation["n_test_unsupported"]:
            st.warning(
                f"{evaluation['n_test_unsupported']:,} of {evaluation['n_test']:,} test rows were "
                f"excluded from scoring because they carry categorical levels absent from the "
                f"training set: {evaluation['unsupported_levels']}. The model has no coefficient "
                "for those levels, so predicting them would invent an effect.",
                icon="⚠️",
            )
        if cv["n_unsupported"]:
            st.warning(
                f"{cv['n_unsupported']:,} fold-validation rows were excluded across folds for the "
                "same reason.",
                icon="⚠️",
            )
        if evaluation["rank_deficient"]:
            st.warning(
                f"The training design matrix is rank deficient: {evaluation['n_params']} columns "
                f"but rank {evaluation['n_effective_params']}. Some coefficients are not separately "
                "identified on these observations.",
                icon="⚠️",
            )

        settings_caption(
            {
                "Specification": spec_label(stage4_used["specification"]),
                "Split": f"{1 - stage4_used['test_size']:.0%} train / {stage4_used['test_size']:.0%} test, random",
                "Seed": stage4_used["seed"],
                "Folds": stage4_used["folds"],
                "Training rows": f"{evaluation['n_train']:,}",
                "Test rows scored": f"{evaluation['n_test_evaluated']:,}",
            }
        )

        register(
            "4 Validation",
            [
                {"model": spec_label(stage4_used["specification"]), "metric": "rmse_train_pp", "value": evaluation["rmse_train_pp"]},
                {"model": spec_label(stage4_used["specification"]), "metric": "rmse_cv_pp", "value": cv["rmse_mean_pp"]},
                {"model": spec_label(stage4_used["specification"]), "metric": "rmse_test_pp", "value": evaluation["rmse_test_pp"]},
                {"model": spec_label(stage4_used["specification"]), "metric": "r2_test", "value": evaluation["r2_test"]},
                {"model": "Naive decile", "metric": "rmse_test_pp", "value": naive["rmse_test_pp"]},
            ],
        )

        with st.expander("Fold-level results, rules and code"):
            st.markdown(f"**Fold-level RMSE** (mean {cv['rmse_mean_pp']:.4f} pp, "
                        f"standard deviation {cv['rmse_std'] * 100:.4f} pp)")
            fold_table = cv["folds"].copy()
            fold_table["rmse_pp"] = fold_table["rmse"] * 100
            st.dataframe(
                fold_table[["fold", "n_fit", "n_validate", "n_evaluated", "n_unsupported", "rmse_pp", "r2"]]
                .style.format({"rmse_pp": "{:.4f}", "r2": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown(
                f"""
**Procedure, in order**

1. A single random test set of {stage4_used['test_size']:.0%} is reserved with seed
   {stage4_used['seed']} and set aside.
2. {stage4_used['folds']}-fold cross-validation runs **inside the training set only**. Each fold
   builds its own design matrix from its own fitting rows, so no preprocessing is shared across
   folds and nothing from a validation fold influences the fit.
3. The specification is refitted on the full training set.
4. That fit is scored once on the untouched test set.

**Rules this stage keeps**

- No parameter is ever selected using the test set. The folds exist so that choices can be made
  without touching it.
- Rows whose categorical levels never appeared in the fitting data are counted and excluded
  rather than being given an invented effect.
- The naive decile benchmark uses the training deciles and training means only; its outer bins
  are widened to cover test loans outside the training range.
"""
            )
            st.code(
                """train, test = train_test_split(df_clean, test_size=test_size, random_state=seed)
folds = KFold(n_splits=k, shuffle=True, random_state=seed)
for fit_index, validate_index in folds.split(train):
    fold_fit = smf.ols(formula, data=train.iloc[fit_index]).fit()
    predicted = fold_fit.predict(train.iloc[validate_index])   # design rebuilt per fold
final = smf.ols(formula, data=train).fit()
test_rmse = root_mean_squared_error(test["int_rate"], final.predict(test))""",
                language="python",
            )

# --------------------------------------------------------------------------- #
# Stage 5 — Learning curve
# --------------------------------------------------------------------------- #

with tabs[4]:
    st.subheader("How much training data the specification actually needs")
    st.write(
        "The test set stays fixed while training subsamples grow, so any change in held-out error "
        "comes from sample size alone."
    )

    controls = st.columns(3)
    stage5_key = controls[0].selectbox(
        "Specification", core.STAGE5_SPECS, index=core.STAGE5_SPECS.index("model3"),
        format_func=spec_label, key="stage5_spec",
    )
    pool_size = int(len(clean) * (1 - TEST_SIZE))
    max_size = controls[1].slider(
        "Maximum training-sample size", 500, pool_size, min(pool_size, 12000), 500, key="stage5_max"
    )
    repeats = controls[2].slider("Repeated samples per size", 1, 6, 3, key="stage5_repeats")

    stage5_params = {
        "specification": stage5_key,
        "max_train_size": int(max_size),
        "repeats": int(repeats),
        "test_size": float(TEST_SIZE),
        "seed": int(SEED),
    }

    stage5, stage5_used = stage_gate(
        "stage5",
        stage5_params,
        lambda: cached_learning_curve(
            clean, core.SPECS[stage5_key].formula, float(TEST_SIZE), int(SEED), int(max_size), int(repeats)
        ),
    )

    if stage5 is not None:
        summary = stage5["summary"]
        largest = summary.iloc[-1]
        metric_row(
            [
                ("Largest sample drawn", f"{int(largest['sample_size']):,}", "Of the training pool."),
                (
                    "Held-out RMSE there",
                    f"{largest['rmse_test_mean_pp']:.3f} pp",
                    "Mean across repeated samples.",
                ),
                (
                    "Effective complexity",
                    f"{int(stage5['full_rank'])}",
                    "Design-matrix rank when fitted on the whole pool.",
                ),
                (
                    "Approximate plateau",
                    f"{stage5['plateau_size']:,}" if stage5["plateau_size"] else "not reached",
                    "First size within 2% of the best held-out RMSE observed.",
                ),
                ("Fixed test rows", f"{stage5['n_test']:,}", None),
            ]
        )

        stage5_figure = go.Figure()
        stage5_figure.add_trace(
            go.Scatter(
                x=pd.concat([summary["sample_size"], summary["sample_size"][::-1]]),
                y=pd.concat([summary["rmse_test_max"] * 100, (summary["rmse_test_min"] * 100)[::-1]]),
                fill="toself",
                fillcolor="rgba(46,134,171,0.15)",
                line=dict(width=0),
                name="Held-out range across repeats",
                hoverinfo="skip",
            )
        )
        stage5_figure.add_trace(
            go.Scatter(
                x=summary["sample_size"], y=summary["rmse_test_mean_pp"],
                mode="lines+markers", name="Held-out RMSE", line=dict(color="#2E86AB", width=3),
            )
        )
        stage5_figure.add_trace(
            go.Scatter(
                x=summary["sample_size"], y=summary["rmse_train_pp"],
                mode="lines+markers", name="Training RMSE", line=dict(color="#7FB069", width=3, dash="dot"),
            )
        )
        unreliable = summary[~summary["estimable"]]
        if not unreliable.empty:
            stage5_figure.add_trace(
                go.Scatter(
                    x=unreliable["sample_size"], y=unreliable["rmse_test_mean_pp"],
                    mode="markers", name="Cannot support reliable estimation",
                    marker=dict(color="#D95D39", size=13, symbol="x"),
                )
            )
        stage5_figure.update_layout(
            template=PLOT_TEMPLATE,
            title=f"Learning curve — {spec_label(stage5_used['specification'])}",
            xaxis=dict(title="Training-sample size", type="log"),
            # Log scale on both axes: tiny samples miss by an order of magnitude
            # more than large ones, and a linear axis would flatten the region
            # where the curve actually settles.
            yaxis=dict(title="RMSE (percentage points)", type="log"),
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(stage5_figure, use_container_width=True)

        if not unreliable.empty:
            st.warning(
                f"Sizes marked with a cross ({', '.join(f'{int(s):,}' for s in unreliable['sample_size'])}) "
                f"either cannot reproduce the full design rank of {stage5['full_rank']} or hold fewer than "
                "two observations per parameter. Their error figures are shown for completeness but are "
                "not reliable estimates.",
                icon="⚠️",
            )
        if stage5["plateau_size"]:
            st.markdown(
                f"**Reading this:** held-out error stops improving appreciably somewhere around "
                f"**{stage5['plateau_size']:,} loans** for this specification. Treat that as an "
                "approximate description of the curve you just computed — it shifts with the seed, "
                "the number of repeats and the specification, and it is not a guaranteed minimum "
                "sample requirement. Beyond it, more of the *same* data buys little; reducing error "
                "further needs better features or a different model."
            )

        settings_caption(
            {
                "Specification": spec_label(stage5_used["specification"]),
                "Training pool": f"{stage5['n_pool']:,} loans",
                "Fixed test set": f"{stage5['n_test']:,} loans",
                "Repeats per size": stage5_used["repeats"],
                "Seed": stage5_used["seed"],
            }
        )

        register(
            "5 Learning curve",
            [
                {"model": spec_label(stage5_used["specification"]), "metric": f"rmse_test_pp @ n={int(row['sample_size'])}", "value": row["rmse_test_mean_pp"]}
                for _, row in summary.iterrows()
            ]
            + [
                {"model": spec_label(stage5_used["specification"]), "metric": "plateau_size", "value": stage5["plateau_size"]},
                {"model": spec_label(stage5_used["specification"]), "metric": "effective_params", "value": stage5["full_rank"]},
            ],
        )

        with st.expander("Curve table, rules and code"):
            display = summary.copy()
            display["rmse_test_range_pp"] = (display["rmse_test_max"] - display["rmse_test_min"]) * 100
            st.dataframe(
                display[
                    ["sample_size", "rmse_train_pp", "rmse_test_mean_pp", "rmse_test_range_pp",
                     "effective_params", "estimable"]
                ].style.format(
                    {"rmse_train_pp": "{:.4f}", "rmse_test_mean_pp": "{:.4f}", "rmse_test_range_pp": "{:.4f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown(
                f"""
**Rules this stage keeps**

- The evaluation set is the Stage 4 test set and never changes across sizes, so the curve is not
  contaminated by a moving target.
- Subsamples are drawn only from the training pool ({stage5['n_pool']:,} loans), never from the
  test set.
- Each size is drawn {stage5_used['repeats']} time(s) with different seeds; the shaded band is the
  spread across those draws and shows how much of any apparent improvement is luck.
- Sizes that cannot support reliable estimation are marked rather than quietly plotted.
- Any plateau is described as approximate. It is a feature of this curve, not a sample-size rule.
"""
            )
            st.code(
                """train_pool, test = train_test_split(df_clean, test_size=test_size, random_state=seed)
for size in sizes:
    for repeat in range(repeats):
        subsample = train_pool.sample(n=size, random_state=seed * 1000 + repeat)
        fit = smf.ols(formula, data=subsample).fit()
        rmse = root_mean_squared_error(test["int_rate"], fit.predict(test))  # test never changes""",
                language="python",
            )

# --------------------------------------------------------------------------- #
# Stage 6 — OLS, LASSO and Ridge
# --------------------------------------------------------------------------- #

with tabs[5]:
    st.subheader("How regularisation changes error and coefficients")
    st.write(
        "A deliberately over-parameterised specification is fitted on a small training sample, "
        "where unpenalised OLS has too little data to behave — the situation regularisation exists for."
    )
    st.caption(f"Specification held fixed for this stage: `{core.COMPLEX_FORMULA}`")

    controls = st.columns(3)
    train_pct = controls[0].slider(
        "Training-sample size (% of loans)", 0.4, 10.0, 0.8, 0.2, key="stage6_train",
        help="The assignment uses 0.8%, which leaves roughly 300 loans to fit 57 predictor columns.",
    )
    alpha_source = controls[1].radio(
        "Penalty strength",
        ["Cross-validated", "Manual — LASSO", "Manual — Ridge"],
        key="stage6_source",
        help="Manual mode overrides one algorithm's alpha; the other keeps its cross-validated value.",
    )
    if alpha_source == "Manual — LASSO":
        exponent = controls[2].slider(
            "LASSO alpha (log10 scale)", -5.0, -1.0, -3.0, 0.1, key="stage6_lasso_alpha"
        )
        manual_lasso, manual_ridge = float(10**exponent), None
        controls[2].caption(f"alpha = {manual_lasso:.6g}  ·  range {core.LASSO_ALPHA_RANGE}")
    elif alpha_source == "Manual — Ridge":
        exponent = controls[2].slider(
            "Ridge alpha (log10 scale)", -2.0, 4.0, 1.0, 0.1, key="stage6_ridge_alpha"
        )
        manual_lasso, manual_ridge = None, float(10**exponent)
        controls[2].caption(f"alpha = {manual_ridge:.6g}  ·  range {core.RIDGE_ALPHA_RANGE}")
    else:
        manual_lasso = manual_ridge = None
        controls[2].caption(
            "Both alphas come from 5-fold cross-validation inside the training set, each on its own range."
        )

    stage6_params = {
        "train_size": round(train_pct / 100, 5),
        "seed": int(SEED),
        "lasso_alpha": manual_lasso,
        "ridge_alpha": manual_ridge,
    }

    stage6, stage6_used = stage_gate(
        "stage6",
        stage6_params,
        lambda: cached_regularisation(
            clean, round(train_pct / 100, 5), int(SEED), manual_lasso, manual_ridge
        ),
    )

    if stage6 is not None:
        performance = stage6["performance"].set_index("model")
        shrinkage = stage6["shrinkage"].set_index("model")
        metric_row(
            [
                ("OLS test RMSE", f"{performance.loc['OLS', 'test_rmse_pp']:.3f} pp", "Unpenalised baseline."),
                ("LASSO test RMSE", f"{performance.loc['LASSO', 'test_rmse_pp']:.3f} pp",
                 f"alpha = {stage6['used_lasso_alpha']:.6g}"),
                ("Ridge test RMSE", f"{performance.loc['Ridge', 'test_rmse_pp']:.3f} pp",
                 f"alpha = {stage6['used_ridge_alpha']:.6g}"),
                ("LASSO zero coefficients",
                 f"{int(shrinkage.loc['LASSO', 'n_zero'])} of {stage6['n_features']}",
                 f"{int(shrinkage.loc['LASSO', 'n_zero_from_selection'])} by variable selection, "
                 f"{int(shrinkage.loc['LASSO', 'n_zero_constant_column'])} constant columns."),
                ("Training rows", f"{stage6['n_train']:,}",
                 f"Design rank {stage6['train_rank']} of {stage6['n_features']} columns."),
            ]
        )

        main_figure = go.Figure(
            go.Bar(
                x=performance.index,
                y=performance["test_rmse_pp"],
                marker_color=["#B0B0B0", "#2E86AB", "#F2C14E"],
                text=[f"{v:.3f}" for v in performance["test_rmse_pp"]],
                textposition="outside",
            )
        )
        main_figure.update_layout(
            template=PLOT_TEMPLATE,
            title="Test RMSE on identical held-out loans",
            yaxis_title="Test RMSE (percentage points)",
            yaxis_type="log",
            height=440,
        )
        st.plotly_chart(main_figure, use_container_width=True)
        st.caption(
            "The axis is logarithmic because unpenalised OLS on this sample can miss by orders of "
            "magnitude more than the regularised fits."
        )

        left, right = st.columns(2)

        cv_figure = go.Figure()
        cv_figure.add_trace(
            go.Scatter(
                x=stage6["lasso_path"]["alpha"], y=stage6["lasso_path"]["cv_rmse_pp"],
                mode="lines", name="LASSO", line=dict(color="#2E86AB", width=3),
            )
        )
        cv_figure.add_trace(
            go.Scatter(
                x=stage6["ridge_path"]["alpha"], y=stage6["ridge_path"]["cv_rmse_pp"],
                mode="lines", name="Ridge", line=dict(color="#F2C14E", width=3), xaxis="x2",
            )
        )
        cv_figure.add_vline(x=stage6["used_lasso_alpha"], line=dict(dash="dash", color="#2E86AB"))
        cv_figure.update_layout(
            template=PLOT_TEMPLATE,
            title="Cross-validation RMSE against alpha",
            xaxis=dict(title="LASSO alpha", type="log"),
            xaxis2=dict(title="Ridge alpha", type="log", overlaying="x", side="top"),
            yaxis=dict(title="CV RMSE (pp)", type="log"),
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.15),
        )
        left.plotly_chart(cv_figure, use_container_width=True)
        left.caption(
            "Two axes, because the two penalties are on different scales: a LASSO alpha and a Ridge "
            "alpha of the same number mean entirely different amounts of shrinkage."
        )

        coefficients = stage6["coefficients"]
        magnitude = (
            coefficients.assign(size=coefficients["OLS_standardised"].abs())
            .nlargest(18, "size")
            .sort_values("size")
        )
        shrink_figure = go.Figure()
        for name, colour in [("OLS", "#B0B0B0"), ("Ridge", "#F2C14E"), ("LASSO", "#2E86AB")]:
            shrink_figure.add_trace(
                go.Bar(
                    y=magnitude["feature"], x=magnitude[f"{name}_standardised"],
                    name=name, orientation="h", marker_color=colour,
                )
            )
        shrink_figure.update_layout(
            template=PLOT_TEMPLATE,
            title="Coefficient shrinkage (standardised scale)",
            xaxis_title="Coefficient on standardised predictors",
            height=420,
            barmode="group",
            yaxis=dict(tickfont=dict(size=9)),
            legend=dict(orientation="h", yanchor="bottom", y=1.1),
        )
        right.plotly_chart(shrink_figure, use_container_width=True)
        right.caption(
            "The 18 predictors with the largest unpenalised coefficients. Comparing on the "
            "standardised scale is what makes the penalties comparable across predictors."
        )

        st.dataframe(
            stage6["shrinkage"].rename(
                columns={
                    "n_zero": "Zero coefficients",
                    "n_nonzero": "Non-zero coefficients",
                    "n_zero_from_selection": "Zeroed by the penalty",
                    "n_zero_constant_column": "Zero because the column is constant",
                    "n_features": "Predictor columns",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        if stage6["constant_columns"]:
            st.info(
                f"{len(stage6['constant_columns'])} design columns are constant in this training "
                "sample (a category combination that never occurs at this sample size). Their "
                "coefficients are zero for every algorithm, including Ridge — that is not LASSO "
                "variable selection, and the table above separates the two.",
                icon="ℹ️",
            )
        if stage6["n_test_unsupported"]:
            st.warning(
                f"{stage6['n_test_unsupported']:,} test rows were excluded because they carry "
                f"categorical levels absent from this small training sample: "
                f"{stage6['unsupported_levels']}. All three algorithms are scored on the same "
                f"{stage6['n_test_evaluated']:,} remaining loans.",
                icon="⚠️",
            )

        settings_caption(
            {
                "Training sample": f"{stage6['n_train']:,} loans ({stage6_used['train_size']:.3%})",
                "Test loans scored": f"{stage6['n_test_evaluated']:,}",
                "Predictor columns": stage6["n_features"],
                "Seed": stage6_used["seed"],
                "CV": f"{stage6['cv_folds']}-fold, seed {stage6['cv_seed']}",
                "LASSO alpha": f"{stage6['used_lasso_alpha']:.6g}",
                "Ridge alpha": f"{stage6['used_ridge_alpha']:.6g}",
            }
        )

        register(
            "6 Regularisation",
            [
                {"model": row["model"], "metric": metric, "value": row[metric]}
                for _, row in stage6["performance"].iterrows()
                for metric in ["alpha", "test_rmse_pp", "test_r2"]
            ],
        )

        with st.expander("Coefficients, rules and code"):
            st.markdown("**Coefficients on the original predictor scale**")
            st.dataframe(
                coefficients[["feature", "OLS", "LASSO", "Ridge"]].style.format(
                    {"OLS": "{:.4e}", "LASSO": "{:.4e}", "Ridge": "{:.4e}"}
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown(
                f"""
**Rules this stage keeps**

- All three algorithms see the same observations and the same
  {stage6['n_features']} predictor columns. The test design matrix is built from the *training*
  `design_info`, which is what guarantees identical coding on both sides. (The notebook calls
  `dmatrices` separately on each split, which can produce different columns when a small training
  sample misses a category — that discrepancy is corrected here rather than reproduced.)
- The OLS baseline is fitted by least squares on the same standardised design, so it is
  numerically stable and comparably scaled rather than being penalised differently.
- Standardisation happens **inside** each pipeline and inside every cross-validation fold, so no
  fold learns the mean or scale of its validation rows.
- `fit_intercept=True` keeps the intercept out of the penalty for both LASSO and Ridge.
- Zeros from LASSO selection are reported separately from zeros caused by constant columns.
- LASSO and Ridge have separate alpha ranges, {core.LASSO_ALPHA_RANGE} and
  {core.RIDGE_ALPHA_RANGE}; the penalties are not on a shared scale.

**Why the numbers move as they do.** With only {stage6['n_train']:,} loans and
{stage6['n_features']} columns (design rank {stage6['train_rank']}), unpenalised OLS fits noise and
can produce a strongly negative test R-squared — worse than predicting the test mean. LASSO's
absolute-value penalty can set coefficients exactly to zero, so it selects variables. Ridge's
squared penalty shrinks coefficients towards zero without reaching it, so it keeps every
predictor but trusts none of them fully.
"""
            )
            st.code(
                """y_train, X_train = dmatrices(formula, train, return_type="dataframe")
(X_test,) = build_design_matrices([X_train.design_info], supported_test, return_type="dataframe")
pipeline = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=alpha))])
# CV for alpha, standardising inside each fold
for fit_index, validate_index in KFold(5, shuffle=True, random_state=4444).split(X_train):
    pipeline.fit(X_train.to_numpy()[fit_index], y_train[fit_index])""",
                language="python",
            )

# --------------------------------------------------------------------------- #
# Stage 7 — Time information
# --------------------------------------------------------------------------- #

with tabs[6]:
    st.subheader("Whether time information helps, and for which question")
    st.write(
        "The same five specifications answer very differently depending on whether you are "
        "predicting other loans from months you have seen, or loans from months you have not."
    )

    controls = st.columns(3)
    stage7_key = controls[0].selectbox(
        "Specification to display", core.TIME_SPECS, format_func=spec_label, key="stage7_spec"
    )
    goal = controls[1].radio(
        "Validation goal",
        ["Within-period prediction", "Future-month forecasting"],
        key="stage7_goal",
        help="Within-period holds out random loans; future-month holds out the latest whole months.",
    )
    mode = "within" if goal.startswith("Within") else "future"
    holdout_months = controls[2].slider(
        "Months held out", 1, 12, 6, key="stage7_months", disabled=(mode == "within"),
        help="Used for future-month forecasting only.",
    )

    stage7_params = {
        "mode": mode,
        "test_size": float(TEST_SIZE),
        "seed": int(SEED),
        "holdout_months": int(holdout_months),
    }

    stage7, stage7_used = stage_gate(
        "stage7",
        stage7_params,
        lambda: cached_time_models(clean, mode, float(TEST_SIZE), int(SEED), int(holdout_months)),
    )

    if stage7 is not None:
        results = stage7["results"].set_index("key")
        row = results.loc[stage7_key]

        if not row["supported"]:
            st.error(f"**{spec_label(stage7_key)} is unavailable for this comparison.** {row['reason']}", icon="🚫")
            metric_row(
                [
                    ("Held-out RMSE", "—", "Not estimable for unseen months."),
                    ("Held-out R-squared", "—", None),
                    ("Effective parameters", "—", None),
                    ("Training rows", f"{stage7['n_train']:,}", None),
                ]
            )
        else:
            metric_row(
                [
                    ("Held-out RMSE", f"{row['rmse_test_pp']:.3f} pp", "On the held-out loans."),
                    ("Held-out R-squared", f"{row['r2_test']:.4f}", None),
                    ("Training RMSE", f"{row['rmse_train_pp']:.3f} pp", "In-sample, for contrast."),
                    ("Effective parameters", f"{int(row['effective_params'])}", "Design-matrix rank."),
                    ("Loans scored", f"{int(row['n_test_evaluated']):,}", None),
                ]
            )

        monthly = stage7["monthly"]
        if not monthly.empty and spec_label(stage7_key) in set(monthly["model"]):
            series = monthly[monthly["model"] == spec_label(stage7_key)].sort_values("ym")
            stage7_figure = go.Figure()
            stage7_figure.add_trace(
                go.Scatter(
                    x=series["ym"], y=series["observed_mean"] * 100, mode="lines+markers",
                    name="Observed monthly mean", line=dict(color="#444", width=3),
                )
            )
            for split, colour in [("Training", "#7FB069"), ("Held out", "#2E86AB")]:
                part = series[series["split"] == split]
                if not part.empty:
                    stage7_figure.add_trace(
                        go.Scatter(
                            x=part["ym"], y=part["fitted_mean"] * 100, mode="lines+markers",
                            name=f"Fitted — {split.lower()}",
                            line=dict(color=colour, width=3, dash="dot"),
                        )
                    )
            stage7_figure.update_layout(
                template=PLOT_TEMPLATE,
                title=f"Monthly observed and fitted mean interest rate — {spec_label(stage7_key)}",
                xaxis_title="Issue month",
                yaxis_title="Mean interest rate (%)",
                height=480,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(stage7_figure, use_container_width=True)

        display = stage7["results"][
            ["model", "supported", "rmse_test_pp", "r2_test", "effective_params", "n_test_evaluated", "reason"]
        ].rename(
            columns={
                "rmse_test_pp": "Held-out RMSE (pp)",
                "r2_test": "Held-out R-squared",
                "effective_params": "Effective parameters",
                "n_test_evaluated": "Loans scored",
                "supported": "Comparable",
                "reason": "If not, why",
            }
        )
        st.dataframe(
            display.style.format(
                {
                    "Held-out RMSE (pp)": "{:.4f}",
                    "Held-out R-squared": "{:.4f}",
                    "Effective parameters": "{:.0f}",
                    "Loans scored": "{:,.0f}",
                },
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
        )

        if stage7_used["mode"] == "future":
            st.info(
                f"Trained on months {stage7['train_month_range'][0]} to {stage7['train_month_range'][1]} "
                f"and scored on {stage7['test_month_range'][0]} to {stage7['test_month_range'][1]} "
                f"({len(stage7['test_months'])} whole months, {stage7['n_test']:,} loans). "
                "The two month-year dummy specifications are disabled here: a free coefficient per "
                "month simply does not exist for a month that has not happened yet.",
                icon="📅",
            )
        else:
            restricted = stage7["results"]["n_test_unsupported"].max()
            st.info(
                f"Random {stage7_used['test_size']:.0%} hold-out across all months, seed "
                f"{stage7_used['seed']}: {stage7['n_train']:,} training and {stage7['n_test']:,} "
                "held-out loans. Every month in the hold-out also appears in training, which is why "
                "month-year dummies can be scored at all here"
                + (
                    f" — {int(restricted):,} held-out loans still had to be excluded for want of a "
                    "trained categorical effect."
                    if restricted
                    else ", and no loans had to be excluded for want of a trained categorical effect."
                ),
                icon="ℹ️",
            )

        settings_caption(
            {
                "Validation goal": goal,
                "Training rows": f"{stage7['n_train']:,}",
                "Held-out rows": f"{stage7['n_test']:,}",
                "Seed": stage7_used["seed"],
                "Months held out": stage7_used["holdout_months"] if stage7_used["mode"] == "future" else "—",
            }
        )

        register(
            "7 Time information",
            [
                {"model": r["model"], "metric": "rmse_test_pp", "value": r["rmse_test_pp"]}
                for _, r in stage7["results"].iterrows()
            ]
            + [
                {"model": r["model"], "metric": "effective_params", "value": r["effective_params"]}
                for _, r in stage7["results"].iterrows()
            ],
        )

        with st.expander("Formulas, complexity and rules"):
            for key in core.TIME_SPECS:
                spec = core.SPECS[key]
                parameters = results.loc[key, "effective_params"]
                complexity = "not estimable here" if pd.isna(parameters) else f"{int(parameters)} effective parameters"
                st.markdown(f"**{spec.label}** — `{spec.formula}`  \n{spec.note} ({complexity}).")

            st.markdown(
                """
**Rules this stage keeps**

- Within-period evaluation requires that the training data contain the categorical effects the
  held-out loans need. Any held-out loan without that support is excluded and the count is
  disclosed above, so the restricted evaluation sample is never hidden.
- Future-month evaluation splits on whole months and uses earlier months to predict later ones.
  No loan from a test month appears in training.
- Month-year dummy models are never presented as able to estimate effects for unseen future
  months; they are disabled with the reason stated.
- Unsupported comparisons are disabled rather than filled in with a number that would not mean
  what it appears to mean.

**The honest summary.** Month-year dummies give the best within-period numbers because they
absorb whatever happened in each month. That is useful for *describing* the period, and useless
for forecasting: the coefficient for next month does not exist. A linear trend, or a trend that
differs by grade, extrapolates — which is a modelling assumption you are choosing to make, not a
fact the data has established. If you need to forecast, the trend specifications are the only
ones in this list that can even be attempted, and the future-month view is the only one of the
two goals that tells you whether they work.
"""
            )
