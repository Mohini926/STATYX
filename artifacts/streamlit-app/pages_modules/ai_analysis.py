import streamlit as st
import pandas as pd

from ai.embedding_model import model
from services.AI_Service import run_ai_analysis

TEST_DESCRIPTIONS = {
    "independent_t_test": "compare mean of a continuous variable between two independent groups",
    "paired_t_test": "compare mean before and after intervention on same subjects",
    "one_sample_t_test": "compare mean against known population value",
    "anova": "compare mean across more than two groups",
    "anova_rm": "compare repeated measurements",
    "mann_whitney_u": "nonparametric comparison between two independent groups",
    "kruskal_wallis": "nonparametric comparison across multiple groups",
    "chi_square": "association between two categorical variables",
    "fisher_exact": "association between two binary variables",
    "chisquare_gof": "goodness of fit test",
    "mcnemar_test": "paired categorical association",
    "cochran_q": "compare proportions across repeated groups",
    "pearson_correlation": "linear correlation",
    "spearman_correlation": "rank based correlation",
    "kendall_tau": "ordinal association",
    "mutual_information": "nonlinear dependency",
    "linear_regression": "predict continuous outcome",
    "logistic_regression": "predict binary outcome",
    "poisson_regression": "model count outcome",
    "probit_regression": "binary outcome with probit",
    "cox_regression": "survival analysis",
    "kaplan_meier": "survival probability",
    "shapiro_test": "normality test",
    "ks_test": "distribution test",
    "levene_test": "variance equality",
    "bartlett_test": "variance homogeneity",
    "jarque_bera": "skewness & kurtosis",
    "variance_ratio": "compare variances",
    "binomial_test": "test proportion",
    "z_test_proportion": "z test for proportion",
    "two_proportion_ztest": "compare proportions",
    "roc_auc": "classification performance",
    "cohens_d": "effect size",
    "hedges_g": "bias corrected effect size",
    "phi_coefficient": "binary association",
    "cramers_v": "categorical association strength",
    "odds_ratio": "odds comparison",
    "relative_risk": "risk comparison",
}


def render():

    st.header("🧠 AI Objective Analysis")

    df = st.session_state.df

    if df is None:
        st.warning("Upload data first")
        st.stop()

    test_names = list(TEST_DESCRIPTIONS.keys())
    test_embeddings = model.encode(
        [f"passage: {TEST_DESCRIPTIONS[t]}" for t in test_names],
        convert_to_tensor=True,

    objective = st.text_input("Enter objective (e.g., 'Is outcome associated with gender?')")

        if st.button("Run Suggested Tests"):
        analysis = run_ai_analysis(df, objective, test_embeddings, test_names)

        if analysis is None:
            st.error("Could not infer analysis plan. Please refine objective and try again.")
            st.stop()

        inferred_target = analysis["target"]
        inferred_group = analysis["group"]

        st.session_state.ai_analysis_payload = analysis
        st.session_state.ai_target = inferred_target
        st.session_state.ai_group = inferred_group


    payload = st.session_state.get("ai_analysis_payload")
    if payload:
        st.markdown("#### Confirm / adjust selected columns")

        columns = list(df.columns)
        target = st.selectbox(
            "Target column",
            options=columns,
            index=columns.index(st.session_state.get("ai_target", columns[0]))
            if st.session_state.get("ai_target") in columns
            else 0,
        )
        group_candidates = [col for col in columns if col != target] or columns
        group = st.selectbox(
            "Relevant/group column",
            options=group_candidates,
            index=group_candidates.index(st.session_state.get("ai_group", group_candidates[0]))
            if st.session_state.get("ai_group") in group_candidates
            else 0,
        )

        rerun = st.button("Run Top 5 on selected columns")
        if rerun:
            payload = run_ai_analysis(
                df,
                payload["objective"],
                payload["test_embeddings"],
                payload["test_names"],
                target_override=target,
                group_override=group,
            )
            st.session_state.ai_analysis_payload = payload

        st.markdown("### Top 5 Suggested Tests (Executed)")
        for test in payload["top_results"]:
            st.subheader(test["test"])
            st.table(pd.DataFrame(list(test["result"].items()), columns=["Metric", "Value"]))

        if payload["remaining_tests"]:
            st.markdown("### Other Possible Tests (Execute on demand)")
            test_choice = st.selectbox("Select additional test", payload["remaining_tests"])
            if st.button("Run Selected Test"):
                extra_result = run_ai_analysis(
                    df,
                    payload["objective"],
                    payload["test_embeddings"],
                    payload["test_names"],
                    target_override=target,
                    group_override=group,
                    force_test_key=test_choice,
                )
                for test in extra_result["top_results"]:
                    st.subheader(f"Additional: {test['test']}")
                    st.table(pd.DataFrame(list(test["result"].items()), columns=["Metric", "Value"]))

    if st.button("⬅ Back"):
        st.session_state.step = "statistics"
        st.rerun()