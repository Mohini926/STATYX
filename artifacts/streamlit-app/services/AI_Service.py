import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy import stats

from ai.column_inference import infer_columns_from_objective
from ai.embedding_model import model
from ai.test_selector import rank_tests
from stats.stats_tests import TEST_REGISTRY


def generate_auto_insights(df: pd.DataFrame) -> List[Dict[str, Any]]:

    insights = []
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)

    insights.append({
        "type": "overview",
        "icon": "chart",
        "title": "Dataset Overview",
        "description": f"Your dataset has {len(df):,} rows and {len(df.columns)} columns. "
                       f"It contains {len(numeric_cols)} numeric and {len(cat_cols)} categorical columns.",
        "severity": "info",
    })
    return insights

def _run_test(df: pd.DataFrame, test_key: str, target: str, group: str) -> Optional[Dict[str, Any]]:
    if test_key not in TEST_REGISTRY:
        return None
    if test_key in {"cox_regression", "kaplan_meier"}:
        return None

    try:
        result = TEST_REGISTRY[test_key](df, target, group)
    except Exception:
        return None

    if not isinstance(result, dict) or len(result) == 0:
        return None
    return result


def run_ai_analysis(
    df: pd.DataFrame,
    objective: str,
    test_embeddings: Any,
    test_names: List[str],
    target_override: Optional[str] = None,
    group_override: Optional[str] = None,
    force_test_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if df is None or not objective:
        return None

    query_emb = model.encode(f"query: {objective}", convert_to_tensor=True)
    ranked_tests = rank_tests(query_emb, test_embeddings, test_names)

    target, group = infer_columns_from_objective(df, objective)
    target = target_override or target
    group = group_override or group

    if not target or not group or target == group:
        return None

    if force_test_key:
        single_result = _run_test(df, force_test_key, target, group)
        top_results = []
        if single_result:
            top_results.append(
                {
                    "test": force_test_key.replace("_", " ").title(),
                    "confidence": None,
                    "result": single_result,
                }
            )
        return {
            "objective": objective,
            "target": target,
            "group": group,
            "top_results": top_results,
            "remaining_tests": [],
            "test_names": test_names,
            "test_embeddings": test_embeddings,
        }

    results = []
    executed_keys = []
    for test_key, confidence in ranked_tests[:20]:
        result = _run_test(df, test_key, target, group)
        if result is None:
            continue
        executed_keys.append(test_key)
        results.append(
            {
                "test": test_key.replace("_", " ").title(),
                "test_key": test_key,
                "confidence": round(float(confidence), 3),
                "result": result,
            }
        )

    top_results = results[:5]
    remaining_tests = [k for k in executed_keys if k not in [r["test_key"] for r in top_results]]

    return {
        "objective": objective,
        "target": target,
        "group": group,
        "top_results": top_results,
        "remaining_tests": remaining_tests,
        "test_names": test_names,
        "test_embeddings": test_embeddings,
    }