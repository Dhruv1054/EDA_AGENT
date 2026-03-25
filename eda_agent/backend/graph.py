from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backend.eda_nodes import (
    apply_imputation,
    detect_duplicates,
    detect_missing,
    generate_charts,
    generate_summary,
    load_data,
    map_semantic_type_to_strategy,
    detect_semantic_type_for_each_null_column,
    profile_columns_with_nulls,
    remove_duplicates,
)
from backend.llm_nodes import generate_llm_insights


# ── State ─────────────────────────────────────────────────────────────────────

class EDAState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────
    file_path:  str
    session_id: str
    job_id:     str

    # ── Deterministic pipeline outputs (LLM nodes never mutate these) ───────
    df:               Any    # pandas DataFrame — not JSON-serialisable
    missing_values:   dict
    null_column_profiles: list
    null_semantic_types:  dict
    imputation_strategies: dict
    fill_logic:       list
    null_handling_report: str
    duplicate_count:  int
    dup_logic:        dict
    charts_generated: list
    summary:          dict

    # ── LLM layer outputs (read-only products) ───────────────────────────────
    llm_insights: dict
    llm_context:  dict


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(EDAState)

    g.add_node("load_data",             load_data)
    g.add_node("detect_missing",        detect_missing)
    g.add_node("profile_columns_with_nulls",              profile_columns_with_nulls)
    g.add_node("detect_semantic_type_for_each_null_column", detect_semantic_type_for_each_null_column)
    g.add_node("map_semantic_type_to_strategy",           map_semantic_type_to_strategy)
    g.add_node("apply_imputation",                        apply_imputation)
    g.add_node("detect_duplicates",     detect_duplicates)
    g.add_node("remove_duplicates",     remove_duplicates)
    g.add_node("generate_charts",       generate_charts)
    g.add_node("generate_summary",      generate_summary)
    g.add_node("generate_llm_insights", generate_llm_insights)

    g.set_entry_point("load_data")
    g.add_edge("load_data",             "detect_missing")
    g.add_edge("detect_missing",        "profile_columns_with_nulls")
    g.add_edge("profile_columns_with_nulls", "detect_semantic_type_for_each_null_column")
    g.add_edge("detect_semantic_type_for_each_null_column", "map_semantic_type_to_strategy")
    g.add_edge("map_semantic_type_to_strategy", "apply_imputation")
    g.add_edge("apply_imputation",      "detect_duplicates")
    g.add_edge("detect_duplicates",     "remove_duplicates")
    g.add_edge("remove_duplicates",     "generate_charts")
    g.add_edge("generate_charts",       "generate_summary")
    g.add_edge("generate_summary",      "generate_llm_insights")
    g.add_edge("generate_llm_insights", END)

    return g.compile()


eda_graph = build_graph()
