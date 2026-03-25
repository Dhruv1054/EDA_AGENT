from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backend.eda_nodes import (
    detect_duplicates,
    detect_missing,
    execute_imputation_plan,
    generate_charts,
    generate_imputation_report,
    generate_summary,
    load_data,
    profile_columns_with_nulls,
    remove_duplicates,
)
from backend.llm_nodes import generate_imputation_plan_using_llm, generate_llm_insights


# ── State ─────────────────────────────────────────────────────────────────────

class EDAState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────
    file_path:  str
    session_id: str
    job_id:     str

    # ── Deterministic pipeline outputs (LLM nodes never mutate these) ───────
    df:                       Any    # pandas DataFrame — not JSON-serialisable
    missing_values:           dict
    null_column_profiles:     list

    # ── LLM imputation plan + execution results ──────────────────────────────
    imputation_plan:          dict   # {"imputation_plan": [...]}
    executed_imputation_steps: list  # per-step execution records
    imputation_report:        dict   # {"columns": [...]} — Column | Type | Condition | Strategy
    fill_logic:               list   # compatible with _build_llm_context
    null_handling_report:     str

    # ── Post-imputation pipeline ─────────────────────────────────────────────
    duplicate_count:          int
    dup_logic:                dict
    charts_generated:         list
    summary:                  dict

    # ── LLM layer outputs (read-only products) ───────────────────────────────
    llm_insights:             dict
    llm_context:              dict


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(EDAState)

    # Register nodes
    g.add_node("load_data",                         load_data)
    g.add_node("detect_missing",                    detect_missing)
    g.add_node("profile_columns_with_nulls",        profile_columns_with_nulls)
    g.add_node("generate_imputation_plan_using_llm", generate_imputation_plan_using_llm)
    g.add_node("execute_imputation_plan",           execute_imputation_plan)
    g.add_node("generate_imputation_report",        generate_imputation_report)
    g.add_node("detect_duplicates",                 detect_duplicates)
    g.add_node("remove_duplicates",                 remove_duplicates)
    g.add_node("generate_charts",                   generate_charts)
    g.add_node("generate_summary",                  generate_summary)
    g.add_node("generate_llm_insights",             generate_llm_insights)

    # Wire edges
    g.set_entry_point("load_data")
    g.add_edge("load_data",                          "detect_missing")
    g.add_edge("detect_missing",                     "profile_columns_with_nulls")
    g.add_edge("profile_columns_with_nulls",         "generate_imputation_plan_using_llm")
    g.add_edge("generate_imputation_plan_using_llm", "execute_imputation_plan")
    g.add_edge("execute_imputation_plan",            "generate_imputation_report")
    g.add_edge("generate_imputation_report",         "detect_duplicates")
    g.add_edge("detect_duplicates",                  "remove_duplicates")
    g.add_edge("remove_duplicates",                  "generate_charts")
    g.add_edge("generate_charts",                    "generate_summary")
    g.add_edge("generate_summary",                   "generate_llm_insights")
    g.add_edge("generate_llm_insights",              END)

    return g.compile()


eda_graph = build_graph()
