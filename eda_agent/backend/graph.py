from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backend.eda_nodes import (
    build_imputation_report,
    detect_duplicates,
    detect_missing,
    execute_generated_code,
    generate_charts,
    generate_summary,
    load_data,
    profile_columns_with_nulls,
    remove_duplicates,
)
from backend.llm_nodes import (
    generate_imputation_plan,
    generate_python_code_from_plan,
    generate_llm_insights,
)


# ── State ─────────────────────────────────────────────────────────────────────

class EDAState(TypedDict, total=False):
    # Inputs
    file_path:  str
    session_id: str
    job_id:     str

    # Node 1-3
    df:                   Any     # pandas DataFrame
    missing_values:       dict    # {"col": count} — pre-fill snapshot
    null_column_profiles: list    # [{column_name, dtype, null_count, skew, ...}]

    # Step 1 — LLM plan (JSON, no code)
    imputation_plan:      dict    # {"imputation_plan": [{column, semantic_type, condition, strategy, reason}]}

    # Step 2 — LLM code (pure Python string)
    generated_code:       str

    # Execution
    execution_status:     str     # "success" | "fallback" | "skipped"
    execution_error:      str

    # Report
    fill_logic:           list    # consumed by _build_llm_context
    imputation_report:    dict    # {"columns": [...], "generated_code": ..., "execution_status": ...}
    null_handling_report: str

    # Post-imputation
    duplicate_count:      int
    dup_logic:            dict
    charts_generated:     list
    summary:              dict

    # LLM insights
    llm_insights:         dict
    llm_context:          dict


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(EDAState)

    # Register nodes
    g.add_node("load_data",                      load_data)
    g.add_node("detect_missing",                 detect_missing)
    g.add_node("profile_columns_with_nulls",     profile_columns_with_nulls)
    g.add_node("generate_imputation_plan",       generate_imputation_plan)        # Step 1
    g.add_node("generate_python_code_from_plan", generate_python_code_from_plan)  # Step 2
    g.add_node("execute_generated_code",         execute_generated_code)
    g.add_node("build_imputation_report",        build_imputation_report)
    g.add_node("detect_duplicates",              detect_duplicates)
    g.add_node("remove_duplicates",              remove_duplicates)
    g.add_node("generate_charts",                generate_charts)
    g.add_node("generate_summary",               generate_summary)
    g.add_node("generate_llm_insights",          generate_llm_insights)

    # Wire edges
    g.set_entry_point("load_data")
    g.add_edge("load_data",                      "detect_missing")
    g.add_edge("detect_missing",                 "profile_columns_with_nulls")
    g.add_edge("profile_columns_with_nulls",     "generate_imputation_plan")
    g.add_edge("generate_imputation_plan",       "generate_python_code_from_plan")
    g.add_edge("generate_python_code_from_plan", "execute_generated_code")
    g.add_edge("execute_generated_code",         "build_imputation_report")
    g.add_edge("build_imputation_report",        "detect_duplicates")
    g.add_edge("detect_duplicates",              "remove_duplicates")
    g.add_edge("remove_duplicates",              "generate_charts")
    g.add_edge("generate_charts",                "generate_summary")
    g.add_edge("generate_summary",               "generate_llm_insights")
    g.add_edge("generate_llm_insights",          END)

    return g.compile()


eda_graph = build_graph()
