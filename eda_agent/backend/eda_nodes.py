import os
import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from backend.job_store import update_job
from backend.llm_nodes import detect_semantic_type_for_column

OUTPUTS_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUTS_BASE, exist_ok=True)

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _session_dir(session_id: str) -> str:
    """Return (and create) a per-session output directory."""
    path = os.path.join(OUTPUTS_BASE, session_id)
    os.makedirs(path, exist_ok=True)
    return path


def _tick(state: dict, progress: int, stage: str) -> None:
    """Update job-store progress without coupling node logic to the store."""
    job_id = state.get("job_id", "")
    if job_id:
        update_job(job_id, progress=progress, stage=stage)


# ── Pipeline nodes ────────────────────────────────────────────────────────────

def load_data(state: dict) -> dict:
    """Node 1: Load CSV file into a pandas DataFrame."""
    _tick(state, 10, "Loading data")
    df = pd.read_csv(state["file_path"])
    return {**state, "df": df}


def detect_missing(state: dict) -> dict:
    """Node 2: Detect missing values per column (pre-fill snapshot)."""
    _tick(state, 20, "Detecting missing values")
    df: pd.DataFrame = state["df"]
    mv = df.isnull().sum()
    mv = {str(k): int(v) for k, v in mv[mv > 0].to_dict().items()}
    return {**state, "missing_values": mv}


def _to_json_safe(v: Any) -> Any:
    """Best-effort conversion of pandas/numpy scalars to JSON-safe primitives."""
    try:
        if hasattr(v, "item"):
            v = v.item()
    except Exception:
        pass

    if v is None:
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def profile_columns_with_nulls(state: dict) -> dict:
    """Node 3: Build compact metadata for columns that contain nulls."""
    _tick(state, 25, "Profiling columns with missing values")
    df: pd.DataFrame = state["df"]
    missing: dict = state.get("missing_values", {})

    profiles: list[dict] = []
    if not missing:
        return {**state, "null_column_profiles": profiles}

    n_rows = max(int(len(df)), 1)
    for col, null_count in missing.items():
        try:
            s = df[col]
        except Exception:
            continue

        non_null = s.dropna()
        sample = [_to_json_safe(x) for x in non_null.head(5).tolist()]

        profiles.append(
            {
                "column_name": str(col),
                "dtype": str(s.dtype),
                "null_percentage": round((int(null_count) / n_rows) * 100.0, 3),
                "unique_values": int(non_null.nunique(dropna=True)),
                "sample_values": sample,
            }
        )

    return {**state, "null_column_profiles": profiles}


def _heuristic_semantic_type(profile: dict) -> tuple[str, str, bool]:
    """Return (semantic_type, reason, is_confident)."""
    name = str(profile.get("column_name", "")).lower()
    dtype = str(profile.get("dtype", "")).lower()
    unique = int(profile.get("unique_values") or 0)
    samples = profile.get("sample_values") or []

    if "datetime" in dtype or "date" in dtype or "time" in dtype:
        return ("datetime", "Datetime dtype detected.", True)
    if dtype.startswith("bool"):
        return ("boolean_flag", "Boolean dtype detected.", True)
    if any(tok in name for tok in ["id", "uuid", "identifier"]):
        # High-ish uniqueness suggests identifier; still keep conservative.
        if unique >= 50:
            return ("identifier", "Column name suggests identifier with many unique values.", True)
        return ("identifier", "Column name suggests identifier.", False)

    if any(tok in name for tok in ["is_", "has_", "flag", "active", "enabled"]):
        return ("boolean_flag", "Column name suggests boolean flag.", False)

    if dtype.startswith(("int", "float")) or "int" in dtype or "float" in dtype:
        # Few unique numeric values often represent categories.
        if 0 < unique <= 10:
            return ("categorical_label", "Numeric dtype but low cardinality suggests labels.", False)
        return ("numeric_measure", "Numeric dtype suggests a measurable quantity.", True)

    # Object/string heuristics: free text vs labels.
    if dtype in {"object", "string"}:
        avg_len = 0.0
        if samples:
            lens = [len(str(x)) for x in samples if x is not None]
            avg_len = sum(lens) / max(len(lens), 1)
        if avg_len >= 30:
            return ("free_text", "Long sample strings suggest free-form text.", True)
        return ("categorical_label", "String dtype with short samples suggests labels.", False)

    return ("categorical_label", "Default fallback.", False)


def detect_semantic_type_for_each_null_column(state: dict) -> dict:
    """Node 4: Detect semantic type for each null-containing column.

    Uses heuristics for obvious cases, otherwise calls Ollama to classify.
    """
    _tick(state, 28, "Detecting semantic types for null columns")
    profiles: list[dict] = state.get("null_column_profiles", [])

    semantic: dict[str, dict] = {}
    for p in profiles:
        col = str(p.get("column_name", ""))
        guess, why, confident = _heuristic_semantic_type(p)
        if confident:
            semantic[col] = {"semantic_type": guess, "reason": why, "source": "heuristic"}
            continue

        llm_out = detect_semantic_type_for_column(p)
        semantic_type = llm_out.get("semantic_type") or guess
        reason = llm_out.get("reason") or why
        semantic[col] = {"semantic_type": semantic_type, "reason": reason, "source": "llm"}

    return {**state, "null_semantic_types": semantic}


STRATEGY_MAP: dict[str, str] = {
    "numeric_measure": "median",
    "categorical_label": "mode",
    "boolean_flag": "mode",
    "identifier": "leave_null",
    "datetime": "forward_fill",
    "free_text": "constant:Unknown",
}

ALLOWED_STRATEGIES = {"median", "mode", "constant:Unknown", "forward_fill", "leave_null"}


def map_semantic_type_to_strategy(state: dict) -> dict:
    """Node 5: Deterministically map semantic types to allowed strategies."""
    _tick(state, 29, "Mapping semantic types to imputation strategies")
    semantic: dict = state.get("null_semantic_types", {})
    strategies: dict[str, str] = {}

    for col, info in semantic.items():
        stype = str(info.get("semantic_type", "categorical_label"))
        strategy = STRATEGY_MAP.get(stype, "mode")
        if strategy not in ALLOWED_STRATEGIES:
            strategy = "mode"
        strategies[str(col)] = strategy

    return {**state, "imputation_strategies": strategies}


def apply_imputation(state: dict) -> dict:
    """Node 6: Apply deterministic imputation strategies using pandas."""
    _tick(state, 30, "Filling missing values")
    df: pd.DataFrame = state["df"].copy()
    missing: dict = state.get("missing_values", {})
    semantic: dict = state.get("null_semantic_types", {})
    strategies: dict = state.get("imputation_strategies", {})

    fill_logic: list[dict] = []
    report_lines: list[str] = []

    for col, null_count in missing.items():
        if col not in df.columns:
            continue
        n_missing = int(null_count)
        if n_missing <= 0:
            continue

        sem = semantic.get(col, {})
        sem_type = str(sem.get("semantic_type", "categorical_label"))
        strategy_key = str(strategies.get(col, STRATEGY_MAP.get(sem_type, "mode")))

        fill_value_str = ""
        reason = str(sem.get("reason", "")).strip() or "Semantic classification."

        try:
            if strategy_key == "leave_null":
                pass
            elif strategy_key == "forward_fill":
                df[col] = df[col].ffill()
            elif strategy_key == "constant:Unknown":
                df[col] = df[col].fillna("Unknown")
                fill_value_str = "Unknown"
            elif strategy_key == "median":
                fill_val = None
                try:
                    fill_val = df[col].median()
                except Exception:
                    fill_val = None
                if fill_val is None or (isinstance(fill_val, float) and pd.isna(fill_val)) or pd.isna(fill_val):
                    # Fallbacks for all-null / non-computable median
                    mode_s = df[col].mode(dropna=True)
                    if not mode_s.empty:
                        fill_val = mode_s.iloc[0]
                        df[col] = df[col].fillna(fill_val)
                    else:
                        strategy_key = "leave_null"
                else:
                    df[col] = df[col].fillna(fill_val)
                if strategy_key != "leave_null":
                    fill_value_str = str(round(float(fill_val), 4)) if isinstance(fill_val, float) else str(fill_val)
            elif strategy_key == "mode":
                mode_s = df[col].mode(dropna=True)
                if not mode_s.empty:
                    fill_val = mode_s.iloc[0]
                    df[col] = df[col].fillna(fill_val)
                    fill_value_str = str(round(float(fill_val), 4)) if isinstance(fill_val, float) else str(fill_val)
                else:
                    # Safe fallback if mode cannot be computed (e.g., all-null)
                    strategy_key = "leave_null"
            else:
                # Should not happen, but keep safe
                strategy_key = "leave_null"
        except Exception as exc:
            log.warning("Imputation failed for column %s with %s: %s", col, strategy_key, exc)
            strategy_key = "leave_null"
            reason = f"{reason} (imputation error: {exc})"

        strategy_display = {
            "median": "Median",
            "mode": "Mode",
            "forward_fill": "Forward Fill",
            "leave_null": "Leave Null",
            "constant:Unknown": "Constant:Unknown",
        }.get(strategy_key, "Leave Null")

        fill_logic.append(
            {
                "column": col,
                "missing_before": n_missing,
                "semantic_type": sem_type,
                "strategy": strategy_display,
                "strategy_key": strategy_key,
                "fill_value": fill_value_str,
                "reason": reason,
            }
        )
        report_lines.append(
            f"- Column: {col} | Null Count: {n_missing} | Detected Type: {sem_type} | Strategy: {strategy_key}"
        )

    return {
        **state,
        "df": df,
        "fill_logic": fill_logic,
        "null_handling_report": "Null Handling Report:\n" + ("\n".join(report_lines) if report_lines else "✅ No missing values to handle."),
    }


# Backwards-compatible wrapper (kept for compatibility; graph no longer uses it)
def fill_missing_values(state: dict) -> dict:
    """Legacy Node: preserved name for compatibility."""
    # If invoked directly, run the new flow using existing state where possible.
    state = profile_columns_with_nulls(state)
    state = detect_semantic_type_for_each_null_column(state)
    state = map_semantic_type_to_strategy(state)
    return apply_imputation(state)


def detect_duplicates(state: dict) -> dict:
    """Node 4: Count duplicate rows (after imputation)."""
    _tick(state, 40, "Detecting duplicate rows")
    df: pd.DataFrame = state["df"]
    return {**state, "duplicate_count": int(df.duplicated().sum())}


def remove_duplicates(state: dict) -> dict:
    """Node 5: Drop duplicate rows — keep first occurrence."""
    _tick(state, 50, "Removing duplicates")
    df: pd.DataFrame  = state["df"]
    dup_count: int    = state.get("duplicate_count", 0)
    df_clean          = df.drop_duplicates().reset_index(drop=True)
    dup_logic = {
        "duplicates_found": dup_count,
        "strategy":         "Keep First Occurrence",
        "reason": (
            "Exact duplicate rows carry no new information and bias "
            "statistical results. The first occurrence is retained "
            "and subsequent copies are dropped."
        ),
    }
    return {**state, "df": df_clean, "dup_logic": dup_logic}


def generate_charts(state: dict) -> dict:
    """Node 6: Histogram, boxplot, correlation heatmap — scoped to session dir."""
    _tick(state, 62, "Generating charts")
    df: pd.DataFrame = state["df"]
    session_id       = state.get("session_id", "default")
    out_dir          = _session_dir(session_id)
    charts: list[str] = []

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Histogram
    if numeric_cols:
        cols = numeric_cols[:4]
        fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#4C72B0")
            ax.set_title(f"Histogram – {col}", fontsize=10)
        fig.tight_layout()
        p = os.path.join(out_dir, "histogram.png")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        charts.append(p)

    # Boxplot
    if numeric_cols:
        cols = numeric_cols[:4]
        fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            sns.boxplot(y=df[col].dropna(), ax=ax, color="#55A868")
            ax.set_title(f"Boxplot – {col}", fontsize=10)
        fig.tight_layout()
        p = os.path.join(out_dir, "boxplot.png")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        charts.append(p)

    # Correlation heatmap
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        sz   = max(6, len(numeric_cols))
        fig, ax = plt.subplots(figsize=(sz, sz - 1))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax, linewidths=0.5, square=True)
        ax.set_title("Correlation Heatmap", fontsize=12)
        fig.tight_layout()
        p = os.path.join(out_dir, "correlation_heatmap.png")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        charts.append(p)

    return {**state, "charts_generated": charts}


def generate_summary(state: dict) -> dict:
    """Node 7: Comprehensive dataset summary + important facts list."""
    _tick(state, 75, "Building dataset summary")
    df: pd.DataFrame  = state["df"]
    missing: dict     = state.get("missing_values", {})
    dup_count: int    = state.get("duplicate_count", 0)
    null_report: str  = state.get("null_handling_report", "")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    summary: dict = {
        "total_rows":               int(len(df)),
        "total_columns":            int(len(df.columns)),
        "numeric_column_count":     int(len(num_cols)),
        "categorical_column_count": int(len(cat_cols)),
        "numeric_columns":          num_cols,
        "categorical_columns":      cat_cols,
        "total_missing_values":     int(sum(missing.values())),
        "duplicates_removed":       dup_count,
    }

    facts: list[str] = []
    mem_kb = round(df.memory_usage(deep=True).sum() / 1024, 2)
    facts.append(f"💾 Dataset occupies **{mem_kb} KB** in memory.")
    facts.append(
        f"📐 **{len(num_cols)} numeric** and "
        f"**{len(cat_cols)} categorical** columns."
    )

    if missing:
        worst = max(missing, key=missing.get)
        facts.append(
            f"🕳️ Most missing: **'{worst}'** — {missing[worst]} cells."
        )
    else:
        facts.append("✅ No missing values in the original dataset.")

    facts.append(
        f"♻️ **{dup_count}** duplicate row(s) removed."
        if dup_count else "✅ No duplicate rows found."
    )

    if num_cols:
        desc = df[num_cols].describe()
        for col in num_cols[:3]:
            facts.append(
                f"📊 **{col}**: mean = {round(float(desc.loc['mean', col]), 3)}, "
                f"std = {round(float(desc.loc['std', col]), 3)}"
            )

    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
        for c in corr.columns:
            corr.loc[c, c] = 0.0
        idx = corr.stack().idxmax()
        facts.append(
            f"🔗 Strongest correlation: **'{idx[0]}'** ↔ **'{idx[1]}'** "
            f"(r = {round(float(corr.loc[idx[0], idx[1]]), 3)})."
        )

    for col in cat_cols[:2]:
        top = df[col].mode()
        facts.append(
            f"🏷️ **{col}**: {df[col].nunique()} unique value(s); "
            f"most common → '{top.iloc[0] if not top.empty else 'N/A'}'."
        )

    summary["dataset_facts"] = facts
    if null_report:
        summary["null_handling_report"] = null_report
    return {**state, "summary": summary}
