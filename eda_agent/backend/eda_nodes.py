import json
import os
import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from backend.job_store import update_job

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
    """Node 3: Build compact metadata (including stats) for columns that contain nulls."""
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

        profile: dict = {
            "column_name": str(col),
            "dtype": str(s.dtype),
            "null_count": int(null_count),
            "null_percentage": round((int(null_count) / n_rows) * 100.0, 3),
            "unique_values": int(non_null.nunique(dropna=True)),
            "sample_values": sample,
            "mean": None,
            "median": None,
            "skew": None,
        }

        if pd.api.types.is_numeric_dtype(s) and not non_null.empty:
            try:
                profile["mean"] = round(float(non_null.mean()), 4)
                profile["median"] = round(float(non_null.median()), 4)
                profile["skew"] = round(float(non_null.skew()), 4) if len(non_null) >= 3 else None
            except Exception:
                pass

        profiles.append(profile)

    return {**state, "null_column_profiles": profiles}


def execute_imputation_plan(state: dict) -> dict:
    """Node 5: Dynamically execute LLM-generated imputation code on the DataFrame."""
    _tick(state, 33, "Executing LLM imputation plan")
    df: pd.DataFrame = state["df"].copy()
    plan: dict = state.get("imputation_plan", {})
    steps: list[dict] = plan.get("imputation_plan", [])

    executed: list[dict] = []
    # Single shared namespace so each step sees mutations from prior steps
    exec_ns: dict = {"df": df, "pd": pd}

    for step in steps:
        col = step.get("column", "")
        code = (step.get("code") or "").strip()

        print("\n---------------------------")
        print("Column:", step.get("column"))
        print("Type:", step.get("semantic_type"))
        print("Condition:", step.get("condition"))
        print("Strategy:", step.get("strategy"))
        print("Code:", code)

        if not code or code.lstrip().startswith("#"):
            executed.append({**step, "status": "skipped", "error": "No executable code"})
            continue

        try:
            exec(code, exec_ns)  # noqa: S102
            df = exec_ns.get("df", df)
            executed.append({**step, "status": "success"})
        except Exception as exc:
            print(f"Error executing code for column {step.get('column')}: {exc}")
            log.warning("Imputation code failed for column '%s': %s | code: %s", col, exc, code)
            # Safe fallback
            if col in df.columns:
                try:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())
                        fallback = "fallback:median"
                    else:
                        mode_s = df[col].mode(dropna=True)
                        df[col] = df[col].fillna(mode_s.iloc[0] if not mode_s.empty else "Unknown")
                        fallback = "fallback:mode"
                    exec_ns["df"] = df
                    executed.append({**step, "status": fallback, "error": str(exc)})
                except Exception as fe:
                    executed.append({**step, "status": "failed", "error": str(exc), "fallback_error": str(fe)})
            else:
                executed.append({**step, "status": "failed", "error": str(exc)})

    print("\n=== DATAFRAME AFTER IMPUTATION ===")
    print(df.head(10))

    print("\n=== NULL COUNTS AFTER IMPUTATION ===")
    print(df.isnull().sum())

    return {**state, "df": df, "executed_imputation_steps": executed}


_STRATEGY_FILL_LABEL: dict[str, str] = {
    "median":           "median value",
    "mean":             "mean value",
    "mode":             "mode value",
    "forward_fill":     "forward filled",
    "leave_null":       "—",
    "constant_unknown": "Unknown",
}


def generate_imputation_report(state: dict) -> dict:
    """Node 6: Build structured imputation report from executed plan steps."""
    _tick(state, 36, "Generating imputation report")
    missing: dict = state.get("missing_values", {})
    executed: list[dict] = state.get("executed_imputation_steps", [])

    fill_logic: list[dict] = []
    report_lines: list[str] = []

    for step in executed:
        col = step.get("column", "")
        null_count = int(missing.get(col, 0))
        sem_type = step.get("semantic_type", "")
        condition = step.get("condition", "")
        strategy = step.get("strategy", "")
        status = step.get("status", "")
        reason = step.get("reason", "")
        # Derive a human-readable fill label for the frontend
        fill_value = _STRATEGY_FILL_LABEL.get(strategy.lower(), strategy or "—")

        fill_logic.append(
            {
                "column": col,
                "missing_before": null_count,
                "semantic_type": sem_type,
                "condition": condition,
                "strategy": strategy,
                "fill_value": fill_value,
                "status": status,
                "reason": reason,
            }
        )
        report_lines.append(
            f"- Column: {col} | Type: {sem_type} | Condition: {condition} "
            f"| Strategy: {strategy} | Status: {status}"
        )

    imputation_report = {
        "columns": [
            {
                "column": s.get("column", ""),
                "semantic_type": s.get("semantic_type", ""),
                "condition": s.get("condition", ""),
                "strategy": s.get("strategy", ""),
            }
            for s in executed
        ]
    }

    null_report = "Null Handling Report:\n" + (
        "\n".join(report_lines) if report_lines else "✅ No missing values to handle."
    )

    return {
        **state,
        "fill_logic": fill_logic,
        "imputation_report": imputation_report,
        "null_handling_report": null_report,
    }


def detect_duplicates(state: dict) -> dict:
    """Node 7: Count duplicate rows (after imputation)."""
    _tick(state, 40, "Detecting duplicate rows")
    df: pd.DataFrame = state["df"]
    return {**state, "duplicate_count": int(df.duplicated().sum())}


def remove_duplicates(state: dict) -> dict:
    """Node 8: Drop duplicate rows — keep first occurrence."""
    _tick(state, 50, "Removing duplicates")
    df: pd.DataFrame = state["df"]
    dup_count: int = state.get("duplicate_count", 0)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    dup_logic = {
        "duplicates_found": dup_count,
        "strategy": "Keep First Occurrence",
        "reason": (
            "Exact duplicate rows carry no new information and bias "
            "statistical results. The first occurrence is retained "
            "and subsequent copies are dropped."
        ),
    }
    return {**state, "df": df_clean, "dup_logic": dup_logic}


def generate_charts(state: dict) -> dict:
    """Node 9: Histogram, boxplot, correlation heatmap — scoped to session dir."""
    _tick(state, 62, "Generating charts")
    df: pd.DataFrame = state["df"]
    session_id = state.get("session_id", "default")
    out_dir = _session_dir(session_id)
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
        sz = max(6, len(numeric_cols))
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
    """Node 10: Comprehensive dataset summary + important facts list."""
    _tick(state, 75, "Building dataset summary")
    df: pd.DataFrame = state["df"]
    missing: dict = state.get("missing_values", {})
    dup_count: int = state.get("duplicate_count", 0)
    null_report: str = state.get("null_handling_report", "")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    summary: dict = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "numeric_column_count": int(len(num_cols)),
        "categorical_column_count": int(len(cat_cols)),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "total_missing_values": int(sum(missing.values())),
        "duplicates_removed": dup_count,
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
