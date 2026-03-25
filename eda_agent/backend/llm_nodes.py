"""LLM layer — strictly read-only interpretation.
Receives serialised stats JSON → returns text/structured JSON.
Never mutates the DataFrame or any deterministic pipeline state.
"""
from __future__ import annotations

import logging
import json
from typing import Any

import ollama
from ollama import Client

from backend.job_store import update_job

log = logging.getLogger(__name__)

# ── Prompts ────────────────────────────────────────────────────────────────────

IMPUTATION_SYSTEM_PROMPT = (
    "You are a data preprocessing assistant. "
    "Return only valid JSON with no extra text or markdown."
)

SYSTEM_PROMPT = """You are a senior data analyst assistant.
You receive a JSON summary of an already-cleaned dataset and provide concise,
factual, non-hallucinated insights.

Rules:
- ONLY reference columns and statistics present in the JSON context.
- Do NOT invent statistics or column names.
- Flag correlations > 0.7 as "strong", 0.4-0.7 as "moderate", < 0.4 as "weak".
- Always respond in valid JSON matching the schema given.
- If uncertain, write "Insufficient data." rather than speculating.
"""

MAX_HISTORY_TURNS = 6

# Initialize global client with a long timeout for local LLM inference
ollama_client = Client(timeout=300)

# ── Missing-value semantic classifier (helper; NOT a LangGraph node) ────────────

SEMANTIC_TYPES = {
    "numeric_measure",
    "categorical_label",
    "boolean_flag",
    "identifier",
    "datetime",
    "free_text",
}


def detect_semantic_type_for_column(profile: dict) -> dict:
    """Return semantic type classification for a single column profile.

    This helper is intentionally side-effect free. It does NOT impute values; it
    only returns a semantic label + reason so deterministic pandas logic can act.
    """
    prompt = f"""You are a data preprocessing assistant.
Your task is to classify a dataset column into one of these semantic types:
- numeric_measure
- categorical_label
- boolean_flag
- identifier
- datetime
- free_text
Column Metadata:
Column Name: {profile.get("column_name")}
Data Type: {profile.get("dtype")}
Null Percentage: {profile.get("null_percentage")}
Unique Values: {profile.get("unique_values")}
Sample Values: {profile.get("sample_values")}
Return strictly valid JSON in this format:
{{
 "semantic_type": "...",
 "reason": "..."
}}
Do not return any extra text.
"""

    try:
        response = ollama_client.chat(
            model="llama3:8b-instruct-q4_K_M",
            messages=[
                {"role": "system", "content": "You are a data preprocessing assistant."},
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={"temperature": 0.0, "num_predict": 200},
        )
        raw = response["message"]["content"]
        data = json.loads(raw) if isinstance(raw, str) else {}
        semantic_type = str(data.get("semantic_type", "")).strip()
        reason = str(data.get("reason", "")).strip() or "No reason provided."
        if semantic_type not in SEMANTIC_TYPES:
            raise ValueError(f"Invalid semantic_type: {semantic_type!r}")
        return {"semantic_type": semantic_type, "reason": reason}
    except Exception as exc:
        log.warning("Semantic detection failed for %s: %s", profile.get("column_name"), exc)
        return {
            "semantic_type": "categorical_label",
            "reason": f"Fallback classification used (LLM unavailable/invalid): {exc}",
        }


# ── Imputation plan generator (LangGraph node) ─────────────────────────────────

def _fallback_imputation_plan(profiles: list[dict]) -> dict:
    """Rule-based fallback plan used when LLM is unavailable or returns invalid JSON."""
    steps: list[dict] = []
    for p in profiles:
        col: str = p.get("column_name", "")
        dtype: str = str(p.get("dtype", "")).lower()
        skew = p.get("skew")
        null_pct: float = float(p.get("null_percentage") or 0)
        unique: int = int(p.get("unique_values") or 0)
        samples: list = p.get("sample_values") or []

        # Escape single quotes in column name for safe code generation
        col_esc = col.replace("\\", "\\\\").replace("'", "\\'")

        if "datetime" in dtype or "date" in dtype or "time" in dtype:
            sem, condition, strategy = "datetime", "datetime_series", "forward_fill"
            code = f"df['{col_esc}'] = df['{col_esc}'].ffill()"

        elif dtype.startswith("bool"):
            sem, condition, strategy = "boolean_flag", "categorical_low_cardinality", "mode"
            code = (
                f"_mv = df['{col_esc}'].mode(); "
                f"df['{col_esc}'] = df['{col_esc}'].fillna(_mv.iloc[0] if not _mv.empty else False)"
            )

        elif dtype.startswith(("int", "float")) or "int" in dtype or "float" in dtype:
            sem = "numeric_measure"
            if null_pct > 50:
                condition, strategy = "high_null_percentage", "median"
                code = f"df['{col_esc}'] = df['{col_esc}'].fillna(df['{col_esc}'].median())"
            elif skew is not None and abs(float(skew)) > 1.0:
                condition, strategy = "numeric_skewed", "median"
                code = f"df['{col_esc}'] = df['{col_esc}'].fillna(df['{col_esc}'].median())"
            else:
                condition, strategy = "numeric_non_skewed", "mean"
                code = f"df['{col_esc}'] = df['{col_esc}'].fillna(df['{col_esc}'].mean())"

        elif dtype in {"object", "string"} or "object" in dtype:
            name_lower = col.lower()
            if any(tok in name_lower for tok in ["id", "uuid", "identifier", "key", "code"]):
                sem, condition, strategy = "identifier", "identifier_column", "leave_null"
                code = f"# '{col_esc}' is an identifier — left as null"
            else:
                avg_len = 0.0
                if samples:
                    lens = [len(str(x)) for x in samples if x is not None]
                    avg_len = sum(lens) / max(len(lens), 1)
                if avg_len >= 30:
                    sem, condition, strategy = "free_text", "text_column", "constant_unknown"
                    code = f"df['{col_esc}'] = df['{col_esc}'].fillna('Unknown')"
                elif unique <= 10:
                    sem, condition, strategy = "categorical_label", "categorical_low_cardinality", "mode"
                    code = (
                        f"_mv = df['{col_esc}'].mode(); "
                        f"df['{col_esc}'] = df['{col_esc}'].fillna(_mv.iloc[0] if not _mv.empty else 'Unknown')"
                    )
                else:
                    sem, condition, strategy = "categorical_label", "categorical_high_cardinality", "mode"
                    code = (
                        f"_mv = df['{col_esc}'].mode(); "
                        f"df['{col_esc}'] = df['{col_esc}'].fillna(_mv.iloc[0] if not _mv.empty else 'Unknown')"
                    )
        else:
            sem, condition, strategy = "categorical_label", "categorical_low_cardinality", "mode"
            code = (
                f"_mv = df['{col_esc}'].mode(); "
                f"df['{col_esc}'] = df['{col_esc}'].fillna(_mv.iloc[0] if not _mv.empty else 'Unknown')"
            )

        steps.append(
            {
                "column": col,
                "semantic_type": sem,
                "condition": condition,
                "strategy": strategy,
                "reason": "Fallback rule-based classification (LLM unavailable or returned invalid JSON).",
                "code": code,
            }
        )
    return {"imputation_plan": steps}


def generate_imputation_plan_using_llm(state: dict) -> dict:
    """LangGraph Node: Ask LLaMA3 to produce a column-level imputation plan with pandas code."""
    job_id = state.get("job_id", "")
    if job_id:
        update_job(job_id, progress=28, stage="Generating LLM imputation plan")

    profiles: list[dict] = state.get("null_column_profiles", [])
    if not profiles:
        return {**state, "imputation_plan": {"imputation_plan": []}}

    user_prompt = f"""You are a data preprocessing assistant.
Your task is to generate a column-level imputation plan for missing values.
For each column, analyze metadata and:
1. Identify semantic type:
   - numeric_measure
   - categorical_label
   - boolean_flag
   - identifier
   - datetime
   - free_text
2. Determine condition with more granularity:
   - numeric_skewed
   - numeric_non_skewed
   - categorical_low_cardinality
   - categorical_high_cardinality
   - identifier_column
   - text_column
   - datetime_series
   - high_null_percentage
3. Select imputation strategy using rules:
- numeric + skewed → median
- numeric + non-skewed → mean
- categorical → mode
- boolean → mode
- free_text → "Unknown"
- identifier → leave null
- datetime → forward fill
4. Generate executable pandas code using df
5. Return STRICT JSON:
{{
  "imputation_plan": [
    {{
      "column": "...",
      "semantic_type": "...",
      "condition": "...",
      "strategy": "...",
      "reason": "...",
      "code": "..."
    }}
  ]
}}
Do NOT return anything outside JSON.

Column profiles:
{json.dumps(profiles, indent=2)}
"""

    try:
        response = ollama_client.chat(
            model="llama3:8b-instruct-q4_K_M",
            messages=[
                {"role": "system", "content": IMPUTATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
            options={"temperature": 0.0, "num_predict": 1200},
        )
        raw = response["message"]["content"]
        plan: dict = json.loads(raw) if isinstance(raw, str) else {}

        # Validate structure
        if "imputation_plan" not in plan or not isinstance(plan["imputation_plan"], list):
            raise ValueError(f"Missing or invalid 'imputation_plan' key in LLM response: {raw[:200]}")

        # Ensure every step has the required keys; patch missing ones defensively
        required_keys = {"column", "semantic_type", "condition", "strategy", "reason", "code"}
        for step in plan["imputation_plan"]:
            for k in required_keys:
                if k not in step:
                    step[k] = ""

        log.info("LLM generated imputation plan with %d steps.", len(plan["imputation_plan"]))

    except Exception as exc:
        log.warning("LLM imputation plan generation failed (%s) — using fallback.", exc)
        plan = _fallback_imputation_plan(profiles)

    return {**state, "imputation_plan": plan}


# ── Context builder ────────────────────────────────────────────────────────────

def _build_llm_context(state: dict) -> dict:
    """Build a compact, token-efficient context dict from pipeline state."""
    df          = state.get("df")
    summary     = state.get("summary", {})
    fill_logic  = state.get("fill_logic", [])

    ctx: dict[str, Any] = {
        "shape": [summary.get("total_rows"), summary.get("total_columns")],
        "columns": {
            "numeric":     summary.get("numeric_columns", []),
            "categorical": summary.get("categorical_columns", []),
        },
        "missing_values_filled": {
            r["column"]: r["missing_before"] for r in fill_logic
        },
        "duplicates_removed": state.get("duplicate_count", 0),
        "dataset_facts": summary.get("dataset_facts", []),
    }

    if df is not None:
        numeric_cols = df.select_dtypes("number").columns.tolist()
        if numeric_cols:
            desc = df[numeric_cols].describe()
            ctx["numeric_stats"] = {
                col: {
                    "mean": round(float(desc.loc["mean", col]), 3),
                    "std":  round(float(desc.loc["std",  col]), 3),
                    "min":  round(float(desc.loc["min",  col]), 3),
                    "max":  round(float(desc.loc["max",  col]), 3),
                    "skew": round(float(df[col].skew()), 3) if hasattr(df[col], "skew") else 0,
                }
                for col in numeric_cols[:5]
            }
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().abs()
            for c in corr.columns:
                corr.loc[c, c] = 0.0
            top = corr.stack().nlargest(3)
            ctx["top_correlations"] = {
                f"{a}↔{b}": round(float(v), 3) for (a, b), v in top.items()
            }
    return ctx


# ── LangGraph Node ─────────────────────────────────────────────────────────────

def generate_llm_insights(state: dict) -> dict:
    """LLM Node: structured AI insights via LLaMA3 (Ollama). Read-only."""
    job_id = state.get("job_id", "")
    if job_id:
        update_job(job_id, progress=85, stage="Generating AI insights with LLaMA3")

    ctx = _build_llm_context(state)

    user_prompt = f"""Dataset summary (JSON):
{json.dumps(ctx, indent=2)}

Respond ONLY as valid JSON matching this exact schema:
{{
  "key_findings":           ["finding 1", "finding 2", "finding 3"],
  "data_quality_notes":     ["note 1", "note 2"],
  "anomalies":              ["anomaly 1", "anomaly 2"],
  "visual_analysis_insights":["insight from charts 1", "insight from charts 2"],
  "recommended_next_steps": ["step 1", "step 2", "step 3"]
}}

For 'visual_analysis_insights', interpret the numeric stats (mean, std, skew) as if you were looking at Histograms, Boxplots, and Heatmaps. 
- Mention distributions based on skewness.
- Mention potential outliers based on min/max vs mean.
- Mention correlation trends.

Include 3-5 items in each list. Be specific — reference actual column names and numbers.
"""

    try:
        response = ollama_client.chat(
            model="llama3:8b-instruct-q4_K_M",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            format="json",
            options={"temperature": 0.1, "num_predict": 600},
        )
        raw      = response["message"]["content"]
        insights = json.loads(raw)
        for key in ["key_findings", "data_quality_notes", "anomalies", "visual_analysis_insights", "recommended_next_steps"]:
            if key not in insights or not isinstance(insights[key], list):
                insights[key] = []

    except Exception as exc:
        insights = {
            "key_findings":           ["⚠️ LLM unavailable — start Ollama and pull llama3:8b-instruct-q4_K_M."],
            "data_quality_notes":     [f"Error detail: {exc}"],
            "anomalies":              [],
            "visual_analysis_insights": [],
            "recommended_next_steps": [
                "Run: ollama pull llama3:8b-instruct-q4_K_M",
                "Then: ollama serve",
            ],
        }

    if job_id:
        update_job(job_id, progress=95, stage="Finalising results")

    return {**state, "llm_insights": insights, "llm_context": ctx}


# ── Conversational helper (not a LangGraph node) ───────────────────────────────

def chat_with_data(query: str, dataset_context: dict, history: list[dict]) -> str:
    """Direct LLM call for follow-up questions. Uses sliding window memory."""
    system = (
        "You are a concise data analyst assistant. "
        "Answer questions based ONLY on the dataset context provided. "
        "Be specific, factual, and brief (3-5 sentences max). "
        "If the question cannot be answered from the provided data, say so clearly."
    )

    recent = history[-MAX_HISTORY_TURNS:]
    messages = [
        {"role": "system",    "content": system},
        {"role": "user",      "content": f"Dataset context:\n{json.dumps(dataset_context, indent=2)}"},
        {"role": "assistant", "content": "I've reviewed the dataset context. Ask me anything about it."},
    ]
    messages.extend(recent)
    messages.append({"role": "user", "content": query})

    try:
        response = ollama_client.chat(
            model="llama3:8b-instruct-q4_K_M",
            messages=messages,
            options={"temperature": 0.3, "num_predict": 350},
        )
        return response["message"]["content"]
    except Exception as exc:
        return (
            f"⚠️ LLM unavailable: {exc}\n\n"
            "Make sure Ollama is running:\n"
            "```\nollama serve\nollama pull llama3:8b-instruct-q4_K_M\n```"
        )
