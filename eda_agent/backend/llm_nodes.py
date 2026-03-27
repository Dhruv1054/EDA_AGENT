"""LLM layer — two-step imputation pipeline + insights + chat.

Step 1: generate_imputation_plan  → LLM returns strict JSON plan (no code)
Step 2: generate_python_code_from_plan → LLM returns pure Python code
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from ollama import Client

from backend.job_store import update_job

log = logging.getLogger(__name__)

# ── Ollama client ──────────────────────────────────────────────────────────────
ollama_client = Client(timeout=300)

# ── System prompts ─────────────────────────────────────────────────────────────
_PLAN_SYSTEM   = "You are a data preprocessing assistant. Return only valid JSON with no extra text."
_CODE_SYSTEM   = "You are a Python code generator. Return only valid Python code. No markdown, no explanations."
_INSIGHT_SYSTEM = """You are a senior data analyst assistant.
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


# ── Internal helpers ───────────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Remove ```python / ``` fences the LLM may add despite instructions."""
    text = re.sub(r"```(?:python)?\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def _fallback_plan(profiles: list[dict]) -> dict:
    """Rule-based plan (Step 1 fallback) — same JSON schema as the LLM."""
    steps: list[dict] = []
    for p in profiles:
        col      = p.get("column_name", "")
        dtype    = str(p.get("dtype", "")).lower()
        skew     = p.get("skew")
        null_pct = float(p.get("null_percentage") or 0)
        unique   = int(p.get("unique_values") or 0)
        samples  = p.get("sample_values") or []

        if "datetime" in dtype or "date" in dtype or "time" in dtype:
            sem, cond, strat = "datetime", "datetime_series", "forward_fill"
        elif dtype.startswith("bool"):
            sem, cond, strat = "boolean_flag", "categorical_low_cardinality", "mode"
        elif dtype.startswith(("int", "float")) or "int" in dtype or "float" in dtype:
            sem = "numeric_measure"
            if null_pct > 50:
                cond, strat = "high_null_percentage", "median"
            elif skew is not None and abs(float(skew)) > 1.0:
                cond, strat = "numeric_skewed", "median"
            else:
                cond, strat = "numeric_non_skewed", "mean"
        elif "object" in dtype or "string" in dtype:
            name_l = col.lower()
            if any(t in name_l for t in ["id", "uuid", "identifier", "key", "code"]):
                sem, cond, strat = "identifier", "identifier_column", "leave_null"
            else:
                avg_len = (sum(len(str(x)) for x in samples if x) / max(len(samples), 1)) if samples else 0
                if avg_len >= 30:
                    sem, cond, strat = "free_text", "text_column", "constant_unknown"
                elif unique <= 10:
                    sem, cond, strat = "categorical_label", "categorical_low_cardinality", "mode"
                else:
                    sem, cond, strat = "categorical_label", "categorical_high_cardinality", "mode"
        else:
            sem, cond, strat = "categorical_label", "categorical_low_cardinality", "mode"

        steps.append({
            "column":        col,
            "semantic_type": sem,
            "condition":     cond,
            "strategy":      strat,
            "reason":        "Fallback rule-based plan (LLM unavailable).",
        })
    return {"imputation_plan": steps}


def _fallback_code_from_plan(plan: dict) -> str:
    """Rule-based code generator (Step 2 fallback) — produces pandas code from plan."""
    lines: list[str] = []
    for step in plan.get("imputation_plan", []):
        col   = step.get("column", "")
        strat = step.get("strategy", "")
        esc   = col.replace("\\", "\\\\").replace("'", "\\'")

        if strat == "median":
            lines.append(f"df['{esc}'] = df['{esc}'].fillna(df['{esc}'].median())")
        elif strat == "mean":
            lines.append(f"df['{esc}'] = df['{esc}'].fillna(df['{esc}'].mean())")
        elif strat == "mode":
            lines.append(f"_mv = df['{esc}'].mode(); df['{esc}'] = df['{esc}'].fillna(_mv.iloc[0] if not _mv.empty else 'Unknown')")
        elif strat == "forward_fill":
            lines.append(f"df['{esc}'] = df['{esc}'].ffill()")
        elif strat in ("constant_unknown", "constant"):
            lines.append(f"df['{esc}'] = df['{esc}'].fillna('Unknown')")
        elif strat == "leave_null":
            lines.append(f"# '{esc}' — identifier, left as null")
        else:
            lines.append(f"df['{esc}'] = df['{esc}'].fillna(df['{esc}'].mode().iloc[0] if not df['{esc}'].mode().empty else 'Unknown')")
    return "\n".join(lines)


# ── LangGraph Node — Step 1: Plan ──────────────────────────────────────────────

def generate_imputation_plan(state: dict) -> dict:
    """LangGraph Node — Step 1: LLM returns strict JSON plan (no pandas code)."""
    job_id = state.get("job_id", "")
    if job_id:
        update_job(job_id, progress=27, stage="Step 1 — Generating imputation plan")

    profiles: list[dict] = state.get("null_column_profiles", [])
    if not profiles:
        log.info("No null columns — skipping imputation plan.")
        return {**state, "imputation_plan": {"imputation_plan": []}}

    user_prompt = f"""You are a data preprocessing assistant.

Your task is to generate a column-level imputation plan.

For each column:

1. Identify semantic type:
   - numeric_measure
   - categorical_label
   - boolean_flag
   - identifier
   - datetime
   - free_text

2. Determine condition:
   - numeric_skewed
   - numeric_non_skewed
   - categorical_low_cardinality
   - categorical_high_cardinality
   - identifier_column
   - text_column
   - datetime_series
   - high_null_percentage

3. Select strategy:
   - numeric + skewed → median
   - numeric + non-skewed → mean
   - categorical → mode
   - boolean → mode
   - free_text → "Unknown"
   - identifier → leave null
   - datetime → forward fill

Return STRICT JSON:

{{
  "imputation_plan": [
    {{
      "column": "...",
      "semantic_type": "...",
      "condition": "...",
      "strategy": "...",
      "reason": "..."
    }}
  ]
}}

Do NOT include code.
Do NOT include explanations outside JSON.

Column profiles:
{json.dumps(profiles, indent=2)}
"""

    plan: dict = {}
    try:
        response = ollama_client.chat(
            model="llama3:8b-instruct-q4_K_M",
            messages=[
                {"role": "system", "content": _PLAN_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            format="json",
            options={"temperature": 0.0, "num_predict": 1000},
        )
        raw  = response["message"]["content"]
        plan = json.loads(raw) if isinstance(raw, str) else {}

        if "imputation_plan" not in plan or not isinstance(plan["imputation_plan"], list):
            raise ValueError(f"Invalid plan structure: {raw[:200]}")

        required = {"column", "semantic_type", "condition", "strategy", "reason"}
        for step in plan["imputation_plan"]:
            for k in required:
                if k not in step:
                    step[k] = ""

        log.info("Step 1 — Plan generated for %d columns.", len(plan["imputation_plan"]))
        print("\n=== IMPUTATION PLAN ===")
        print(json.dumps(plan, indent=2))

    except Exception as exc:
        log.warning("Step 1 LLM failed (%s) — using fallback plan.", exc)
        plan = _fallback_plan(profiles)

    return {**state, "imputation_plan": plan}


# ── LangGraph Node — Step 2: Code ─────────────────────────────────────────────

def generate_python_code_from_plan(state: dict) -> dict:
    """LangGraph Node — Step 2: LLM returns pure Python pandas code from the plan."""
    job_id = state.get("job_id", "")
    if job_id:
        update_job(job_id, progress=30, stage="Step 2 — Generating pandas code")

    plan: dict = state.get("imputation_plan", {"imputation_plan": []})
    if not plan.get("imputation_plan"):
        return {**state, "generated_code": ""}

    user_prompt = f"""You are a data preprocessing assistant.

Given the following imputation plan, generate valid Python pandas code to apply it on a DataFrame named `df`.

Rules:
- Use df[...] syntax
- Use fillna with mean/median/mode/constant/forward fill
- Do NOT include JSON
- Do NOT include explanations
- Do NOT include markdown
- Return ONLY Python code

Imputation plan:
{json.dumps(plan, indent=2)}
"""

    code: str = ""
    try:
        response = ollama_client.chat(
            model="llama3:8b-instruct-q4_K_M",
            messages=[
                {"role": "system", "content": _CODE_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            options={"temperature": 0.0, "num_predict": 800},
        )
        raw  = response["message"]["content"].strip()
        code = _strip_markdown(raw)

        if not code:
            raise ValueError("LLM returned empty code.")

        log.info("Step 2 — Code generated (%d chars).", len(code))
        print("\n=== GENERATED CODE ===")
        print(code)

    except Exception as exc:
        log.warning("Step 2 LLM failed (%s) — using fallback code.", exc)
        code = _fallback_code_from_plan(plan)
        print("\n=== GENERATED CODE (fallback) ===")
        print(code)

    return {**state, "generated_code": code}


# ── Context builder (unchanged) ───────────────────────────────────────────────

def _build_llm_context(state: dict) -> dict:
    df         = state.get("df")
    summary    = state.get("summary", {})
    fill_logic = state.get("fill_logic", [])

    ctx: dict[str, Any] = {
        "shape":   [summary.get("total_rows"), summary.get("total_columns")],
        "columns": {
            "numeric":     summary.get("numeric_columns", []),
            "categorical": summary.get("categorical_columns", []),
        },
        "missing_values_filled": {r["column"]: r["missing_before"] for r in fill_logic},
        "duplicates_removed":    state.get("duplicate_count", 0),
        "dataset_facts":         summary.get("dataset_facts", []),
    }

    if df is not None:
        num_cols = df.select_dtypes("number").columns.tolist()
        if num_cols:
            desc = df[num_cols].describe()
            ctx["numeric_stats"] = {
                col: {
                    "mean": round(float(desc.loc["mean", col]), 3),
                    "std":  round(float(desc.loc["std",  col]), 3),
                    "min":  round(float(desc.loc["min",  col]), 3),
                    "max":  round(float(desc.loc["max",  col]), 3),
                    "skew": round(float(df[col].skew()), 3),
                }
                for col in num_cols[:5]
            }
        if len(num_cols) >= 2:
            corr = df[num_cols].corr().abs()
            for c in corr.columns:
                corr.loc[c, c] = 0.0
            top = corr.stack().nlargest(3)
            ctx["top_correlations"] = {
                f"{a}↔{b}": round(float(v), 3) for (a, b), v in top.items()
            }
    return ctx


# ── LangGraph Node — LLM Insights (unchanged) ─────────────────────────────────

def generate_llm_insights(state: dict) -> dict:
    """LLM Node: structured AI insights via LLaMA3. Read-only."""
    job_id = state.get("job_id", "")
    if job_id:
        update_job(job_id, progress=85, stage="Generating AI insights with LLaMA3")

    ctx         = _build_llm_context(state)
    user_prompt = f"""Dataset summary (JSON):
{json.dumps(ctx, indent=2)}

Respond ONLY as valid JSON matching this exact schema:
{{
  "key_findings":            ["finding 1", "finding 2", "finding 3"],
  "data_quality_notes":      ["note 1", "note 2"],
  "anomalies":               ["anomaly 1"],
  "visual_analysis_insights":["insight 1", "insight 2"],
  "recommended_next_steps":  ["step 1", "step 2"]
}}

For visual_analysis_insights interpret skew/mean/std as histogram/boxplot signals.
Include 3-5 items per list. Reference actual column names and numbers.
"""

    try:
        response = ollama_client.chat(
            model="llama3:8b-instruct-q4_K_M",
            messages=[
                {"role": "system", "content": _INSIGHT_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            format="json",
            options={"temperature": 0.1, "num_predict": 600},
        )
        insights = json.loads(response["message"]["content"])
        for k in ["key_findings", "data_quality_notes", "anomalies",
                  "visual_analysis_insights", "recommended_next_steps"]:
            if k not in insights or not isinstance(insights[k], list):
                insights[k] = []
    except Exception as exc:
        insights = {
            "key_findings":            ["⚠️ LLM unavailable — start Ollama and pull llama3:8b-instruct-q4_K_M."],
            "data_quality_notes":      [f"Error: {exc}"],
            "anomalies":               [],
            "visual_analysis_insights":[],
            "recommended_next_steps":  ["Run: ollama pull llama3:8b-instruct-q4_K_M", "Then: ollama serve"],
        }

    if job_id:
        update_job(job_id, progress=95, stage="Finalising results")

    return {**state, "llm_insights": insights, "llm_context": ctx}


# ── Conversational helper (unchanged) ─────────────────────────────────────────

def chat_with_data(query: str, dataset_context: dict, history: list[dict]) -> str:
    system = (
        "You are a concise data analyst assistant. "
        "Answer questions based ONLY on the dataset context provided. "
        "Be specific, factual, and brief (3-5 sentences max). "
        "If the question cannot be answered from the data, say so clearly."
    )
    messages = [
        {"role": "system",    "content": system},
        {"role": "user",      "content": f"Dataset context:\n{json.dumps(dataset_context, indent=2)}"},
        {"role": "assistant", "content": "I've reviewed the dataset context. Ask me anything about it."},
        *history[-MAX_HISTORY_TURNS:],
        {"role": "user", "content": query},
    ]
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
            "Make sure Ollama is running:\n```\nollama serve\nollama pull llama3:8b-instruct-q4_K_M\n```"
        )
