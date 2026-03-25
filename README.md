# EDA Agent

This repository contains a simple AI-powered Exploratory Data Analysis (EDA) tool. You upload a CSV file and the backend cleans, analyzes, visualizes, and explains the dataset, with optional LLM insights. A Streamlit frontend lets you interact with the pipeline.

## High‑level overview

- **backend**: FastAPI service implementing the pipeline, job tracking and caching.
- **frontend**: Streamlit app that lets users upload files, poll job status, display results and chat with the dataset.
- **outputs**: Directory where uploaded files, processed results and generated charts are stored.

## Important files

### Root
- `README.md` – this file (project overview and explanation).
- `eda_agent/requirements.txt` – Python dependencies needed to run the app.

### Backend (`eda_agent/backend/`)

- `main.py` – FastAPI application. Handles uploads, status/result endpoints, file validation, job lifecycle, caching, and orchestrates the background pipeline.
- `models.py` – Pydantic models used for request/response validation (upload responses, job status, chat payloads).
- `job_store.py` – In‑memory record of running jobs and a simple file‑hash cache. Provides CRUD helpers for job state and cache.
- `graph.py` – Defines the LangGraph state graph that strings together the individual processing nodes into a pipeline. Also declares the shared `EDAState` typed dictionary.
- `eda_nodes.py` – Deterministic pipeline steps (load CSV, detect/fill missing values, deduplicate, generate charts and summary). Each function accepts a state dict, updates progress, and returns new state.
- `llm_nodes.py` – LLM integration layer. Builds a compact JSON context from the pipeline state and calls an Ollama LLaMA3 model to produce structured insights. Also contains a helper for conversational follow‑ups.

### Frontend (`eda_agent/frontend/`)

- `app.py` – Streamlit user interface. Handles file upload, starts jobs via API, polls for progress, renders results (summary metrics, charts, insights) and provides download/chat features.

### Miscellaneous

- `outputs/` – Generated at runtime. Contains subfolders for session outputs (`charts, cleaned files, etc.`) and an `uploads/` folder holding raw CSVs that were processed.

## How it works (in plain terms)

1. **Upload a CSV** through the Streamlit front end.
2. The frontend POSTs the file to `/upload`; the backend saves it and starts a background pipeline job.
3. The pipeline (implemented as a LangGraph) runs several nodes in order:
   - Read the CSV into a pandas DataFrame.
   - Detect missing values (pre-clean snapshot).
   - **Profile only columns with nulls** (name, dtype, null %, unique count, sample values).
   - **Semantic type detection (LLM-assisted)** for each null-containing column using Ollama + `llama3:8b-instruct-q4_K_M`:
     - semantic types: `numeric_measure`, `categorical_label`, `boolean_flag`, `identifier`, `datetime`, `free_text`
     - the LLM **does not** impute values row-by-row; it only classifies the column
   - **Deterministic strategy mapping** (fixed code) from semantic type → imputation strategy:
     - `numeric_measure` → `median`
     - `categorical_label` → `mode`
     - `boolean_flag` → `mode`
     - `identifier` → `leave_null`
     - `datetime` → `forward_fill`
     - `free_text` → `constant:Unknown`
   - Apply imputation using pandas (safe fallbacks for all-null columns / empty mode / median failures).
   - Count and remove duplicate rows.
   - Generate basic charts (histogram, boxplot, correlation heatmap).
   - Build a textual summary of the dataset (row/column counts, memory usage, basic stats and facts).
   - (Optionally) pass the summary to an LLM to get human‑readable insights.
4. Job progress is tracked in `job_store`; clients can poll `/status/{job_id}`.
5. Once complete, results are cached by file hash so re‑uploads of the same file return instantly.
6. The frontend displays the results and lets you download the cleaned CSV or ask follow‑up questions via `/chat`.

### Missing-value outputs (what you’ll see in results/UI)

- The API response includes:
  - `missing_values`: missing counts **before** cleaning
  - `fill_logic`: per-column imputation decisions (strategy + fill value), plus optional semantic metadata:
    - `semantic_type` (e.g. `identifier`, `free_text`)
    - `strategy_key` (e.g. `leave_null`, `constant:Unknown`)
  - `summary["null_handling_report"]`: a compact, human-readable report of null handling decisions
- The Streamlit UI can optionally display semantic details and the null-handling report.

## Getting started

1. Create a Python virtual environment.
2. Install dependencies from `eda_agent/requirements.txt`.
3. Start the backend server:
   ```bash
   cd eda_agent
   uvicorn backend.main:app --reload
   ```
4. In a separate terminal start the frontend:
   ```bash
   cd eda_agent
   streamlit run frontend/app.py
   ```
5. Open the Streamlit URL shown in the terminal (usually `http://localhost:8501`).

## Notes & tips

- The LLM functionality requires [Ollama](https://ollama.com/) running with the `llama3:8b-instruct-q4_K_M` model pulled and served.
- The rate limiter prevents abuse during development (`10 uploads/min`, `30 chat requests/min`).
- `job_store` is in‑memory; restart the server to clear job history. For production use a shared backend like Redis.
- The pipeline is intentionally simple and deterministic; the LLM node only reads state and never mutates the DataFrame.

Enjoy exploring your data!
