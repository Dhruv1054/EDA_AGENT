import os
import time

import pandas as pd
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_URL   = "http://127.0.0.1:8000"
UPLOAD_URL = f"{BASE_URL}/upload"
STATUS_URL = f"{BASE_URL}/status"
RESULT_URL = f"{BASE_URL}/result"
CHAT_URL   = f"{BASE_URL}/chat"

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "analysis_done": False,
    "eda_result":    {},
    "session_id":    "",
    "llm_context":      {},
    "chat_history":     [],
    "cleaned_csv_data": None,
    "job_id":           "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#0a0d16;color:#e4e6eb;}

.hero{background:linear-gradient(135deg,#1a1f35 0%,#0d1b2a 100%);
      border:1px solid #2a3550;border-radius:20px;padding:3rem 2rem 2.5rem;
      margin-bottom:2rem;text-align:center;}
.hero h1{font-size:2.8rem;font-weight:800;
          background:linear-gradient(90deg,#4f9cf9,#a78bfa,#f472b6);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;}
.hero p{color:#8899aa;font-size:1.05rem;margin-top:.6rem;}

.metric-card{background:linear-gradient(145deg,#1a1f35,#151929);
             border:1px solid #2a3550;border-radius:14px;
             padding:1.3rem 1.5rem;margin-bottom:1rem;
             transition:transform .2s,border-color .2s;}
.metric-card:hover{transform:translateY(-2px);border-color:#4f9cf9;}
.metric-card h3{color:#4f9cf9;font-size:.8rem;font-weight:600;
                text-transform:uppercase;letter-spacing:.09em;margin:0 0 .35rem;}
.metric-card p{color:#e4e6eb;font-size:1.9rem;font-weight:700;margin:0;}

.section-header{color:#a78bfa;font-size:1.05rem;font-weight:700;
                border-bottom:2px solid #2a3550;padding-bottom:.45rem;
                margin:2rem 0 1rem;text-transform:uppercase;letter-spacing:.07em;}

.phase-banner{background:linear-gradient(135deg,#0d1b2a,#1a1035);
              border-left:4px solid #a78bfa;border-radius:10px;
              padding:.9rem 1.4rem;margin:.5rem 0 1.2rem;
              font-size:.95rem;color:#c4b5fd;font-weight:500;}

.logic-pill{display:inline-block;background:#1e2d40;border:1px solid #3b5272;
            border-radius:20px;padding:.25rem .75rem;font-size:.78rem;
            color:#7dd3fc;font-weight:600;margin-right:.4rem;}

.insight-card{background:#131828;border-left:3px solid #4f9cf9;
              border-radius:8px;padding:.65rem 1rem;margin:.4rem 0;
              font-size:.9rem;color:#d1d5db;line-height:1.5;}

.insight-card.purple{border-left-color:#a78bfa;}
.insight-card.pink  {border-left-color:#f472b6;}
.insight-card.green {border-left-color:#34d399;}

.fact-card{background:#131828;border-left:3px solid #4f9cf9;
           border-radius:8px;padding:.65rem 1rem;margin:.45rem 0;
           font-size:.9rem;color:#d1d5db;line-height:1.5;}

/* ── Merged EDA Report card ── */
.report-card{
  background:linear-gradient(145deg,#111827,#0d1420);
  border:1px solid #2a3550;border-radius:20px;
  padding:2rem 2.2rem;margin-top:1.5rem;}

.report-title{
  font-size:1.55rem;font-weight:800;
  background:linear-gradient(90deg,#4f9cf9,#a78bfa,#f472b6);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  margin:0 0 .3rem;}

.report-subtitle{
  color:#6b7a8f;font-size:.88rem;margin-bottom:1.8rem;
  border-bottom:1px solid #1e2d40;padding-bottom:1rem;}

.report-sub{
  font-size:.72rem;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;color:#4f9cf9;
  border-bottom:1px solid #1e2d40;padding-bottom:.4rem;
  margin:1.6rem 0 .9rem;}

.report-sub.v{color:#a78bfa;}
.report-sub.w{color:#f472b6;}
.report-sub.g{color:#34d399;}

.summary-grid{
  display:grid;grid-template-columns:repeat(4,1fr);
  gap:.8rem;margin-bottom:1.4rem;}

.sg-item{
  background:#1a1f35;border:1px solid #2a3550;
  border-radius:12px;padding:.9rem 1rem;
  text-align:center;transition:border-color .2s;}
.sg-item:hover{border-color:#4f9cf9;}
.sg-item .sg-val{font-size:1.6rem;font-weight:800;color:#e4e6eb;}
.sg-item .sg-lbl{font-size:.72rem;font-weight:600;letter-spacing:.07em;
                  text-transform:uppercase;color:#6b7a8f;margin-top:.25rem;}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    display: flex;
    width: 100%;
    gap: 0.5rem;
    background: #111827;
    padding: .5rem;
    border-radius: 12px;
    border: 1px solid #2a3550;
}
.stTabs [data-baseweb="tab"] {
    flex: 1 1 0%;
    min-width: 0;
    height: 54px;
    white-space: nowrap;
    background-color: transparent !important;
    border-radius: 8px;
    color: #94a3b8 !important;
    font-weight: 800;
    font-size: 1rem;
    border: none;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    transition: all .3s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    text-overflow: ellipsis;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #4f9cf9 !important;
    background: #1e293b !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e293b, #0f172a) !important;
    color: #4f9cf9 !important;
    border-bottom: 2px solid #4f9cf9 !important;
}

div.stButton>button{background:linear-gradient(135deg,#4f9cf9,#a78bfa);
                    color:white;border:none;border-radius:10px;
                    padding:.6rem 2.8rem;font-weight:700;font-size:1rem;
                    transition:opacity .2s,transform .15s;}
div.stButton>button:hover{opacity:.85;transform:translateY(-1px);}

.download-btn-container {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 1rem;
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔬 EDA Agent</h1>
  <p>Upload a CSV — the AI pipeline cleans, analyses, visualises, and explains your data with LLaMA3.</p>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📂 Upload Dataset</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose a CSV file", type=["csv"],
    help="Max 50 MB. The pipeline will clean, chart, and explain your data automatically.",
)
run_btn = st.button("🚀 Run EDA Analysis")

# ── Run Pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if uploaded_file is None:
        st.warning("⚠️ Please upload a CSV file first.")
    else:
        st.session_state.analysis_done = False
        st.session_state.chat_history  = []

        # ── Step 1: submit job ────────────────────────────────────────────────
        try:
            resp = requests.post(
                UPLOAD_URL,
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                timeout=30,
            )
            resp.raise_for_status()
            meta       = resp.json()
            job_id     = meta["job_id"]
            session_id = meta["session_id"]
            st.session_state.session_id = session_id
        except requests.exceptions.ConnectionError:
            st.error("❌ Backend unreachable. Start it with:\n```\nuvicorn backend.main:app --reload\n```")
            st.stop()
        except Exception as exc:
            st.error(f"❌ Upload failed: {exc}")
            st.stop()

        # ── Step 2: poll for completion ───────────────────────────────────────
        progress_bar  = st.progress(0)
        stage_label   = st.empty()

        while True:
            try:
                s = requests.get(f"{STATUS_URL}/{job_id}", timeout=10).json()
            except Exception:
                time.sleep(1)
                continue

            pct   = s.get("progress", 0)
            stage = s.get("stage", "…")
            progress_bar.progress(pct / 100)
            stage_label.markdown(
                f'<div class="phase-banner">⚙️ <strong>{stage}</strong></div>',
                unsafe_allow_html=True,
            )

            if s["status"] == "done":
                break
            elif s["status"] == "failed":
                st.error(f"❌ Pipeline failed: {s.get('error', 'unknown error')}")
                st.stop()
            time.sleep(1)

        progress_bar.empty()
        stage_label.empty()

        # ── Step 3: fetch result ──────────────────────────────────────────────
        try:
            data = requests.get(f"{RESULT_URL}/{job_id}", timeout=30).json()
        except Exception as exc:
            st.error(f"❌ Failed to fetch result: {exc}")
            st.stop()

        # ── Step 4: fetch cleaned CSV data ────────────────────────────────────
        try:
            dl_resp = requests.get(f"{BASE_URL}/download/{job_id}", timeout=10)
            if dl_resp.status_code == 200:
                st.session_state.cleaned_csv_data = dl_resp.content
            else:
                st.session_state.cleaned_csv_data = None
        except Exception:
            st.session_state.cleaned_csv_data = None

        st.session_state.analysis_done = True
        st.session_state.eda_result    = data
        st.session_state.llm_context   = data.get("llm_context", {})
        st.session_state.job_id        = job_id
        st.success("✅ Analysis complete!")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.analysis_done:
    data          = st.session_state.eda_result
    summary       = data.get("summary", {})
    missing       = data.get("missing_values", {})
    fill_logic    = data.get("fill_logic", [])
    dup_logic     = data.get("dup_logic", {})
    dup_removed   = data.get("duplicates_removed", 0)
    charts        = data.get("charts_generated", [])
    llm_insights  = data.get("llm_insights", {})
    dataset_facts = summary.get("dataset_facts", [])
    null_report   = summary.get("null_handling_report", "")

    # ── Download Button & Link ──
    st.markdown('<div class="download-btn-container">', unsafe_allow_html=True)
    if st.session_state.cleaned_csv_data:
        st.download_button(
            label="📥 DOWNLOAD CLEANED DATA (CSV)",
            data=st.session_state.cleaned_csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            key="dl_top"
        )
    else:
        st.error("⚠️ Download data missing. Try running analysis again.")
        # Fallback direct link
        d_url = f"{BASE_URL}/download/{st.session_state.job_id}"
        st.markdown(f'<a href="{d_url}" target="_blank" style="color:#4f9cf9;font-weight:600;">🔗 Direct Download Link</a>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Interactive Tabs ──
    tab1, tab2, tab3 = st.tabs(["📊 RAW DATASET OVERVIEW", "📈 CHARTS GENERATED", "📄 EDA REPORT"])

    with tab1:
        # ══════════════════════════════════════════════════════════════════════════
        # SECTION A — Raw Dataset Overview
        # ══════════════════════════════════════════════════════════════════════════
        st.markdown('<p class="section-header">📊 Raw Dataset — Overview</p>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (lbl, val) in enumerate([
            ("Total Rows",           summary.get("total_rows", "—")),
            ("Total Columns",        summary.get("total_columns", "—")),
            ("Numeric Columns",      summary.get("numeric_column_count", "—")),
            ("Categorical Columns",  summary.get("categorical_column_count", "—")),
            ("Total Missing Values", summary.get("total_missing_values", "—")),
            ("Duplicates Found",     dup_removed),
        ]):
            with cols[i % 3]:
                st.markdown(
                    f'<div class="metric-card"><h3>{lbl}</h3><p>{val}</p></div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<p class="section-header">🕳️ Missing Values (Pre-Cleaning)</p>', unsafe_allow_html=True)
        if missing:
            mv_df = (
                pd.DataFrame(list(missing.items()), columns=["Column", "Missing Count"])
                .sort_values("Missing Count", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(mv_df, use_container_width=True)
        else:
            st.info("✅ No missing values detected in the original dataset.")

        # Cleaning Steps (Section B moves inside tab 1 too for context)
        st.markdown('<p class="section-header">🧹 Cleaning Pipeline Applied</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="phase-banner">
        ⚙️ <strong>Automatic Processing:</strong>&nbsp;
        Missing value imputation & Duplicate removal were performed to prepare the data.
        </div>""", unsafe_allow_html=True)

        # B1 — Fill logic
        if fill_logic:
            st.markdown("#### 🩹 Missing Value Imputation")
            show_semantic = st.toggle("Show semantic details (semantic type + strategy key)", value=False)
            for row in fill_logic:
                strat = row.get("strategy", "Mode")
                strat_key = row.get("strategy_key", "")
                sem_type = row.get("semantic_type", "")
                colour = "#4f9cf9" if strat == "Median" else "#a78bfa"
                meta = ""
                if show_semantic:
                    bits = []
                    if sem_type:
                        bits.append(f"semantic_type: <code>{sem_type}</code>")
                    if strat_key:
                        bits.append(f"strategy_key: <code>{strat_key}</code>")
                    if bits:
                        meta = (
                            "<div style='margin-top:.25rem;color:#94a3b8;font-size:.85rem;'>"
                            + " · ".join(bits)
                            + "</div>"
                        )
                st.markdown(
                    f'<div style="margin:.35rem 0;padding:.55rem 1rem;background:#131828;'
                    f'border-radius:8px;border-left:3px solid {colour};">'
                    f'<span class="logic-pill">{strat}</span>'
                    f'<strong style="color:#e2e8f0;">{row.get("column","")}</strong>'
                    f'<span style="color:white;">&nbsp;→ filled with</span> <code>{row["fill_value"]}</code>'
                    f'{meta}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        
        if dup_removed > 0:
            st.markdown("#### ♻️ Duplicate Removal")
            st.markdown(
                f'<div style="padding:.8rem 1.2rem;background:#131828;border-radius:10px;'
                f'border-left:3px solid #f472b6;margin:.5rem 0;">'
                f'<strong style="color:#f9a8d4;">{dup_removed} duplicates removed.</strong> Strategy: Keep First Occurrence.'
                f'</div>',
                unsafe_allow_html=True,
            )

    with tab2:
        # ══════════════════════════════════════════════════════════════════════════
        # SECTION B — Charts
        # ══════════════════════════════════════════════════════════════════════════
        st.markdown('<p class="section-header">📈 Charts — Generated on Cleaned Data</p>', unsafe_allow_html=True)
        chart_meta = {
            "histogram":           ("📊 Histogram — Distribution",     "Shows frequency distribution with KDE overlay."),
            "boxplot":             ("📦 Boxplot — Spread & Outliers",  "Visualises median, IQR, and outliers."),
            "correlation_heatmap": ("🌡️ Correlation Heatmap",         "Pearson correlations between numeric features."),
        }
        if charts:
            for path in charts:
                norm = os.path.normpath(path)
                key  = next((k for k in chart_meta if k in os.path.basename(norm)), None)
                lbl, caption = chart_meta.get(key, (os.path.basename(norm), ""))
                if os.path.exists(norm):
                    st.markdown(f"**{lbl}**")
                    st.caption(f"ℹ️ {caption}")
                    st.image(norm, use_container_width=True)
                    st.markdown("---")
        else:
            st.info("ℹ️ No charts generated — dataset contains no numeric columns.")

    with tab3:
        # ══════════════════════════════════════════════════════════════════════════
        # SECTION C — Unified EDA Report
        # ══════════════════════════════════════════════════════════════════════════
        st.markdown('<p class="section-header">📄 Automated AI Insights Report</p>', unsafe_allow_html=True)

        filled_total = sum(r["missing_before"] for r in fill_logic) if fill_logic else 0

        # ── Summary grid metrics ──
        st.markdown(
            f"""
            <div class="report-card">
              <div class="report-title">📊 Dataset Summary</div>
              <div class="report-subtitle">
                Cleaned dataset snapshot &nbsp;·&nbsp;
                {filled_total} cell(s) imputed &nbsp;·&nbsp;
                {dup_removed} duplicate(s) removed
              </div>

              <div class="summary-grid">
                <div class="sg-item">
                  <div class="sg-val">{summary.get('total_rows', '—')}</div>
                  <div class="sg-lbl">Total Rows</div>
                </div>
                <div class="sg-item">
                  <div class="sg-val">{summary.get('total_columns', '—')}</div>
                  <div class="sg-lbl">Total Columns</div>
                </div>
                <div class="sg-item">
                  <div class="sg-val">{summary.get('numeric_column_count', '—')}</div>
                  <div class="sg-lbl">Numeric Cols</div>
                </div>
                <div class="sg-item">
                  <div class="sg-val">{summary.get('categorical_column_count', '—')}</div>
                  <div class="sg-lbl">Categorical Cols</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if dataset_facts:
            st.markdown('<p class="report-sub" style="margin-top:1.4rem;">💡 Important Facts</p>', unsafe_allow_html=True)
            for fact in dataset_facts:
                st.markdown(f'<div class="fact-card">{fact}</div>', unsafe_allow_html=True)

        if null_report:
            with st.expander("🧾 Null Handling Report (semantic type → strategy)", expanded=False):
                st.code(null_report)

        if llm_insights:
            st.markdown('<p class="report-sub">🔍 Key Findings</p>', unsafe_allow_html=True)
            for item in llm_insights.get("key_findings", []):
                st.markdown(f'<div class="insight-card">{item}</div>', unsafe_allow_html=True)

            st.markdown('<p class="report-sub v">🧪 Data Quality Report</p>', unsafe_allow_html=True)
            dq_c1, dq_c2 = st.columns(2)
            with dq_c1:
                st.markdown("**Notes**")
                for item in llm_insights.get("data_quality_notes", []):
                    st.markdown(f'<div class="insight-card purple">{item}</div>', unsafe_allow_html=True)
            with dq_c2:
                st.markdown("**Anomalies**")
                for item in llm_insights.get("anomalies", []):
                    st.markdown(f'<div class="insight-card pink">{item}</div>', unsafe_allow_html=True)

            st.markdown('<p class="report-sub w">🎨 Visual Analysis (Chart-Based Insights)</p>', unsafe_allow_html=True)
            for item in llm_insights.get("visual_analysis_insights", []):
                st.markdown(f'<div class="insight-card pink">{item}</div>', unsafe_allow_html=True)

            st.markdown('<p class="report-sub g">🚀 Recommended Next Steps</p>', unsafe_allow_html=True)
            for idx, step in enumerate(llm_insights.get("recommended_next_steps", []), 1):
                st.markdown(f'<div class="insight-card green"><b>{idx}.</b> {step}</div>', unsafe_allow_html=True)
        else:
            st.info("⚠️ LLM insights unavailable. Make sure Ollama is running.")

        # Center-aligned download button at the very end of the report
        st.markdown("<br><hr>", unsafe_allow_html=True)
        if st.session_state.get("cleaned_csv_data"):
            col_left, col_mid, col_right = st.columns([1, 2, 1])
            with col_mid:
                st.download_button(
                    label="📥 DOWNLOAD FINAL CLEANED DATASET",
                    data=st.session_state.cleaned_csv_data,
                    file_name="cleaned_dataset.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_btn_bottom"
                )
        else:
            st.caption("Preparing download...")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION E — Conversational Chat
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<p class="section-header">💬 Ask Your Data — Chat with LLaMA3</p>', unsafe_allow_html=True)
    st.caption("Ask follow-up questions about your dataset. LLaMA3 answers using only the EDA context — no hallucination.")

    # Render existing conversation
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # New user input
    user_input = st.chat_input("e.g. Which column has the most variance? Any strong correlations?")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("LLaMA3 is thinking…"):
                try:
                    r = requests.post(
                        CHAT_URL,
                        json={
                            "query":           user_input,
                            "session_id":      st.session_state.session_id,
                            "history":         st.session_state.chat_history[-6:],
                            "dataset_context": st.session_state.llm_context,
                        },
                        timeout=300,
                    )
                    r.raise_for_status()
                    answer = r.json().get("answer", "No response.")
                except Exception as exc:
                    answer = f"⚠️ Error: {exc}"
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "user",      "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<br><center style='color:#3a4a5c;font-size:.8rem;'>"
    "EDA Agent v3 · FastAPI + LangGraph + LLaMA3 (Ollama)</center>",
    unsafe_allow_html=True,
)
