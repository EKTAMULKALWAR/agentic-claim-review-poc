"""Streamlit UI — Cotiviti Agentic Claim Review POC."""

from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from agents import (
    compute_outlier_flags,
    compute_peer_stats,
    get_available_models,
    ollama_running,
    run_orchestrator,
    stream_policy_agent,
    stream_stats_agent,
)
from load_cms_data import clean_claims, load_or_download
from pipeline import (
    TOP_FLAGGED,
    build_claim_query,
    detect_anomalies,
    engineer_features,
)
from vector_store import PolicyVectorStore

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cotiviti – Agentic Claim Review",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }

  .stage-pill {
    display: flex; align-items: center; gap: 14px;
    padding: 14px 20px;
    background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%);
    border-left: 4px solid #38bdf8;
    border-radius: 0 10px 10px 0;
    margin: 1.8rem 0 1rem;
  }
  .pill-num {
    background: #38bdf8; color: #0f172a;
    font-weight: 900; font-size: 1rem;
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
  }
  .pill-title { font-size: 1.15rem; font-weight: 700; color: #e2e8f0; }
  .pill-sub   { font-size: 0.82rem; color: #94a3b8; margin-top: 3px; }

  .agent-box {
    border: 1px solid #334155; border-radius: 10px;
    padding: 16px 20px; margin-bottom: 12px;
    background: #1e293b;
  }
  .agent-label {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #38bdf8; margin-bottom: 8px;
  }

  .verdict-audit {
    display: inline-block; padding: 12px 28px;
    background: #422006; border: 2px solid #fbbf24;
    border-radius: 8px; color: #fbbf24;
    font-size: 1.1rem; font-weight: 800; letter-spacing: 0.07em;
  }
  .verdict-clinical {
    display: inline-block; padding: 12px 28px;
    background: #064e3b; border: 2px solid #34d399;
    border-radius: 8px; color: #34d399;
    font-size: 1.1rem; font-weight: 800; letter-spacing: 0.07em;
  }

  .policy-card {
    background: #0f172a; border-left: 4px solid #38bdf8;
    border-radius: 0 10px 10px 0; padding: 16px 20px; margin: 10px 0;
  }
  .policy-id    { color: #38bdf8; font-weight: 700; font-size: 0.82rem;
                  text-transform: uppercase; letter-spacing: 0.08em; }
  .policy-title { color: #e2e8f0; font-size: 1rem; font-weight: 600; margin: 5px 0; }
  .policy-text  { color: #cbd5e1; font-size: 0.88rem; line-height: 1.7; }

  .reasoning-step {
    display: flex; gap: 14px;
    padding: 11px 16px; background: #0f172a;
    border-radius: 8px; margin-bottom: 8px;
    font-size: 0.9rem; color: #cbd5e1; line-height: 1.5;
  }
  .step-num { color: #38bdf8; font-weight: 700; flex-shrink: 0; min-width: 22px; }

  div[data-testid="metric-container"] {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; padding: 14px 18px;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached pipeline (Stages 1–2 are slow; cache for the session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="⏳ Loading & scoring CMS data — takes ~20 s on first run…")
def load_scored() -> pd.DataFrame:
    raw = load_or_download()
    df = clean_claims(raw)
    return detect_anomalies(engineer_features(df))


@st.cache_resource(show_spinner="📚 Building policy vector index…")
def get_vector_store() -> PolicyVectorStore:
    return PolicyVectorStore()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 Cotiviti")
    st.markdown("**Agentic Claim Review POC**")
    st.caption("Healthcare Payment Integrity · Internship Demo")
    st.divider()

    st.markdown("### Agent Architecture")
    st.markdown("""
```
Flagged Claim
      │
      ▼
┌─────────────────┐
│   Orchestrator  │
└────────┬────────┘
         │ delegates
    ┌────┴─────┐
    ▼          ▼
 Stats      Policy
 Agent      Agent
    │          │
    └────┬─────┘
         │ synthesizes
         ▼
  Determination
  → Human Auditor
```
""")
    st.divider()

    # Ollama model picker
    st.markdown("### Local LLM (Ollama)")
    _running = ollama_running()
    if _running:
        _models = get_available_models()
        if _models:
            selected_model = st.selectbox("Model", _models, index=0)
            st.success(f"Ollama running · {len(_models)} model(s) available", icon="✅")
        else:
            st.warning("Ollama running but no models found. Run:\n```\nollama pull llama3.2\n```")
            selected_model = None
    else:
        st.error("Ollama not detected. Start it with:\n```\nollama serve\n```")
        selected_model = None
    st.divider()

    st.markdown("### Tech Stack")
    for t in ["scikit-learn · Isolation Forest", "ChromaDB · vector store",
              "sentence-transformers · embeddings", "Ollama · local LLM",
              "Streamlit · UI", "Plotly · charts"]:
        st.markdown(f"• `{t}`")
    st.divider()

    st.markdown("### Data")
    st.markdown(
        "**CMS Medicare Physician & Other Practitioners** · 2024  \n"
        "data.cms.gov · 50 000-row sample  \n"
        "Public data only · No PHI"
    )
    st.divider()
    st.caption(f"Run at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# Agentic Claim Review")
st.markdown("#### Payment Integrity Proof-of-Concept — CMS Medicare Data")
st.warning(
    "**Human-in-the-loop guarantee:** This agent assists auditors and **never auto-denies** "
    "claims. Every determination routes to a certified human reviewer before any action.",
    icon="⚠️",
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="stage-pill">
  <div class="pill-num">1</div>
  <div>
    <div class="pill-title">Anomaly Detection</div>
    <div class="pill-sub">Isolation Forest · 200 estimators · 2% contamination · StandardScaler</div>
  </div>
</div>
""", unsafe_allow_html=True)

scored = load_scored()
n_total   = len(scored)
n_flagged = int(scored["is_anomaly"].sum())
flagged   = scored[scored["is_anomaly"]].head(TOP_FLAGGED).reset_index(drop=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows Analyzed",     f"{n_total:,}")
k2.metric("Anomalies Flagged", f"{n_flagged:,}")
k3.metric("Flag Rate",         f"{n_flagged / n_total:.1%}")
k4.metric("Top Anomaly Score", f"{scored['anomaly_score'].min():.4f}")

st.markdown("<br>", unsafe_allow_html=True)

# Charts
BG, CARD, FG = "#0f172a", "#1e293b", "#e2e8f0"
c_left, c_right = st.columns([3, 2], gap="large")

with c_left:
    normal_sample = scored[~scored["is_anomaly"]].sample(2000, random_state=42)
    plot_df = pd.concat([
        normal_sample.assign(Status="Normal"),
        scored[scored["is_anomaly"]].assign(Status="Anomaly"),
    ], ignore_index=True)
    fig = px.scatter(
        plot_df, x="services_per_bene", y="charge_to_payment_ratio",
        color="Status", color_discrete_map={"Normal": "#475569", "Anomaly": "#f97316"},
        hover_data={"Rndrng_NPI": True, "Rndrng_Prvdr_Type": True,
                    "HCPCS_Cd": True, "anomaly_score": ":.4f"},
        labels={"services_per_bene": "Services per Beneficiary",
                "charge_to_payment_ratio": "Charge-to-Payment Ratio"},
        title="Anomaly Landscape", log_y=True,
    )
    fig.update_traces(marker=dict(size=5, opacity=0.75))
    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=CARD, font_color=FG, title_font_size=13,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=46, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

with c_right:
    top_spec = (
        scored[scored["is_anomaly"]]
        .groupby("Rndrng_Prvdr_Type").size()
        .sort_values(ascending=True).tail(10)
        .reset_index(name="count")
    )
    fig2 = px.bar(
        top_spec, x="count", y="Rndrng_Prvdr_Type", orientation="h",
        title="Top 10 Specialties by Anomaly Count",
        color="count", color_continuous_scale="Blues",
        labels={"Rndrng_Prvdr_Type": "", "count": "Anomalies"},
    )
    fig2.update_layout(
        plot_bgcolor=BG, paper_bgcolor=CARD, font_color=FG,
        title_font_size=13, coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=46, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

# Flagged claims table
st.markdown(f"**Top {TOP_FLAGGED} flagged claims** — click a row to select it for multi-agent analysis")

COLS = {
    "Rndrng_NPI": "NPI", "Rndrng_Prvdr_Type": "Specialty",
    "HCPCS_Cd": "HCPCS", "HCPCS_Desc": "Description",
    "Tot_Srvcs": "Services", "Tot_Benes": "Patients",
    "services_per_bene": "Svc/Patient",
    "Avg_Sbmtd_Chrg": "Avg Charge ($)", "Avg_Mdcr_Pymt_Amt": "Avg Payment ($)",
    "charge_to_payment_ratio": "Chg/Pay", "anomaly_score": "Score",
}
disp = flagged[list(COLS)].rename(columns=COLS).copy()
disp["Avg Charge ($)"]  = disp["Avg Charge ($)"].map("${:,.2f}".format)
disp["Avg Payment ($)"] = disp["Avg Payment ($)"].map("${:,.2f}".format)
disp["Svc/Patient"]     = disp["Svc/Patient"].map("{:.2f}".format)
disp["Chg/Pay"]         = disp["Chg/Pay"].map("{:.2f}".format)
disp["Score"]           = disp["Score"].map("{:.4f}".format)

selection = st.dataframe(
    disp, use_container_width=True, hide_index=False,
    on_select="rerun", selection_mode="single-row",
    column_config={
        "NPI":         st.column_config.TextColumn(width="small"),
        "HCPCS":       st.column_config.TextColumn(width="small"),
        "Score":       st.column_config.TextColumn(width="small"),
        "Description": st.column_config.TextColumn(width="large"),
    },
)

# default to the oxaliplatin claim (J9263) - cleanest example to walk through
_j9263 = flagged[flagged["HCPCS_Cd"] == "J9263"]
_default_idx = int(_j9263.index[0]) if not _j9263.empty else 0

sel = selection.selection.rows
idx = sel[0] if sel else _default_idx
row = flagged.iloc[idx]

st.info(
    f"**Selected:** NPI `{row['Rndrng_NPI']}` · {row['Rndrng_Prvdr_Type']} · "
    f"HCPCS `{row['HCPCS_Cd']}` — *{row['HCPCS_Desc']}*"
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Policy Retrieval
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="stage-pill">
  <div class="pill-num">2</div>
  <div>
    <div class="pill-title">Policy Retrieval</div>
    <div class="pill-sub">ChromaDB · sentence-transformers (all-MiniLM-L6-v2) · 15 CMS 2026 policies</div>
  </div>
</div>
""", unsafe_allow_html=True)

store   = get_vector_store()
query   = build_claim_query(row)
results = store.retrieve(query, n_results=3)   # top-3 by semantic similarity
policy  = results[0]                            # primary policy for agents

# Top match — full card
p_left, p_right = st.columns([3, 1], gap="large")
with p_left:
    st.markdown(f"""
<div class="policy-card">
  <div class="policy-id">{policy['id']} &nbsp;·&nbsp; {policy.get('effective','')}</div>
  <div class="policy-title">{policy['title']}</div>
  <div class="policy-text">{policy['text']}</div>
</div>""", unsafe_allow_html=True)

with p_right:
    st.metric("Semantic Similarity", f"{policy['similarity_score']:.3f}")
    st.progress(min(policy['similarity_score'] * 2, 1.0))
    st.caption(f"Model: `{store.model_name}`")
    st.caption(f"Index: {store.policy_count} policies")
    st.caption("Local embeddings — would swap to OpenAI + Pinecone for a real deployment")

# Runner-up policies
if len(results) > 1:
    with st.expander("Other relevant policies (ranked by similarity)"):
        for r in results[1:]:
            st.markdown(
                f"**{r['id']}** — {r['title']} &nbsp; `similarity {r['similarity_score']:.3f}`  \n"
                f"<small>{r['text'][:200]}…</small>",
                unsafe_allow_html=True,
            )
            st.markdown("---")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Multi-Agent Determination
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="stage-pill">
  <div class="pill-num">3</div>
  <div>
    <div class="pill-title">Multi-Agent Determination</div>
    <div class="pill-sub">Stats Agent · Policy Agent · Orchestrator → Human Auditor</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Peer stats table — deterministic, shown immediately
peer_stats = compute_peer_stats(row["Rndrng_Prvdr_Type"], row["HCPCS_Cd"], scored)

if not peer_stats.get("insufficient"):
    ps = peer_stats
    outlier_flags = compute_outlier_flags(row, ps)
    f = outlier_flags["flags"]

    metrics      = ["Services", "Services/bene", "Avg charge ($)", "Avg payment ($)", "Chg/pay ratio"]
    flag_keys    = ["Services", "Services/beneficiary", "Avg submitted charge", None, "Charge/payment ratio"]
    provider_row = [
        int(row["Tot_Srvcs"]),
        round(row["services_per_bene"], 2),
        round(row["Avg_Sbmtd_Chrg"], 2),
        round(row["Avg_Mdcr_Pymt_Amt"], 2),
        round(row["charge_to_payment_ratio"], 2),
    ]
    peer_med_row = [ps["median_services"], ps["median_svc_bene"], ps["median_charge"], ps["median_payment"], ps["median_ratio"]]
    peer_p95_row = [ps["p95_services"],    ps["p95_svc_bene"],    ps["p95_charge"],    ps["p95_payment"],    ps["p95_ratio"]]
    status_row   = [
        ("EXCEEDS P95" if f.get(k) else "within range") if k else "—"
        for k in flag_keys
    ]

    peer_df = pd.DataFrame({
        "Metric":          metrics,
        "Peer Median":     peer_med_row,
        "Peer P95":        peer_p95_row,
        "This Provider":   provider_row,
        "Status":          status_row,
    })

    with st.expander(f"📊 Peer Benchmark Comparison ({ps['scope']})", expanded=True):
        st.dataframe(peer_df, use_container_width=True, hide_index=True)

else:
    # not enough peers in this 50k sample to do percentiles, so just show raw numbers
    spb = row["services_per_bene"]
    abs_df = pd.DataFrame({
        "Metric":         ["Services total", "Beneficiaries", "Services / beneficiary", "Avg submitted charge ($)", "Avg Medicare payment ($)", "Charge/payment ratio"],
        "This Provider":  [int(row["Tot_Srvcs"]), int(row["Tot_Benes"]), round(spb, 2),
                           round(row["Avg_Sbmtd_Chrg"], 2), round(row["Avg_Mdcr_Pymt_Amt"], 2),
                           round(row["charge_to_payment_ratio"], 2)],
        "Clinical Flag":  ["—", "—",
                           "⚠ Clinically extraordinary" if spb > 10 else "within normal range",
                           "—", "—",
                           "⚠ Elevated" if row["charge_to_payment_ratio"] > 3.0 else "within range"],
    })
    with st.expander(
        f"📊 Absolute Claim Metrics — {peer_stats.get('scope', 'insufficient peer data')} "
        f"(peer percentile unavailable in 50K sample)",
        expanded=True,
    ):
        st.dataframe(abs_df, use_container_width=True, hide_index=True)

# Run button
st.markdown("<br>", unsafe_allow_html=True)

cache_key = f"result_{row['Rndrng_NPI']}_{row['HCPCS_Cd']}"

if selected_model is None:
    st.error(
        "No Ollama model available. Install Ollama from **ollama.com**, then run:\n"
        "```bash\nollama pull llama3.2\nollama serve\n```"
    )
else:
    run_col, _ = st.columns([2, 5])
    if run_col.button(
        "🤖 Run Multi-Agent Analysis",
        type="primary",
        use_container_width=True,
        disabled=(selected_model is None),
    ):
        # Clear any previous result for this claim
        if cache_key in st.session_state:
            del st.session_state[cache_key]

        stats_text, policy_text, determination = "", "", {}

        # ── Stats Agent ──────────────────────────────────────────────────────
        with st.status("🔍 Stats Agent — analyzing peer benchmarks...", expanded=True) as s_status:
            st.markdown('<div class="agent-label">Stats Agent</div>', unsafe_allow_html=True)
            stats_text = st.write_stream(
                stream_stats_agent(row, peer_stats, selected_model)
            )
            s_status.update(label="Stats Agent ✓", state="complete", expanded=False)

        # ── Policy Agent ─────────────────────────────────────────────────────
        with st.status("📋 Policy Agent — reviewing policy compliance...", expanded=True) as p_status:
            st.markdown('<div class="agent-label">Policy Agent</div>', unsafe_allow_html=True)
            policy_text = st.write_stream(
                stream_policy_agent(row, results, stats_text, selected_model)
            )
            p_status.update(label="Policy Agent ✓", state="complete", expanded=False)

        # ── Orchestrator ─────────────────────────────────────────────────────
        with st.spinner("🧠 Orchestrator synthesizing findings..."):
            determination = run_orchestrator(
                row, results, peer_stats, stats_text, policy_text, selected_model
            )

        st.session_state[cache_key] = {
            "stats_text":    stats_text,
            "policy_text":   policy_text,
            "determination": determination,
        }
        st.rerun()

    # ── Show cached determination ─────────────────────────────────────────────
    if cache_key in st.session_state:
        cached = st.session_state[cache_key]
        det    = cached["determination"]
        verdict = det.get("verdict", "FLAG_FOR_AUDIT")

        st.markdown("---")
        st.markdown("#### Agent Findings")
        exp_l, exp_r = st.columns(2, gap="large")

        with exp_l:
            with st.expander("Stats Agent findings", expanded=False):
                st.markdown(
                    f'<div class="agent-box"><div class="agent-label">Stats Agent</div>'
                    f'{cached["stats_text"]}</div>',
                    unsafe_allow_html=True,
                )

        with exp_r:
            with st.expander("Policy Agent findings", expanded=False):
                st.markdown(
                    f'<div class="agent-box"><div class="agent-label">Policy Agent</div>'
                    f'{cached["policy_text"]}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("#### Orchestrator — Final Determination")

        if verdict == "FLAG_FOR_AUDIT":
            st.markdown(
                '<div class="verdict-audit">🚩 &nbsp;FLAG FOR AUDIT</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="verdict-clinical">🔬 &nbsp;ROUTE TO CLINICAL REVIEW</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        d1, d2, d3 = st.columns(3)
        d1.metric("Confidence",  f"{det.get('confidence', 0):.0%}")
        d2.metric("Routes To",   "Human Auditor")
        d3.metric("Source",      "Ollama · local LLM")

        st.markdown(f"> {det.get('verdict_reason', '')}")
        st.caption(f"Policy citation: **{det.get('policy_citation', '')}**")

        st.markdown("**Chain-of-Thought Reasoning**")
        for i, step in enumerate(det.get("reasoning", []), 1):
            st.markdown(
                f'<div class="reasoning-step">'
                f'<span class="step-num">{i}.</span><span>{step}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.caption(f"Source: {det.get('source', '')}")
    else:
        st.info(
            "Click **Run Multi-Agent Analysis** above to activate the Stats Agent, "
            "Policy Agent, and Orchestrator for the selected claim.",
            icon="👆",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Data: CMS Medicare Physician & Other Practitioners – by Provider and Service · "
    "data.cms.gov · 2024 · 50 000-row sample · Public data · No PHI · Demonstration only."
)
