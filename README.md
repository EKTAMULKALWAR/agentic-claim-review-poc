# Agentic Claim Review — Payment Integrity POC

A proof-of-concept multi-agent system that flags anomalous Medicare claims, retrieves the relevant payer policy, and produces a cited, human-routed determination using a local LLM.

Built for the Cotiviti internship assignment using the real CMS Medicare Physician & Other Practitioners dataset.

---

## What it does

### Stage 1 — Anomaly Detection
Loads 50 000 rows from the CMS Medicare dataset and engineers **7 features**: the four raw metrics (avg submitted charge, avg Medicare payment, services per beneficiary, charge-to-payment ratio) plus **three peer-group z-scores** that measure how far each provider deviates from other providers billing the *same procedure code in the same specialty*. A scikit-learn **Isolation Forest** (200 estimators, 2% contamination) is trained on all 7 features so it flags providers who are outliers within their own peer cohort — not just globally noisy rows.

### Stage 2 — Policy Retrieval
Encodes each flagged claim into a dense vector and performs **semantic similarity search** over a 15-policy knowledge base grounded in real 2026 CMS guidelines. Uses **ChromaDB** (persistent, embedded) as the vector store and **`all-MiniLM-L6-v2`** (sentence-transformers) as the local embedding model. Returns the top-3 ranked policies with cosine similarity scores; the highest-ranked policy is passed to the agents.
> Production equivalent: Pinecone / pgvector + OpenAI or Cohere embeddings.

### Stage 3 — Multi-Agent Determination
Three coordinated agents produce a final determination:

```
Flagged Claim
      │
      ▼
┌─────────────────┐
│   Orchestrator  │
└────────┬────────┘
         │ delegates to
    ┌────┴─────┐
    ▼          ▼
 Stats      Policy
 Agent      Agent
    │          │
    └────┬─────┘
         │ synthesizes into
         ▼
  Determination JSON
  → Human Auditor
```

- **Stats Agent** — compares the provider's metrics against peer benchmarks (same specialty + HCPCS) and narrates the statistical outliers.
- **Policy Agent** — retrieves the relevant payer policy and identifies specific compliance concerns.
- **Orchestrator** — synthesizes both findings into a structured JSON verdict: `FLAG_FOR_AUDIT` or `ROUTE_TO_CLINICAL_REVIEW`. **Never auto-denies.**

All reasoning is streamed live in the UI so you can watch each agent work in real time.

---

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd "Cotiviti Internship Assignment"
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r code/requirements.txt
```

### 4. Install Ollama and pull a model

Ollama runs the LLM locally — no API key or internet connection needed for inference.

```bash
# Install Ollama from https://ollama.com
# Then pull a model (llama3.2 is fast; llama3.1:8b gives better reasoning):
ollama pull llama3.2
```

Start the Ollama server (it may already be running as a background service after install):

```bash
ollama serve
```

### 5. Run the app

```bash
streamlit run code/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Data

**CMS Medicare Physician & Other Practitioners – by Provider and Service (2024)**  
Source: [data.cms.gov](https://data.cms.gov)  
A 50 000-row sample is cached at `data/cms_sample_50k.csv` on first run. Public data only — no PHI.

Key columns used: `Rndrng_NPI`, `Rndrng_Prvdr_Type`, `HCPCS_Cd`, `Tot_Srvcs`, `Tot_Benes`, `Avg_Sbmtd_Chrg`, `Avg_Mdcr_Pymt_Amt`.

---

## Project structure

```
code/
  app.py               ← Streamlit UI (entry point)
  agents.py            ← Stats Agent, Policy Agent, Orchestrator
  pipeline.py          ← Isolation Forest + peer z-scores, query builder
  vector_store.py      ← ChromaDB + sentence-transformers policy index
  load_cms_data.py     ← CMS data fetching and caching
  policies.py          ← 15-policy knowledge base (2026 CMS guidelines)
  requirements.txt     ← Python dependencies
  data/
    cms_sample_50k.csv ← cached CMS sample (auto-downloaded if missing, gitignored)
  chroma_db/           ← persisted vector index (auto-built on first run, gitignored)
pdf/
  Cotiviti_Agentic_AI_Report.pdf ← written report
ppt/
  Cotiviti_Presentation.pptx     ← presentation slides
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas` / `numpy` | Data loading and feature engineering |
| `scikit-learn` | Isolation Forest, StandardScaler |
| `chromadb` | Persistent embedded vector database |
| `sentence-transformers` | Local embedding model (`all-MiniLM-L6-v2`) |
| `ollama` | Local LLM inference via Ollama |
| `streamlit` | Interactive web UI |
| `plotly` | Anomaly scatter chart and specialty bar chart |

---

## Human-in-the-loop guarantee

This agent **never auto-denies** a claim. Every determination is one of:

- `FLAG_FOR_AUDIT` — routes to a payment-integrity auditor
- `ROUTE_TO_CLINICAL_REVIEW` — routes to a clinical reviewer

The agent assists; a certified human makes the final decision.
