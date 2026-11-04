# Solari Analytics — Support Ticket Similarity Engine

A fast, deterministic similarity matching engine built for Solari Analytics support agents. The system indexes historical support tickets using TF-IDF term weighting and Cosine Similarity, surfacing relevant past resolutions in **<5ms** with zero hallucination risk or LLM API overhead.

Includes a core Python processing engine, unit test coverage, a FastAPI REST server, and a modern agent workspace UI.

---

## System Architecture

```text
Incoming Support Ticket (Subject + Description)
                      │
                      ▼
        ┌───────────────────────────┐
        │  FastAPI REST Server      │
        │  POST /api/match          │
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │  TicketSimilarityEngine   │
        │  • TF-IDF Vectorizer      │
        │  • Sublinear TF Scaling   │
        │  • Unigrams + Bigrams     │
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │  Cosine Similarity Matrix │
        │  Threshold Cutoff: ≥ 0.30 │
        └─────────────┬─────────────┘
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
[ Valid Matches Found ]   [ Score < 0.30 Cutoff ]
Top 3 Ranked Resolutions  "Novel Issue" Fallback Alert
```

---

## Project Structure

```text
solari-similarity-engine/
├── app.py               # Core TF-IDF & similarity engine logic
├── server.py            # FastAPI REST server exposing /api/match
├── index.html           # Production light-theme agent portal UI
├── test_app.py          # Pytest unit tests for threshold & matching logic
├── SOLUTION_DESIGN.md   # Architectural choices, trade-offs, and V2 roadmap
├── README.md            # Project documentation
└── starter_kit/
    └── data/
        ├── tickets.json             # Seed historical resolution corpus
        └── new_tickets_example.json # Evaluation query suite
```

---

## Quickstart Guide

### 1. Requirements & Setup
Requires **Python 3.9+**. Install dependencies:

```bash
pip install scikit-learn fastapi uvicorn pytest
```

---

### 2. Run the CLI Pipeline
Verify core vectorization and threshold evaluation directly in the terminal:

```bash
python app.py
```

---

### 3. Run the Full-Stack Agent Portal

**Step 1: Start the FastAPI Backend**
```bash
python -m uvicorn server:app --reload --port 8000
```
*The server trains the TF-IDF matrix on boot and logs confirmation at `http://127.0.0.1:8000`.*

**Step 2: Launch the Frontend UI**
* **Direct:** Double-click `index.html` in your file explorer to open it in any browser.
* **Via HTTP Server:** Open a second terminal and run:
  ```bash
  python -m http.server 3000
  ```
  Navigate to `http://localhost:3000` in your browser.

---

## Key Design Decisions

* **Why TF-IDF over GenAI/RAG:** Support agents require exact, verified historical resolution steps. Routing queries through an LLM introduces 1–3s latency, recurring token costs, and hallucination risks. TF-IDF delivers sub-5ms vector retrieval with 100% deterministic outputs.
* **Sublinear TF Scaling (`1 + log(tf)`):** Prevents repetitive non-technical words in longer ticket descriptions from dominating similarity scores.
* **$0.20$ Confidence Cutoff:** If an incoming ticket doesn't reach a 20% similarity score against past records, the engine explicitly alerts the agent that the issue is novel rather than surfacing misleading fixes.

---

## Evaluation Test Queries

Use the quick-preset buttons in the UI or test these raw inputs:

1. **NT-1 (Password Issue):** Matches `T-1001` with high confidence (~50%+ match).
2. **NT-2 (Billing Issue):** Matches `T-1303` duplicate charge resolution notes.
3. **NT-3 (SSO Issue):** Matches `T-1202` Okta redirect callback solution.
4. **NT-4 (Snowflake Sync Timeout):** Drops below the $0.30$ cutoff, triggering the amber **"Novel Issue — Manual Investigation Required"** alert.