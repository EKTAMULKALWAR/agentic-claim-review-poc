"""
Stats Agent + Policy Agent + Orchestrator, all running on local Ollama.
The orchestrator calls the other two and merges their output into one
JSON verdict that gets routed to a human auditor.
"""

import json
import re
from typing import Generator

import ollama
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_available_models() -> list[str]:
    """Return models available in the local Ollama instance."""
    try:
        result = ollama.list()
        return [m["model"] for m in result.get("models", [])]
    except Exception:
        return []


def ollama_running() -> bool:
    try:
        ollama.list()
        return True
    except Exception:
        return False


def compute_peer_stats(specialty: str, hcpcs: str, df: pd.DataFrame) -> dict:
    """Return peer benchmark statistics for the same specialty + HCPCS code."""
    peers = df[(df["Rndrng_Prvdr_Type"] == specialty) & (df["HCPCS_Cd"] == hcpcs)]
    scope = f"same specialty + HCPCS ({len(peers)} providers)"

    if len(peers) < 3:
        peers = df[df["HCPCS_Cd"] == hcpcs]
        scope = f"same HCPCS cross-specialty ({len(peers)} providers)"

    if len(peers) < 3:
        return {"n_peers": len(peers), "scope": scope, "insufficient": True}

    return {
        "n_peers":            len(peers),
        "scope":              scope,
        "insufficient":       False,
        "median_services":    round(float(peers["Tot_Srvcs"].median()), 1),
        "p95_services":       round(float(peers["Tot_Srvcs"].quantile(0.95)), 1),
        "median_svc_bene":    round(float(peers["services_per_bene"].median()), 2),
        "p95_svc_bene":       round(float(peers["services_per_bene"].quantile(0.95)), 2),
        "median_charge":      round(float(peers["Avg_Sbmtd_Chrg"].median()), 2),
        "p95_charge":         round(float(peers["Avg_Sbmtd_Chrg"].quantile(0.95)), 2),
        "median_payment":     round(float(peers["Avg_Mdcr_Pymt_Amt"].median()), 2),
        "p95_payment":        round(float(peers["Avg_Mdcr_Pymt_Amt"].quantile(0.95)), 2),
        "median_ratio":       round(float(peers["charge_to_payment_ratio"].median()), 2),
        "p95_ratio":          round(float(peers["charge_to_payment_ratio"].quantile(0.95)), 2),
    }


def compute_outlier_flags(row: pd.Series, ps: dict) -> dict:
    """Check which metrics actually clear the peer p95 — doing this in
    plain python because llama3.2 kept getting the comparisons wrong."""
    if ps.get("insufficient"):
        return {"insufficient": True, "any_outlier": False, "n_outliers": 0, "flags": {}}

    flags = {
        "Services":             bool(row["Tot_Srvcs"] > ps["p95_services"]),
        "Services/beneficiary": bool(row["services_per_bene"] > ps["p95_svc_bene"]),
        "Avg submitted charge": bool(row["Avg_Sbmtd_Chrg"] > ps["p95_charge"]),
        "Charge/payment ratio": bool(row["charge_to_payment_ratio"] > ps["p95_ratio"]),
    }
    return {
        "insufficient": False,
        "any_outlier":  any(flags.values()),
        "n_outliers":   sum(flags.values()),
        "flags":        flags,
    }


def _chat_stream(model: str, system: str, user: str) -> Generator[str, None, None]:
    for chunk in ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        stream=True,
    ):
        yield chunk["message"]["content"]


def _chat(model: str, system: str, user: str, as_json: bool = False) -> str:
    kwargs = {"format": "json"} if as_json else {}
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        **kwargs,
    )
    return response["message"]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# Stats Agent
# ─────────────────────────────────────────────────────────────────────────────

_STATS_SYSTEM = """You are a Medicare data analyst specializing in payment-integrity audits.
Compare a flagged provider's billing metrics against peer benchmarks.
The table you receive already includes a code-computed "Status" column (EXCEEDS P95 or within range) —
treat that column as ground truth and never assert your own threshold comparison that contradicts it.
Be concise, quantitative, and direct. Highlight the most significant outliers and by what factor.
Keep your response to 3-5 sentences."""


def _build_peer_table_text(row: pd.Series, ps: dict) -> str:
    if ps.get("insufficient"):
        # not enough peers to do percentiles, so just dump the raw numbers
        ratio = row["charge_to_payment_ratio"]
        spb   = row["services_per_bene"]
        return (
            f"Peer group: {ps.get('scope', 'N/A')} — too few providers for percentile benchmarking.\n\n"
            f"Absolute claim metrics (evaluate against clinical norms):\n"
            f"  Services total:       {int(row['Tot_Srvcs'])}\n"
            f"  Beneficiaries:        {int(row['Tot_Benes'])}\n"
            f"  Services/beneficiary: {spb:.2f}"
            + (f"  ← EXTREME if >10 for this code; {spb:.0f} is clinically extraordinary\n" if spb > 10 else "\n")
            + f"  Avg submitted charge: ${row['Avg_Sbmtd_Chrg']:.2f}\n"
            f"  Avg Medicare payment: ${row['Avg_Mdcr_Pymt_Amt']:.2f}\n"
            f"  Charge/payment ratio: {ratio:.2f}"
            + (f"  ← elevated (>3.0)\n" if ratio > 3.0 else "\n")
            + f"\nNote: services_per_bene of {spb:.0f} means each beneficiary received that many "
            f"units on average — state this plainly in your analysis."
        )

    outliers = compute_outlier_flags(row, ps)
    f = outliers["flags"]
    status = lambda name: "EXCEEDS P95" if f[name] else "within range"

    z_svc  = f"  (z={row['z_svc_bene_peer']:+.1f})" if "z_svc_bene_peer"  in row.index else ""
    z_chg  = f"  (z={row['z_charge_peer']:+.1f})"   if "z_charge_peer"    in row.index else ""
    z_rat  = f"  (z={row['z_ratio_peer']:+.1f})"    if "z_ratio_peer"     in row.index else ""

    return (
        f"Peer group: {ps['scope']}\n"
        f"Metric                 | Peer Median | Peer P95    | This Provider  | Status (code-computed)\n"
        f"Services               | {ps['median_services']:<11} | {ps['p95_services']:<11} | {row['Tot_Srvcs']:<14} | {status('Services')}\n"
        f"Services/beneficiary   | {ps['median_svc_bene']:<11} | {ps['p95_svc_bene']:<11} | {row['services_per_bene']:<14.2f} | {status('Services/beneficiary')}{z_svc}\n"
        f"Avg submitted charge   | ${ps['median_charge']:<10.2f} | ${ps['p95_charge']:<10.2f} | ${row['Avg_Sbmtd_Chrg']:<13.2f} | {status('Avg submitted charge')}{z_chg}\n"
        f"Avg Medicare payment   | ${ps['median_payment']:<10.2f} | ${ps['p95_payment']:<10.2f} | ${row['Avg_Mdcr_Pymt_Amt']:<13.2f} | —\n"
        f"Charge/payment ratio   | {ps['median_ratio']:<11} | {ps['p95_ratio']:<11} | {row['charge_to_payment_ratio']:<14.2f} | {status('Charge/payment ratio')}{z_rat}\n\n"
        f"{outliers['n_outliers']} of 4 metrics exceed the peer 95th percentile (code-computed — do not recompute)."
    )


def stream_stats_agent(
    row: pd.Series, peer_stats: dict, model: str
) -> Generator[str, None, None]:
    """Yield token chunks — Stats Agent analyzing peer benchmarks."""
    user = (
        f"FLAGGED CLAIM:\n"
        f"NPI {row['Rndrng_NPI']} | {row['Rndrng_Prvdr_Type']} | "
        f"HCPCS {row['HCPCS_Cd']} — {row['HCPCS_Desc']}\n\n"
        f"PEER BENCHMARK COMPARISON:\n"
        f"{_build_peer_table_text(row, peer_stats)}\n\n"
        f"Which metrics are anomalous, by how much, and what does that suggest?"
    )
    yield from _chat_stream(model, _STATS_SYSTEM, user)


# ─────────────────────────────────────────────────────────────────────────────
# Policy Agent
# ─────────────────────────────────────────────────────────────────────────────

_POLICY_SYSTEM = """You are a Medicare billing compliance expert.
Given a flagged claim, statistical peer context, and a retrieved payer policy,
identify specific compliance concerns grounded in the policy text.
Cite policy thresholds or criteria explicitly. Keep your response to 3-5 sentences."""


def stream_policy_agent(
    row: pd.Series, policies: list[dict], stats_summary: str, model: str
) -> Generator[str, None, None]:
    """Yield token chunks — Policy Agent interpreting claim against retrieved policies."""
    policy_block = "\n\n".join(
        f"[{p['id']} — {p['title']} | similarity {p['similarity_score']:.3f}]\n{p['text']}"
        for p in policies
    )
    user = (
        f"CLAIM: {row['Rndrng_Prvdr_Type']} | HCPCS {row['HCPCS_Cd']} — {row['HCPCS_Desc']}\n"
        f"Services/bene: {row['services_per_bene']:.2f} | "
        f"Charge/payment ratio: {row['charge_to_payment_ratio']:.2f}\n\n"
        f"STATISTICAL CONTEXT (from Stats Agent):\n{stats_summary}\n\n"
        f"RETRIEVED POLICIES (semantic search, ranked by relevance):\n{policy_block}\n\n"
        f"Which policy is most applicable? Does this claim raise compliance concerns? "
        f"Cite specific thresholds from the policy text. 3-5 sentences."
    )
    yield from _chat_stream(model, _POLICY_SYSTEM, user)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

_ORCH_SYSTEM = """You are a senior Medicare payment-integrity auditor.
Synthesize findings from a Stats Agent and a Policy Agent into a final structured claim determination.
STRICT RULES:
- Never auto-deny. Verdicts must be exactly FLAG_FOR_AUDIT or ROUTE_TO_CLINICAL_REVIEW.
- FLAG_FOR_AUDIT: clear billing anomaly suggesting possible fraud or abuse.
- ROUTE_TO_CLINICAL_REVIEW: financial outlier that may have legitimate clinical explanation.
- Output ONLY valid JSON. No extra text, no markdown fences.
- Output EXACTLY 3 to 5 reasoning steps — no more. Each step must state a distinct, specific finding (a number, a policy threshold, or a conclusion). Do NOT repeat or restate previous steps. Do NOT describe the audit process itself (no "an audit will be conducted" filler). If you have made your case in 3 steps, stop at 3."""


def run_orchestrator(
    row: pd.Series,
    policies: list[dict],
    peer_stats: dict,
    stats_summary: str,
    policy_summary: str,
    model: str,
) -> dict:
    """Merge the stats + policy findings into one final verdict.

    llama3.2 would sometimes cite a policy other than the one we actually
    showed in stage 2, or flag something as fraud when nothing crossed a
    peer threshold. Patched both with checks below instead of trusting
    the model's JSON as-is.
    """
    top_policy = policies[0]
    outliers = compute_outlier_flags(row, peer_stats)
    policy_refs = "; ".join(f"{p['id']}: {p['title']}" for p in policies)

    if outliers["insufficient"]:
        outlier_line = (
            f"Peer percentile benchmarks unavailable (too few peers in this dataset sample). "
            f"Evaluate on absolute values: services_per_bene={row['services_per_bene']:.2f}, "
            f"charge/payment ratio={row['charge_to_payment_ratio']:.2f}."
        )
    else:
        outlier_line = (
            f"{outliers['n_outliers']} of 4 metrics exceed the peer 95th percentile "
            f"(code-computed — do not contradict)."
        )

    user = (
        f"CLAIM: NPI {row['Rndrng_NPI']} | {row['Rndrng_Prvdr_Type']} | HCPCS {row['HCPCS_Cd']}\n"
        f"Services: {int(row['Tot_Srvcs'])} | Benes: {int(row['Tot_Benes'])} | "
        f"Svc/Bene: {row['services_per_bene']:.2f}\n"
        f"Avg Charge: ${row['Avg_Sbmtd_Chrg']:.2f} | Avg Payment: ${row['Avg_Mdcr_Pymt_Amt']:.2f} | "
        f"Ratio: {row['charge_to_payment_ratio']:.2f}\n\n"
        f"CODE-COMPUTED OUTLIER CHECK (ground truth — do not recompute or contradict):\n"
        f"{outlier_line}\n\n"
        f"STATS AGENT FINDINGS:\n{stats_summary}\n\n"
        f"POLICY AGENT FINDINGS:\n{policy_summary}\n\n"
        f"POLICY REFERENCES (cite ONLY the top one — {top_policy['id']}): {policy_refs}\n\n"
        f"Respond with this exact JSON structure:\n"
        f'{{"reasoning":["step 1","step 2","step 3","step 4"],'
        f'"verdict":"FLAG_FOR_AUDIT",'
        f'"verdict_reason":"one sentence for the human auditor",'
        f'"confidence":0.85,'
        f'"policy_citation":"{top_policy["id"]}: {top_policy["title"]}"}}'
    )

    raw = _chat(model, _ORCH_SYSTEM, user, as_json=True).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(m.group()) if m else _fallback_determination(row, policies)

    # cap it at 5 steps no matter what the model gave back
    result["reasoning"] = result.get("reasoning", [])[:5]

    if result.get("verdict") not in ("FLAG_FOR_AUDIT", "ROUTE_TO_CLINICAL_REVIEW"):
        result["verdict"] = "FLAG_FOR_AUDIT"

    # always cite the policy we actually displayed in stage 2, not whatever the model picked
    result["policy_citation"] = f"{top_policy['id']}: {top_policy['title']}"

    # don't let it flag for audit if nothing actually crossed the peer p95
    if not outliers["insufficient"] and not outliers["any_outlier"]:
        result["verdict"] = "ROUTE_TO_CLINICAL_REVIEW"
        result["confidence"] = min(float(result.get("confidence", 0.5)), 0.55)
        result.setdefault("reasoning", []).append(
            "No metric exceeds the peer 95th percentile here, so this is "
            "downgraded to clinical review instead of a clear anomaly flag."
        )

    result["source"] = f"multi-agent · Ollama / {model}"
    return result


def _fallback_determination(row: pd.Series, policies: list[dict]) -> dict:
    top = policies[0]
    verdict = "FLAG_FOR_AUDIT" if (row["services_per_bene"] > 2 or row["charge_to_payment_ratio"] > 5) else "ROUTE_TO_CLINICAL_REVIEW"
    return {
        "reasoning": [
            f"Provider {row['Rndrng_NPI']} ({row['Rndrng_Prvdr_Type']}) billed HCPCS {row['HCPCS_Cd']}.",
            f"Services per beneficiary: {row['services_per_bene']:.2f} — significantly above norms.",
            f"Charge-to-payment ratio: {row['charge_to_payment_ratio']:.2f} — statistical outlier.",
            f"Pattern aligns with {top['id']}: {top['title']}.",
        ],
        "verdict": verdict,
        "verdict_reason": "Anomalous billing pattern detected; routing to human auditor for review.",
        "confidence": 0.72,
        "policy_citation": f"{top['id']}: {top['title']}",
    }
