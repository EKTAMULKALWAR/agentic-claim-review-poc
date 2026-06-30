"""
Stage 1/2 helpers shared by the Streamlit app: feature engineering,
Isolation Forest scoring, and the text query builder for policy retrieval.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

TOP_FLAGGED = 15


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def _peer_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    """Z-score of col within each (specialty, HCPCS) peer group, clipped to [-10, 10]."""
    grp  = df.groupby(["Rndrng_Prvdr_Type", "HCPCS_Cd"])[col]
    mean = grp.transform("mean")
    std  = grp.transform("std").fillna(0)
    z    = (df[col] - mean) / std.replace(0, 1)
    return z.clip(-10, 10).fillna(0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the features Isolation Forest trains on: raw metrics plus
    peer-group z-scores so it catches providers who are weird relative to
    others billing the same code in the same specialty, not just globally."""
    out = df.copy()
    out["services_per_bene"]       = out["Tot_Srvcs"] / out["Tot_Benes"]
    out["charge_to_payment_ratio"] = out["Avg_Sbmtd_Chrg"] / out["Avg_Mdcr_Pymt_Amt"]

    out["z_charge_peer"]   = _peer_zscore(out, "Avg_Sbmtd_Chrg")
    out["z_svc_bene_peer"] = _peer_zscore(out, "services_per_bene")
    out["z_ratio_peer"]    = _peer_zscore(out, "charge_to_payment_ratio")
    return out


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    """Train Isolation Forest and score rows (lower score = more anomalous)."""
    feature_cols = [
        "Avg_Sbmtd_Chrg",
        "Avg_Mdcr_Pymt_Amt",
        "services_per_bene",
        "charge_to_payment_ratio",
        "z_charge_peer",
        "z_svc_bene_peer",
        "z_ratio_peer",
    ]

    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    preds = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)

    result = df.copy()
    result["anomaly_score"] = scores
    result["is_anomaly"] = preds == -1
    result = result.sort_values("anomaly_score", ascending=True).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Query builder for Stage 2 (policy retrieval)
# ---------------------------------------------------------------------------

def build_claim_query(row: pd.Series) -> str:
    return (
        f"Provider specialty: {row['Rndrng_Prvdr_Type']}. "
        f"HCPCS {row['HCPCS_Cd']}: {row['HCPCS_Desc']}. "
        f"Services: {row['Tot_Srvcs']}, beneficiaries: {row['Tot_Benes']}, "
        f"services per beneficiary: {row['services_per_bene']:.2f}. "
        f"Avg submitted charge ${row['Avg_Sbmtd_Chrg']:.2f}, "
        f"avg Medicare payment ${row['Avg_Mdcr_Pymt_Amt']:.2f}, "
        f"charge-to-payment ratio {row['charge_to_payment_ratio']:.2f}."
    )