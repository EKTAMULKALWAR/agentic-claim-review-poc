"""Load CMS Medicare Physician & Other Practitioners – by Provider and Service data."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

CMS_DATASET_ID = "92396110-2aed-4d63-a6a2-5d6207d46a29"
CMS_API_URL = f"https://data.cms.gov/data-api/v1/dataset/{CMS_DATASET_ID}/data"
CMS_CSV_URL = (
    "https://data.cms.gov/sites/default/files/2026-05/"
    "b5ebab5a-f490-418a-9bce-4b9f31419356/PHY_R26_P05_V10_D24_Prov_Svc.csv"
)

KEY_COLUMNS = [
    "Rndrng_NPI",
    "Rndrng_Prvdr_Type",
    "HCPCS_Cd",
    "HCPCS_Desc",
    "Tot_Srvcs",
    "Tot_Benes",
    "Avg_Sbmtd_Chrg",
    "Avg_Mdcr_Pymt_Amt",
]

DEFAULT_SAMPLE_PATH = Path(__file__).parent / "data" / "cms_sample_50k.csv"
API_PAGE_SIZE = 6500


def fetch_from_api(target_rows: int = 50_000) -> pd.DataFrame:
    """Fetch rows from the CMS JSON API (paginated; max ~6500 per request)."""
    frames: list[pd.DataFrame] = []
    offset = 0

    while offset < target_rows:
        size = min(API_PAGE_SIZE, target_rows - offset)
        url = f"{CMS_API_URL}?size={size}&offset={offset}"
        print(f"Fetching CMS API offset={offset:,} size={size:,} ...")

        with urlopen(url, timeout=120) as response:
            rows = json.load(response)

        if not rows:
            break

        frames.append(pd.DataFrame(rows))
        offset += len(rows)
        if len(rows) < size:
            break

    if not frames:
        raise RuntimeError("CMS API returned no data.")

    df = pd.concat(frames, ignore_index=True)
    print(f"Fetched {len(df):,} rows from CMS API.")
    return df


def load_or_download(
    csv_path: Path | str = DEFAULT_SAMPLE_PATH,
    sample_size: int = 50_000,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load cached sample CSV or download from CMS API."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and not force_refresh:
        print(f"Loading cached sample from {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        return df[KEY_COLUMNS].copy()

    df = fetch_from_api(sample_size)
    df = df[KEY_COLUMNS].copy()
    df.to_csv(csv_path, index=False)
    print(f"Saved sample to {csv_path}")
    return df


def clean_claims(df: pd.DataFrame) -> pd.DataFrame:
    """Clean numeric fields and drop invalid rows."""
    df = df.copy()
    numeric_cols = ["Tot_Srvcs", "Tot_Benes", "Avg_Sbmtd_Chrg", "Avg_Mdcr_Pymt_Amt"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)
    df = df[(df["Tot_Benes"] > 0) & (df["Tot_Srvcs"] > 0)]
    df = df[df["Avg_Mdcr_Pymt_Amt"] > 0]
    df["Rndrng_NPI"] = df["Rndrng_NPI"].astype(str)
    df["Rndrng_Prvdr_Type"] = df["Rndrng_Prvdr_Type"].fillna("Unknown")
    return df.reset_index(drop=True)
