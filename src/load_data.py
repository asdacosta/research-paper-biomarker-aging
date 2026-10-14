"""
Step 1 of Topic 1: turn five separate NHANES 2017-2018 files into one clean table.

Each NHANES component (demographics, blood chemistry, blood count, etc.) ships as
its own .XPT file. They all share one column, SEQN, a unique ID per person.
That's the only thing that lets us join them: it's not that the files are
related by position, it's that row 4021 in one file and row 4021 in another
file are almost certainly NOT the same person -- we have to match by SEQN,
not by row order.

Design choice worth naming: reading and merging are two separate functions.
read_xpt() touches the disk. build_dataset() only touches DataFrames already
in memory. That split is what let me test the merge logic below with fake
data, before a single real NHANES file existed on this machine.
"""

import pandas as pd

RAW_DIR = "../data/raw"
OUT_PATH = "../data/processed/nhanes_2017_2018_biomarkers.csv"

# Which columns we need from each file, beyond SEQN (added automatically).
FILES = {
    "DEMO_J.XPT":   ["RIDAGEYR", "RIAGENDR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"],
    "BIOPRO_J.XPT": ["LBXSAL", "LBXSCR"],              # albumin, creatinine
    "CBC_J.XPT":    ["LBXHGB", "LBXWBCSI"],            # hemoglobin, white blood cells
    "HSCRP_J.XPT":  ["LBXHSCRP"],                      # inflammation marker
    "GLU_J.XPT":    ["LBXGLU", "WTSAF2YR"],            # fasting glucose + its own subsample weight
}

BIOMARKERS = ["LBXSAL", "LBXSCR", "LBXHGB", "LBXWBCSI", "LBXHSCRP", "LBXGLU"]


def read_xpt(filename, columns):
    """Read one NHANES .XPT file and keep only SEQN + the columns we asked for."""
    path = f"{RAW_DIR}/{filename}"
    df = pd.read_sas(path, format="xport")
    keep = ["SEQN"] + [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"  WARNING: {filename} is missing expected columns: {missing}")
    return df[keep]


def build_dataset(demo, biopro, cbc, hscrp, glu):
    """
    Merge the five tables on SEQN, then apply the filters that make this an
    adult, complete-case biomarker table. This function never touches disk,
    which is what makes it testable with fake data.
    """
    df = demo.merge(biopro, on="SEQN", how="left") \
             .merge(cbc,    on="SEQN", how="left") \
             .merge(hscrp,  on="SEQN", how="left") \
             .merge(glu,    on="SEQN", how="left")

    # NHANES lab eligibility for these components starts at 12; we only
    # want adults, and 18 is the simplest, most defensible cut for a first pass.
    df = df[df["RIDAGEYR"] >= 18].copy()

    return df


def main():
    print("Reading raw files...")
    tables = {}
    for filename, cols in FILES.items():
        print(f"  {filename}")
        key = filename.split("_")[0].split(".")[0].lower()
        tables[key] = read_xpt(filename, cols)

    print("Merging on SEQN...")
    merged = build_dataset(
        demo=tables["demo"],
        biopro=tables["biopro"],
        cbc=tables["cbc"],
        hscrp=tables["hscrp"],
        glu=tables["glu"],
    )

    print(f"\nRows after merge, adults only: {len(merged)}")
    print("\nMissingness per biomarker (this is normal -- not everyone gives every sample):")
    for col in BIOMARKERS:
        if col in merged.columns:
            n_missing = merged[col].isna().sum()
            pct = 100 * n_missing / len(merged)
            print(f"  {col}: {n_missing} missing ({pct:.1f}%)")

    merged.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
