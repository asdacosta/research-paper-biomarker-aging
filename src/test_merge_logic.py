"""
Proves build_dataset() works BEFORE any real NHANES file has been downloaded.
This is the habit worth keeping: test the logic on fake data you made up,
so that when real data shows up, you're only debugging the data, not the code.
"""

import pandas as pd
from load_data import build_dataset

# Five people, on purpose including cases that break naive code:
# - SEQN 5 exists in demo but in NO lab file (simulates someone who didn't give blood)
# - SEQN 4 is a minor (age 15) -- should get filtered out
# - SEQN 3 is missing just one biomarker -- should survive, just with a NaN

demo = pd.DataFrame({
    "SEQN": [1, 2, 3, 4, 5],
    "RIDAGEYR": [45, 62, 30, 15, 70],
    "RIAGENDR": [1, 2, 1, 2, 1],
})
biopro = pd.DataFrame({
    "SEQN": [1, 2, 4, 5],
    "LBXSAL": [4.1, 3.9, 4.5, 4.0],
    "LBXSCR": [0.9, 1.1, 0.8, 1.0],
})
cbc = pd.DataFrame({
    "SEQN": [1, 2, 3, 4],
    "LBXHGB": [14.2, 13.8, 15.1, 13.0],
    "LBXWBCSI": [6.1, 7.2, 5.9, 6.5],
})
hscrp = pd.DataFrame({
    "SEQN": [1, 3, 4],
    "LBXHSCRP": [0.8, 1.2, 0.5],
})
glu = pd.DataFrame({
    "SEQN": [1, 2, 3, 4],
    "LBXGLU": [92, 101, 88, 95],
})

result = build_dataset(demo, biopro, cbc, hscrp, glu)

print(result[["SEQN", "RIDAGEYR", "LBXSAL", "LBXHSCRP"]])
print()

assert len(result) == 4, f"expected 4 adults after dropping the minor, got {len(result)}"
assert 4 not in result["SEQN"].values, "the 15-year-old should have been filtered out"
assert 5 in result["SEQN"].values, "SEQN 5 should survive the merge even with no lab data, just NaNs"
row5 = result[result["SEQN"] == 5].iloc[0]
assert pd.isna(row5["LBXHGB"]), "SEQN 5 has no CBC row, so this should be NaN, not dropped"

print("All checks passed: age filter works, left-joins preserve people with partial data.")
