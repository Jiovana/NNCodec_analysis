import pandas as pd
import numpy as np
import re

PROFILE_CSV = "profile.csv"      # <-- your file
OUT_PREFIX  = "profile_analysis" # output prefix

df = pd.read_csv(PROFILE_CSV)

# ------------------------------
# Normalize section names
# ------------------------------
df["section"] = df["section"].str.strip()

# ------------------------------
# Functional grouping rules
# ------------------------------

def categorize(section):
    s = section.lower()

    # ---- Context modeling ----
    if "ctx" in s or "contextmodeler" in s:
        return "context_modeling"

    # ---- Binarizer core (weights, coeffs, etc.) ----
    if s.startswith("xenc") or s.startswith("pseudoenc") or "remabs" in s:
        return "binarizer"

    # ---- Standard CABAC bin encoding ----
    if s == "encodebin":
        return "cabac_standard"

    # ---- Bypass (EP) encoding ----
    if "ep" in s:
        return "cabac_bypass"

    # ---- Top-level encode functions ----
    if "encodelayer" in s or "encodeweights" in s or "xencodeweights" in s:
        return "top_level"

    # ---- Catch-all ----
    return "other"


df["category"] = df["section"].apply(categorize)

# ------------------------------
# Summary per section
# ------------------------------
sum_section = (
    df.groupby("section")
      .agg(total_time_us=("total_time_us","sum"),
           calls=("calls","sum"),
           avg_us=("avg_time_us","mean"))
      .sort_values("total_time_us", ascending=False)
      .reset_index()
)

# ------------------------------
# Summary per category
# ------------------------------
sum_category = (
    df.groupby("category")
      .agg(total_time_us=("total_time_us","sum"),
           calls=("calls","sum"))
      .sort_values("total_time_us", ascending=False)
      .reset_index()
)

sum_category["pct_of_total"] = (
    100 * sum_category["total_time_us"] / sum_category["total_time_us"].sum()
)

# ------------------------------
# Save results
# ------------------------------
sum_section.to_csv(f"{OUT_PREFIX}_by_section.csv", index=False)
sum_category.to_csv(f"{OUT_PREFIX}_by_category.csv", index=False)

# ------------------------------
# Print summaries
# ------------------------------
print("\n=== TOP SECTIONS (sorted by total time) ===")
print(sum_section.head(20))

print("\n=== CATEGORY BREAKDOWN ===")
print(sum_category)
