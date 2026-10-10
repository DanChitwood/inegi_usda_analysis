import os
import numpy as np
import pandas as pd

# ==========================================
# 0. ENSURE OUTPUT DIRECTORY EXISTS
# ==========================================
output_dir = "../outputs"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 1. READ AND MERGE DATA
# ==========================================
print("Reading raw NAWS data files...")
A2E = pd.read_csv("../data/NAWS/NAWS_A2E1103.csv", low_memory=False)
F2Y = pd.read_csv("../data/NAWS/NAWS_F2Y1103.csv", low_memory=False)

# Change the ID variable to a string for merging
A2E["FWID"] = A2E["FWID"].astype(str)
F2Y["FWID"] = F2Y["FWID"].astype(str)

print(f"The length of A2E is {len(A2E)}")
print(f"The length of F2Y is {len(F2Y)}")

# Merge the two dataframes
merged_df = pd.merge(A2E, F2Y, on="FWID", how="inner")
print(f"The length of the merged_df is {len(merged_df)}")

# Save to pickle in outputs directory
pickle_path = os.path.join(output_dir, "NAWS_df.pkl")
merged_df.to_pickle(pickle_path)

# Read back from pickle (validating pipeline flow)
df = pd.read_pickle(pickle_path)

# ==========================================
# 2. DEFINE VARIABLES & MAPPINGS
# ==========================================
fiscal_yrs = [
    [1989, 1990],
    [1991, 1992],
    [1993, 1994],
    [1995, 1996],
    [1997, 1998],
    [1999, 2000],
    [2001, 2002],
    [2003, 2004],
    [2005, 2006],
    [2007, 2008],
    [2009, 2010],
    [2011, 2012],
    [2013, 2014],
    [2015, 2016],
    [2017, 2018],
    [2019, 2020],
    [2021, 2022],
]

var = "B18"  # variable of interest (state/province lived before US)

var_vals = ["34.0", "30.0", "33.0", "38.0", "31.0", "46.0", "50.0"]
var_names = [
    "Michoacán",
    "Guanajuato",
    "Jalisco",
    "Oaxaca",
    "Guerrero",
    "Tamaulipas",
    "Zacatecas",
]

regs = [5, 3, 1, 6, 4, 2]
reg_names = [
    "Northwest",
    "Midwest",
    "East",
    "California",
    "Southwest",
    "Southeast",
]

# Map codes to human-readable names
state_mapping = dict(zip(var_vals, var_names))
state_mapping["other"] = "Other"
region_mapping = dict(zip(regs, reg_names))
tracked_states = var_names + ["Other"]

# Flatten all valid fiscal years into a single list (excluding index 0 / 1989-1990 if B18 data is missing)
all_tracked_fy = [fy for period in fiscal_yrs[1:] for fy in period]

# Filter dataset for all tracked fiscal years combined
df_filtered = df[df["FY_x"].isin(all_tracked_fy)].copy()
df_filtered[var] = df_filtered[var].astype(str)

# Assign human-readable names for Mexican states and US regions
df_filtered["Mexican_State"] = df_filtered[var].map(state_mapping).fillna("Other")
df_filtered["US_Region"] = df_filtered["REGION6"].map(region_mapping)

# ==========================================
# 3. OVERALL WEIGHTED AGGREGATION (COLLAPSING FISCAL YEARS)
# ==========================================
print("Calculating overall weighted worker proportions across all fiscal years...")

# Group by Mexican State and US Region across ALL combined fiscal years, summing the sampling weights (PWTYCRD_x)
grouped = (
    df_filtered[df_filtered["Mexican_State"].isin(tracked_states)]
    .groupby(["Mexican_State", "US_Region"])["PWTYCRD_x"]
    .sum()
    .reset_index()
)

state_region_records = []

# Calculate overall proportion of workers from each region for each state (summing to 1 across regions)
for state in tracked_states:
    state_subset = grouped[grouped["Mexican_State"] == state]
    total_state_weight = state_subset["PWTYCRD_x"].sum()

    for reg_name in reg_names:
        reg_match = state_subset[state_subset["US_Region"] == reg_name]
        weight = reg_match["PWTYCRD_x"].values[0] if not reg_match.empty else 0.0

        proportion = (weight / total_state_weight) if total_state_weight > 0 else 0.0

        state_region_records.append({
            "Mexican_State": state,
            "US_Region": reg_name,
            "Weighted_Workers": weight,
            "Proportion": proportion,
        })

# Create the final dataframe (without Fiscal_Year_Period)
mex_state_region_df = pd.DataFrame(state_region_records)

# ==========================================
# 4. EXPORT OUTPUTS
# ==========================================
output_csv_path = os.path.join(output_dir, "proportion_mex_state.csv")
mex_state_region_df.to_csv(output_csv_path, index=False)
print(f"Successfully saved overall proportion dataframe to {output_csv_path}")
