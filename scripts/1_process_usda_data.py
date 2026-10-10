import os
import pandas as pd

def main():
    # Determine correct paths based on execution context (supporting both ./scripts and root execution)
    if os.path.exists(os.path.join("..", "data")):
        match_path = os.path.join("..", "data", "USDA_INEGI_MATCHES.csv")
        survey_path = os.path.join("..", "data", "USDA_2022_survey_by_state.csv")
        output_dir = os.path.join("..", "outputs")
    else:
        match_path = os.path.join("data", "USDA_INEGI_MATCHES.csv")
        survey_path = os.path.join("data", "USDA_2022_survey_by_state.csv")
        output_dir = "outputs"

    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, "USDA_state_crop_hectares.csv")

    print(f"Loading matches from {match_path}...")
    matches_df = pd.read_csv(match_path)

    print(f"Loading survey data from {survey_path}...")
    survey_df = pd.read_csv(survey_path)

    # Get unique valid measurements (excluding 'NA')
    unique_measurements = matches_df['usda_specific_measurement'].dropna().unique()
    unique_measurements = [m for m in unique_measurements if m.upper() != 'NA']

    acres_to_hectares = 0.404686
    results = []

    for measurement in unique_measurements:
        # Filter survey data for the specific Data Item matching measurement
        subset = survey_df[survey_df['Data Item'].str.strip() == measurement.strip()].copy()
        
        if subset.empty:
            print(f"Warning: No matching survey rows found for '{measurement}'")
            continue

        # Clean and convert values to numeric hectares
        subset['Value_num'] = (
            pd.to_numeric(subset['Value'].astype(str).str.replace(',', ''), errors='coerce')
            * acres_to_hectares
        )

        # Aggregate by State
        state_values = subset.groupby('State')['Value_num'].sum().to_dict()

        row_data = {
            'usda_specific_measurement': measurement
        }
        row_data.update(state_values)
        results.append(row_data)

    summary_df = pd.DataFrame(results)

    if summary_df.empty:
        print("Error: No data matched between USDA_INEGI_MATCHES.csv and USDA_2022_survey_by_state.csv.")
        return

    # Reorder columns: usda_specific_measurement, then alphabetically by state
    state_cols = sorted([c for c in summary_df.columns if c != 'usda_specific_measurement'])
    summary_df = summary_df[['usda_specific_measurement'] + state_cols]

    # Fill empty/NaN cells with 0
    summary_df[state_cols] = summary_df[state_cols].fillna(0)

    print(f"Saving summary hectares matrix to {output_csv_path}...")
    summary_df.to_csv(output_csv_path, index=False)
    print("Done!")

if __name__ == '__main__':
    main()