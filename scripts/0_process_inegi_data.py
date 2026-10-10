import os
import pandas as pd

def process_inegi_dataset(file_path, category_header):
    """Helper function to process either annual or perennial INEGI agricultural datasets for harvested area in hectares."""
    df_raw = pd.read_excel(file_path, header=None)

    # Ingest data starting after the header/metadata block (row index 10)
    data = df_raw.iloc[10:].copy()
    
    data.columns = [
        'entidad_code', 'municipio_code', 'cultivo', 'descripcion',
        'unidades_prod_total', 'unidades_prod_con_prod', 'sup_cultivada',
        'sup_cosechada', 'produccion_total_ton',
        'mod_hidrica_1', 'mod_hidrica_2', 'mod_hidrica_3', 'mod_hidrica_4',
        'mod_hidrica_5', 'mod_hidrica_6', 'mod_hidrica_7', 'mod_hidrica_8'
    ][:data.shape[1]]

    # Exclude national rows ('00 NAL')
    regional_data = data[data['entidad_code'] != '00 NAL'].copy()

    # Isolate State Totals (municipio_code is NaN, cultivo is valid, excluding category header)
    state_crop_rows = regional_data[
        regional_data['municipio_code'].isna() &
        regional_data['cultivo'].notna() &
        (regional_data['cultivo'] != category_header)
    ].copy()

    state_crop_rows['cultivo'] = state_crop_rows['cultivo'].astype(str).str.strip()

    # Extract state names cleanly from the state header rows (where cultivo is NaN)
    state_headers = regional_data[
        regional_data['municipio_code'].isna() &
        regional_data['cultivo'].isna()
    ][['entidad_code', 'descripcion']].drop_duplicates()
    
    state_map = dict(zip(state_headers['entidad_code'], state_headers['descripcion']))
    state_crop_rows['state_name'] = state_crop_rows['entidad_code'].map(state_map)

    # Convert harvested area figures to numeric values (in hectares)
    state_crop_rows['sup_cosechada'] = pd.to_numeric(
        state_crop_rows['sup_cosechada'], errors='coerce'
    ).fillna(0)

    # Pivot table: Harvested Hectares (sup_cosechada)
    ha_matrix = state_crop_rows.pivot_table(
        index='cultivo',
        columns='state_name',
        values='sup_cosechada',
        aggfunc='sum',
        fill_value=0
    )

    return ha_matrix

def main():
    # Relative paths from ./scripts to ./data and ./outputs
    annual_path = os.path.join('..', 'data', 'ca2022_agr02.xlsx')
    perennial_path = os.path.join('..', 'data', 'ca2022_agr05.xlsx')
    output_dir = os.path.join('..', 'outputs')
    output_ha_path = os.path.join(output_dir, 'INEGI_state_crop_hectares.csv')

    os.makedirs(output_dir, exist_ok=True)

    print("Processing annual crops dataset for harvested area...")
    annual_ha = process_inegi_dataset(annual_path, 'Anuales')

    print("Processing perennial crops dataset for harvested area...")
    perennial_ha = process_inegi_dataset(perennial_path, 'Perennes')

    # Combine harvested hectares matrices
    combined_ha = pd.concat([annual_ha, perennial_ha])
    combined_ha = combined_ha.groupby(combined_ha.index).sum()

    print(f"Combined Harvested Hectares Matrix created with shape: {combined_ha.shape} (Crops x States)")

    # Export master harvested hectares matrix to CSV
    combined_ha.to_csv(output_ha_path)
    print(f"Successfully saved master harvested hectares matrix to {output_ha_path}")

if __name__ == '__main__':
    main()
