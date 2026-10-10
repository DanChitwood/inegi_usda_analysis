import os
import pandas as pd

def main():
    # Paths assuming execution from ./scripts
    match_path = os.path.join("..", "data", "USDA_INEGI_MATCHES.csv")
    inegi_path = os.path.join("..", "outputs", "INEGI_state_crop_hectares.csv")
    usda_path = os.path.join("..", "outputs", "USDA_state_crop_hectares.csv")
    output_dir = os.path.join("..", "outputs")
    output_csv_path = os.path.join(output_dir, "INEGI_USDA_combined_data.csv")

    # Fallback paths if run from root instead of ./scripts
    if not os.path.exists(match_path):
        match_path = os.path.join("data", "USDA_INEGI_MATCHES.csv")
    if not os.path.exists(inegi_path):
        inegi_path = os.path.join("outputs", "INEGI_state_crop_hectares.csv")
    if not os.path.exists(usda_path):
        usda_path = os.path.join("outputs", "USDA_state_crop_hectares.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Loading matches from {match_path}...")
    matches_df = pd.read_csv(match_path)

    # Ensure blank/NaN in usda_specific_measurement are set to 'NA'
    matches_df['usda_specific_measurement'] = matches_df['usda_specific_measurement'].fillna('NA')

    # Apply collapse mapping for INEGI commodities in matches table
    matches_df['inegi_commodity'] = matches_df['inegi_commodity'].replace({
        'Maíz grano amarillo': 'Maíz grano',
        'Maíz grano blanco': 'Maíz grano',
        'Jitomate (Tomate rojo)': 'Jitomate/tomatillo',
        'Tomate de cáscara (tomatillo)': 'Jitomate/tomatillo'
    })
    # Drop duplicates created by collapsing
    matches_df = matches_df.drop_duplicates(subset=['inegi_commodity', 'usda_specific_measurement']).reset_index(drop=True)

    print(f"Loading INEGI data from {inegi_path}...")
    inegi_df = pd.read_csv(inegi_path)

    print(f"Loading USDA data from {usda_path}...")
    usda_df = pd.read_csv(usda_path)

    # State code mappings
    mx_state_map = {
        'Aguascalientes': 'AGS', 'Baja California': 'BCN', 'Baja California Sur': 'BCS',
        'Campeche': 'CAM', 'Chiapas': 'CHP', 'Chihuahua': 'CHH', 'Ciudad de México': 'CMX',
        'Coahuila de Zaragoza': 'COA', 'Colima': 'COL', 'Durango': 'DUR', 'Guanajuato': 'GTO',
        'Guerrero': 'GRO', 'Hidalgo': 'HGO', 'Jalisco': 'JAL', 'Michoacán de Ocampo': 'MIC',
        'Morelos': 'MOR', 'México': 'MEX', 'Nayarit': 'NAY', 'Nuevo León': 'NLE',
        'Oaxaca': 'OAX', 'Puebla': 'PUE', 'Querétaro': 'QRO', 'Quintana Roo': 'ROO',
        'San Luis Potosí': 'SLP', 'Sinaloa': 'SIN', 'Sonora': 'SON', 'Tabasco': 'TAB',
        'Tamaulipas': 'TAM', 'Tlaxcala': 'TLA', 'Veracruz de Ignacio de la Llave': 'VER',
        'Yucatán': 'YUC', 'Zacatecas': 'ZAC'
    }

    us_state_map = {
        'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR', 'CALIFORNIA': 'CA',
        'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE', 'FLORIDA': 'FL', 'GEORGIA': 'GA',
        'HAWAII': 'HI', 'IDAHO': 'ID', 'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA',
        'KANSAS': 'KS', 'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
        'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS', 'MISSOURI': 'MO',
        'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV', 'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ',
        'NEW MEXICO': 'NM', 'NEW YORK': 'NY', 'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH',
        'OKLAHOMA': 'OK', 'OREGON': 'OR', 'OTHER STATES': 'OT', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI',
        'SOUTH CAROLINA': 'SC', 'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT',
        'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI',
        'WYOMING': 'WY'
    }

    # Rename INEGI columns to 3-letter codes
    inegi_rename = {'cultivo': 'inegi_cultivo'}
    for col in inegi_df.columns:
        if col in mx_state_map:
            inegi_rename[col] = mx_state_map[col]
    inegi_df = inegi_df.rename(columns=inegi_rename)

    # Rename USDA columns to 2-letter codes
    usda_rename = {'usda_specific_measurement': 'usda_specific_measurement'}
    for col in usda_df.columns:
        upper_col = col.strip().upper()
        if upper_col in us_state_map:
            usda_rename[col] = us_state_map[upper_col]
    usda_df = usda_df.rename(columns=usda_rename)

    # Handle collapse rules in INEGI data
    inegi_df['inegi_cultivo_mapped'] = inegi_df['inegi_cultivo'].replace({
        'Maíz grano amarillo': 'Maíz grano',
        'Maíz grano blanco': 'Maíz grano',
        'Jitomate (Tomate rojo)': 'Jitomate/tomatillo',
        'Tomate de cáscara (tomatillo)': 'Jitomate/tomatillo'
    })

    # Group by the mapped crop name (summing states if collapsed)
    mx_state_cols = list(mx_state_map.values())
    inegi_collapsed = inegi_df.groupby('inegi_cultivo_mapped')[mx_state_cols].sum().reset_index()
    inegi_collapsed = inegi_collapsed.rename(columns={'inegi_cultivo_mapped': 'inegi_cultivo'})

    # Merge matches dataframe with collapsed INEGI and USDA data
    merged = pd.merge(matches_df, inegi_collapsed, left_on='inegi_commodity', right_on='inegi_cultivo', how='left')
    merged = pd.merge(merged, usda_df, on='usda_specific_measurement', how='left')

    # USDA commodity mapping dictionary
    usda_commodity_map = {
        'Agave': 'Agave', 'Aguacate': 'Avocado', 'Alfalfa': 'Alfalfa', 'Algodón': 'Cotton',
        'Amaranto': 'Amaranth', 'Arroz': 'Rice', 'Avena forrajera': 'Oats', 'Brócoli': 'Broccoli',
        'Cacao': 'Chocolate', 'Café': 'Coffee', 'Calabaza/calabacita': 'Squash', 'Caña de azúcar': 'Sugarcane',
        'Cebada grano': 'Barley', 'Cebolla': 'Onion', 'Chile': 'Chile', 'Coco': 'Coconut',
        'Espárrago': 'Asparagus', 'Fresa': 'Strawberry', 'Frijol': 'Bean', 'Guayaba': 'Guava',
        'Jitomate/tomatillo': 'Tomato', 'Limón': 'Lime', 'Mango': 'Mango', 'Manzana': 'Apple',
        'Maíz forrajero': 'Corn silage', 'Maíz grano': 'Corn grain', 'Melón': 'Melon', 'Naranja': 'Orange',
        'Nopal verdura': 'Nopal', 'Nuez': 'Nuts', 'Papa': 'Potato', 'Papaya': 'Papaya',
        'Pasto cultivado': 'Hay', 'Piña': 'Pineapple', 'Plátano': 'Banana', 'Sandía': 'Watermelon',
        'Sorgo forrajero': 'Sorghum silage', 'Sorgo grano': 'Sorghum grain', 'Soya': 'Soy',
        'Trigo grano': 'Wheat', 'Uva': 'Grape'
    }

    merged['usda_commodity'] = merged['inegi_commodity'].map(usda_commodity_map).fillna(merged['inegi_commodity'])

    # Clean up columns and order: inegi_commodity, usda_commodity, usda_specific_measurement, then MX states, then US states
    us_state_cols = list(us_state_map.values())
    existing_mx = [c for c in mx_state_cols if c in merged.columns]
    existing_us = [c for c in us_state_cols if c in merged.columns]

    final_columns = ['inegi_commodity', 'usda_commodity', 'usda_specific_measurement'] + existing_mx + existing_us
    final_df = merged[final_columns].copy()

    # Fill NaN with 0 for numeric hectare values
    final_df[existing_mx + existing_us] = final_df[existing_mx + existing_us].fillna(0)

    print(f"Saving combined dataset to {output_csv_path}...")
    final_df.to_csv(output_csv_path, index=False)
    print("Done!")

if __name__ == '__main__':
    main()
