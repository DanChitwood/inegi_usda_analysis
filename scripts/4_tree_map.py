import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import geopandas as gpd

def safe_compute_rects(items, x, y, dx, dy):
    if not items:
        return []
    if len(items) == 1:
        idx, val = items[0]
        return [(idx, x, y, dx, dy)]
    
    total = sum(v for _, v in items)
    if total <= 0:
        n = len(items)
        rects = []
        if dx >= dy:
            w = dx / n
            for i, (idx, val) in enumerate(items):
                rects.append((idx, x + i*w, y, w, dy))
        else:
            h = dy / n
            for i, (idx, val) in enumerate(items):
                rects.append((idx, x, y + i*h, dx, h))
        return rects
    
    mid = max(1, len(items) // 2)
    group1 = items[:mid]
    group2 = items[mid:]
    
    sum1 = sum(v for _, v in group1)
    sum2 = sum(v for _, v in group2)
    
    rects = []
    if dx >= dy:
        w1 = dx * (sum1 / total) if total > 0 else dx
        w2 = dx - w1
        rects.extend(safe_compute_rects(group1, x, y, w1, dy))
        rects.extend(safe_compute_rects(group2, x + w1, y, w2, dy))
    else:
        h1 = dy * (sum1 / total) if total > 0 else dy
        h2 = dy - h1
        rects.extend(safe_compute_rects(group1, x, y, dx, h1))
        rects.extend(safe_compute_rects(group2, x, y + h1, dx, h2))
        
    return rects

def format_crop_label(name):
    words = name.split()
    if len(words) <= 1:
        return name
    elif len(words) == 2:
        return f"{words[0]}\n{words[1]}"
    else:
        mid = len(words) // 2
        return f"{' '.join(words[:mid])}\n{' '.join(words[mid:])}"

def prepare_plot_dataframe(df_source, val_col, name_col, top_n=10):
    """
    Filters and groups data so that any crop whose box would be too small 
    to label is automatically collapsed into 'Other'.
    """
    df_sorted = df_source[df_source[val_col] > 0].sort_values(by=val_col, ascending=False).reset_index(drop=True)
    if len(df_sorted) <= top_n:
        return df_sorted

    # Iteratively ensure all included items generate boxes above the minimum threshold (dx > 8 and dy > 8 out of 100)
    for n in range(top_n, 0, -1):
        top_df = df_sorted.iloc[:n].copy()
        other_sum = df_sorted.iloc[n:][val_col].sum()
        if other_sum > 0:
            other_row = pd.DataFrame([{name_col: 'Other', val_col: other_sum}])
            test_df = pd.concat([top_df, other_row], ignore_index=True)
        else:
            test_df = top_df.copy()
            
        sizes = test_df[val_col].values
        indexed = sorted(enumerate(sizes), key=lambda x: x[1], reverse=True)
        rects = safe_compute_rects(indexed, 0, 0, 100, 100)
        dict_rects = {idx: (x, y, dx, dy) for idx, x, y, dx, dy in rects}
        
        # Check if all plotted boxes meet the minimum size for text labeling
        all_labeled = True
        for idx, (_, dx, dy) in enumerate([(rect[1], rect[3], rect[4]) for rect in rects]):
            if dx <= 8 or dy <= 8:
                all_labeled = False
                break
        if all_labeled:
            return test_df
            
    # Fallback to top_n if strict sizing isn't met
    top_df = df_sorted.iloc[:top_n].copy()
    other_sum = df_sorted.iloc[top_n:][val_col].sum()
    other_row = pd.DataFrame([{name_col: 'Other', val_col: other_sum}])
    return pd.concat([top_df, other_row], ignore_index=True)

def main():
    output_dir = "outputs" if os.path.exists("outputs") else os.path.join("..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_img_path = os.path.join(output_dir, "INEGI_USDA_crop_labor_figure.png")

    combined_path = os.path.join(output_dir, "INEGI_USDA_combined_data.csv")
    if not os.path.exists(combined_path):
        combined_path = "INEGI_USDA_combined_data.csv"

    prop_path = "proportion_mex_state.csv" if os.path.exists("proportion_mex_state.csv") else os.path.join("..", "outputs", "proportion_mex_state.csv")
    if not os.path.exists(prop_path):
        prop_path = os.path.join(output_dir, "proportion_mex_state.csv")

    df_comb = pd.read_csv(combined_path)
    prop_df = pd.read_csv(prop_path)

    ne_url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip"
    states_gdf = gpd.read_file(ne_url)

    us_states = states_gdf[
        (states_gdf["admin"] == "United States of America") &
        (~states_gdf["name"].isin(["Alaska", "Hawaii", "District of Columbia"]))
    ].copy()
    mx_states = states_gdf[states_gdf["admin"] == "Mexico"].copy()

    region_mapping_names = {
        "California": ["California"],
        "Northwest": ["Idaho", "Montana", "Wyoming", "Colorado", "Nevada", "Utah", "Oregon", "Washington"],
        "Midwest": ["Illinois", "Indiana", "Ohio", "Iowa", "Missouri", "Kansas", "Nebraska", "North Dakota", "South Dakota", "Michigan", "Minnesota", "Wisconsin"],
        "East": ["North Carolina", "Virginia", "Kentucky", "Tennessee", "West Virginia", "Connecticut", "Maine", "Massachusetts", "New Hampshire", "New York", "Rhode Island", "Vermont", "Delaware", "Maryland", "New Jersey", "Pennsylvania"],
        "Southeast": ["Arkansas", "Louisiana", "Mississippi", "Alabama", "Georgia", "South Carolina", "Florida"],
        "Southwest": ["Arizona", "New Mexico", "Oklahoma", "Texas"]
    }

    state_to_region = {}
    for reg, st_list in region_mapping_names.items():
        for st in st_list:
            state_to_region[st] = reg
    us_states["US_Region"] = us_states["name"].map(state_to_region)

    proj_crs = "ESRI:102008"
    us_states_proj = us_states.to_crs(proj_crs)
    mx_states_proj = mx_states.to_crs(proj_crs)

    name_to_code = {
        'California': 'CA',
        'Idaho': 'ID', 'Montana': 'MT', 'Wyoming': 'WY', 'Colorado': 'CO', 'Nevada': 'NV', 'Utah': 'UT', 'Oregon': 'OR', 'Washington': 'WA',
        'Illinois': 'IL', 'Indiana': 'IN', 'Ohio': 'OH', 'Iowa': 'IA', 'Missouri': 'MO', 'Kansas': 'KS', 'Nebraska': 'NE', 'North Dakota': 'ND', 'South Dakota': 'SD', 'Michigan': 'MI', 'Minnesota': 'MN', 'Wisconsin': 'WI',
        'North Carolina': 'NC', 'Virginia': 'VA', 'Kentucky': 'KY', 'Tennessee': 'TN', 'West Virginia': 'WV', 'Connecticut': 'CT', 'Maine': 'ME', 'Massachusetts': 'MA', 'New Hampshire': 'NH', 'New York': 'NY', 'Rhode Island': 'RI', 'Vermont': 'VT', 'Delaware': 'DE', 'Maryland': 'MD', 'New Jersey': 'NJ', 'Pennsylvania': 'PA',
        'Arkansas': 'AR', 'Louisiana': 'LA', 'Mississippi': 'MS', 'Alabama': 'AL', 'Georgia': 'GA', 'South Carolina': 'SC', 'Florida': 'FL',
        'Arizona': 'AZ', 'New Mexico': 'NM', 'Oklahoma': 'OK', 'Texas': 'TX'
    }

    region_mapping_codes = {
        reg: [name_to_code[st] for st in st_list if st in name_to_code]
        for reg, st_list in region_mapping_names.items()
    }

    reg_names = ["Northwest", "Midwest", "East", "California", "Southwest", "Southeast"]
    
    region_crop_df = pd.DataFrame({'usda_commodity': df_comb['usda_commodity']})
    for reg_name, st_codes in region_mapping_codes.items():
        valid_sts = [st for st in st_codes if st in df_comb.columns]
        if valid_sts:
            region_crop_df[reg_name] = df_comb[valid_sts].sum(axis=1)
        else:
            region_crop_df[reg_name] = 0.0

    df_comb['crop_collapsed'] = df_comb['usda_commodity']
    region_crop_df['crop_collapsed'] = df_comb['usda_commodity']
    reg_crop_collapsed = region_crop_df.groupby('crop_collapsed')[reg_names].sum()

    # Semantic color mapping with 'Other'
    crop_color_mapping = {
        'Agave': '#5B8266',
        'Alfalfa': '#8FBC8F',
        'Avocado': '#4C7D23',
        'Barley': '#D4AF37',
        'Bean': '#8B4513',
        'Broccoli': '#2E8B57',
        'Chile': '#C0392B',
        'Coconut': '#B48A57',
        'Coffee': '#6F4E37',
        'Corn grain': '#F1C40F',
        'Corn silage': '#85929E',
        'Cotton': '#EAECEE',
        'Grape': '#6C3483',
        'Guava': '#E59866',
        'Hay': '#D4AC0D',
        'Lime': '#27AE60',
        'Mango': '#D35400',
        'Oats': '#D2B48C',
        'Other': '#BDC3C7',
        'Rice': '#F9E79F',
        'Sorghum grain': '#B9770E',
        'Soy': '#52BE80',
        'Squash': '#EB984E',
        'Sugarcane': '#7DCEA0',
        'Wheat': '#DEB887'
    }

    state_config = {
        "Michoacán": {"code": "MIC", "color": "#1b9e77", "shape_kw": ["Michoacán"]},
        "Guanajuato": {"code": "GTO", "color": "#d95f02", "shape_kw": ["Guanajuato"]},
        "Jalisco": {"code": "JAL", "color": "#7570b3", "shape_kw": ["Jalisco"]},
        "Oaxaca": {"code": "OAX", "color": "#e7298a", "shape_kw": ["Oaxaca"]},
        "Guerrero": {"code": "GRO", "color": "#66a61e", "shape_kw": ["Guerrero"]},
        "Tamaulipas": {"code": "TAM", "color": "#e6ab02", "shape_kw": ["Tamaulipas"]},
        "Zacatecas": {"code": "ZAC", "color": "#a6761d", "shape_kw": ["Zacatecas"]}
    }

    fig, axes = plt.subplots(len(state_config), 3, figsize=(10, 14), dpi=300)

    fig.text(0.50, 0.97, "Mexico", ha='center', va='center', fontsize=11, fontweight='normal')
    fig.text(0.83, 0.97, "USA", ha='center', va='center', fontsize=11, fontweight='normal')

    rendered_text_labels = set()

    for row_idx, (state_name, cfg) in enumerate(state_config.items()):
        code = cfg["code"]
        hex_color = cfg["color"]
        shape_keywords = cfg["shape_kw"]
        
        # COLUMN 1: Map
        ax_map = axes[row_idx, 0] if len(state_config) > 1 else axes[0]
        st_labor_data = prop_df[prop_df['Mexican_State'] == state_name]
        labor_props = dict(zip(st_labor_data['US_Region'], st_labor_data['Proportion']))

        for region_name in reg_names:
            reg_subset = us_states_proj[us_states_proj['US_Region'] == region_name]
            p = labor_props.get(region_name, 0.0)
            alpha_val = max(0.12, min(1.0, p * 1.4))
            reg_subset.plot(ax=ax_map, color=hex_color, alpha=alpha_val, edgecolor="white", linewidth=0.2)

        mx_bg = mx_states_proj.copy()
        mx_bg.plot(ax=ax_map, color="#e8e8e8", edgecolor="white", linewidth=0.2, alpha=0.8)

        mx_names = mx_states_proj['name'].fillna('')
        mask = np.column_stack([mx_names.str.contains(kw, case=False) for kw in shape_keywords]).any(axis=1)
        mx_target = mx_states_proj[mask]
        
        if not mx_target.empty:
            mx_target.plot(ax=ax_map, color=hex_color, alpha=1.0, edgecolor="black", linewidth=0.5)

        ax_map.set_title(f"{state_name}", fontsize=10, fontweight='normal', color=hex_color, pad=2)
        ax_map.axis("off")

        sm = cm.ScalarMappable(cmap=mcolors.LinearSegmentedColormap.from_list("", [(1,1,1,0.1), hex_color]), norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_map, fraction=0.035, pad=0.02, ticks=[0, 0.5, 1.0])
        cbar.ax.tick_params(labelsize=5)

        # COLUMN 2: Home Treemap (Mexico)
        ax_treemap_home = axes[row_idx, 1] if len(state_config) > 1 else axes[1]
        
        home_crop_df = df_comb.groupby('usda_commodity')[code].sum().reset_index()
        home_plot_df = prepare_plot_dataframe(home_crop_df, val_col=code, name_col='usda_commodity', top_n=10)

        sizes_home = home_plot_df[code].values
        indexed_home = sorted(enumerate(sizes_home), key=lambda x: x[1], reverse=True)
        rects_home = safe_compute_rects(indexed_home, 0, 0, 100, 100)
        dict_home = {idx: (x, y, dx, dy) for idx, x, y, dx, dy in rects_home}

        for i, row in home_plot_df.iterrows():
            if i not in dict_home:
                continue
            x, y, dx, dy = dict_home[i]
            crop_name = row['usda_commodity']
            box_color = crop_color_mapping.get(crop_name, '#BDC3C7')
            
            ax_treemap_home.add_patch(patches.Rectangle((x, y), dx, dy, facecolor=box_color, edgecolor='white', linewidth=1.0))
            
            if dx > 8 and dy > 8:
                dyn_fontsize = max(7, min(9, int(np.sqrt(dx * dy) * 0.35)))
                label_str = format_crop_label(crop_name)
                ax_treemap_home.text(x + dx/2, y + dy/2, label_str, color='black', fontsize=dyn_fontsize, fontweight='normal', ha='center', va='center')
                rendered_text_labels.add(crop_name)

        ax_treemap_home.set_xlim(0, 100)
        ax_treemap_home.set_ylim(0, 100)
        ax_treemap_home.axis("off")

        # COLUMN 3: Weighted US Footprint Treemap
        ax_treemap_us = axes[row_idx, 2] if len(state_config) > 1 else axes[2]

        us_weighted_series = pd.Series(0.0, index=reg_crop_collapsed.index)
        for reg_name, p in labor_props.items():
            if reg_name in reg_crop_collapsed.columns:
                us_weighted_series += reg_crop_collapsed[reg_name] * p

        us_crop_df = us_weighted_series.reset_index(name='US_Weighted_Hectares')
        us_plot_df = prepare_plot_dataframe(us_crop_df, val_col='US_Weighted_Hectares', name_col='crop_collapsed', top_n=10)

        sizes_us = us_plot_df['US_Weighted_Hectares'].values
        indexed_us = sorted(enumerate(sizes_us), key=lambda x: x[1], reverse=True)
        rects_us = safe_compute_rects(indexed_us, 0, 0, 100, 100)
        dict_us = {idx: (x, y, dx, dy) for idx, x, y, dx, dy in rects_us}

        for i, row in us_plot_df.iterrows():
            if i not in dict_us:
                continue
            x, y, dx, dy = dict_us[i]
            crop_name = row['crop_collapsed']
            box_color = crop_color_mapping.get(crop_name, '#BDC3C7')

            ax_treemap_us.add_patch(patches.Rectangle((x, y), dx, dy, facecolor=box_color, edgecolor='white', linewidth=1.0))
            
            if dx > 8 and dy > 8:
                dyn_fontsize = max(7, min(9, int(np.sqrt(dx * dy) * 0.35)))
                label_str = format_crop_label(crop_name)
                ax_treemap_us.text(x + dx/2, y + dy/2, label_str, color='black', fontsize=dyn_fontsize, fontweight='normal', ha='center', va='center')
                rendered_text_labels.add(crop_name)

        ax_treemap_us.set_xlim(0, 100)
        ax_treemap_us.set_ylim(0, 100)
        ax_treemap_us.axis("off")

    plt.subplots_adjust(top=0.94, bottom=0.03, left=0.03, right=0.97, hspace=0.20, wspace=0.12)
    plt.savefig(output_img_path, bbox_inches='tight')
    plt.close()

    print(f"Successfully saved 7-state tri-panel figure to {output_img_path}")
    print("\n=== USDA Commodities Rendered with Text Labels in Figure ===")
    for crop in sorted(rendered_text_labels):
        print(f" - {crop}")

if __name__ == '__main__':
    main()
