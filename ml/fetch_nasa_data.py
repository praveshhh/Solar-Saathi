"""
Solar Saathi — Authentic Data Acquisition from NASA POWER API
=============================================================
Downloads 10+ years (2014-2024) of real climate data from NASA POWER API
for 120 diverse global locations covering different climate zones.

Parameters fetched:
  - T2M: Temperature at 2 Meters (°C)
  - T2M_MAX: Max Temperature (°C)  
  - T2M_MIN: Min Temperature (°C)
  - RH2M: Relative Humidity at 2 Meters (%)
  - WS2M: Wind Speed at 2 Meters (m/s)
  - ALLSKY_SFC_SW_DWN: Surface Shortwave Downward Irradiance (kWh/m²/day) ~ GHI
  - ALLSKY_SFC_UV_INDEX: UV Index
  - PRECTOTCORR: Precipitation (mm/day)

Data is monthly for efficiency and covers diverse regions:
  - India (multiple cities), Middle East, Africa, Southeast Asia,
  - Southern Europe, Americas, Australia/Oceania, Cold regions
"""

import requests
import pandas as pd
import numpy as np
import os
import time
import json

# -------------------------------------------------------------------
# 120 diverse global locations across climate zones
# Each tuple: (name, latitude, longitude, climate_zone, country)
# -------------------------------------------------------------------
LOCATIONS = [
    # === INDIA (30 locations across diverse climates) ===
    ("Mumbai", 19.076, 72.878, "tropical_humid", "India"),
    ("Delhi", 28.614, 77.209, "semi_arid_hot", "India"),
    ("Chennai", 13.083, 80.270, "tropical_humid", "India"),
    ("Bangalore", 12.972, 77.595, "tropical_moderate", "India"),
    ("Hyderabad", 17.385, 78.487, "semi_arid_hot", "India"),
    ("Kolkata", 22.573, 88.364, "tropical_humid", "India"),
    ("Ahmedabad", 23.023, 72.572, "semi_arid_hot", "India"),
    ("Pune", 18.520, 73.856, "semi_arid_hot", "India"),
    ("Jaipur", 26.912, 75.787, "arid_hot", "India"),
    ("Lucknow", 26.847, 80.947, "semi_arid_hot", "India"),
    ("Jodhpur", 26.239, 73.024, "arid_hot", "India"),
    ("Varanasi", 25.318, 82.988, "semi_arid_hot", "India"),
    ("Nagpur", 21.146, 79.088, "semi_arid_hot", "India"),
    ("Bhopal", 23.260, 77.413, "semi_arid_hot", "India"),
    ("Leh", 34.153, 77.577, "cold_arid", "India"),
    ("Srinagar", 34.084, 74.797, "cold_temperate", "India"),
    ("Shimla", 31.105, 77.172, "cold_temperate", "India"),
    ("Gangtok", 27.339, 88.607, "cold_humid", "India"),
    ("Coimbatore", 11.017, 76.956, "tropical_moderate", "India"),
    ("Visakhapatnam", 17.687, 83.218, "tropical_humid", "India"),
    ("Kochi", 9.931, 76.267, "tropical_wet", "India"),
    ("Thiruvananthapuram", 8.524, 76.936, "tropical_wet", "India"),
    ("Guwahati", 26.144, 91.736, "tropical_humid", "India"),
    ("Chandigarh", 30.733, 76.779, "semi_arid_hot", "India"),
    ("Udaipur", 24.585, 73.712, "semi_arid_hot", "India"),
    ("Rann_of_Kutch", 23.733, 69.860, "arid_hot", "India"),
    ("Ladakh_Nubra", 34.688, 77.568, "cold_arid", "India"),
    ("Bikaner", 28.023, 73.312, "arid_hot", "India"),
    ("Mangalore", 12.914, 74.856, "tropical_wet", "India"),
    ("Patna", 25.612, 85.145, "semi_arid_hot", "India"),
    
    # === MIDDLE EAST (10 locations - extreme heat/arid) ===
    ("Dubai", 25.205, 55.271, "arid_hot", "UAE"),
    ("Riyadh", 24.714, 46.675, "arid_hot", "Saudi Arabia"),
    ("Abu_Dhabi", 24.454, 54.377, "arid_hot", "UAE"),
    ("Doha", 25.276, 51.520, "arid_hot", "Qatar"),
    ("Muscat", 23.588, 58.383, "arid_hot", "Oman"),
    ("Kuwait_City", 29.376, 47.977, "arid_hot", "Kuwait"),
    ("Amman", 31.949, 35.933, "semi_arid_hot", "Jordan"),
    ("Tehran", 35.689, 51.389, "semi_arid_hot", "Iran"),
    ("Baghdad", 33.313, 44.366, "arid_hot", "Iraq"),
    ("Jeddah", 21.543, 39.198, "arid_hot", "Saudi Arabia"),
    
    # === AFRICA (15 locations - diverse climates) ===
    ("Cairo", 30.044, 31.236, "arid_hot", "Egypt"),
    ("Nairobi", -1.286, 36.817, "tropical_moderate", "Kenya"),
    ("Cape_Town", -33.925, 18.424, "mediterranean", "South Africa"),
    ("Lagos", 6.524, 3.379, "tropical_humid", "Nigeria"),
    ("Johannesburg", -26.204, 28.047, "subtropical", "South Africa"),
    ("Casablanca", 33.573, -7.590, "mediterranean", "Morocco"),
    ("Addis_Ababa", 9.025, 38.747, "tropical_moderate", "Ethiopia"),
    ("Dar_es_Salaam", -6.793, 39.208, "tropical_humid", "Tanzania"),
    ("Accra", 5.614, -0.186, "tropical_humid", "Ghana"),
    ("Dakar", 14.716, -17.467, "semi_arid_hot", "Senegal"),
    ("Tunis", 36.807, 10.166, "mediterranean", "Tunisia"),
    ("Khartoum", 15.501, 32.560, "arid_hot", "Sudan"),
    ("Windhoek", -22.560, 17.084, "semi_arid_hot", "Namibia"),
    ("Maputo", -25.966, 32.573, "tropical_humid", "Mozambique"),
    ("Ouagadougou", 12.372, -1.525, "semi_arid_hot", "Burkina Faso"),

    # === SOUTHEAST ASIA (10 locations - tropical) ===
    ("Bangkok", 13.756, 100.502, "tropical_humid", "Thailand"),
    ("Singapore", 1.352, 103.820, "tropical_wet", "Singapore"),
    ("Jakarta", -6.175, 106.845, "tropical_wet", "Indonesia"),
    ("Manila", 14.600, 120.984, "tropical_humid", "Philippines"),
    ("Hanoi", 21.029, 105.853, "tropical_humid", "Vietnam"),
    ("Kuala_Lumpur", 3.139, 101.687, "tropical_wet", "Malaysia"),
    ("Phnom_Penh", 11.557, 104.917, "tropical_humid", "Cambodia"),
    ("Yangon", 16.871, 96.199, "tropical_humid", "Myanmar"),
    ("Ho_Chi_Minh", 10.823, 106.630, "tropical_humid", "Vietnam"),
    ("Bali", -8.340, 115.092, "tropical_wet", "Indonesia"),

    # === EUROPE / MEDITERRANEAN (15 locations) ===
    ("Madrid", 40.417, -3.704, "mediterranean", "Spain"),
    ("Rome", 41.903, 12.496, "mediterranean", "Italy"),
    ("Athens", 37.984, 23.728, "mediterranean", "Greece"),
    ("Lisbon", 38.722, -9.139, "mediterranean", "Portugal"),
    ("Marseille", 43.297, 5.382, "mediterranean", "France"),
    ("Berlin", 52.520, 13.405, "temperate", "Germany"),
    ("London", 51.507, -0.128, "temperate_oceanic", "UK"),
    ("Stockholm", 59.329, 18.069, "cold_temperate", "Sweden"),
    ("Oslo", 59.914, 10.752, "cold_temperate", "Norway"),
    ("Helsinki", 60.170, 24.941, "cold_temperate", "Finland"),
    ("Warsaw", 52.230, 21.012, "temperate", "Poland"),
    ("Bucharest", 44.427, 26.103, "temperate", "Romania"),
    ("Istanbul", 41.009, 28.978, "mediterranean", "Turkey"),
    ("Barcelona", 41.386, 2.168, "mediterranean", "Spain"),
    ("Seville", 37.389, -5.984, "mediterranean", "Spain"),

    # === AMERICAS (15 locations) ===
    ("Phoenix", 33.449, -112.074, "arid_hot", "USA"),
    ("Los_Angeles", 34.052, -118.244, "mediterranean", "USA"),
    ("Miami", 25.762, -80.192, "tropical_humid", "USA"),
    ("Denver", 39.739, -104.990, "semi_arid_cold", "USA"),
    ("Houston", 29.760, -95.370, "subtropical", "USA"),
    ("Santiago", -33.449, -70.669, "mediterranean", "Chile"),
    ("Sao_Paulo", -23.551, -46.634, "subtropical", "Brazil"),
    ("Mexico_City", 19.433, -99.133, "tropical_moderate", "Mexico"),
    ("Buenos_Aires", -34.604, -58.382, "subtropical", "Argentina"),
    ("Bogota", 4.711, -74.072, "tropical_moderate", "Colombia"),
    ("Lima", -12.046, -77.043, "arid_mild", "Peru"),
    ("Tucson", 32.222, -110.975, "arid_hot", "USA"),
    ("Las_Vegas", 36.169, -115.140, "arid_hot", "USA"),
    ("Atacama", -23.650, -68.200, "arid_cold", "Chile"),
    ("Brasilia", -15.780, -47.930, "tropical_moderate", "Brazil"),

    # === AUSTRALIA / OCEANIA (10 locations) ===
    ("Sydney", -33.869, 151.209, "subtropical", "Australia"),
    ("Melbourne", -37.814, 144.963, "temperate_oceanic", "Australia"),
    ("Brisbane", -27.468, 153.024, "subtropical", "Australia"),
    ("Perth", -31.953, 115.861, "mediterranean", "Australia"),
    ("Alice_Springs", -23.698, 133.881, "arid_hot", "Australia"),
    ("Darwin", -12.463, 130.846, "tropical_humid", "Australia"),
    ("Adelaide", -34.929, 138.601, "mediterranean", "Australia"),
    ("Townsville", -19.259, 146.817, "tropical_humid", "Australia"),
    ("Auckland", -36.849, 174.763, "temperate_oceanic", "New Zealand"),
    ("Fiji_Suva", -18.142, 178.442, "tropical_wet", "Fiji"),

    # === COLD / EXTREME (5 locations) ===
    ("Reykjavik", 64.147, -21.943, "subarctic", "Iceland"),
    ("Anchorage", 61.218, -149.900, "subarctic", "USA"),
    ("Ulaanbaatar", 47.921, 106.906, "cold_arid", "Mongolia"),
    ("Murmansk", 68.958, 33.090, "subarctic", "Russia"),
    ("Tromsø", 69.649, 18.956, "subarctic", "Norway"),
]

# NASA POWER API parameters
NASA_PARAMS = "T2M,T2M_MAX,T2M_MIN,RH2M,WS2M,ALLSKY_SFC_SW_DWN,ALLSKY_SFC_UV_INDEX,PRECTOTCORR"
START_YEAR = 2014
END_YEAR = 2024
BASE_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"


def fetch_location_data(name, lat, lon, climate_zone, country, retries=3):
    """Fetch monthly climate data from NASA POWER API for a single location."""
    params = {
        "parameters": NASA_PARAMS,
        "community": "RE",  # Renewable Energy community
        "longitude": lon,
        "latitude": lat,
        "start": START_YEAR,
        "end": END_YEAR,
        "format": "JSON"
    }
    
    for attempt in range(retries):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            
            properties = data["properties"]["parameter"]
            records = []
            
            # Process each month (skip annual entries ending with "13")
            for key in sorted(properties["T2M"].keys()):
                if key.endswith("13"):  # Skip annual averages
                    continue
                    
                year = int(key[:4])
                month = int(key[4:])
                
                record = {
                    "location": name,
                    "country": country,
                    "latitude": lat,
                    "longitude": lon,
                    "climate_zone": climate_zone,
                    "year": year,
                    "month": month,
                    "avg_temp_C": properties["T2M"].get(key, np.nan),
                    "max_temp_C": properties["T2M_MAX"].get(key, np.nan),
                    "min_temp_C": properties["T2M_MIN"].get(key, np.nan),
                    "humidity_pct": properties["RH2M"].get(key, np.nan),
                    "wind_speed_ms": properties["WS2M"].get(key, np.nan),
                    "ghi_kwh_m2_day": properties["ALLSKY_SFC_SW_DWN"].get(key, np.nan),
                    "uv_index": properties["ALLSKY_SFC_UV_INDEX"].get(key, np.nan),
                    "precipitation_mm_day": properties["PRECTOTCORR"].get(key, np.nan),
                }
                
                # Replace NASA fill values (-999) with NaN
                for col in record:
                    if isinstance(record[col], (int, float)) and record[col] == -999:
                        record[col] = np.nan
                
                records.append(record)
            
            return records
            
        except Exception as e:
            print(f"  Attempt {attempt+1}/{retries} failed for {name}: {e}")
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
    
    print(f"  FAILED to fetch data for {name} after {retries} attempts")
    return []


def compute_degradation_features(df):
    """
    Compute physics-informed solar panel degradation features from real climate data.
    Based on established PV degradation literature:
      - Arrhenius equation for temperature-dependent degradation
      - Humidity-induced corrosion (acetic acid formation)
      - UV photodegradation of encapsulant
      - Thermal cycling fatigue
      - Dust/soiling effects from wind + precipitation patterns
    """
    
    # --- Thermal stress factor (Arrhenius-based) ---
    # Higher temperatures accelerate chemical degradation exponentially
    # Ea = 0.7 eV (typical for PV encapsulant degradation)
    Ea = 0.7  # activation energy in eV
    k_B = 8.617e-5  # Boltzmann constant in eV/K
    T_ref = 25 + 273.15  # Reference temperature in Kelvin
    T_abs = df["avg_temp_C"] + 273.15
    df["thermal_stress"] = np.exp((Ea / k_B) * (1/T_ref - 1/T_abs))
    
    # --- Thermal cycling fatigue ---
    # Daily temperature swing causes mechanical stress (expansion/contraction)
    df["temp_range_C"] = df["max_temp_C"] - df["min_temp_C"]
    df["thermal_cycle_stress"] = (df["temp_range_C"] / 20.0) ** 1.5  # normalized
    
    # --- Humidity-induced degradation ---
    # Moisture ingress rate follows Fick's law; accelerated above 60% RH
    df["humidity_stress"] = np.where(
        df["humidity_pct"] > 60,
        ((df["humidity_pct"] - 60) / 40.0) ** 2 * df["thermal_stress"],
        0.01
    )
    
    # --- Damp heat index (combined temperature + humidity) ---
    # Industry standard IEC 61215 uses 85°C/85% RH as accelerated test
    df["damp_heat_index"] = (df["avg_temp_C"] / 85.0) * (df["humidity_pct"] / 85.0)
    
    # --- UV degradation ---
    # UV exposure causes yellowing of encapsulant (EVA), reducing transmittance
    df["uv_stress"] = df["uv_index"] / 10.0  # Normalized to 0-1 range
    
    # --- Cumulative irradiance exposure ---
    # Total solar energy dose (proxy for photon-induced degradation)
    df["irradiance_dose_kwh_m2_month"] = df["ghi_kwh_m2_day"] * 30  # approx monthly
    
    # --- Soiling/dust index ---
    # Low precipitation + moderate wind = high dust accumulation
    # High precipitation = cleaning effect
    df["dust_soiling_index"] = np.where(
        df["precipitation_mm_day"] < 1.0,
        (1 - df["precipitation_mm_day"]) * (1 + df["wind_speed_ms"] / 5.0),
        0.1
    )
    
    # --- Wind mechanical stress ---
    # High winds cause micro-cracking and mounting fatigue
    df["wind_stress"] = (df["wind_speed_ms"] / 10.0) ** 2
    
    # --- Combined degradation rate estimate (% per year) ---
    # Base degradation ~0.5%/year (industry standard for quality panels)
    # Stress multipliers based on environmental conditions
    base_rate = 0.5
    df["degradation_rate_pct_yr"] = base_rate * (
        0.25 * df["thermal_stress"].clip(0, 5) +
        0.15 * df["thermal_cycle_stress"].clip(0, 3) +
        0.20 * df["humidity_stress"].clip(0, 3) +
        0.15 * df["damp_heat_index"].clip(0, 2) +
        0.10 * df["uv_stress"].clip(0, 2) +
        0.10 * df["dust_soiling_index"].clip(0, 3) +
        0.05 * df["wind_stress"].clip(0, 2)
    )
    
    # Clip to realistic range [0.3%, 3.5%] based on literature
    df["degradation_rate_pct_yr"] = df["degradation_rate_pct_yr"].clip(0.3, 3.5)
    
    return df


def simulate_panel_configurations(df):
    """
    For each location's climate data, simulate multiple solar panel configurations
    to enrich the dataset with system-level features.
    """
    np.random.seed(42)
    
    module_types = ["Mono-Si", "Poly-Si", "CdTe", "CIGS"]
    module_efficiency = {"Mono-Si": 0.21, "Poly-Si": 0.17, "CdTe": 0.18, "CIGS": 0.15}
    module_degradation_modifier = {"Mono-Si": 1.0, "Poly-Si": 1.08, "CdTe": 0.95, "CIGS": 1.12}
    
    mounting_types = ["fixed_tilt", "rooftop", "open_rack"]
    mounting_temp_modifier = {"fixed_tilt": 1.0, "rooftop": 1.15, "open_rack": 0.90}
    
    enriched_rows = []
    
    locations = df["location"].unique()
    
    for loc in locations:
        loc_data = df[df["location"] == loc].copy()
        
        # For each location, simulate 3-4 panel configurations
        n_configs = np.random.choice([3, 4])
        
        for config_id in range(n_configs):
            module = np.random.choice(module_types)
            mounting = np.random.choice(mounting_types)
            tilt = np.random.uniform(10, 45)
            panel_wattage = np.random.choice([250, 300, 350, 400, 450, 500, 550])
            installation_year = np.random.choice(range(2014, 2020))
            
            for _, row in loc_data.iterrows():
                panel_age = row["year"] - installation_year
                if panel_age < 0:
                    continue
                
                new_row = row.to_dict()
                new_row["module_type"] = module
                new_row["module_efficiency"] = module_efficiency[module]
                new_row["mounting_type"] = mounting
                new_row["tilt_angle"] = round(tilt, 1)
                new_row["panel_wattage_W"] = panel_wattage
                new_row["installation_year"] = installation_year
                new_row["panel_age_years"] = panel_age
                
                # Adjust degradation based on panel type and mounting
                deg_modifier = (
                    module_degradation_modifier[module] *
                    mounting_temp_modifier[mounting]
                )
                new_row["degradation_rate_pct_yr"] *= deg_modifier
                new_row["degradation_rate_pct_yr"] = min(max(
                    new_row["degradation_rate_pct_yr"], 0.3
                ), 3.5)
                
                # Cumulative degradation
                cumulative_deg = new_row["degradation_rate_pct_yr"] * panel_age / 100.0
                cumulative_deg = min(cumulative_deg, 0.80)  # Max 80% degradation
                
                # Current power output
                new_row["current_efficiency"] = module_efficiency[module] * (1 - cumulative_deg)
                new_row["dc_power_kw"] = (
                    panel_wattage / 1000.0 *
                    (1 - cumulative_deg) *
                    row["ghi_kwh_m2_day"] / 5.0  # Normalized to peak sun hours
                )
                new_row["ac_power_kw"] = new_row["dc_power_kw"] * 0.96  # Inverter efficiency
                new_row["daily_yield_kwh"] = new_row["dc_power_kw"] * 5.0  # Approx 5 peak sun hours
                
                # Module temperature (NOCT-based estimation)
                NOCT = 45 if mounting == "rooftop" else (42 if mounting == "fixed_tilt" else 40)
                new_row["module_temp_C"] = row["avg_temp_C"] + (NOCT - 20) * row["ghi_kwh_m2_day"] / 0.8
                
                # Remaining Useful Life (RUL) estimation
                # Assuming panel end-of-life at 80% of original power (20% degradation)
                typical_lifespan = 25.0  # Industry standard warranty period
                if new_row["degradation_rate_pct_yr"] > 0:
                    estimated_total_life = min(20.0 / new_row["degradation_rate_pct_yr"], 40.0)
                else:
                    estimated_total_life = 40.0
                
                new_row["estimated_total_life_years"] = round(estimated_total_life, 2)
                new_row["remaining_useful_life_years"] = round(
                    max(estimated_total_life - panel_age, 0), 2
                )
                
                # Performance Ratio
                if row["ghi_kwh_m2_day"] > 0:
                    new_row["performance_ratio"] = (1 - cumulative_deg) * 0.85  # Base PR = 0.85
                else:
                    new_row["performance_ratio"] = 0
                
                enriched_rows.append(new_row)
    
    return pd.DataFrame(enriched_rows)


def main():
    os.makedirs("ml/data", exist_ok=True)
    
    print("=" * 70)
    print("  SOLAR SAATHI — NASA POWER API Data Acquisition")
    print("  Fetching AUTHENTIC climate data for 120 global locations")
    print("  Period: 2014-2024 (10+ years of monthly data)")
    print("=" * 70)
    
    all_records = []
    total = len(LOCATIONS)
    
    for i, (name, lat, lon, climate, country) in enumerate(LOCATIONS):
        print(f"\n[{i+1}/{total}] Fetching: {name} ({country}) | {lat:.3f}, {lon:.3f} | {climate}")
        records = fetch_location_data(name, lat, lon, climate, country)
        all_records.extend(records)
        print(f"  ✓ Got {len(records)} monthly records")
        
        # Rate limiting — NASA API recommends max 10 requests/minute
        if (i + 1) % 8 == 0:
            print("  ⏳ Rate limit pause (6 seconds)...")
            time.sleep(6)
        else:
            time.sleep(1.5)
    
    # Create raw DataFrame
    df_raw = pd.DataFrame(all_records)
    
    # Replace -999 fill values
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
    df_raw[numeric_cols] = df_raw[numeric_cols].replace(-999, np.nan)
    
    # Drop rows with too many missing values
    df_raw = df_raw.dropna(thresh=10)
    
    # Fill remaining NaN with forward fill within each location
    df_raw = df_raw.sort_values(["location", "year", "month"])
    df_raw[numeric_cols] = df_raw.groupby("location")[
        numeric_cols.intersection(df_raw.columns)
    ].transform(lambda x: x.ffill().bfill())
    
    # Save raw climate data
    raw_path = "ml/data/nasa_climate_data_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    print(f"\n✓ Raw climate data saved: {raw_path} ({len(df_raw)} records)")
    
    # Compute degradation features
    print("\n➜ Computing physics-informed degradation features...")
    df_features = compute_degradation_features(df_raw.copy())
    
    # Simulate panel configurations (enriches dataset ~3-4x)
    print("➜ Simulating diverse panel configurations...")
    df_final = simulate_panel_configurations(df_features)
    
    # Save final dataset
    final_path = "ml/data/solar_saathi_dataset.csv"
    df_final.to_csv(final_path, index=False)
    
    print(f"\n{'=' * 70}")
    print(f"  DATASET COMPLETE")
    print(f"  Total records: {len(df_final):,}")
    print(f"  Locations: {df_final['location'].nunique()}")
    print(f"  Countries: {df_final['country'].nunique()}")
    print(f"  Time span: {df_final['year'].min()}-{df_final['year'].max()}")
    print(f"  Features: {len(df_final.columns)} columns")
    print(f"  Saved to: {final_path}")
    print(f"{'=' * 70}")
    
    # Print column info
    print("\n📊 Dataset Columns:")
    for col in df_final.columns:
        dtype = df_final[col].dtype
        if dtype in ['float64', 'int64']:
            print(f"  {col}: {dtype} | range [{df_final[col].min():.3f}, {df_final[col].max():.3f}]")
        else:
            print(f"  {col}: {dtype} | {df_final[col].nunique()} unique values")


if __name__ == "__main__":
    main()
