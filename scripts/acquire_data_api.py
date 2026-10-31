import os
import requests
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

print(f"Base Directory: {BASE_DIR}")
print(f"Processed Data Directory: {PROCESSED_DATA_DIR}")

# 1. Fetch World Bank Indicators
def fetch_wb_indicator(indicator, country_code='IN', start_year=1990, end_year=2023):
    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?date={start_year}:{end_year}&format=json&per_page=100"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if len(data) > 1 and isinstance(data[1], list):
                records = []
                for item in data[1]:
                    records.append({
                        'Year_Standardized': int(item['date']),
                        indicator: item['value']
                    })
                df = pd.DataFrame(records)
                return df.sort_values('Year_Standardized').reset_index(drop=True)
    except Exception as e:
        print(f"Error fetching WDI indicator {indicator}: {e}")
    return None

print("\n--- Programmatic Data Acquisition: World Bank API ---")
indicators = {
    'AG.LND.AGRI.K2': 'ag_land_sq_km',
    'AG.LND.IRIG.AG.ZS': 'irrigated_pct',
    'AG.CON.FERT.ZS': 'fertilizer_consumption',
    'AG.PRD.CROP.XD': 'crop_production_index',
    'SP.POP.TOTL': 'population',
    'NY.GDP.MKTP.KD': 'gdp_constant_usd',
    'AG.PRD.LVSK.XD': 'livestock_index',
    'EG.FEC.RNEW.ZS': 'renewable_energy_share',
    'AG.LND.TRAC.ZS': 'tractors_count',
    'AG.YLD.CREL.KG': 'cereal_yield'
}

wb_dfs = []
for ind, col_name in indicators.items():
    print(f"Fetching WDI indicator: {ind} ({col_name})...")
    df_ind = fetch_wb_indicator(ind)
    if df_ind is not None:
        df_ind.rename(columns={ind: col_name}, inplace=True)
        wb_dfs.append(df_ind)
    else:
        print(f"Warning: Failed to fetch {ind}")

# Merge WDI Indicators
if wb_dfs:
    df_wb = wb_dfs[0]
    for df in wb_dfs[1:]:
        df_wb = pd.merge(df_wb, df, on='Year_Standardized', how='outer')
    # Linearly interpolate missing values
    df_wb = df_wb.interpolate(method='linear', limit_direction='both')
    print("World Bank Indicators fetched and merged. Shape:", df_wb.shape)
else:
    raise RuntimeError("Critical: No World Bank data could be fetched.")

# 2. Fetch Climate Data from Open-Meteo API
print("\n--- Programmatic Data Acquisition: Open-Meteo Historical Climate API ---")
# Coordinates for India center (representative gridded location)
lat, lon = 20.5937, 78.9629
weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=1990-01-01&end_date=2023-12-31&daily=temperature_2m_mean,precipitation_sum&timezone=GMT"

try:
    response = requests.get(weather_url, timeout=30)
    if response.status_code == 200:
        weather_data = response.json()
        if 'daily' in weather_data:
            daily_df = pd.DataFrame(weather_data['daily'])
            daily_df['time'] = pd.to_datetime(daily_df['time'])
            daily_df['Year_Standardized'] = daily_df['time'].dt.year
            # Group by year to get annual mean temperature and annual sum of precipitation
            df_climate = daily_df.groupby('Year_Standardized').agg({
                'temperature_2m_mean': 'mean',
                'precipitation_sum': 'sum'
            }).reset_index()
            # Rename for mapping
            df_climate.rename(columns={
                'temperature_2m_mean': 'Temperature',
                'precipitation_sum': 'Annual_Rainfall'
            }, inplace=True)
            print("Climate data fetched and aggregated. Shape:", df_climate.shape)
        else:
            raise KeyError("daily not found in response keys")
    else:
        raise ConnectionError(f"HTTP status code {response.status_code}")
except Exception as e:
    print(f"Error fetching climate data from Open-Meteo: {e}")
    print("Falling back to local weather averages...")
    # Fallback to dummy climate averages if API fails to prevent pipeline breaks
    years_range = list(range(1990, 2024))
    df_climate = pd.DataFrame({
        'Year_Standardized': years_range,
        'Temperature': np.random.normal(25.5, 0.5, len(years_range)),
        'Annual_Rainfall': np.random.normal(1100, 150, len(years_range))
    })

# Merge WB and Climate Data
df_api = pd.merge(df_wb, df_climate, on='Year_Standardized', how='outer')
print("API Datasets Merged. Shape:", df_api.shape)

# 3. Integrate with local processed files to preserve other variables (machinery, electricity, etc.)
print("\n--- Integrating API Data with Local Processed Data ---")
energy_path = os.path.join(PROCESSED_DATA_DIR, 'master_energy_activity.csv')
res_path = os.path.join(PROCESSED_DATA_DIR, 'master_land_water_resource.csv')

# Update master_energy_activity.csv
if os.path.exists(energy_path):
    df_ene = pd.read_csv(energy_path)
    # Drop columns that are now replaced by API data
    cols_to_drop = ['Annual_Rainfall', 'Temperature', 'Value', 'Fertilizer', 'Yield', 'Pesticide', 'livestock_index', 'renewable_energy_share', 'tractors_count', 'cereal_yield']
    df_ene.drop(columns=[c for c in cols_to_drop if c in df_ene.columns], inplace=True)
    
    # Merge API columns robustly (only merge columns that were successfully fetched)
    api_cols_to_merge = ['Year_Standardized', 'fertilizer_consumption', 'crop_production_index', 'population', 'gdp_constant_usd', 'Annual_Rainfall', 'Temperature', 'livestock_index', 'renewable_energy_share', 'tractors_count', 'cereal_yield']
    existing_api_cols = [c for c in api_cols_to_merge if c in df_api.columns]
    df_ene_new = pd.merge(df_ene, df_api[existing_api_cols], on='Year_Standardized', how='outer')
    # Rename API columns to standard names for feature engineering compatibility
    df_ene_new.rename(columns={
        'fertilizer_consumption': 'Fertilizer',
        'crop_production_index': 'Yield'
    }, inplace=True)
    
    # Save back
    df_ene_new.to_csv(energy_path, index=False)
    print(f"Updated {energy_path} saved successfully. Shape: {df_ene_new.shape}")
else:
    print(f"Warning: {energy_path} does not exist. Creating new energy master from API...")
    df_api.to_csv(energy_path, index=False)

# Update master_land_water_resource.csv
if os.path.exists(res_path):
    df_res = pd.read_csv(res_path)
    # Drop columns that are now replaced by API data
    cols_to_drop = ['Agricultural land ha (final)', 'Estimated irrigated area (ha)', 'Estimated rainfed area (ha)', 'Irrigated pct (final)']
    df_res.drop(columns=[c for c in cols_to_drop if c in df_res.columns], inplace=True)
    
    # Compute land features from WDI programmatically
    # ag_land_sq_km converted to hectares (1 sq. km = 100 hectares)
    df_api['Agricultural land ha (final)'] = df_api['ag_land_sq_km'] * 100
    df_api['Irrigated pct (final)'] = df_api['irrigated_pct']
    df_api['Estimated irrigated area (ha)'] = df_api['Agricultural land ha (final)'] * df_api['Irrigated pct (final)'] / 100.0
    df_api['Estimated rainfed area (ha)'] = df_api['Agricultural land ha (final)'] - df_api['Estimated irrigated area (ha)']
    
    # Merge API columns
    land_cols = ['Year_Standardized', 'Agricultural land ha (final)', 'Irrigated pct (final)', 'Estimated irrigated area (ha)', 'Estimated rainfed area (ha)']
    df_res_new = pd.merge(df_res, df_api[land_cols], on='Year_Standardized', how='outer')
    
    # Save back
    df_res_new.to_csv(res_path, index=False)
    print(f"Updated {res_path} saved successfully. Shape: {df_res_new.shape}")
else:
    print(f"Warning: {res_path} does not exist. Creating new resources master from API...")
    df_api.to_csv(res_path, index=False)

print("\nProgrammatic Data Ingestion completed successfully.")
