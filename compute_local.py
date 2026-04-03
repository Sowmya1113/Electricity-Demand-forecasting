import os
import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import NASAPowerClient

LOCALITY_PROFILES = {
    "Mumbai, MH": { "base_demand": 4500, "lat": 19.08, "lng": 72.88 },
    "Delhi, DL": { "base_demand": 12000, "lat": 28.61, "lng": 77.21 },
    "Bengaluru, KA": { "base_demand": 8000, "lat": 12.97, "lng": 77.59 },
    "Chennai, TN": { "base_demand": 6000, "lat": 13.08, "lng": 80.27 },
    "Kolkata, WB": { "base_demand": 5500, "lat": 22.57, "lng": 88.36 },
    "Hyderabad, TS": { "base_demand": 7000, "lat": 17.39, "lng": 78.49 },
    "Ahmedabad, GJ": { "base_demand": 5000, "lat": 23.03, "lng": 72.59 },
    "Pune, MH": { "base_demand": 5800, "lat": 18.52, "lng": 73.86 },
    "Jaipur, RJ": { "base_demand": 4200, "lat": 26.91, "lng": 75.79 },
    "Lucknow, UP": { "base_demand": 4000, "lat": 26.85, "lng": 80.95 },
    "Chandigarh, CH": { "base_demand": 1800, "lat": 30.74, "lng": 76.79 },
    "Kochi, KL": { "base_demand": 2200, "lat": 9.93, "lng": 76.26 },
    "Indore, MP": { "base_demand": 2800, "lat": 22.72, "lng": 75.86 },
    "Bhopal, MP": { "base_demand": 2500, "lat": 23.26, "lng": 77.41 },
    "Patna, BR": { "base_demand": 2500, "lat": 25.6, "lng": 85.1 },
    "Surat, GJ": { "base_demand": 3500, "lat": 21.17, "lng": 72.83 },
    "Vadodara, GJ": { "base_demand": 3000, "lat": 22.31, "lng": 73.18 },
    "Rajkot, GJ": { "base_demand": 2200, "lat": 22.3, "lng": 70.8 },
    "Nagpur, MH": { "base_demand": 3200, "lat": 21.14, "lng": 79.08 },
    "Coimbatore, TN": { "base_demand": 2800, "lat": 11.02, "lng": 76.96 },
}

def generate_actual_profiles():
    # 1. Load actual national demand
    csv_path = os.path.join(os.path.dirname(__file__), "actual_demand.csv")
    df_demand = pd.read_csv(csv_path)
    df_demand["datetime"] = pd.to_datetime(df_demand["datetime"])
    df_demand["date"] = df_demand["datetime"].dt.normalize()
    daily_demand = df_demand.groupby("date")["demand_mw"].mean().reset_index()
    
    # Restrict to string year 2023 for faster API load
    start_date = "20230101"
    end_date = "20231231"
    
    mask = (daily_demand["date"] >= "2023-01-01") & (daily_demand["date"] <= "2023-12-31")
    daily_demand = daily_demand[mask]

    client = NASAPowerClient()
    new_profiles = {}
    
    for city, prof in LOCALITY_PROFILES.items():
        print(f"Fetching actual weather for {city}...")
        try:
            weather_df = client.fetch_daily_data(prof["lat"], prof["lng"], start_date, end_date)
            if weather_df.empty:
                print(f"Empty weather for {city}, using defaults...")
                new_profiles[city] = prof
                continue
                
            weather_df = weather_df.reset_index()
            merged = pd.merge(daily_demand, weather_df, left_on="date", right_on="datetime")
            merged = merged.dropna(subset=["demand_mw", "temperature_2m", "relative_humidity", "wind_speed_10m"])
            
            X = merged[["temperature_2m", "relative_humidity", "wind_speed_10m"]]
            y = merged["demand_mw"]
            
            model = LinearRegression()
            model.fit(X, y)
            
            nat_base = model.intercept_
            scale = prof["base_demand"] / (nat_base if nat_base > 0 else 1.0)
            
            # Predict actual local coefficients based on local weather vs demand patterns
            temp_coef = int(model.coef_[0] * scale)
            humid_coef = int(model.coef_[1] * scale)
            wind_coef = int(model.coef_[2] * scale)
            
            new_profiles[city] = {
                "base_demand": prof["base_demand"],
                "temp_coef": temp_coef,
                "humid_coef": humid_coef,
                "wind_coef": wind_coef,
                "lat": prof["lat"],
                "lng": prof["lng"]
            }
        except Exception as e:
            print(f"Failed {city}: {e}")
            new_profiles[city] = prof
            
        time.sleep(0.5) # rate limiting
        
    print("FINISHED")
    import pprint
    pprint.pprint(new_profiles)
    
if __name__ == "__main__":
    generate_actual_profiles()
