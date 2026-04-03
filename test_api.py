import sys
import os

# Add the directory containing data_pipeline.py to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import EmberEnergyClient

client = EmberEnergyClient()

# Test if Ember has state-level data
for code in ["IND", "IN-MH", "IN-DL", "Maharashtra", "MH"]:
    df = client.fetch_generation_mix(code, start_date="2023-01-01", end_date="2023-12-31")
    if not df.empty:
        print(f"Data found for {code}!")
    else:
        print(f"No data for {code}.")
