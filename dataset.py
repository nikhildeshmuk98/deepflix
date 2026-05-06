import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker for Indian locale
fake = Faker('en_IN')

# --- Define Indian-specific data for features ---
indian_airlines = [
    "IndiGo", "Air India", "Vistara", "SpiceJet", "Akasa Air", "Alliance Air", "Star Air", "FlyBig"
]

aircraft_types = [
    "Airbus A320", "Boeing 737", "ATR 72", "Airbus A321", "Boeing 787", "Embraer E190", "Dornier Do 228"
]

indian_airports = {
    "DEL": {"name": "Indira Gandhi International Airport", "city": "Delhi", "elevation_ft": 777},
    "BOM": {"name": "Chhatrapati Shivaji Maharaj International Airport", "city": "Mumbai", "elevation_ft": 39},
    "BLR": {"name": "Kempegowda International Airport", "city": "Bengaluru", "elevation_ft": 3000},
    "MAA": {"name": "Chennai International Airport", "city": "Chennai", "elevation_ft": 33},
    "CCU": {"name": "Netaji Subhas Chandra Bose International Airport", "city": "Kolkata", "elevation_ft": 18},
    "HYD": {"name": "Rajiv Gandhi International Airport", "city": "Hyderabad", "elevation_ft": 2024},
    "AMD": {"name": "Sardar Vallabhbhai Patel International Airport", "city": "Ahmedabad", "elevation_ft": 192},
    "PNQ": {"name": "Pune Airport", "city": "Pune", "elevation_ft": 1863},
    "COK": {"name": "Cochin International Airport", "city": "Kochi", "elevation_ft": 30},
    "GOI": {"name": "Dabolim Airport", "city": "Goa", "elevation_ft": 189},
    "IXB": {"name": "Bagdogra International Airport", "city": "Siliguri", "elevation_ft": 412},
    "BBI": {"name": "Biju Patnaik International Airport", "city": "Bhubaneswar", "elevation_ft": 85},
    "GAU": {"name": "Lokpriya Gopinath Bordoloi International Airport", "city": "Guwahati", "elevation_ft": 158},
    "ATQ": {"name": "Sri Guru Ram Dass Jee International Airport", "city": "Amritsar", "elevation_ft": 765}
}

flight_phases = ["Takeoff", "En_Route", "Landing", "Taxi", "Parked"]
seasons = ["Monsoon", "Winter", "Summer", "Post-Monsoon"]
weather_conditions = ["Clear", "Cloudy", "Rain", "Fog", "Thunderstorms", "Dust_Storm"]
turbulence_severities = ["None", "Light", "Moderate", "Severe"]
route_complexities = ["Direct", "Multiple_Waypoints", "Mountainous_Terrain"]
air_traffic_densities = ["Low", "Medium", "High"]
accident_severities = ["Minor_Incident", "Serious_Incident", "Fatal_Accident"]
cause_categories = ["Human_Error", "Mechanical_Failure", "Weather", "External_Factors", "Bird_Strike"]

# --- Function to generate a single flight record ---
def generate_flight_record(flight_id):
    airline = random.choice(indian_airlines)
    aircraft_type = random.choice(aircraft_types)
    aircraft_age = random.randint(1, 25)
    flight_phase = random.choice(flight_phases)

    dep_airport_code = random.choice(list(indian_airports.keys()))
    arr_airport_code = random.choice([code for code in indian_airports.keys() if code != dep_airport_code])

    dep_airport_elevation = indian_airports[dep_airport_code]["elevation_ft"]
    arr_airport_elevation = indian_airports[arr_airport_code]["elevation_ft"]

    flight_duration = round(random.uniform(0.5, 5.0), 2)
    last_maintenance_hours = random.randint(100, 2000)
    engine_hours_since_overhaul = random.randint(500, 8000)
    
    pilot_experience_hours = random.randint(1000, 20000)
    copilot_experience_hours = random.randint(500, pilot_experience_hours - 500) if pilot_experience_hours > 1000 else random.randint(100, 500)
    
    num_crew = random.randint(4, 12)
    total_onboard = random.randint(50, 400)
    cargo_weight_kg = random.randint(0, 10000)

    flight_date = fake.date_between(start_date='-5y', end_date='today')
    flight_time_utc = fake.time(pattern='%H:%M', end_datetime=None)

    # Determine season based on month (simplified for Indian context)
    month = flight_date.month
    if 6 <= month <= 9:
        season = "Monsoon"
    elif 10 <= month <= 2:
        season = "Winter"
    else:
        season = "Summer"
    
    # Adjust weather conditions based on season
    if season == "Monsoon":
        weather_cond = random.choices(["Rain", "Thunderstorms", "Cloudy"], weights=[0.4, 0.2, 0.4], k=1)[0]
        precipitation = round(random.uniform(0.1, 15.0), 2)
    elif season == "Winter":
        weather_cond = random.choices(["Fog", "Clear", "Cloudy"], weights=[0.3, 0.4, 0.3], k=1)[0]
        precipitation = 0 if weather_cond != "Fog" else round(random.uniform(0.0, 0.1), 2) # Light fog
    else: # Summer/Post-Monsoon
        weather_cond = random.choices(["Clear", "Cloudy", "Dust_Storm"], weights=[0.6, 0.3, 0.1], k=1)[0]
        precipitation = 0 if weather_cond != "Dust_Storm" else 0 # Dust storm doesn't have precipitation

    visibility_km = round(random.uniform(0.5, 10.0), 2)
    wind_speed_knots = random.randint(0, 40)
    wind_direction_degrees = random.randint(0, 359)
    temperature_celsius = round(random.uniform(10, 45), 1)
    
    turbulence_severity = random.choice(turbulence_severities)
    
    route_complexity = random.choice(route_complexities)
    air_traffic_density = random.choice(air_traffic_densities)

    # --- Accident Simulation (imbalance handling) ---
    accident = 0
    accident_severity = None
    fatalities = 0
    injuries = 0
    aircraft_damage = None
    cause_category = None

    # Introduce accidents rarely (e.g., 2% of flights)
    if random.random() < 0.02: # Adjust this probability to control number of accidents
        accident = 1
        accident_severity = random.choices(accident_severities, weights=[0.5, 0.3, 0.2], k=1)[0] # More minor incidents
        aircraft_damage = random.choices(["Minor", "Substantial", "Destroyed"], weights=[0.5, 0.3, 0.2], k=1)[0]
        cause_category = random.choices(cause_categories, weights=[0.4, 0.3, 0.2, 0.05, 0.05], k=1)[0]
        
        if accident_severity == "Fatal_Accident":
            fatalities = random.randint(1, total_onboard)
            injuries = random.randint(0, total_onboard - fatalities) if total_onboard > fatalities else 0
        elif accident_severity == "Serious_Incident":
            fatalities = 0
            injuries = random.randint(1, int(total_onboard * 0.2)) # Up to 20% injured
        else: # Minor_Incident
            fatalities = 0
            injuries = random.randint(0, 5)

    record = {
        "Flight_ID": f"F{flight_date.strftime('%Y%m%d')}{flight_id:04d}",
        "Airline": airline,
        "Aircraft_Type": aircraft_type,
        "Aircraft_Age_Years": aircraft_age,
        "Flight_Phase": flight_phase,
        "Departure_Airport_Code": dep_airport_code,
        "Arrival_Airport_Code": arr_airport_code,
        "Flight_Duration_Hours": flight_duration,
        "Last_Maintenance_Hours": last_maintenance_hours,
        "Engine_Hours_Since_Overhaul": engine_hours_since_overhaul,
        "Pilot_Experience_Hours": pilot_experience_hours,
        "CoPilot_Experience_Hours": copilot_experience_hours,
        "Number_of_Crew": num_crew,
        "Total_Onboard": total_onboard,
        "Cargo_Weight_KG": cargo_weight_kg,
        "Date": flight_date.strftime('%Y-%m-%d'),
        "Time_UTC": flight_time_utc,
        "Season": season,
        "Visibility_KM": visibility_km,
        "Wind_Speed_KNOTS": wind_speed_knots,
        "Wind_Direction_Degrees": wind_direction_degrees,
        "Temperature_Celsius": temperature_celsius,
        "Precipitation_MM": precipitation,
        "Weather_Condition": weather_cond,
        "Turbulence_Severity": turbulence_severity,
        "Departure_Airport_Elevation_FT": dep_airport_elevation,
        "Arrival_Airport_Elevation_FT": arr_airport_elevation,
        "Route_Complexity": route_complexity,
        "Air_Traffic_Density": air_traffic_density,
        "Accident": accident,
        "Accident_Severity": accident_severity,
        "Fatalities": fatalities,
        "Injuries": injuries,
        "Aircraft_Damage": aircraft_damage,
        "Cause_Category": cause_category
    }
    return record

# --- Generate the dataset ---
num_rows = 2500  # You asked for 2000+ rows
data = []
for i in range(num_rows):
    data.append(generate_flight_record(i + 1))

df = pd.DataFrame(data)

# --- Save to CSV ---
output_filename = "flight_accidents_india_synthetic.csv"
df.to_csv(output_filename, index=False)

print(f"Dataset generated successfully with {len(df)} rows and saved to {output_filename}")
print("\nFirst 5 rows of the generated dataset:")
print(df.head())
print("\nAccident distribution:")
print(df['Accident'].value_counts())