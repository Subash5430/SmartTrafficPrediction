import pandas as pd
import numpy as np

# ---------------- INDIA AREA → LAT/LNG MAP ----------------
AREA_TO_COORDS = {
    "Residential areas": (12.9716, 77.5946),
    "Office areas": (17.3850, 78.4867),
    "Recreational areas": (28.6139, 77.2090),
    "Industrial areas": (19.0760, 72.8777),
    "Market areas": (13.0827, 80.2707)
}

# ---------------- PROCESS US DATASET ----------------
us_file = "../dataset/us-accidents/US_Accidents_March23.csv"
us_chunks = []

for chunk in pd.read_csv(
    us_file,
    usecols=["Start_Lat", "Start_Lng", "Start_Time", "Severity"],
    chunksize=200_000
):
    chunk["Start_Time"] = pd.to_datetime(chunk["Start_Time"], errors="coerce")

    chunk["hour"] = chunk["Start_Time"].dt.hour
    chunk["day"] = chunk["Start_Time"].dt.weekday + 1

    chunk.rename(columns={
        "Start_Lat": "latitude",
        "Start_Lng": "longitude"
    }, inplace=True)

    chunk["traffic_volume"] = chunk["Severity"] * 200
    chunk["accident"] = (chunk["Severity"] >= 3).astype(int)
    chunk["country"] = "US"

    final = chunk[[
        "latitude", "longitude", "hour", "day",
        "traffic_volume", "accident", "country"
    ]].dropna()

    final = final.sample(n=min(3000, len(final)), random_state=42)
    us_chunks.append(final)

us_df = pd.concat(us_chunks, ignore_index=True)

# ---------------- PROCESS INDIAN DATASET ----------------
india_file = "../dataset/india-accidents/Road.csv"
india_raw = pd.read_csv(india_file)

# Filter usable areas
india_raw = india_raw[india_raw["Area_accident_occured"].isin(AREA_TO_COORDS.keys())]

# Extract hour from Time
india_raw["hour"] = pd.to_datetime(india_raw["Time"], errors="coerce").dt.hour

# Map day of week to 1–7
day_map = {
    "Monday": 1, "Tuesday": 2, "Wednesday": 3,
    "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7
}
india_raw["day"] = india_raw["Day_of_week"].map(day_map)

# Map area to coordinates
india_raw["latitude"] = india_raw["Area_accident_occured"].map(
    lambda x: AREA_TO_COORDS[x][0]
)
india_raw["longitude"] = india_raw["Area_accident_occured"].map(
    lambda x: AREA_TO_COORDS[x][1]
)

# Traffic proxy
india_raw["traffic_volume"] = india_raw["Number_of_vehicles_involved"] * 150

# Binary accident label
india_raw["accident"] = india_raw["Accident_severity"].apply(
    lambda x: 1 if x in ["Serious injury", "Fatal injury"] else 0
)

india_raw["country"] = "INDIA"

india_df = india_raw[[
    "latitude", "longitude", "hour", "day",
    "traffic_volume", "accident", "country"
]].dropna()

# ---------------- COMBINE ----------------
final_df = pd.concat([us_df, india_df], ignore_index=True)

final_df.to_csv("processed_combined_accidents.csv", index=False)

print("Combined dataset created")
print("US records:", len(us_df))
print("India records:", len(india_df))
print("Total records:", len(final_df))
