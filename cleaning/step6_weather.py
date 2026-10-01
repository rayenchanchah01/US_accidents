"""Step 6 - Weather_Condition (108 labels) -> Weather_Group (8 groups).

The weather vocabulary changed in 2019: "Clear" and "Scattered Clouds" stop being used and "Fair"
and "Cloudy" replace them. So the raw label partly tells the model which year a row comes from,
not what the weather was. Grouping removes that artefact and turns 108 sparse labels into 8.

When a label has several parts ("Light Rain / Windy") the strongest one wins, in this order:
thunderstorm > snow/ice > rain > fog/haze > windy > cloudy > clear.
"""
import pandas as pd

GROUP_KEYWORDS = [
    ("Heavy rain/Thunderstorm", ["thunder", "t-storm", "tornado", "funnel", "squall", "hail", "heavy rain"]),
    ("Snow/Ice", ["snow", "sleet", "ice", "freezing", "wintry"]),
    ("Rain", ["rain", "drizzle", "shower"]),
    ("Fog/Haze/Smoke", ["fog", "mist", "haze", "smoke", "dust", "sand", "ash"]),
    ("Windy", ["windy"]),
    ("Cloudy", ["cloud", "overcast"]),
    ("Clear", ["fair", "clear"]),
]


def weather_group(label):
    if pd.isna(label) or label == "N/A Precipitation":
        return "Unknown"
    text = label.lower()
    for group, keywords in GROUP_KEYWORDS:
        if any(k in text for k in keywords):
            return group
    raise ValueError(f"unmapped weather label: {label!r}")


def run(df, log):
    before = df.shape
    mapping = {label: weather_group(label) for label in df["Weather_Condition"].dropna().unique()}
    df = df.assign(Weather_Group=df["Weather_Condition"].map(mapping).fillna("Unknown"))
    df = df.drop(columns="Weather_Condition")
    log.notes["weather_mapping"] = mapping
    log.add("6. Weather grouping", "Weather_Condition (108 labels) -> Weather_Group (8 groups)", before, df.shape)
    return df
