"""Step 4 - Missing values.

| Column                         | Missing        | Action                        |
|--------------------------------|----------------|-------------------------------|
| City                           | 19 rows        | drop the rows                 |
| Sunrise_Sunset                 | ~1.5K (0.3%)   | fill "Unknown"                |
| Street                         | ~0.7K (0.1%)   | leave empty, strip spaces     |
| Weather_Condition              | 2.2%           | "Unknown" group in step 6     |
| Temperature, Humidity,         | 1.8-7.4%       | leave empty (NaN)             |
| Pressure, Visibility, Wind_Speed                |                               |

Why
- Sunrise_Sunset: 11% of the rows missing it are Severity 4 (vs 2.6% overall). Dropping them would
  remove ~165 rows of our rarest class, so we keep them as "Unknown".
- Numeric weather: filling with the median must be fitted on the training set only (inside the
  model Pipeline, Phase 9). Filling it here would let test data leak into training.
- Street: only used later to build is_highway; an empty street is simply "not a highway".
"""


def run(df, log):
    before = df.shape
    df = df.dropna(subset=["City"])
    log.add("4a. Missing City", "dropped rows with no City", before, df.shape)

    before = df.shape
    df = df.assign(Sunrise_Sunset=df["Sunrise_Sunset"].fillna("Unknown"),
                   Street=df["Street"].str.strip())
    log.add("4b. Fill / tidy", "Sunrise_Sunset NaN -> 'Unknown'; Street stripped of spaces", before, df.shape)
    return df
