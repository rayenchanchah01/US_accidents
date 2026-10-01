"""Step 3 - Column audit: drop the columns we must not or need not use.

The project rule: a column is allowed only if its value is known when the accident is first
reported. On top of that we drop columns that carry no information, repeat another column, are a
recording artefact, or are almost never True.

Every non-leakage drop was checked with a model (reports/model_check.py): putting the column back
into the cleaned dataset does not change macro F1 by more than retraining noise.
The evidence and charts are in reports/Data_Cleaning_Report.pdf, section 5.
"""

DROPPED = {
    "Leakage": {
        "End_Time":     "Known only once the road is cleared. Duration follows Severity "
                        "(Severity 4: median 129 min, others 44-77 min)",
        "End_Lat":      "End of the affected stretch of road, measured afterwards. "
                        "Also 100% missing for Source2 and Source3",
        "End_Lng":      "Same as End_Lat",
        "Distance(mi)": "Length of road affected, part of how Severity is judged "
                        "(Severity 4: median 0.49 mi, others 0.07 mi or less)",
        "Description":  "Free text written afterwards ('Road closed', 'Lane blocked')",
    },
    "No information": {
        "ID":           "Identifier, different on every row",
        "Country":      "Constant: always 'US'",
        "Turning_Loop": "Constant: always False",
    },
    "Redundant": {
        "Wind_Chill(F)":         "Correlation 0.994 with Temperature (77% identical values), 26% missing",
        "Civil_Twilight":        "Equal to Sunrise_Sunset in 95% of rows",
        "Nautical_Twilight":     "Equal to Sunrise_Sunset in 90% of rows",
        "Astronomical_Twilight": "Equal to Sunrise_Sunset in 86% of rows",
        "Timezone":              "4 values, implied by State and longitude",
        "Airport_Code":          "Nearest weather station; the location is already in lat/lng, City, County, State",
        "Zipcode":               "~128K raw / ~17K 5-digit values (median 7 rows each); "
                                 "the location is already in lat/lng, City, County, State",
        "Weather_Timestamp":     "Time of the weather reading; Start_Time already gives the time",
    },
    "Recording artefact": {
        "Wind_Direction":    "Spelling changed in 2019 ('Calm' -> 'CALM', 'West' -> 'W'), so it mostly tells the year. "
                             "No plausible link to traffic delay",
        "Precipitation(in)": "29% missing, and missing mostly before 2019 (87-91% in 2016-18 vs ~4% from 2020). "
                             "90% of known values are 0; ~30 are impossible (~10 in/h in clear weather). "
                             "Rain is kept through Weather_Group",
    },
    "Almost never True": {
        "Roundabout":      "True in 13 rows (0.003%)",
        "Bump":            "True in ~210 rows (0.04%)",
        "Traffic_Calming": "True in ~460 rows (0.09%)",
        "No_Exit":         "True in 0.25% of rows",
        "Give_Way":        "True in 0.47% of rows",
        "Railway":         "True in 0.86% of rows",
    },
}

DROP_REASONS = {col: reason for group in DROPPED.values() for col, reason in group.items()}
DROP_CATEGORY = {col: category for category, group in DROPPED.items() for col in group}


def run(df, log):
    before = df.shape
    df = df.drop(columns=list(DROP_REASONS))
    log.add("3. Column audit", f"dropped {len(DROP_REASONS)} columns", before, df.shape)
    return df
