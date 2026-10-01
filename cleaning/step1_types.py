"""Step 1 - Fix the data types.

Times
    Start_Time, End_Time and Weather_Timestamp are stored as text in three formats:
        "2019-06-12 10:10:56"   "2022-12-03 23:37:14.000000"   "2022-12-03 23:37:14.000000000"
    The fractional part is always zero, so we parse with format="mixed" and keep whole seconds.
    End_Time and Weather_Timestamp are dropped in step 3, but parsing them here first means the
    duplicate check in step 2 compares real times, not two spellings of the same time.

Road features
    The 13 road-feature columns (Amenity ... Turning_Loop) are True/False -> 1/0.
"""
import pandas as pd

TIME_COLS = ["Start_Time", "End_Time", "Weather_Timestamp"]
ROAD_COLS = ["Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
             "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop"]


def run(df, log):
    before = df.shape
    df = df.copy()
    for col in TIME_COLS:
        df[col] = pd.to_datetime(df[col], format="mixed").dt.floor("s")
    df[ROAD_COLS] = df[ROAD_COLS].astype(int)
    log.add("1. Types", "times parsed to datetime; road flags True/False -> 1/0", before, df.shape)
    return df
