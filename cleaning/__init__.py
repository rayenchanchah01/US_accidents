"""Data cleaning for the US Accidents project, one file per step.

    step1_types.py       fix data types
    step2_duplicates.py  remove duplicate rows and accidents reported twice
    step3_columns.py     drop leakage / constant / redundant / artefact / rare columns
    step4_missing.py     missing values
    step5_outliers.py    impossible values -> NaN
    step6_weather.py     Weather_Condition -> Weather_Group
    step7_save.py        final checks, save US_Accidents_cleaned.csv, write reports/cleaning_log.md

Run everything from the repo root:   python -m cleaning
Run it step by step:                 02_cleaning.ipynb
"""
import pandas as pd

from cleaning import (step1_types, step2_duplicates, step3_columns, step4_missing,
                      step5_outliers, step6_weather, step7_save)
from cleaning.common import RAW_PATH, CleaningLog, md5

STEPS = [step1_types, step2_duplicates, step3_columns, step4_missing, step5_outliers, step6_weather]


def run_all(save=True):
    """Original CSV -> cleaned DataFrame. With save=True also writes the CSV and the log."""
    raw_md5 = md5(RAW_PATH)
    df = pd.read_csv(RAW_PATH)
    log = CleaningLog()
    log.add("0. Load", RAW_PATH.name, df.shape, df.shape)
    for step in STEPS:
        df = step.run(df, log)
    if save:
        df = step7_save.run(df, log, raw_md5)
    return df, log
