"""Step 5 - Physically impossible values -> NaN.

| Column          | Valid range | Why                                                                     |
|-----------------|-------------|-------------------------------------------------------------------------|
| Temperature(F)  | -50 to 130  | Contiguous-US records are -70 F and 134 F. Removed: 140, 196, 207       |
|                 |             | (in "Fair"/"Cloudy" weather) and -77.8                                  |
| Wind_Speed(mph) | 0 to 100    | 127-823 mph appear with "Fair", "Clear", "Overcast". Real storms in the |
|                 |             | data peak at 77-82 mph (e.g. Hurricane Ian, FL, 28 Sep 2022)            |
| Visibility(mi)  | 0 to 100    | Most stations cap at 10 mi, some report up to 80; 100+ is not plausible |
| Pressure(in)    | 17 to 31.5  | Station pressure: 20-25 inHg is normal at altitude (Colorado, Utah).    |
|                 |             | 0.12, 2.99 and 38.44 are sensor errors                                  |
| Humidity(%)     | 0 to 100    | sanity check                                                            |

The bad values become NaN (the rows stay) and are imputed later inside the model Pipeline.
"""
import numpy as np

LIMITS = {
    "Temperature(F)": (-50, 130),
    "Wind_Speed(mph)": (0, 100),
    "Visibility(mi)": (0, 100),
    "Pressure(in)": (17, 31.5),
    "Humidity(%)": (0, 100),
}


def run(df, log):
    before = df.shape
    df = df.copy()
    n_fixed = {}
    for col, (lo, hi) in LIMITS.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        n_fixed[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    log.notes["impossible_values"] = n_fixed
    log.add("5. Impossible values",
            "set to NaN: " + ", ".join(f"{c} {n}" for c, n in n_fixed.items()), before, df.shape)
    return df
