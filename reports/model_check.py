"""Model check for the column decisions in step 3 (feeds section 5 of the PDF report).

Question, for every column we dropped: if we put it back into the cleaned dataset, does a model
predict Severity better?

Method
- Rows: the cleaned rows (steps 1, 2, 4, 5 applied), so only the columns differ between runs.
- Model: HistGradientBoostingClassifier with class_weight="balanced". It handles NaN and
  categories natively, so every column can be tested as it is.
- Metric: macro F1 on a stratified 20% hold-out, repeated on 3 different splits.
- Baseline: all 21 kept columns in a model-ready form (time parts from Start_Time; City and County
  as "accidents per city/county", counted on the training rows; an is_highway flag from Street).
  Each dropped column is added back alone, then all together.
- Comparison: every run is compared with the baseline trained with the same seed on the same split.
- Noise: the baseline is retrained with 4 seeds on every split. A change smaller than the largest
  change caused by the seed alone is no real difference.
- Leakage: End_Time (as duration), Distance(mi) and Description (mentions "closed"/"blocked") are
  added to show how much they would inflate the score.
This is a screening check; the official split, features and models come in Phases 6-9.

Run from the repo root:  python reports/model_check.py   (~15-20 min)
Output: reports/model_check.csv
"""
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleaning import step1_types, step2_duplicates, step4_missing, step5_outliers  # noqa: E402
from cleaning.common import RAW_PATH, CleaningLog  # noqa: E402
from cleaning.step6_weather import weather_group  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "model_check.csv"
FREQ_ENCODED = ["City", "County", "Airport_Code", "Zipcode"]  # too many values for a category: "rows per value" on train
SPLIT_SEEDS = [0, 1, 2]
NOISE_SEEDS = [0, 1, 2, 3]


def load():
    df = pd.read_csv(RAW_PATH)
    log = CleaningLog()
    for step in (step1_types, step2_duplicates, step4_missing, step5_outliers):
        df = step.run(df, log)
    return df.reset_index(drop=True)


def build_features(df):
    t = df["Start_Time"]
    base = pd.DataFrame({
        "Source": df["Source"], "State": df["State"],
        "Start_Lat": df["Start_Lat"], "Start_Lng": df["Start_Lng"],
        "hour": t.dt.hour, "weekday": t.dt.weekday, "month": t.dt.month, "year": t.dt.year,
        "Temperature(F)": df["Temperature(F)"], "Humidity(%)": df["Humidity(%)"],
        "Pressure(in)": df["Pressure(in)"], "Visibility(mi)": df["Visibility(mi)"],
        "Wind_Speed(mph)": df["Wind_Speed(mph)"],
        "Weather_Group": df["Weather_Condition"].map(weather_group),
        "Sunrise_Sunset": df["Sunrise_Sunset"],
        **{c: df[c] for c in ["Amenity", "Crossing", "Junction", "Station", "Stop", "Traffic_Signal"]},
        "City": df["City"] + ", " + df["State"],
        "County": df["County"] + ", " + df["State"],
        "is_highway": df["Street"].str.contains(r"I-|Interstate|Hwy|Highway|US-|Fwy", na=False).astype(int),
    })
    dropped = pd.DataFrame({
        **{c: df[c] for c in ["Roundabout", "Bump", "Traffic_Calming", "No_Exit", "Give_Way", "Railway",
                              "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight",
                              "Wind_Direction", "Wind_Chill(F)", "Precipitation(in)", "Timezone",
                              "Weather_Condition", "Airport_Code"]},
        "Zipcode": df["Zipcode"].str[:5],
        "Weather_Timestamp": (t - df["Weather_Timestamp"]).dt.total_seconds() / 60,
    })
    leakage = pd.DataFrame({
        "End_Time": (df["End_Time"] - t).dt.total_seconds() / 60,
        "Distance(mi)": df["Distance(mi)"],
        "Description": df["Description"].str.contains("closed|blocked", case=False, na=False).astype(int),
    })
    X = pd.concat([base, dropped, leakage], axis=1)
    for c in X.columns:
        if X[c].dtype == object and c not in FREQ_ENCODED:
            X[c] = X[c].astype("category")
    return X, list(base.columns), list(dropped.columns), list(leakage.columns)




def encode(Xtr, Xte, cols):
    Xtr, Xte = Xtr[cols].copy(), Xte[cols].copy()
    for c in FREQ_ENCODED:
        if c in cols:
            counts = Xtr[c].value_counts()
            Xtr[c] = Xtr[c].map(counts).astype(float)
            Xte[c] = Xte[c].map(counts).astype(float)
    return Xtr, Xte


def fit_score(Xtr, Xte, ytr, yte, cols, seed=0):
    Xtr, Xte = encode(Xtr, Xte, cols)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                           class_weight="balanced", categorical_features="from_dtype",
                                           early_stopping=True, validation_fraction=0.1, random_state=seed)
    start = time.time()
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {"macro_f1": f1_score(yte, pred, average="macro"),
            "balanced_acc": balanced_accuracy_score(yte, pred),
            "recall_sev4": recall_score(yte, pred, labels=[4], average="macro"),
            "seconds": round(time.time() - start, 1)}


def main():
    df = load()
    X, base, dropped, leakage = build_features(df)
    y = df["Severity"]
    runs = [("noise", "baseline", base, s) for s in NOISE_SEEDS]
    runs += [("dropped", c, base + [c], 0) for c in dropped]
    runs += [("dropped", "All dropped columns together", base + dropped, 0)]
    runs += [("leakage", c, base + [c], 0) for c in leakage]
    runs += [("leakage", "All three leakage columns", base + leakage, 0)]

    rows = []
    for split in SPLIT_SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=split)
        for kind, name, cols, seed in runs:
            rows.append({"split": split, "kind": kind, "column": name, "seed": seed,
                         **fit_score(Xtr, Xte, ytr, yte, cols, seed)})
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print("saved", OUT_PATH)


if __name__ == "__main__":
    main()
