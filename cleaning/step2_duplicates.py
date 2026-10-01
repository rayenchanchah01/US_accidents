"""Step 2 - Remove duplicates.

2a. Exact copies
    exec.ipynb reported 0 duplicates because it compared the ID column too, and ID is different on
    every row. Ignoring ID, some rows are exact copies of each other.

2b. The same accident reported more than once
    Some rows share the exact same Start_Time, Start_Lat and Start_Lng (to the second and to 6
    decimals). They are the same accident: they only differ in after-the-fact columns (End_Time,
    Description, Distance) and sometimes in Severity. Keeping both would let one accident appear in
    the training set and the test set at the same time, which inflates the scores.
    We keep one row per accident: the one with the highest Severity.
"""

EVENT_KEY = ["Start_Time", "Start_Lat", "Start_Lng"]


def run(df, log):
    before = df.shape
    df = df[~df.drop(columns="ID").duplicated()]
    log.add("2a. Exact duplicates", "rows identical on every column except ID", before, df.shape)

    repeated = df[df.duplicated(EVENT_KEY, keep=False)]
    log.notes["repeated_event_rows"] = len(repeated)
    log.notes["repeated_event_conflicting_severity"] = int(
        (repeated.groupby(EVENT_KEY)["Severity"].nunique() > 1).sum())

    before = df.shape
    df = (df.sort_values("Severity", ascending=False, kind="stable")
            .drop_duplicates(EVENT_KEY)
            .sort_index())
    log.add("2b. Same accident reported twice",
            "same Start_Time + Start_Lat + Start_Lng; kept the highest Severity", before, df.shape)
    return df
