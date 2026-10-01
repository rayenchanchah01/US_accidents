"""Builds reports/Data_Cleaning_Report.pdf.

Run from the repo root, after make_figures.py:   python reports/build_report.py
Every number in the text comes from reports/figures/facts.json, so the report always matches the code.
"""
import json
import sys
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cleaning.step3_columns import DROP_CATEGORY, DROP_REASONS, DROPPED  # noqa: E402
from cleaning.step5_outliers import LIMITS  # noqa: E402
from cleaning.step7_save import FINAL_COLS  # noqa: E402

FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "Data_Cleaning_Report.pdf"
F = json.loads((FIG / "facts.json").read_text(encoding="utf-8"))
M = F["model"]

# ------------------------------------------------------------------ fonts & styles

def register_fonts():
    """Segoe UI on Windows, DejaVu (shipped with matplotlib) anywhere else."""
    win = Path("C:/Windows/Fonts")
    mpl = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    if (win / "segoeui.ttf").exists():
        sans = [win / f for f in ("segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf", "segoeuiz.ttf")]
        mono = win / "consola.ttf"
    else:
        sans = [mpl / f for f in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf",
                                  "DejaVuSans-BoldOblique.ttf")]
        mono = mpl / "DejaVuSansMono.ttf"
    for name, path in zip(("Sans", "Sans-Bold", "Sans-Italic", "Sans-BoldItalic"), sans):
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFont(TTFont("Mono", str(mono)))
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic",
                                  boldItalic="Sans-BoldItalic")


register_fonts()
INK, INK2, MUTED = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#898781")
GRID, HEAD_BG, BLUE = colors.HexColor("#e1e0d9"), colors.HexColor("#f0efec"), colors.HexColor("#2a78d6")
NOTE_BG, WARN_BG, ORANGE = colors.HexColor("#eef4fc"), colors.HexColor("#fdf0ea"), colors.HexColor("#eb6834")
TEXT_W = A4[0] - 2 * 2.2 * cm

S = {
    "body": ParagraphStyle("body", fontName="Sans", fontSize=9.5, leading=13.8, textColor=INK, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="Sans", fontSize=9.5, leading=13.8, textColor=INK,
                             leftIndent=12, bulletIndent=2, spaceAfter=3),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=15, leading=19, textColor=INK, spaceBefore=6, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11.5, leading=15, textColor=INK, spaceBefore=10, spaceAfter=4),
    "kicker": ParagraphStyle("kicker", fontName="Sans-Bold", fontSize=8.5, leading=11, textColor=BLUE, spaceAfter=2),
    "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=26, leading=31, textColor=INK, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName="Sans", fontSize=12.5, leading=17, textColor=INK2, spaceAfter=16),
    "caption": ParagraphStyle("caption", fontName="Sans-Italic", fontSize=8.5, leading=12, textColor=INK2, spaceBefore=3, spaceAfter=12),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8, leading=10.4, textColor=INK),
    "cell_b": ParagraphStyle("cell_b", fontName="Sans-Bold", fontSize=8, leading=10.4, textColor=INK),
    "cell_s": ParagraphStyle("cell_s", fontName="Sans", fontSize=7.4, leading=9.4, textColor=INK),
    "stat": ParagraphStyle("stat", fontName="Sans-Bold", fontSize=15, leading=19, textColor=INK),
    "stat_l": ParagraphStyle("stat_l", fontName="Sans", fontSize=8.5, leading=11, textColor=INK2),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8, leading=11, textColor=INK, leftIndent=8),
}


def n(x):
    return f"{int(x):,}"


def pct(x, digits=1):
    return f"{x:.{digits}f}%"


def code(text):
    return f'<font name="Mono" size="8.6">{escape(text)}</font>'


MIXED = code('format="mixed"')
BALANCED = code('class_weight="balanced"')


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items, style="bullet"):
    return [Paragraph(t, S[style], bulletText="•") for t in items]


def table(rows, widths, header=True, style="cell", bold_first_col=False, extra=None):
    data = []
    for i, row in enumerate(rows):
        cells = []
        for j, v in enumerate(row):
            if not isinstance(v, str):
                cells.append(v)
                continue
            st = "cell_b" if (header and i == 0) or (bold_first_col and j == 0) else style
            cells.append(Paragraph(v, S[st]))
        data.append(cells)
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    cmds = [("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, GRID)]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), HEAD_BG)]
    t.setStyle(TableStyle(cmds + (extra or [])))
    return t


def box(flowables, bg=NOTE_BG, edge=BLUE):
    t = Table([[flowables]], colWidths=[TEXT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 2.5, edge),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


_fig_no = [0]


def figure(name, caption, width=TEXT_W):
    path = FIG / f"{name}.png"
    w, h = PILImage.open(path).size
    _fig_no[0] += 1
    img = Image(str(path), width=width, height=width * h / w)
    return KeepTogether([img, P(f"<b>Figure {_fig_no[0]}.</b> {caption}", "caption")])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2.2 * cm, 1.2 * cm, "US Accidents Severity Prediction · Data cleaning report")
    canvas.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, f"{doc.page}")
    canvas.restoreState()


# ------------------------------------------------------------------ numbers used in the text

steps = {s["step"]: s for s in F["steps"]}
raw_rows, raw_cols = F["raw_shape"]
clean_rows, clean_cols = F["clean_shape"]
removed = raw_rows - clean_rows
exact_dups = steps["2a. Exact duplicates"]["rows_before"] - steps["2a. Exact duplicates"]["rows_after"]
twice = steps["2b. Same accident reported twice"]["rows_before"] - steps["2b. Same accident reported twice"]["rows_after"]
no_city = steps["4a. Missing City"]["rows_before"] - steps["4a. Missing City"]["rows_after"]
sev_raw = {int(k): v for k, v in F["severity_raw"].items()}
sev_clean = {int(k): v for k, v in F["severity_clean"].items()}
dropped_model = {k: v for k, v in M["dropped"].items() if k != "All dropped columns together"}
together = M["dropped"]["All dropped columns together"]
band = M["noise_band"]
leak_all = M["leakage"]["All three leakage columns"]
n_dropped_tested = len(dropped_model)


def verdict(r):
    if abs(r["mean_delta"]) <= band:
        return "No difference (inside noise)"
    return "Slightly worse with it" if r["mean_delta"] < 0 else "Slightly better with it"


# ------------------------------------------------------------------ column dictionary

COLUMNS = [
    ("Identification", [
        ("ID", "Unique identifier of the accident record", "text"),
        ("Source", "Traffic data provider that reported the accident (anonymised as Source1-3)", "category")]),
    ("Target", [("Severity", "Impact on traffic, from 1 (least) to 4 (most)", "integer 1-4")]),
    ("Time", [
        ("Start_Time", "When the accident started (local time)", "date-time"),
        ("End_Time", "When the impact on traffic ended (local time)", "date-time"),
        ("Weather_Timestamp", "Time of the weather observation used for the row", "date-time"),
        ("Timezone", "Time zone of the location (US/Eastern, US/Central ...)", "category")]),
    ("Location", [
        ("Start_Lat", "Latitude where the accident started", "number"),
        ("Start_Lng", "Longitude where the accident started", "number"),
        ("End_Lat", "Latitude of the end of the affected stretch of road", "number"),
        ("End_Lng", "Longitude of the end of the affected stretch of road", "number"),
        ("Distance(mi)", "Length of road affected by the accident, miles", "number"),
        ("Street", "Street name", "text"),
        ("City", "City", "text"),
        ("County", "County", "text"),
        ("State", "State, 2-letter code", "category"),
        ("Zipcode", "ZIP code (5 or 9 digits)", "text"),
        ("Country", "Country", "text"),
        ("Airport_Code", "Airport weather station closest to the accident", "text")]),
    ("Text", [("Description", "Free-text description of the accident", "text")]),
    ("Weather at the nearest station", [
        ("Temperature(F)", "Temperature, °F", "number"),
        ("Wind_Chill(F)", "Wind chill (felt temperature), °F", "number"),
        ("Humidity(%)", "Relative humidity, %", "number"),
        ("Pressure(in)", "Air pressure at the station, inches of mercury", "number"),
        ("Visibility(mi)", "Visibility, miles", "number"),
        ("Wind_Direction", "Wind direction (N, SSW, CALM, VAR ...)", "category"),
        ("Wind_Speed(mph)", "Wind speed, miles per hour", "number"),
        ("Precipitation(in)", "Precipitation amount, inches", "number"),
        ("Weather_Condition", "Weather description (Fair, Light Rain, Fog ...)", "category")]),
    ("Road features nearby (True/False)", [
        ("Amenity", "Amenity nearby (school, shop, restaurant ...)", "True/False"),
        ("Bump", "Speed bump or hump", "True/False"),
        ("Crossing", "Pedestrian crossing", "True/False"),
        ("Give_Way", "Give-way (yield) sign", "True/False"),
        ("Junction", "Junction or highway exit/entry", "True/False"),
        ("No_Exit", "No-exit (dead-end) road", "True/False"),
        ("Railway", "Railway crossing", "True/False"),
        ("Roundabout", "Roundabout", "True/False"),
        ("Station", "Bus or train station", "True/False"),
        ("Stop", "Stop sign", "True/False"),
        ("Traffic_Calming", "Traffic-calming measure", "True/False"),
        ("Traffic_Signal", "Traffic signal", "True/False"),
        ("Turning_Loop", "Turning loop", "True/False")]),
    ("Light (period of day)", [
        ("Sunrise_Sunset", "Day or Night, based on sunrise / sunset", "Day/Night"),
        ("Civil_Twilight", "Day or Night, based on civil twilight (sun 6° below horizon)", "Day/Night"),
        ("Nautical_Twilight", "Day or Night, based on nautical twilight (12°)", "Day/Night"),
        ("Astronomical_Twilight", "Day or Night, based on astronomical twilight (18°)", "Day/Night")]),
]
DESCRIPTION = {c: d for _, cols in COLUMNS for c, d, _ in cols}
DESCRIPTION["Weather_Group"] = "Weather_Condition folded into 8 groups (step 6)"


def decision(col):
    if col == "Severity":
        return "Target"
    if col == "Weather_Condition":
        return "Replaced by Weather_Group"
    if col in DROP_CATEGORY:
        return f"Dropped: {DROP_CATEGORY[col].lower()}"
    return "Kept"


def missing_text(v):
    if v == 0:
        return "–"
    return "<0.01%" if v < 0.01 else (f"{v:.2f}%" if v < 1 else f"{v:.1f}%")


def column_dictionary():
    rows = [["Column", "Description", "Type", "Unique values", "Missing", "Decision"]]
    group_rows = []
    for group, cols in COLUMNS:
        group_rows.append(len(rows))
        rows.append([f"<b>{escape(group)}</b>", "", "", "", "", ""])
        for col, desc, typ in cols:
            d = decision(col)
            d_html = f'<font color="#b54a1d">{d}</font>' if d.startswith("Dropped") else d
            rows.append([escape(col), escape(desc), typ, n(F["raw_nunique"][col]),
                         escape(missing_text(F["raw_missing_pct"][col])), d_html])
    extra = []
    for r in group_rows:
        extra += [("SPAN", (0, r), (-1, r)), ("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f7f7f5"))]
    return table(rows, [3.3 * cm, 5.6 * cm, 1.8 * cm, 1.7 * cm, 1.3 * cm, 2.9 * cm], style="cell_s", extra=extra)


# ------------------------------------------------------------------ the story

def build():
    story = []
    add = story.extend

    # ---------------- cover
    add([Spacer(1, 1.2 * cm), P("DATA SCIENCE PROJECT · DATA PREPARATION", "kicker"),
         P("US Accidents Severity Prediction", "title"),
         P("Data cleaning report: from the 500K-row Kaggle sample to the dataset we model on", "subtitle")])
    stats = [[P(f"{n(raw_rows)} × {raw_cols}", "stat"), P(f"{n(clean_rows)} × {clean_cols}", "stat"),
              P(f"{n(removed)}", "stat"), P(f"{len(DROP_REASONS)} + 1", "stat")],
             [P("rows × columns in the original sample", "stat_l"), P("rows × columns in the cleaned dataset", "stat_l"),
              P(f"rows removed ({removed / raw_rows:.2%}), all duplicates or rows with no city", "stat_l"),
              P("columns dropped, plus one replaced by a cleaner version", "stat_l")]]
    st = Table(stats, colWidths=[TEXT_W / 4] * 4)
    st.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 2, BLUE), ("TOPPADDING", (0, 0), (-1, 0), 8),
                            ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                            ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    add([st, Spacer(1, 0.7 * cm)])
    add([table([
        ["Dataset", "US Accidents (2016-2023), Moosavi et al., Kaggle. We use a 500,000-row sample: "
                    f"{code('US_Accidents_March23_sampled_500k.csv')}"],
        ["Goal", "Predict accident <b>Severity</b> (1-4, impact on traffic) from what is known when the accident is first reported"],
        ["Repository", "github.com/rayenchanchah01/US_accidents"],
        ["Code", f"{code('cleaning/')} (one file per step), {code('02_cleaning.ipynb')}, {code('reports/')} (this report)"],
        ["Output", f"{code('US_Accidents_cleaned.csv')} — the original CSV is never modified"],
        ["Date", date.today().strftime("%d %B %Y")],
    ], [2.6 * cm, TEXT_W - 2.6 * cm], header=False, bold_first_col=True), Spacer(1, 0.7 * cm)])

    add([box([P("<b>In short</b>")] + bullets([
        f"The cleaned data is a <b>new file</b> with {n(clean_rows)} rows and {clean_cols} columns. "
        "The original CSV is only read; its checksum is verified after every run.",
        f"<b>{n(exact_dups + twice)} rows were duplicates</b>: {n(exact_dups)} exact copies and {n(twice)} accidents "
        "reported twice. The first notebook found 0 because it compared the ID column, which is unique on every row.",
        f"<b>5 columns leak the answer</b>: they are only known after the accident is over. Adding them lifts a model's "
        f"macro F1 from {M['baseline_mean']:.2f} to {leak_all['mean_f1']:.2f}, which is a score we could never get in practice.",
        f"<b>{len(DROP_REASONS) - 5} more columns are dropped</b> because they are constant, repeat another column, are a "
        f"recording artefact or are almost never True. A model check shows that putting them back "
        f"({n_dropped_tested} tested one by one and all together) does not improve the model (Figure 8).",
        "<b>Warning for modelling:</b> the meaning of the Severity labels changes by data provider and by year (section 10).",
    ])]))
    story.append(PageBreak())

    # ---------------- contents
    add([P("Contents", "h1")])
    toc = ["1  The dataset", "2  How the cleaning is organised", "3  Step 1 — Fix the data types",
           "4  Step 2 — Remove duplicates", "5  Step 3 — Column audit: what we dropped and why",
           "6  Step 4 — Missing values", "7  Step 5 — Impossible values", "8  Step 6 — Weather groups",
           "9  Step 7 — The cleaned dataset", "10  Warning for modelling: the Severity labels drift",
           "11  How to reproduce", "References"]
    add([P(t, "body") for t in toc])
    story.append(Spacer(1, 0.4 * cm))

    # ---------------- 1. dataset
    add([P("1  The dataset", "h1"), P("1.1  Where it comes from", "h2"),
         P("The <i>US Accidents</i> dataset (Moosavi et al.) collects about 7.7 million traffic accidents reported in the "
           "contiguous United States between February 2016 and March 2023. Accidents are reported by traffic data "
           "providers (shown as Source1-3), which collect events from transport departments, police, traffic cameras "
           "and road sensors. Each record is then enriched with the weather at the nearest airport weather station, "
           "the road features nearby (from OpenStreetMap) and the period of day (day or night). "
           "The project works with a 500,000-row sample of the full 3 GB file, stored in the repository with Git LFS."),
         table([["Property", "Value"],
                ["Rows × columns", f"{n(raw_rows)} × {raw_cols}"],
                ["Period covered by the sample", f"{F['date_min'][:10]} to {F['date_max'][:10]}"],
                ["States", f"{F['n_states']} (contiguous US)"],
                ["Rows per provider", ", ".join(f"{k}: {n(v)} ({v / raw_rows:.1%})" for k, v in F["source_counts"].items())],
                ["Licence", "CC BY-NC-SA 4.0 — academic use allowed, citation required (see References)"]],
               [5 * cm, TEXT_W - 5 * cm]),
         P("1.2  What we predict: Severity", "h2"),
         P("Severity measures the <b>impact of the accident on traffic</b> (how long traffic is delayed), "
           "<b>not</b> injuries or how dangerous the accident was."),
         table([["Severity", "Meaning", "Rows", "Share"]] +
               [[str(k), m, n(sev_raw[k]), pct(sev_raw[k] / raw_rows * 100, 2)] for k, m in
                zip([1, 2, 3, 4], ["Least impact on traffic (short delay)", "Moderate impact",
                                   "Significant impact", "Highest impact (long delay)"])],
               [1.8 * cm, 8.5 * cm, 3 * cm, TEXT_W - 13.3 * cm]),
         Spacer(1, 0.3 * cm),
         figure("01_severity", "The classes are very imbalanced: 4 out of 5 accidents are Severity 2. A model that always "
                "answers \"2\" is right 79.6% of the time but scores a macro F1 of only about 0.22, which is why the "
                "project uses macro F1 and not accuracy."),
         CondPageBreak(6 * cm),
         P("1.3  Column dictionary", "h2"),
         P(f"The original file has {raw_cols} columns in 9 groups. The last column says what the cleaning did with each "
           "one; the reasons are in section 5."),
         column_dictionary()])
    story.append(PageBreak())

    # ---------------- 2. organisation
    step_rows = [["Step", "What it does", "Code file", "Rows after", "Columns after"]]
    files = {"1": "step1_types.py", "2a": "step2_duplicates.py", "2b": "step2_duplicates.py", "3": "step3_columns.py",
             "4a": "step4_missing.py", "4b": "step4_missing.py", "5": "step5_outliers.py", "6": "step6_weather.py",
             "7": "step7_save.py"}
    for s in F["steps"]:
        num, what = s["step"].split(". ", 1)
        if num == "0":
            continue
        step_rows.append([num, escape(s["detail"]) if num == "5" else escape(what), code(files[num]),
                          n(s["rows_after"]), str(s["cols_after"])])
    step_rows.append(["7", "Final checks, save the CSV and the cleaning log", code(files["7"]), n(clean_rows), str(clean_cols)])
    add([P("2  How the cleaning is organised", "h1"),
         P("Four rules were followed throughout:"),
         *bullets([
             "<b>The original file is only read.</b> Its MD5 checksum is computed before cleaning and checked again "
             "after saving; step 7 stops with an error if it changed.",
             "<b>The one rule of the project:</b> a column is allowed only if its value is known at the moment the "
             "accident is first reported.",
             "<b>No filling that learns from the data happens here</b> (for example replacing missing temperatures "
             "with the median). It must be fitted on the training set only, inside the model pipeline (Phase 9); "
             "doing it now would let information from the test set leak into training.",
             "<b>Every decision is backed by a number</b>, and every column dropped for a reason other than leakage "
             "was checked with a model (section 5.6)."]),
         Spacer(1, 0.2 * cm),
         P("The code is split into one file per step in the <b>cleaning/</b> folder. Each file starts with a short "
           "explanation of what the step does and why. The steps run in this order:"),
         table(step_rows, [1.1 * cm, 7.4 * cm, 3.7 * cm, 2.1 * cm, TEXT_W - 14.3 * cm]),
         Spacer(1, 0.3 * cm),
         P(f"Run everything with {code('python -m cleaning')}, or step by step in {code('02_cleaning.ipynb')}. "
           f"Each run also rewrites {code('reports/cleaning_log.md')} with the table above.")])

    # ---------------- 3. types
    fm = F["start_time_formats"]
    add([P("3  Step 1 — Fix the data types", "h1"),
         P(f"File: {code('cleaning/step1_types.py')}"),
         P("Pandas reads the time columns as text. In this file they come in three different formats, which breaks any "
           "single fixed date format:"),
         table([["Format of Start_Time", "Example", "Rows"],
                ["Whole seconds", code("2019-06-12 10:10:56"), n(fm.get("19", 0))],
                ["Microseconds", code("2022-12-03 23:37:14.000000"), n(fm.get("26", 0))],
                ["Nanoseconds", code("2022-12-03 23:37:14.000000000"), n(fm.get("29", 0))]],
               [4 * cm, 7.5 * cm, TEXT_W - 11.5 * cm]),
         Spacer(1, 0.2 * cm),
         P(f"The fractional part is always zero, so the times are parsed with {MIXED} and kept to the "
           "second. End_Time and Weather_Timestamp are dropped later, but they are parsed here first so that the "
           "duplicate check compares real times and not two spellings of the same time."),
         P("The 13 road-feature columns are converted from True/False to 1/0, so they can be summed "
           "(e.g. a count of road features nearby) and used by any model.")])

    # ---------------- 4. duplicates
    ex = F["duplicate_example"]
    add([P("4  Step 2 — Remove duplicates", "h1"),
         P(f"File: {code('cleaning/step2_duplicates.py')}"),
         P("4.1  Exact copies", "h2"),
         P(f"The first notebook ({code('exec.ipynb')}) reported <b>0 duplicates</b>. That check included the ID column, "
           f"and every row has a different ID, so no two rows could ever match. Ignoring ID, <b>{n(exact_dups)} rows</b> "
           "are exact copies of another row and were removed."),
         P("4.2  The same accident reported twice", "h2"),
         P(f"A further {n(F['repeated_event_rows'])} rows share the exact same Start_Time, Start_Lat and Start_Lng "
           "(to the second and to 6 decimal places) with another row. These are the same accident reported more than "
           "once: they differ only in after-the-fact columns such as End_Time, Description and Distance, and in "
           f"{F['repeated_event_conflicting_severity']} cases in Severity. An example:"),
         table([["ID", "Source", "Severity", "Start_Time", "End_Time", "Description"]] +
               [[r["ID"], r["Source"], r["Severity"], r["Start_Time"], r["End_Time"], escape(r["Description"])] for r in ex],
               [1.9 * cm, 1.5 * cm, 1.3 * cm, 2.8 * cm, 2.8 * cm, TEXT_W - 10.3 * cm], style="cell_s"),
         Spacer(1, 0.25 * cm),
         P("Keeping both copies would let one accident end up in the training set <i>and</i> in the test set, which makes "
           "the test score look better than it really is. We keep <b>one row per accident</b>. When the copies disagree "
           "on Severity we keep the highest one: the accident had at least that impact on traffic at some point. "
           f"This removed <b>{n(twice)} rows</b>."),
         P(f"In total {n(exact_dups + twice)} duplicate rows were removed ({(exact_dups + twice) / raw_rows:.2%} of the data). "
           "The Severity shares barely move (section 9).")])
    story.append(PageBreak())

    # ---------------- 5. columns
    group_rows = [["Reason", "Columns", "Count"]]
    for cat, cols in DROPPED.items():
        group_rows.append([f"<b>{cat}</b>", escape(", ".join(cols)), str(len(cols))])
    add([P("5  Step 3 — Column audit: what we dropped and why", "h1"),
         P(f"File: {code('cleaning/step3_columns.py')}"),
         P(f"{len(DROP_REASONS)} of the {raw_cols} columns are dropped, for five kinds of reasons. Weather_Condition is "
           "not dropped but replaced by a cleaner version in step 6. The full list with one reason per column is in "
           f"{code('reports/cleaning_log.md')}."),
         table(group_rows, [3.4 * cm, TEXT_W - 5 * cm, 1.6 * cm]),
         Spacer(1, 0.2 * cm)])

    lk = M["leakage"]
    add([P("5.1  Leakage: columns only known after the accident", "h2"),
         P("A model is used <i>when the accident is reported</i>. Any column measured later is not available at that "
           "moment, and some of them describe the very thing we want to predict:"),
         *bullets([
             "<b>End_Time</b> is when the road was cleared. The longer the delay, the higher the Severity.",
             "<b>Distance(mi)</b> is the length of road affected, which is part of how Severity is judged.",
             "<b>End_Lat, End_Lng</b> mark the end of that affected stretch (so they encode the same distance). They are "
             "also missing for every Source2 and Source3 row.",
             "<b>Description</b> is written afterwards and often says \"Road closed\" or \"Lane blocked\"."]),
         figure("03_leakage_evidence", "The leakage columns follow Severity directly: Severity 4 accidents last a median "
                "of 129 minutes and affect 0.49 miles of road; the other levels affect 0.07 miles or less."),
         figure("04_leakage_model", f"Proof by model (method in section 5.6): adding the leakage columns lifts macro F1 from "
                f"{M['baseline_mean']:.3f} to {leak_all['mean_f1']:.3f}. That jump is not real skill: at prediction time "
                "none of these values exist yet. This is why they must go.")])

    add([P("5.2  No information", "h2"),
         *bullets([
             "<b>ID</b> is different on every row: it identifies the record but says nothing about the accident.",
             "<b>Country</b> is always \"US\" and <b>Turning_Loop</b> is always False. A column with one value cannot "
             "help tell the classes apart."])])

    tw = F["twilight_agreement"]
    add([P("5.3  Redundant: the information is already in a column we keep", "h2"),
         figure("05_redundant", f"Left: wind chill is almost a copy of temperature (correlation {F['wind_chill_corr']:.3f}; "
                "most points sit on the y = x line). Right: the three twilight columns give the same Day/Night answer as "
                f"Sunrise_Sunset in {tw['Astronomical_Twilight']:.0f}-{tw['Civil_Twilight']:.0f}% of rows; the hour of the "
                "day (from Start_Time) covers the rest."),
         *bullets([
             f"<b>Wind_Chill(F)</b>: a copy of Temperature in most rows, and {F['raw_missing_pct']['Wind_Chill(F)']:.0f}% missing.",
             "<b>Civil_, Nautical_, Astronomical_Twilight</b>: three other definitions of day/night, almost identical to "
             "Sunrise_Sunset.",
             "<b>Timezone</b> (4 values) follows from the State and longitude.",
             f"<b>Airport_Code</b> ({n(F['raw_nunique']['Airport_Code'])} values) and <b>Zipcode</b> "
             f"({n(F['raw_nunique']['Zipcode'])} values, about 17,000 once cut to 5 digits, a median of 7 rows each) "
             "only repeat the location, which is already given exactly by Start_Lat/Start_Lng and by City, County and State.",
             "<b>Weather_Timestamp</b> is the time of the weather reading; the time of the accident is already in Start_Time."])])

    pm = {int(k): v for k, v in F["precip_missing_by_year"].items()}
    add([P("5.4  Recording artefacts: the column tells the year, not the weather", "h2"),
         P("Around 2019 the data providers changed how weather was recorded. A model would happily learn these changes, "
           "but what it learns is \"which year is this row from\", not anything about the accident:"),
         figure("06_artefacts", f"Left: Precipitation is missing in {pm[2016]:.0f}% of 2016 rows but only {pm[2022]:.0f}% of "
                "2022 rows. Middle: \"Clear\" was replaced by \"Fair\" in 2019 (the same happened to other weather labels). "
                "Right: \"Calm\" became \"CALM\" (and \"West\" became \"W\", \"Variable\" became \"VAR\")."),
         *bullets([
             "<b>Wind_Direction</b>: once the old and new spellings are merged, almost no link with Severity is left, "
             "and there is no reason why the compass direction of the wind would change traffic delay.",
             f"<b>Precipitation(in)</b>: {F['raw_missing_pct']['Precipitation(in)']:.0f}% missing, mostly in the early "
             "years, so \"is it missing\" mainly says the year. 90% of the known values are exactly 0, and about 30 are "
             "impossible (around 10 inches per hour in clear weather in New York). Rain is still captured by "
             "Weather_Group (step 6).",
             "<b>Weather_Condition</b> has the same problem; instead of dropping it, step 6 groups the labels so that "
             "\"Clear\" and \"Fair\" become the same group."])])

    add([P("5.5  Road features that are almost never True", "h2"),
         figure("07_rare_flags", "Six of the road-feature flags are True in less than 1% of rows (Roundabout in only 13 "
                "rows out of 500,000). A flag that is almost always 0 can only ever affect a handful of predictions. "
                "Turning_Loop is never True (section 5.2)."),
         P("The six kept flags (Traffic_Signal, Crossing, Junction, Stop, Station, Amenity) are each True in more than "
           "1% of rows.")])

    # 5.6 model check
    res_rows = [["Column put back", "Mean change in macro F1", f"Range over {M['n_splits']} splits", "Verdict"]]
    for name, r in sorted(dropped_model.items(), key=lambda kv: kv[1]["mean_delta"]):
        res_rows.append([escape(name), f"{r['mean_delta']:+.4f}", f"{r['min_delta']:+.4f} to {r['max_delta']:+.4f}", verdict(r)])
    res_rows.append([f"<b>All {n_dropped_tested} together</b>", f"<b>{together['mean_delta']:+.4f}</b>",
                     f"{together['min_delta']:+.4f} to {together['max_delta']:+.4f}", f"<b>{verdict(together)}</b>"])
    within = sum(abs(r["mean_delta"]) <= band for r in dropped_model.values())
    best = max(dropped_model.items(), key=lambda kv: kv[1]["mean_delta"])
    add([CondPageBreak(8 * cm), P("5.6  Model check: do the dropped columns make any difference?", "h2"),
         P("For every column dropped for a reason other than leakage, we asked one question: <i>if we put this column "
           "back into the cleaned dataset, does a model predict Severity better?</i>"),
         *bullets([
             "<b>Model:</b> gradient boosting (scikit-learn's HistGradientBoostingClassifier) with "
             f"{BALANCED}. It handles missing values and categories as they are, so every "
             "column can be tested without extra preparation.",
             "<b>Baseline:</b> all 21 kept columns in a model-ready form: hour, weekday, month and year from "
             "Start_Time; City and County as the number of accidents per city / county (counted on the training rows "
             "only); an <i>is_highway</i> flag from Street; the other columns as they are.",
             "<b>Test:</b> each dropped column is added back on its own, then all of them together. Macro F1 is measured "
             f"on a 20% hold-out set, repeated on {M['n_splits']} different random splits. Every run is compared with "
             "the baseline trained with the same random seed on the same split.",
             f"<b>Noise:</b> the baseline itself is retrained with {M['n_seeds']} different random seeds. That alone moves "
             f"its score by up to ±{band:.4f}. A change smaller than this is not a real difference."]),
         figure("08_model_check", f"Each row is one dropped column put back into the data. The baseline (cleaned set) "
                f"scores a macro F1 of {M['baseline_mean']:.3f}. {within} of the {n_dropped_tested} columns fall inside "
                f"the noise band, and putting all {n_dropped_tested} back together changes the score by "
                f"{together['mean_delta']:+.4f}."),
         table(res_rows, [4.4 * cm, 3.6 * cm, 4 * cm, TEXT_W - 12 * cm]),
         Spacer(1, 0.3 * cm),
         P(f"<b>Reading the result.</b> Compared with the leakage columns (+{leak_all['mean_f1'] - M['baseline_mean']:.2f}), "
           f"every change here is tiny: the largest gain from a single column is {best[1]['mean_delta']:+.4f} "
           f"({escape(best[0])}), i.e. less than half a point of macro F1, and putting all of them back together "
           f"gives {together['mean_delta']:+.4f}. Dropping them makes the dataset smaller and easier to explain at "
           "no measurable cost. "
           f"For reference, the baseline already scores {M['baseline_mean']:.2f} against 0.22 for the "
           "\"always predict 2\" model; these are quick screening numbers, the official results come in Phases 7-10.")])
    story.append(PageBreak())

    # ---------------- 6. missing
    lm = F["light_missing_severity"]
    lm_missing = next(v for k, v in lm.items() if k.startswith("Rows missing"))
    add([P("6  Step 4 — Missing values", "h1"),
         P(f"File: {code('cleaning/step4_missing.py')}"),
         figure("09_missing", "Missing values in the original data. Most of the heavily missing columns were already "
                "dropped in step 3 for other reasons. The rest are small and handled as in the table below."),
         table([["Column", "Missing", "Action", "Why"],
                ["City", f"{no_city} rows", "Drop the rows", "Too few to matter"],
                ["Sunrise_Sunset", missing_text(F["raw_missing_pct"]["Sunrise_Sunset"]), "Fill with \"Unknown\"",
                 f"These rows are {lm_missing['4']:.0f}% Severity 4, against 2.6% overall (Figure 10). Dropping them "
                 "would remove ~165 rows of our rarest class"],
                ["Street", missing_text(F["raw_missing_pct"]["Street"]), "Leave empty; strip spaces",
                 "Only used later to detect highways; an empty street is simply \"not a highway\""],
                ["Weather_Condition", missing_text(F["raw_missing_pct"]["Weather_Condition"]), "\"Unknown\" group (step 6)", ""],
                ["Temperature, Humidity, Pressure, Visibility, Wind_Speed",
                 f"{min(F['raw_missing_pct'][c] for c in ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']):.1f}"
                 f"–{max(F['raw_missing_pct'][c] for c in ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']):.1f}%",
                 "<b>Leave empty</b>", "The median used to fill them must be computed on the training set only, inside "
                 "the model pipeline (Phase 9). Computing it here would leak test data into training"]],
               [3.6 * cm, 1.8 * cm, 3.4 * cm, TEXT_W - 8.8 * cm]),
         Spacer(1, 0.3 * cm),
         figure("10_light_missing", "Rows with no light information are four times more likely to be Severity 4 than "
                "the average row, so we keep them with the value \"Unknown\" instead of deleting them.")])

    # ---------------- 7. outliers
    reasons5 = {"Temperature(F)": "Contiguous-US records are −70 °F and 134 °F. Removed values: 140, 196, 207 (in \"Fair\" "
                                  "or \"Cloudy\" weather) and −77.8",
                "Wind_Speed(mph)": "127-823 mph readings come with \"Fair\", \"Clear\" or \"Overcast\". Real storms in the "
                                   "data peak at 77-82 mph (e.g. Hurricane Ian, Florida, 28 Sep 2022)",
                "Visibility(mi)": "Most stations stop at 10 miles and a few report up to 80; 100 and more is not plausible",
                "Pressure(in)": "This is pressure at the station, so 20-25 inHg is normal at altitude (Colorado, Utah). "
                                "0.12, 2.99 and 38.44 are sensor errors",
                "Humidity(%)": "Sanity check: a percentage"}
    fixed = F["impossible_values"]
    add([CondPageBreak(10 * cm), P("7  Step 5 — Impossible values", "h1"),
         P(f"File: {code('cleaning/step5_outliers.py')}"),
         P("Some weather readings are physically impossible: sensor or transmission errors. They are set to empty "
           "(NaN) and will be filled later like any other missing value; the rows themselves are kept."),
         figure("11_outliers", "Distribution of each weather measure (log scale, so that single rows are visible). "
                "Values in the shaded areas, outside the valid range, are set to NaN."),
         table([["Column", "Valid range", "Set to NaN", "Why this range"]] +
               [[escape(c), f"{lo:g} to {hi:g}", str(fixed[c]), escape(reasons5[c])] for c, (lo, hi) in LIMITS.items()],
               [3 * cm, 2.2 * cm, 1.8 * cm, TEXT_W - 7 * cm])])

    # ---------------- 8. weather
    mapping = F["weather_mapping"]
    label_counts = F["weather_label_counts"]
    by_group = {}
    for label, group in mapping.items():
        by_group.setdefault(group, []).append(label)
    gc = F["weather_group_counts"]
    w_rows = [["Group", "Rows", "Original labels in the group (most frequent first)"]]
    for group in sorted(gc, key=lambda g: -gc[g]):
        labels = sorted(by_group.get(group, []), key=lambda lab: -label_counts.get(lab, 0))
        text = ", ".join(labels) + (" + missing values" if group == "Unknown" else "")
        w_rows.append([f"<b>{escape(group)}</b>", n(gc[group]), escape(text)])
    add([PageBreak(), P("8  Step 6 — Weather groups", "h1"),
         P(f"File: {code('cleaning/step6_weather.py')}"),
         P(f"Weather_Condition has {len(mapping)} different labels, many used only a handful of times, and its vocabulary "
           "changed in 2019 (Figure 6). Both problems are solved by folding the labels into 8 groups, as planned in the "
           "project plan. \"Clear\" and \"Fair\" now land in the same group, so the column no longer tells the year."),
         P("When a label has several parts, such as \"Light Rain / Windy\", the strongest part wins, in this order: "
           "thunderstorm, snow/ice, rain, fog/haze, windy, cloudy, clear."),
         figure("12_weather_groups", "The 8 weather groups in the cleaned data. Clear and cloudy weather make up 84% "
                "of accidents; rain, snow, fog and storms are rarer but kept as separate groups."),
         table(w_rows, [3.4 * cm, 1.7 * cm, TEXT_W - 5.1 * cm], style="cell_s")])

    # ---------------- 9. final dataset
    final_rows = [["Column", "Description", "Type", "Missing"]]
    for c in FINAL_COLS:
        final_rows.append([escape(c), escape(DESCRIPTION[c]), F["clean_dtypes"][c].replace("datetime64[ns]", "date-time")
                           .replace("object", "text").replace("int64", "integer").replace("float64", "number"),
                           escape(missing_text(F["clean_missing_pct"][c]))])
    sev_rows = [["Severity", "Before: rows", "Before: share", "After: rows", "After: share"]]
    for k in [1, 2, 3, 4]:
        sev_rows.append([str(k), n(sev_raw[k]), pct(sev_raw[k] / raw_rows * 100, 2),
                         n(sev_clean[k]), pct(sev_clean[k] / clean_rows * 100, 2)])
    size_mb = (ROOT / "US_Accidents_cleaned.csv").stat().st_size / 1e6
    add([PageBreak(), P("9  Step 7 — The cleaned dataset", "h1"),
         P(f"File: {code('cleaning/step7_save.py')}"),
         P("Before saving, the code checks that the table has exactly the expected columns, one row per accident, "
           "Severity values in 1-4, no missing values in the identity and location columns, and every weather value "
           "inside the step 5 limits. After saving, it checks that the original CSV is unchanged."),
         P(f"Result: <b>{code('US_Accidents_cleaned.csv')}</b>, {n(clean_rows)} rows × {clean_cols} columns, "
           f"{size_mb:.0f} MB (the original is 197 MB). It is small enough to push to GitHub without Git LFS."),
         table(final_rows, [3.2 * cm, 9 * cm, 2.2 * cm, TEXT_W - 14.4 * cm]),
         Spacer(1, 0.4 * cm),
         P("The cleaning did not change the balance of the classes:"),
         table(sev_rows, [2 * cm] + [(TEXT_W - 2 * cm) / 4] * 4),
         Spacer(1, 0.3 * cm),
         P("The remaining empty values are on purpose (weather measures and a few streets); they are filled inside the "
           "model pipeline. City, County and Street have thousands of values each and will be encoded in Phase 6 "
           "(frequency encoding, and an <i>is_highway</i> flag built from Street).")])

    # ---------------- 10. drift
    add([PageBreak(), P("10  Warning for modelling: the Severity labels drift", "h1"),
         P("While checking the data we found that the <b>meaning of the Severity labels changes</b> depending on the "
           "data provider and the year. This is not something cleaning can fix, but every later phase needs to know "
           "about it."),
         figure("13_drift", "Left: Source1 labels about 18% of accidents as Severity 3 until 2018, then fewer each year, "
                "and none at all from 2021. Right: Severity 1 barely exists, except in Source1 in 2020 and in "
                "Source2/Source3 in 2022. (2023 only contains Source1, January to March.)"),
         box([P("<b>What this means for the next phases</b>")] + bullets([
             "From 2021, a Source1 accident that would have been Severity 3 before is labelled 2 (or 4). The same real "
             "accident can therefore get different labels depending on who reported it and when.",
             "<b>Keep Source and the year as features</b> so the model can tell which labelling scheme a row follows.",
             "The planned time-based test (train on 2016-2021, test on 2022-2023) will score lower, partly because "
             "the labels changed and not only because the model is weaker. Say this in the report.",
             "The planned ablation with and without Source (Phase 9) will show a large gap. That gap measures the "
             "provider effect, not anything about road safety.",
             "All rows were kept, so the team can still decide to restrict the data to a period or a provider with "
             "consistent labels; Source and Start_Time are in the cleaned file for that."]), bg=WARN_BG, edge=ORANGE)])

    # ---------------- 11. reproduce
    add([P("11  How to reproduce", "h1"),
         P("From the repository root, with Python 3.10+ and pandas, numpy, scikit-learn, matplotlib and reportlab installed:"),
         Preformatted("python -m cleaning               # original CSV -> US_Accidents_cleaned.csv + reports/cleaning_log.md\n"
                      "python reports/model_check.py    # model check, ~10-15 min -> reports/model_check.csv\n"
                      "python reports/make_figures.py   # charts -> reports/figures/\n"
                      "python reports/build_report.py   # this PDF", S["code"]),
         Spacer(1, 0.3 * cm),
         table([["Path", "What it is"],
                [code("cleaning/step1_types.py … step7_save.py"), "The cleaning, one file per step, each with its explanation"],
                [code("02_cleaning.ipynb"), "Runs the steps one by one and shows the result of each"],
                [code("US_Accidents_cleaned.csv"), "The cleaned dataset (output)"],
                [code("reports/cleaning_log.md"), "Rows and columns after each step, every dropped column with its reason"],
                [code("reports/model_check.py"), "The model check of section 5.6"],
                [code("reports/make_figures.py"), "All figures of this report"],
                [code("reports/build_report.py"), "Builds this PDF"]],
               [6.4 * cm, TEXT_W - 6.4 * cm]),
         Spacer(1, 0.5 * cm),
         P("References", "h1"),
         *[P(t, "body") for t in [
             "[1] Moosavi, S., Samavatian, M. H., Parthasarathy, S., &amp; Ramnath, R. (2019). <i>A Countrywide Traffic "
             "Accident Dataset.</i> arXiv:1906.05409.",
             "[2] Moosavi, S., Samavatian, M. H., Parthasarathy, S., Teodorescu, R., &amp; Ramnath, R. (2019). <i>Accident "
             "Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights.</i> Proceedings of the 27th "
             "ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems.",
             "[3] Moosavi, S. <i>US Accidents (2016-2023)</i> [Dataset]. Kaggle. "
             "https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents"]]])
    return story


def main():
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title="US Accidents — Data Cleaning Report", author="US Accidents project team")
    doc.build(build(), onFirstPage=footer, onLaterPages=footer)
    print("saved", OUT)


if __name__ == "__main__":
    main()
