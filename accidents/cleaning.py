"""Deterministic, row-local cleaning. No dataset-fitted statistics live here."""
from __future__ import annotations

import numpy as np
import pandas as pd

ROAD = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
        'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal']
WEATHER_NUM = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
               'Wind_Speed(mph)', 'Precipitation(in)']
WEATHER_CAT = ['Weather_Condition', 'Wind_Direction']
CAT = ['State', 'City', 'County', 'Weather_Group', 'Wind_Direction', 'Sunrise_Sunset']
NUM = ['Start_Lat', 'Start_Lng', *WEATHER_NUM, *ROAD, 'Year', 'Month', 'Hour',
       'Weekday', 'Is_Weekend', 'Is_Rush_Hour', 'Is_Highway', 'Road_Feature_Count',
       'Weather_Unavailable', 'Precipitation_Missing']
FEATURES = NUM + CAT
META = ['ID', 'Source', 'Severity', 'Start_Time', 'event_hash', 'row_number']
DROP_REASONS = {
    'End_Time': 'Known after the accident: outcome leakage.',
    'End_Lat': 'End-of-incident extent: outcome leakage.',
    'End_Lng': 'End-of-incident extent: outcome leakage.',
    'Distance(mi)': 'Reported affected extent may encode outcome.',
    'Description': 'Narrative can describe outcome and closure.',
    'ID': 'Identifier, retained only for traceability.',
    'Source': 'Reporting-provider shortcut; retained only for auditing.',
    'Weather_Timestamp': 'Used only to reject future/stale observations.',
    'Wind_Chill(F)': 'Redundant derived temperature, often absent.',
    'Zipcode': 'Redundant high-cardinality location identifier.',
    'Street': 'Only a deterministic highway flag is retained.',
    'Country': 'Constant in this US dataset.',
    'Timezone': 'Local time and state already represented.',
    'Airport_Code': 'Weather-station proxy, unnecessary location shortcut.',
    'Turning_Loop': 'Uninformative road feature in the supplied dataset.',
    'Civil_Twilight': 'Redundant light indicator.',
    'Nautical_Twilight': 'Redundant light indicator.',
    'Astronomical_Twilight': 'Redundant light indicator.',
}


def weather_group(values: pd.Series) -> pd.Series:
    s = values.fillna('Unknown').astype('string').str.lower()
    conditions = [s.str.contains('thunder|tstorm|heavy rain|squall'),
                  s.str.contains('snow|ice|sleet|wintry|freezing|hail'),
                  s.str.contains('rain|drizzle|shower'),
                  s.str.contains('fog|haze|smoke|mist|dust|sand'),
                  s.str.contains('cloud|overcast'), s.str.contains('fair|clear'),
                  s.str.contains('wind|breezy'), s.eq('unknown')]
    return pd.Series(np.select([c.to_numpy(dtype=bool) for c in conditions],
                     ['Storm', 'Snow / ice', 'Rain', 'Low visibility', 'Cloudy',
                      'Clear', 'Windy', 'Unknown'], default='Other'), index=values.index)


def clean_chunk(raw: pd.DataFrame, offset: int = 0, *, snapshots: dict | None = None) -> tuple[pd.DataFrame, dict]:
    required = {'Severity', 'Start_Time', 'Start_Lat', 'Start_Lng', 'ID'}
    if missing := required - set(raw.columns):
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    d = raw.copy()
    log = {'input_rows': len(d)}
    d['row_number'] = np.arange(offset, offset + len(d), dtype=np.int64)
    d['Severity'] = pd.to_numeric(d['Severity'], errors='coerce')
    d['Start_Time'] = pd.to_datetime(d['Start_Time'], format='mixed', errors='coerce').astype('datetime64[ns]')
    for c in ['Start_Lat', 'Start_Lng']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    valid = (d['Severity'].isin([1, 2, 3, 4]) & d['Start_Time'].notna()
             & d.Start_Lat.between(-90, 90) & d.Start_Lng.between(-180, 180))
    log['invalid_target_time_or_coordinates'] = int((~valid).sum())
    d = d.loc[valid].copy()
    if snapshots is not None:
        snapshots['validated'] = d.copy()
    # Hash original precision, before float32 conversion. Same time+point is one event
    # regardless of provider/ID/label. Keep first deterministically across chunks.
    d['event_hash'] = pd.util.hash_pandas_object(
        d[['Start_Time', 'Start_Lat', 'Start_Lng']], index=False).astype('uint64')
    d['Severity'] = d['Severity'].astype('int8')
    for c in WEATHER_NUM:
        d[c] = pd.to_numeric(d.get(c, pd.Series(np.nan, index=d.index)), errors='coerce')
    # Conservative physical checks, not statistical trimming of severe events.
    bounds = {'Temperature(F)': (-100, 140), 'Humidity(%)': (0, 100),
              'Pressure(in)': (15, 35), 'Visibility(mi)': (0, 150),
              'Wind_Speed(mph)': (0, 250), 'Precipitation(in)': (0, 50)}
    for c, (lo, hi) in bounds.items():
        invalid = d[c].notna() & ~d[c].between(lo, hi)
        log['invalid_' + c] = int(invalid.sum())
        d.loc[invalid, c] = np.nan
    wt = pd.to_datetime(d.get('Weather_Timestamp', pd.Series(pd.NaT, index=d.index)),
                        format='mixed', errors='coerce')
    age = (d.Start_Time - wt).dt.total_seconds() / 3600
    unavailable = age.isna() | (age < 0) | (age > 3)
    log['weather_future'] = int((age < 0).sum())
    log['weather_stale_or_missing_time'] = int((age.isna() | (age > 3)).sum())
    d['Weather_Unavailable'] = unavailable.astype('float32')
    for c in WEATHER_CAT:
        if c not in d:
            d[c] = 'Unknown'
    d.loc[unavailable, WEATHER_NUM] = np.nan
    d.loc[unavailable, WEATHER_CAT] = 'Unknown'
    d['Precipitation_Missing'] = d['Precipitation(in)'].isna().astype('float32')
    # Missing precipitation remains missing: zero would falsely mean dry weather.
    wind = d.Wind_Direction.astype('string').str.strip().str.upper()
    wind = wind.replace({'NORTH': 'N', 'SOUTH': 'S', 'WEST': 'W', 'EAST': 'E',
                         'VARIABLE': 'VAR'})
    allowed = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW',
               'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'CALM', 'VAR']
    d['Wind_Direction'] = wind.where(wind.isin(allowed), 'Unknown')
    if snapshots is not None:
        snapshots['weather'] = d.copy()
    d['Weather_Group'] = weather_group(d.Weather_Condition)
    for c in ROAD:
        v = d.get(c, pd.Series(np.nan, index=d.index)).astype('string').str.lower()
        d[c] = v.map({'true': 1, 'false': 0, '1': 1, '0': 0,
                       '1.0': 1, '0.0': 0}).astype('float32')
    d['Road_Feature_Count'] = d[ROAD].sum(axis=1, min_count=len(ROAD))
    dt = d.Start_Time.dt
    for c, v in [('Year', dt.year), ('Month', dt.month), ('Hour', dt.hour), ('Weekday', dt.dayofweek)]:
        d[c] = v
    d['Is_Weekend'] = (d.Weekday >= 5).astype('float32')
    d['Is_Rush_Hour'] = ((d.Weekday < 5) & d.Hour.isin([7, 8, 9, 16, 17, 18, 19])).astype('float32')
    street = d.get('Street', pd.Series('', index=d.index)).astype('string')
    d['Is_Highway'] = street.str.contains(r'\b(?:I-\d+|US-?\d+|Interstate|Hwy|Highway|Fwy|Freeway)\b',
                                            case=False, na=False).astype('float32')
    for c in CAT + ['ID', 'Source']:
        if c not in d:
            d[c] = 'Unknown'
        d[c] = d[c].astype('string').str.strip().replace('', 'Unknown').fillna('Unknown')
    d['State'] = d.State.str.upper()
    d[NUM] = d[NUM].astype('float32')
    if snapshots is not None:
        snapshots['engineered'] = d.copy()
    log['output_rows'] = len(d)
    return d[FEATURES + META], log
