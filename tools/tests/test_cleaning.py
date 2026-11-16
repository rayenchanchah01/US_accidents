import numpy as np
import pandas as pd
from accidents.cleaning import clean_chunk, FEATURES


def rows():
    return pd.DataFrame({'ID': ['a', 'b', 'c', 'd'], 'Severity': [1, 2, 3, 4],
        'Start_Time': ['2020-01-01 12:00:00', '2020-01-01 12:00:00.123456',
                       '2020-01-02 12:00:00', '2020-01-03 12:00:00'],
        'Start_Lat': [35., 36., 37., 38.], 'Start_Lng': [-100.] * 4,
        'Temperature(F)': [60, 65, 999, 50], 'Humidity(%)': [50, 51, 102, 20],
        'Precipitation(in)': [np.nan, 1, 0, 0],
        'Weather_Timestamp': ['2020-01-01 11:30:00', '2020-01-01 12:30:00',
                              '2020-01-02 11:30:00', '2020-01-03 06:00:00'],
        'Wind_Direction': ['West', 'S', 'Calm', 'Variable'],
        'Weather_Condition': ['Fair', 'Rain', 'Snow', 'Cloudy'],
        'Street': ['I-95', 'US-101', 'Main Street', 'State Highway 2'],
        'City': [None, 'Town', 'Town', 'Town']})


def test_missing_values_and_future_weather():
    d, log = clean_chunk(rows())
    assert len(d) == 4
    assert d.City.iloc[0] == 'Unknown'
    assert np.isnan(d['Precipitation(in)'].iloc[0])
    assert d.Precipitation_Missing.iloc[0] == 1
    assert np.isnan(d['Temperature(F)'].iloc[1])
    assert d.Weather_Group.iloc[1] == 'Unknown'
    assert np.isnan(d['Temperature(F)'].iloc[2])
    assert np.isnan(d['Temperature(F)'].iloc[3])
    assert log['weather_future'] == 1
    assert d.Wind_Direction.iloc[0] == 'W'
    assert d.Is_Highway.tolist() == [1, 1, 0, 1]


def test_invalid_target_never_imputed():
    r = rows(); r.loc[1, 'Severity'] = np.nan
    d, log = clean_chunk(r)
    assert len(d) == 3
    assert log['invalid_target_time_or_coordinates'] == 1


def test_leakage_and_row_local_invariance():
    r = rows(); r['End_Time'] = r.Start_Time; r['Distance(mi)'] = r.Severity
    r['Description'] = 'severity four'; r['Source'] = 'provider'
    all_rows, _ = clean_chunk(r)
    one_row, _ = clean_chunk(r.iloc[:1])
    pd.testing.assert_frame_equal(all_rows.iloc[:1], one_row)
    assert not set(['Severity', 'Source', 'ID', 'End_Time', 'Distance(mi)', 'Description']) & set(FEATURES)


def test_event_key_ignores_label_and_provider():
    r = rows().iloc[[0, 0]].copy(); r.index = [0, 1]
    r.loc[1, 'ID'] = 'other'; r.loc[1, 'Severity'] = 4
    d, _ = clean_chunk(r)
    assert d.event_hash.nunique() == 1


def test_event_hash_consistent_across_datetime_precision():
    raw = rows()
    whole, _ = clean_chunk(raw)
    separate, _ = clean_chunk(raw.iloc[[0]])
    assert whole.event_hash.iloc[0] == separate.event_hash.iloc[0]
    assert str(whole.Start_Time.dtype) == 'datetime64[ns]'


def test_visualization_snapshots_do_not_change_cleaning():
    raw = rows()
    expected, expected_log = clean_chunk(raw)
    snapshots = {}
    actual, actual_log = clean_chunk(raw, snapshots=snapshots)
    pd.testing.assert_frame_equal(expected, actual)
    assert expected_log == actual_log
    assert snapshots['validated']['Temperature(F)'].iloc[1] == 65
    assert np.isnan(snapshots['weather']['Temperature(F)'].iloc[1])
    assert snapshots['engineered'].Is_Highway.tolist() == actual.Is_Highway.tolist()
