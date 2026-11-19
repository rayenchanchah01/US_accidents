"""Inference parity, missingness, input validation and real HTTP predictions."""
from functools import partial
from http.server import ThreadingHTTPServer
import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pytest

from accidents.cleaning import FEATURES, ROAD, WEATHER_NUM
from accidents.paths import DASHBOARD, MODELS
from accidents.predict import get_predictor, prepare_input
from serve_dashboard import DashboardHandler


def incident():
    return {'State': 'OH', 'City': 'Dayton', 'County': 'Montgomery',
            'Start_Time': '2022-06-06T08:30', 'Start_Lat': 39.758274,
            'Start_Lng': -84.230507, 'Is_Highway': 0}


def test_missing_context_and_derived_features():
    frame, warnings = prepare_input(incident())
    row = frame.iloc[0]
    assert list(frame.columns) == FEATURES
    assert row[WEATHER_NUM + ROAD].isna().all()
    assert pd.isna(row.Road_Feature_Count)
    assert row.Weather_Unavailable == row.Precipitation_Missing == 1
    assert (row.Year, row.Month, row.Hour, row.Weekday, row.Is_Weekend, row.Is_Rush_Hour) == (2022, 6, 8, 0, 0, 1)
    assert row.Weather_Group == 'Unknown'
    assert len(warnings) == 2


def test_known_context_is_not_masked_or_replaced():
    values = {**incident(), **dict.fromkeys(ROAD, 0), 'Traffic_Signal': 1,
              'Start_Time': '2022-06-05T08:30', 'Is_Highway': 1,
              'Pressure(in)': 29.66, 'Precipitation(in)': 0,
              'Weather_Group': 'Snow / ice', 'Wind_Direction': 'N'}
    frame, _ = prepare_input(values)
    row = frame.iloc[0]
    assert row.Road_Feature_Count == row.Is_Highway == row.Is_Weekend == 1
    assert row.Is_Rush_Hour == row.Weather_Unavailable == row.Precipitation_Missing == 0
    assert row['Pressure(in)'] == pytest.approx(29.66)
    assert row.Weather_Group == 'Snow / ice'


@pytest.mark.parametrize('changes', [
    {'Start_Lat': None}, {'Start_Lat': 91}, {'Start_Lng': float('inf')},
    {'Pressure(in)': 99}, {'Humidity(%)': -1}, {'Visibility(mi)': True},
    {'Temperature(F)': 'warm'}, {'Traffic_Signal': 'false'}, {'Is_Highway': None},
    {'Start_Time': '2022-02-30T08:30'}, {'Start_Time': '2022-01-01'},
    {'Start_Time': '2022-01-01T08:30Z'}, {'State': ''}, {'City': '   '},
    {'County': ['Montgomery']}, {'Sunrise_Sunset': 'Dawn'},
    {'Weather_Group': 'Blizzard'}, {'Wind_Direction': 'Northern'},
    {'Severity': 4}, {'Year': 2022}, {'Distance(mi)': 1},
])
def test_reject_invalid_or_leaking_inputs(changes):
    with pytest.raises(ValueError):
        prepare_input({**incident(), **changes})


@pytest.fixture(scope='module')
def predictor():
    if not (MODELS / 'severity.joblib').exists():
        pytest.skip('Run training first')
    return get_predictor()


def test_predictions_match_saved_model_and_preview(predictor):
    example = predictor.example()['inputs']
    preview = json.loads((DASHBOARD / 'data/pipeline.json').read_text(encoding='utf-8'))['cleaned'][0]
    expected_frame = pd.DataFrame([preview])[FEATURES]
    expected = predictor.model.predict_proba(expected_frame)[0]
    response = predictor.predict(example)
    np.testing.assert_allclose([x['score'] for x in response['scores']], expected, atol=1e-12)
    assert response['severity'] == int(predictor.model.classes_[expected.argmax()])
    assert response['scores_calibrated'] is False
    assert len(response['features']) == 36
    assert set(response['features']) == set(FEATURES)
    changed = {**example, 'Start_Time': '2022-08-05T17:15', 'Is_Highway': 1}
    altered = predictor.predict(changed)
    direct = predictor.model.predict_proba(prepare_input(changed)[0])[0]
    np.testing.assert_allclose([x['score'] for x in altered['scores']], direct, atol=1e-12)
    assert not np.allclose(expected, direct)


def test_schema_uses_training_vocabulary_and_measured_importance(predictor):
    schema = predictor.schema()
    assert schema['importance'][0]['feature'] == 'State'
    # Values depend on the uploaded dataset; require a real, descending measurement.
    drops = [row['macro_f1_drop'] for row in schema['importance']]
    assert drops[0] > 0.05 and drops == sorted(drops, reverse=True)
    assert 'Dayton' in schema['locations']['OH']['cities']
    assert 'Montgomery' in schema['locations']['OH']['counties']
    assert 'Severity' not in schema['example']['inputs']
    json.dumps(schema, allow_nan=False)


def test_unknown_categories_and_out_of_period_are_visible(predictor):
    result = predictor.predict({**incident(), 'Start_Time': '2026-11-19T08:30', 'City': 'Unseen town'})
    assert any('outside' in message for message in result['warnings'])
    assert any('not seen' in message for message in result['warnings'])
    assert result['features']['Temperature(F)'] is None
    json.dumps(result, allow_nan=False)
    normalized = predictor.predict({**incident(), 'State': 'oh', 'City': 'dayton', 'County': 'montgomery'})
    assert normalized['features']['City'] == 'Dayton'
    assert not any('not seen' in message for message in normalized['warnings'])
    with pytest.raises(ValueError, match='state'):
        predictor.predict({**incident(), 'State': 'ZZ'})


@pytest.fixture(scope='module')
def server(predictor):
    handler = partial(DashboardHandler, directory=str(DASHBOARD))
    with ThreadingHTTPServer(('127.0.0.1', 0), handler) as httpd:
        worker = Thread(target=httpd.serve_forever, daemon=True)
        worker.start()
        yield f'http://127.0.0.1:{httpd.server_port}'
        httpd.shutdown()
        worker.join()


def call_api(server, path, payload=None, content_type='application/json'):
    request = Request(server + path, data=payload, headers={'Content-Type': content_type})
    try:
        response = urlopen(request, timeout=30)
    except HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)


def test_http_serves_predictions_and_validation_errors(server):
    status, schema = call_api(server, '/api/prediction-schema')
    assert status == 200
    status, result = call_api(server, '/api/predict', json.dumps(schema['example']['inputs']).encode())
    assert status == 200 and result['severity'] in (1, 2, 3, 4)
    assert sum(x['score'] for x in result['scores']) == pytest.approx(1)
    for raw, expected in [(b'{', 400), (b'[]', 400), (b'null', 400), (b'{}', 400), (b' ' * 16385, 413)]:
        status, result = call_api(server, '/api/predict', raw)
        assert status == expected and result['error']
    assert call_api(server, '/api/predict', b'{}', 'text/plain')[0] == 415
    assert call_api(server, '/api/no-such-endpoint', b'{}')[0] == 404
    with urlopen(server + '/', timeout=5) as response:
        assert response.status == 200
        assert b'id="try-model"' in response.read()


def test_http_reports_missing_model(server, monkeypatch):
    def unavailable():
        raise FileNotFoundError('missing model')
    monkeypatch.setattr('accidents.predict.get_predictor', unavailable)
    status, result = call_api(server, '/api/predict', json.dumps(incident()).encode())
    assert status == 503 and 'saved model is unavailable' in result['error']
