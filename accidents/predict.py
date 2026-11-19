"""Validated, single-incident inference through the saved training pipeline."""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import json
import math
from threading import Lock

import joblib
import pandas as pd

from .cleaning import CAT, FEATURES, ROAD, WEATHER_NUM, clean_chunk
from .paths import DASHBOARD, MODELS, REPORTS


NUMERIC_LIMITS = {
    'Start_Lat': (-90, 90), 'Start_Lng': (-180, 180),
    'Temperature(F)': (-100, 140), 'Humidity(%)': (0, 100),
    'Pressure(in)': (15, 35), 'Visibility(mi)': (0, 150),
    'Wind_Speed(mph)': (0, 250), 'Precipitation(in)': (0, 50),
}
WEATHER_CONDITIONS = {
    'Unknown': 'Unknown', 'Clear': 'Clear', 'Cloudy': 'Overcast',
    'Rain': 'Rain', 'Snow / ice': 'Snow', 'Storm': 'Thunderstorm',
    'Low visibility': 'Fog', 'Windy': 'Windy', 'Other': 'Other',
}
WIND_DIRECTIONS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S',
                   'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'CALM', 'VAR', 'Unknown']
INPUTS = {'Start_Time', 'Is_Highway', *NUMERIC_LIMITS, *ROAD, *CAT}
_load_lock = Lock()


def text_value(payload, name, *, required=False):
    value = payload.get(name)
    if value is None or value == '':
        if required:
            raise ValueError(f'{name} is required.')
        return 'Unknown'
    if not isinstance(value, str) or len(value) > 120:
        raise ValueError(f'{name} must be text of at most 120 characters.')
    value = value.strip()
    if not value:
        if required:
            raise ValueError(f'{name} is required.')
        return 'Unknown'
    return value


def prepare_input(payload):
    """Reuse row-local cleaning; no target or arbitrary default enters the model."""
    if not isinstance(payload, dict):
        raise ValueError('Send one incident as a JSON object.')
    if extra := set(payload) - INPUTS:
        raise ValueError(f'Unsupported inputs: {", ".join(sorted(extra))}.')
    timestamp = text_value(payload, 'Start_Time', required=True)
    try:
        when = datetime.fromisoformat(timestamp)
    except ValueError:
        raise ValueError('Enter a valid local incident date and time.') from None
    if when.tzinfo is not None or 'T' not in timestamp or not 1900 <= when.year <= 2100:
        raise ValueError('Use a local date and time without a timezone, between 1900 and 2100.')

    # clean_chunk validates labeled training rows. These placeholders are discarded
    # before inference, and are never accepted as user inputs or model features.
    raw = {'ID': 'interactive-input', 'Severity': 1, 'Start_Time': timestamp}
    for name, (lo, hi) in NUMERIC_LIMITS.items():
        value = payload.get(name)
        if value is None or value == '':
            if name in ('Start_Lat', 'Start_Lng'):
                raise ValueError(f'{name} is required.')
            raw[name] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{name} must be a number.')
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'{name} must be between {lo} and {hi}.')
        raw[name] = value
    for name in ROAD + ['Is_Highway']:
        value = payload.get(name)
        if value is not None and (type(value) not in (int, bool) or value not in (0, 1)):
            raise ValueError(f'{name} must be yes, no, or unknown.')
        if name == 'Is_Highway':
            if value is None:
                raise ValueError('Road type (Is_Highway) is required.')
            raw['Street'] = 'Highway' if value else 'Local road'
        else:
            raw[name] = value
    for name in ('State', 'City', 'County'):
        raw[name] = text_value(payload, name, required=True)
    raw['State'] = raw['State'].upper()
    weather = text_value(payload, 'Weather_Group')
    wind = text_value(payload, 'Wind_Direction')
    light = text_value(payload, 'Sunrise_Sunset')
    for name, value, options in [('Weather_Group', weather, WEATHER_CONDITIONS),
                                  ('Wind_Direction', wind, WIND_DIRECTIONS),
                                  ('Sunrise_Sunset', light, ['Day', 'Night', 'Unknown'])]:
        if value not in options:
            raise ValueError(f'Choose a supported {name} value.')
    raw.update(Weather_Condition=WEATHER_CONDITIONS[weather], Wind_Direction=wind,
               Sunrise_Sunset=light)
    has_weather = weather != 'Unknown' or wind != 'Unknown' or any(raw[c] is not None for c in WEATHER_NUM)
    raw['Weather_Timestamp'] = timestamp if has_weather else None
    frame, _ = clean_chunk(pd.DataFrame([raw]))
    warnings = []
    if not datetime(2016, 1, 1) <= when < datetime(2023, 4, 1):
        warnings.append('This date is outside the 2016–March 2023 study period. Performance on this date is unverified.')
    missing_roads = sum(payload.get(c) is None for c in ROAD)
    if missing_roads:
        warnings.append(f'{missing_roads} road features are unknown; their values and the total road-feature count stay missing.')
    if not has_weather:
        warnings.append('Weather is unknown. The model receives missing weather values, not assumed conditions.')
    return frame[FEATURES], warnings


class Predictor:
    def __init__(self):
        self.model = joblib.load(MODELS / 'severity.joblib')
        self.card = json.loads((MODELS / 'model_card.json').read_text(encoding='utf-8'))
        self.results = json.loads((REPORTS / 'results.json').read_text(encoding='utf-8'))
        self.categories = dict(zip(CAT, self.model.named_steps['features'].categories_.categories_))
        self.lock = Lock()

    def schema(self):
        locations = {str(state): {'cities': [], 'counties': []} for state in self.categories['State']}
        for field, key in [('City', 'cities'), ('County', 'counties')]:
            for category in self.categories[field]:
                state, _, name = str(category).partition('/')
                if state in locations and name != 'Unknown':
                    locations[state][key].append(name)
        return {
            'model': self.card['selected_model'], 'locations': locations,
            'importance': self.results['importance'], 'numeric_limits': NUMERIC_LIMITS,
            'weather_groups': list(WEATHER_CONDITIONS), 'wind_directions': WIND_DIRECTIONS,
            'road_features': ROAD, 'example': self.example(),
        }

    @staticmethod
    def example():
        """A real, retained preview row; its label is deliberately omitted."""
        path = DASHBOARD / 'data' / 'pipeline.json'
        if not path.exists():
            return None
        preview = json.loads(path.read_text(encoding='utf-8'))
        row = preview['cleaned'][0]
        values = {name: row[name] for name in INPUTS}
        for name in ROAD + ['Is_Highway']:
            values[name] = int(values[name]) if values[name] is not None else None
        return {'source': f"Dataset preview incident {row['ID']}", 'inputs': values}

    def predict(self, payload):
        frame, warnings = prepare_input(payload)
        if frame.iloc[0]['State'] not in self.categories['State']:
            raise ValueError('Choose a state represented in the trained model.')
        # Match capitalization to the fitted vocabulary without fitting anything.
        for name in ('City', 'County'):
            state = frame.iloc[0]['State']
            value = frame.iloc[0][name]
            full = f'{state}/{value}'
            known = {str(c).casefold(): str(c) for c in self.categories[name]}
            if full.casefold() in known:
                frame.loc[0, name] = known[full.casefold()].split('/', 1)[1]
            else:
                warnings.append(f'{name} "{value}" was not seen in training for {state}; it is encoded as an unknown category.')
        with self.lock:
            scores = self.model.predict_proba(frame)[0]
        classes = self.model.classes_
        predicted = int(classes[scores.argmax()])
        # DataFrame.to_json converts missing values to JSON null, never NaN.
        features = json.loads(frame.to_json(orient='records'))[0]
        return {
            'severity': predicted, 'model': self.card['selected_model'],
            'scores': [{'severity': int(label), 'score': float(score)} for label, score in zip(classes, scores)],
            'scores_calibrated': False, 'warnings': warnings, 'features': features,
        }


@lru_cache(maxsize=1)
def _cached_predictor():
    return Predictor()


def get_predictor():
    # lru_cache alone permits duplicate construction on concurrent first requests.
    with _load_lock:
        return _cached_predictor()
