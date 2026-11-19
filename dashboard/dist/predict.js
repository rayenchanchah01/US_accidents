'use strict';
(() => {
  const el = id => document.getElementById(id);
  const form = el('predict-form');
  const controls = el('predict-controls');
  let schema, hasResult = false, busy = false;
  const labels = {
    'Pressure(in)': 'Pressure (inHg)', 'Visibility(mi)': 'Visibility (mi)',
    'Precipitation(in)': 'Precipitation (in)', 'Temperature(F)': 'Temperature (°F)',
    'Humidity(%)': 'Humidity (%)', 'Wind_Speed(mph)': 'Wind speed (mph)',
    Weather_Group: 'Weather conditions', Wind_Direction: 'Wind direction',
    Amenity: 'Amenity', Bump: 'Speed bump', Crossing: 'Crossing', Give_Way: 'Yield sign',
    No_Exit: 'No exit', Railway: 'Railway', Roundabout: 'Roundabout', Station: 'Station',
    Stop: 'Stop sign', Traffic_Calming: 'Traffic calming',
  };
  const featureLabels = {State:'State', Year:'Year', Month:'Month', County:'County', City:'City',
    Is_Highway:'Highway status', Hour:'Hour', Traffic_Signal:'Traffic signal'};
  const scoreColors = ['#386579', '#426044', '#8e6026', '#9b4938'];

  async function request(url, payload) {
    const options = {signal: AbortSignal.timeout(60000)};
    if (payload !== undefined) Object.assign(options, {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload),
    });
    const response = await fetch(url, options);
    if (!(response.headers.get('content-type') || '').includes('application/json')) {
      throw new Error('Live predictions need the local Python server. Run python serve_dashboard.py and open http://127.0.0.1:8765.');
    }
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'The model is unavailable. Please retry.');
    return result;
  }

  function addField(container, name, options) {
    const label = document.createElement('label');
    label.textContent = labels[name] || name.replaceAll('_', ' ');
    const field = document.createElement(options ? 'select' : 'input');
    field.name = name;
    if (options) {
      options.forEach(([value, title]) => field.add(new Option(title, value)));
    } else {
      field.type = 'number'; field.step = 'any'; field.placeholder = 'Unknown';
      [field.min, field.max] = schema.numeric_limits[name];
    }
    label.append(field); el(container).append(label);
  }

  function locationOptions(clear = false) {
    const location = schema.locations[el('predict-state').value];
    for (const [field, list, key] of [['predict-city','predict-cities','cities'], ['predict-county','predict-counties','counties']]) {
      el(list).replaceChildren(...(location?.[key] || []).map(value => new Option(value, value)));
      el(field).placeholder = location ? 'Type or choose a suggestion' : 'Choose a state first';
      if (clear) el(field).value = '';
    }
  }

  function clearResult(message = 'Complete the incident details and run a prediction.') {
    hasResult = false;
    el('predict-result').hidden = true;
    el('predict-error').hidden = true;
    el('predict-output-title').textContent = 'A scenario to explore.';
    el('predict-status').textContent = message;
  }

  function readInputs() {
    const values = Object.fromEntries(new FormData(form));
    for (const name of Object.keys(schema.numeric_limits)) values[name] = values[name] === '' ? null : Number(values[name]);
    for (const name of [...schema.road_features, 'Is_Highway']) values[name] = values[name] === '' ? null : Number(values[name]);
    return values;
  }

  function renderResult(result) {
    const level = result.severity;
    el('predict-output-title').textContent = 'The model’s prediction.';
    el('predict-status').textContent = 'Scored using your submitted incident details.';
    el('predict-level').textContent = `Level ${level}`;
    el('predict-level').style.color = scoreColors[level - 1];
    el('predict-impact').textContent = {1:'Least traffic impact', 2:'Traffic impact · level two', 3:'Traffic impact · level three', 4:'Highest traffic impact'}[level];
    el('predict-scores').replaceChildren(...result.scores.map(item => {
      const row = document.createElement('div'); row.className = 'predict-score';
      row.classList.toggle('is-selected', item.severity === level);
      const label = document.createElement('span'); label.textContent = `Level ${item.severity}`;
      const track = document.createElement('span'); track.className = 'predict-score-track'; track.setAttribute('aria-hidden','true');
      const fill = document.createElement('i'); fill.style.width = `${item.score * 100}%`; fill.style.background = scoreColors[item.severity - 1]; track.append(fill);
      const score = document.createElement('strong'); score.textContent = `${(item.score * 100).toFixed(1)}%`;
      row.append(label, track, score); return row;
    }));
    el('predict-warnings').replaceChildren(...result.warnings.map(message => {
      const li = document.createElement('li'); li.textContent = message; return li;
    }));
    el('predict-feature-values').replaceChildren(...Object.entries(result.features).map(([name, value]) => {
      const row = document.createElement('div'), key = document.createElement('dt'), val = document.createElement('dd');
      key.textContent = name;
      val.textContent = value === null ? 'Missing' : typeof value === 'number' ? String(Number(value.toFixed(6))) : value;
      row.append(key, val); return row;
    }));
    el('predict-result').hidden = false; hasResult = true;
  }

  async function connect() {
    el('predict-retry').hidden = true;
    el('predict-connection').textContent = 'Loading the saved model…';
    try {
      schema = await request('/api/prediction-schema');
      el('predict-state').replaceChildren(new Option('Choose a state', ''), ...Object.keys(schema.locations)
        .sort((a,b) => stateName(a).localeCompare(stateName(b))).map(state => new Option(stateName(state), state)));
      el('predict-weather').replaceChildren(); el('predict-roads').replaceChildren();
      for (const name of ['Pressure(in)', 'Visibility(mi)', 'Precipitation(in)', 'Temperature(F)', 'Humidity(%)', 'Wind_Speed(mph)']) addField('predict-weather', name);
      addField('predict-weather', 'Weather_Group', schema.weather_groups.map(value => [value,value]));
      addField('predict-weather', 'Wind_Direction', ['Unknown', ...schema.wind_directions.filter(v => v !== 'Unknown')].map(value => [value,value]));
      for (const name of schema.road_features.filter(name => !['Traffic_Signal','Junction'].includes(name))) {
        addField('predict-roads', name, [['','Unknown'], ['1','Yes'], ['0','No']]);
      }
      el('predict-importance').replaceChildren(...schema.importance.slice(0,8).map(item => {
        const li = document.createElement('li'); li.textContent = featureLabels[item.feature] || item.feature;
        const drop = document.createElement('span'); drop.textContent = `${item.macro_f1_drop.toFixed(3)} F1 drop`; li.append(drop); return li;
      }));
      el('predict-model-name').textContent = schema.model;
      el('predict-connection').textContent = 'Model ready. Predictions run locally using the saved training pipeline.';
      el('predict-example').hidden = !schema.example;
      controls.disabled = false;
    } catch (error) {
      el('predict-connection').textContent = error.name === 'TypeError'
        ? 'Could not reach the model. Start python serve_dashboard.py, then open http://127.0.0.1:8765 and retry.'
        : error.message;
      el('predict-retry').hidden = false;
    }
  }

  el('predict-retry').addEventListener('click', connect);
  el('predict-state').addEventListener('change', () => locationOptions(true));
  el('predict-example').addEventListener('click', () => {
    for (const [name, value] of Object.entries(schema.example.inputs)) {
      form.elements.namedItem(name).value = value === null ? '' : String(value);
    }
    locationOptions();
    el('predict-example-note').textContent = `${schema.example.source}. All inputs are editable; the original label is not sent to the model.`;
    clearResult('Example loaded. Run the model, or change the inputs first.');
  });
  form.addEventListener('input', () => {
    if (hasResult) clearResult('Inputs changed. Run a new prediction to update the result.');
    el('predict-error').hidden = true;
    el('predict-example-note').textContent = 'Your scenario. Unknown optional values remain missing.';
  });
  form.addEventListener('reset', () => {
    if (busy) return;
    clearResult();
    el('predict-example-note').textContent = 'Enter your own values, or start with an incident from the dataset preview.';
    setTimeout(() => locationOptions(), 0);
  });
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (busy || !form.reportValidity()) return;
    const payload = readInputs();
    busy = true; controls.disabled = true;
    clearResult('Running the saved model on your inputs…');
    el('predict-submit').textContent = 'Predicting…';
    el('predict-output-title').textContent = 'Reading your scenario.';
    el('predict-result').setAttribute('aria-busy', 'true');
    try {
      renderResult(await request('/api/predict', payload));
    } catch (error) {
      clearResult('Prediction was not completed. Check the inputs or retry.');
      el('predict-error').textContent = error.name === 'TypeError' ? 'Could not reach the local model server. Check that it is running, then retry.' : error.message;
      el('predict-error').hidden = false;
    } finally {
      busy = false; controls.disabled = false;
      el('predict-submit').textContent = 'Predict severity';
      el('predict-result').setAttribute('aria-busy', 'false');
    }
  });
  connect();
})();
