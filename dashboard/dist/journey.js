'use strict';
let journey, datasetPage = 0, datasetAscending = true, visibleFeatures = [], filteredDataset = [];
const cellValue = value => value === null || value === undefined ? '<span class="missing-value">Missing</span>' : esc(typeof value === 'number' ? Number(value.toFixed(5)) : value);
function simpleTable(rows, columns = Object.keys(rows[0] || {})) {
  return `<div class="table-scroll"><table class="data-table"><thead><tr>${columns.map(c=>`<th scope="col">${esc(c)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${columns.map(c=>`<td>${cellValue(r[c])}</td>`).join('')}</tr>`).join('') || `<tr><td colspan="${columns.length || 1}">No matching rows.</td></tr>`}</tbody></table></div>`;
}
function pairedTables(before, after, note = '') {
  return `<div class="transformation-pair"><div><div class="table-label"><span>BEFORE</span> Source values</div>${simpleTable(before.slice(0,10))}</div><div><div class="table-label after"><span>AFTER</span> Transformed values</div>${simpleTable(after.slice(0,10))}</div></div><p class="panel-footnote">${esc(note || 'Up to 10 real example rows. Scroll each table horizontally to inspect every feature. Missing means null, not zero.')}</p>`;
}
function datasetView() { return $('dataset-view').value; }
function refreshDatasetOptions(resetColumns = true) {
  const view = datasetView(), columns = journey.columns[view];
  if(resetColumns) visibleFeatures = [...journey.defaults[view]];
  $('dataset-state').innerHTML = '<option value="all">All states</option>' + [...new Set(journey[view].map(r=>r.State))].filter(Boolean).sort().map(s=>`<option value="${esc(s)}">${esc(stateName(s))}</option>`).join('');
  $('feature-filter').innerHTML = '<option value="">Any feature</option>' + columns.map(c=>`<option>${esc(c)}</option>`).join('');
  $('dataset-sort').innerHTML = '<option value="row_number">Source row order</option>' + columns.filter(c=>c!=='row_number').map(c=>`<option>${esc(c)}</option>`).join('');
  $('dataset-columns').innerHTML = columns.map(c=>`<label><input type="checkbox" value="${esc(c)}" ${visibleFeatures.includes(c)?'checked':''}>${esc(c)}</label>`).join('');
  $('sample-scope').textContent = `${fmt(journey.sampling.sample_rows)} records sampled across ${fmt(journey.sampling.source_rows)} source rows. Filters apply to this preview; study metrics use the full dataset.`;
  datasetPage = 0; renderDataset();
}
function renderDataset() {
  const view = datasetView(), stateValue = $('dataset-state').value, severity = $('dataset-severity').value;
  const query = $('dataset-search').value.trim().toLowerCase(), feature = $('feature-filter').value, op = $('feature-operator').value, value = $('feature-value').value.trim();
  $('feature-value').disabled = op === 'missing';
  filteredDataset = journey[view].filter(r=>{
    if(stateValue !== 'all' && r.State !== stateValue) return false;
    if(severity !== 'all' && Number(r.Severity) !== Number(severity)) return false;
    if(query && !Object.values(r).some(v=>v !== null && String(v).toLowerCase().includes(query))) return false;
    if(feature && (value || op === 'missing')) {
      const v = r[feature];
      if(op === 'missing') return v === null || v === undefined || v === '';
      if(v === null || v === undefined) return false;
      if(op === 'contains') return String(v).toLowerCase().includes(value.toLowerCase());
      if(op === 'eq') return String(v).toLowerCase() === value.toLowerCase();
      if(!Number.isFinite(Number(v)) || !Number.isFinite(Number(value))) return false;
      return op === 'gt' ? Number(v) > Number(value) : Number(v) < Number(value);
    }
    return true;
  });
  const sort = $('dataset-sort').value;
  filteredDataset.sort((a,b)=>{
    if(a[sort] == null) return b[sort] == null ? 0 : 1;
    if(b[sort] == null) return -1;
    const compared = typeof a[sort] === 'number' && typeof b[sort] === 'number' ? a[sort]-b[sort] : String(a[sort]).localeCompare(String(b[sort]));
    return datasetAscending ? compared : -compared;
  });
  const size = Number($('dataset-size').value), count = filteredDataset.length, pages = Math.max(1,Math.ceil(count/size));
  datasetPage = Math.min(datasetPage,pages-1);
  const start = datasetPage*size, rows = filteredDataset.slice(start,start+size);
  $('dataset-head').innerHTML = `<tr>${visibleFeatures.map(c=>`<th scope="col">${esc(c)}</th>`).join('')}</tr>`;
  $('dataset-rows').innerHTML = rows.map(r=>`<tr>${visibleFeatures.map(c=>`<td>${cellValue(r[c])}</td>`).join('')}</tr>`).join('') || `<tr><td colspan="${visibleFeatures.length}">No records match. Reset the view or change your filters.</td></tr>`;
  $('dataset-range').textContent = count ? `${fmt(start+1)}–${fmt(Math.min(start+size,count))} of ${fmt(count)} matching preview records` : 'No matching preview records';
  $('dataset-sample-note').textContent = `${fmt(journey[view].length)} ${view === 'raw' ? 'source' : 'retained'} preview rows · seed 42`;
  $('dataset-page').textContent = `${datasetPage+1} / ${pages}`;
  $('dataset-prev').disabled = datasetPage === 0; $('dataset-next').disabled = datasetPage+1 >= pages;
  $('dataset-export').disabled = !count;
  $('column-count').textContent = `${visibleFeatures.length} selected`;
  document.querySelectorAll('#dataset-columns input').forEach(input=>{input.checked = visibleFeatures.includes(input.value); input.disabled = input.checked && visibleFeatures.length === 1;});
}
function renderCleaning(stage = 'validation') {
  const e = journey.examples, c = data.cleaning;
  const stages = {
    validation: ['Require a valid target, time and location.', 'Parse timestamps and numeric coordinates. Keep Severity 1–4 and valid coordinate bounds. Optional missing fields do not cause row deletion.', `${fmt(c.counts.invalid_target_time_or_coordinates)} invalid rows removed in this run. These valid examples remain usable.`, e.validation_before, e.validation_after],
    weather: ['Use weather that was available at incident time.', 'Mask weather observed after the incident, more than three hours before it, or without a timestamp. Values outside conservative physical ranges become missing. Normalize wind direction; missing precipitation stays missing.', `${fmt(c.counts.weather_future)} future-weather rows and ${fmt(c.counts.weather_stale_or_missing_time)} stale/missing-time rows masked, before deduplication.`, e.weather_before, e.weather_after],
    duplicates: ['One exact time and location, one retained event.', 'Deduplicate across the entire source archive by original-precision Start_Time + latitude + longitude. Keep the first occurrence, regardless of provider, ID or severity.', `${fmt(c.counts.repeated_event_rows_removed)} repeated events removed; ${fmt(c.clean_rows)} rows retained.`, journey.raw.filter(r=>!r.Retained).slice(0,10).map(r=>({ID:r.ID,Start_Time:r.Start_Time,Start_Lat:r.Start_Lat,Start_Lng:r.Start_Lng,Decision:'Removed: earlier event retained'})), journey.cleaned.slice(0,10).map(r=>({ID:r.ID,Start_Time:r.Start_Time,Start_Lat:r.Start_Lat,Start_Lng:r.Start_Lng,Decision:'Retained'}))],
    selection: ['Keep predictors that do not reveal the outcome.', 'Exclude end time, incident extent and narrative text. IDs and provider Source remain traceability metadata only; the target Severity never enters the feature matrix.', `${journey.features.length} model inputs retained. ID is shown in examples only to match records.`, e.selection_before, e.selection_after],
  };
  const [title,description,result,before,after] = stages[stage];
  $('clean-stage-title').textContent = title; $('clean-stage-description').textContent = description; $('clean-stage-result').textContent = result;
  $('clean-tables').innerHTML = stage === 'duplicates' ? `<div class="transformation-pair"><div><div class="table-label">ACTUAL REMOVED ROWS</div>${simpleTable(before)}</div><div><div class="table-label after">ACTUAL RETAINED ROWS</div>${simpleTable(after)}</div></div><p class="panel-footnote">Two different groups from the preview, checked against the full saved row manifest. Removed examples do not reappear in the retained table. This rule may merge independent incidents with identical time and coordinates.</p>` : pairedTables(before,after);
  document.querySelectorAll('[data-clean-stage]').forEach(b=>{const selected=b.dataset.cleanStage===stage;b.classList.toggle('selected',selected);b.setAttribute('aria-pressed',String(selected));});
}
function renderJourney() {
  const s=data.stats,c=data.cleaning,e=journey.examples;
  $('dataset-metrics').innerHTML = [[fmt(c.counts.input_rows),'SOURCE RECORDS','Entire source archive'],[fmt(c.clean_rows),'RETAINED EVENTS','After validation and deduplication'],[journey.features.length,'MODEL INPUTS','Location, weather, time & roads'],['4','SEVERITY CLASSES','Original traffic-impact labels']].map(([value,label,note],i)=>`<article class="metric ${i===0?'primary':''}"><div>${label}</div><strong>${value}</strong><p>${note}</p></article>`).join('');
  $('cleaning-flow').innerHTML = [[c.counts.input_rows,'Source records'],[c.counts.invalid_target_time_or_coordinates,'Invalid rows removed'],[c.counts.repeated_event_rows_removed,'Repeated events removed'],[c.clean_rows,'Retained events']].map(([n,label])=>`<div><strong>${fmt(n)}</strong><span>${label}</span></div>`).join('');
  $('missing-table').innerHTML = simpleTable(Object.entries(c.raw_missing_counts).sort((a,b)=>b[1]-a[1]).map(([Feature,Count])=>({Feature,Count,'Missing share':pct(Count/c.counts.input_rows)})));
  $('excluded-table').innerHTML = simpleTable(Object.entries(journey.excluded).map(([Column,Reason])=>({Column,Reason})));
  $('feature-rules').innerHTML = simpleTable(journey.feature_rules.map(([Feature,Source,Rule])=>({Feature,Source,Rule})));
  $('engineering-tables').innerHTML = pairedTables(e.engineering_before,e.engineering_after);
  $('selection-tables').innerHTML = pairedTables(e.selection_before,e.selection_after);
  const eventRows = (rows, decision) => rows.slice(0,10).map(r=>({ID:r.ID,Start_Time:r.Start_Time,Start_Lat:r.Start_Lat,Start_Lng:r.Start_Lng,Decision:decision}));
  $('duplicate-tables').innerHTML = `<div class="stage-result">${fmt(c.counts.repeated_event_rows_removed)} repeated events removed → ${fmt(c.clean_rows)} retained events</div><div class="transformation-pair"><div><div class="table-label">REMOVED ROWS</div>${simpleTable(eventRows(journey.raw.filter(r=>!r.Retained),'Removed: earlier event retained'))}</div><div><div class="table-label after">RETAINED ROWS</div>${simpleTable(eventRows(journey.cleaned,'Retained'))}</div></div>`;
  $('encoding-tables').innerHTML = pairedTables(e.encoding_before,e.encoding_after);
  $('development-label').innerHTML = `<b>80% DEVELOPMENT</b><span>${fmt(s.development_rows)} records</span>`;
  $('test-label').innerHTML = `<b>20% TEST</b><span>${fmt(s.test_rows)} records</span>`;
  $('split-description').textContent = `A stratified split with seed ${s.seed} preserves class proportions. Development is split again 80/20, then stratified subsets are capped for candidate fitting and validation. The winner is refitted on all ${fmt(s.development_rows)} development records before the final test evaluation.`;
  $('split-counts').innerHTML = simpleTable([{Partition:'Candidate fitting',Rows:s.tuning_rows,Use:'Fit each candidate'},{Partition:'Validation',Rows:s.validation_rows,Use:'Choose by macro F1'},{Partition:'Full development',Rows:s.development_rows,Use:'Refit the winner'},{Partition:'Test',Rows:s.test_rows,Use:'Final evaluation only'}]);
  $('split-table').innerHTML = simpleTable(e.split.slice(0,10));
  const sorted=[...s.validation_models].sort((a,b)=>b.macro_f1-a.macro_f1), second=sorted.find(r=>r.name!==s.selected_model);
  const reasoning=document.createElement('article'); reasoning.className='panel model-rationale';
  reasoning.innerHTML=`<div class="eyebrow">WHY THIS MODEL?</div><h3>${esc(s.selected_model)}</h3><p>It achieved the highest validation macro F1: <b>${s.selection.macro_f1.toFixed(4)}</b>${second ? `, compared with ${second.macro_f1.toFixed(4)} for ${esc(second.name)}` : ''}. Macro F1 weights all four severity classes equally, so the dominant level 2 cannot hide weak minority-class performance.</p><p>Boosted trees can model nonlinear interactions between location, time and weather, while this pipeline supports categorical inputs and missing numeric values. The comparison tests whether that flexibility improves on the majority baseline and a linear classifier. Selection comes from the measured validation results.</p>${simpleTable(sorted.map(r=>({Model:r.name,'Macro F1':r.macro_f1.toFixed(4),Accuracy:pct(r.accuracy),'Balanced accuracy':pct(r.balanced_accuracy),'Level 4 recall':pct(r.classification_report['4'].recall),'Fit + validation':`${r.seconds.toFixed(1)} s`,Decision:r.name===s.selected_model?'Selected':'Compared'})))}<p class="panel-footnote">Saved completed experiments on identical fitting and validation rows. Four LightGBM settings compare tree complexity and class weighting. This is a bounded comparison, not a claim that every possible model was tested. Test results never determine the winner.</p>`;
  $('method').querySelector('.section-heading').after(reasoning);
  renderCleaning(); refreshDatasetOptions();
}
async function initJourney() {
  journey=await getJSON('data/pipeline.json');
  if(!Array.isArray(journey.raw)||!journey.examples) throw new Error('Pipeline preview is incomplete.');
  renderJourney();
  $('dataset-view').addEventListener('change',()=>{ $('dataset-search').value='';$('feature-value').value='';refreshDatasetOptions(); });
  ['dataset-state','dataset-severity','dataset-size','feature-filter','feature-operator','dataset-sort'].forEach(id=>$(id).addEventListener('change',()=>{datasetPage=0;renderDataset();}));
  ['dataset-search','feature-value'].forEach(id=>$(id).addEventListener('input',()=>{datasetPage=0;renderDataset();}));
  $('dataset-columns').addEventListener('change',event=>{const input=event.target;if(input.checked)visibleFeatures.push(input.value);else visibleFeatures=visibleFeatures.filter(c=>c!==input.value);renderDataset();});
  $('columns-default').addEventListener('click',()=>{visibleFeatures=[...journey.defaults[datasetView()]];renderDataset();});
  $('columns-all').addEventListener('click',()=>{visibleFeatures=[...journey.columns[datasetView()]];renderDataset();});
  $('dataset-direction').addEventListener('click',()=>{datasetAscending=!datasetAscending;$('dataset-direction').textContent=datasetAscending?'Ascending ↑':'Descending ↓';renderDataset();});
  $('dataset-reset').addEventListener('click',()=>{for(const id of ['dataset-search','feature-value'])$(id).value='';$('dataset-severity').value='all';$('feature-operator').value='contains';$('dataset-size').value='10';datasetAscending=true;$('dataset-direction').textContent='Ascending ↑';refreshDatasetOptions();});
  $('dataset-prev').addEventListener('click',()=>{datasetPage--;renderDataset();});
  $('dataset-next').addEventListener('click',()=>{datasetPage++;renderDataset();});
  $('dataset-export').addEventListener('click',()=>{
    const csv=[visibleFeatures.map(csvCell).join(','),...filteredDataset.map(r=>visibleFeatures.map(c=>csvCell(r[c])).join(','))].join('\r\n');
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=`impact-${datasetView()}-filtered-preview.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),60000);
    notify(`Exported ${fmt(filteredDataset.length)} preview rows with ${visibleFeatures.length} features.`);
  });
  document.querySelectorAll('[data-clean-stage]').forEach(b=>b.addEventListener('click',()=>renderCleaning(b.dataset.cleanStage)));
}
