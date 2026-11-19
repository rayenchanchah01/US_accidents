'use strict';
const $ = (id) => document.getElementById(id);
const fmt = (n) => Number(n).toLocaleString('en-US');
const pct = (n, digits = 1) => `${(100 * n).toFixed(digits)}%`;
const short = (n) => n >= 1e6 ? `${(n / 1e6).toFixed(2)}m` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : fmt(n);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const colors = ['#6c91a1', '#5d7d64', '#c38a41', '#ac5746'];
const names = {AL:'Alabama',AZ:'Arizona',AR:'Arkansas',CA:'California',CO:'Colorado',CT:'Connecticut',DE:'Delaware',DC:'District of Columbia',FL:'Florida',GA:'Georgia',ID:'Idaho',IL:'Illinois',IN:'Indiana',IA:'Iowa',KS:'Kansas',KY:'Kentucky',LA:'Louisiana',ME:'Maine',MD:'Maryland',MA:'Massachusetts',MI:'Michigan',MN:'Minnesota',MS:'Mississippi',MO:'Missouri',MT:'Montana',NE:'Nebraska',NV:'Nevada',NH:'New Hampshire',NJ:'New Jersey',NM:'New Mexico',NY:'New York',NC:'North Carolina',ND:'North Dakota',OH:'Ohio',OK:'Oklahoma',OR:'Oregon',PA:'Pennsylvania',RI:'Rhode Island',SC:'South Carolina',SD:'South Dakota',TN:'Tennessee',TX:'Texas',UT:'Utah',VT:'Vermont',VA:'Virginia',WA:'Washington',WV:'West Virginia',WI:'Wisconsin',WY:'Wyoming'};
let data, state = 'ALL', outcome = 'all', actual = 'all', page = 0, version = 0, currentRows = [], exporting = false;
const pageSize = 10, cache = new Map();

async function getJSON(url) {
  // Revalidate every time: results change after each pipeline run.
  const response = await fetch(url, {cache: 'no-cache'});
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status})`);
  return response.json();
}
function notify(message) {
  $('toast').textContent = message; $('toast').hidden = false;
  clearTimeout(notify.timer); notify.timer = setTimeout(() => { $('toast').hidden = true; }, 6500);
}
function badge(level) { return `<span class="severity sev-${level}">Level ${level}</span>`; }
function stateName(s) { return s === 'ALL' ? 'All states' : (names[s] || s); }

function renderOverview() {
  const c = data.cohorts[state], stats = data.stats;
  $('scope-count').textContent = `${fmt(c.rows)} test records`;
  $('reset-state').hidden = state === 'ALL';
  $('macro').textContent = c.macro_f1.toFixed(3);
  // Majority baseline for the cohort uses the globally fixed training-majority label.
  const majority = Number(Object.entries(stats.split_class_counts.development).sort((a,b) => b[1]-a[1])[0][0]);
  const majorityCount = c.confusion[majority-1].reduce((a,b) => a+b, 0);
  const baselineF1 = c.rows ? (2 * majorityCount / (c.rows + majorityCount)) / 4 : 0;
  $('macro-note').textContent = `+${(c.macro_f1 - baselineF1).toFixed(3)} vs majority baseline`;
  $('accuracy').textContent = pct(c.accuracy);
  $('accuracy-note').textContent = `${pct(majorityCount / c.rows)} by always choosing level ${majority}`;
  const sev4 = c.classification_report['4'];
  $('recall').textContent = sev4.support ? pct(sev4.recall) : '—';
  $('recall-note').textContent = `${fmt(c.confusion[3][3])} of ${fmt(sev4.support)} level 4 events found`;
  $('test-count').textContent = short(c.rows);
  $('count-note').textContent = state === 'ALL' ? '20% of the cleaned dataset' : `${pct(c.rows / stats.test_rows)} of the national test set`;
  renderMatrix(c); renderRecall(c); renderMap(); renderTimeline(c);
}
function renderMatrix(c) {
  let html = '<div class="axis" aria-label="Actual severity down rows">↓</div>';
  for (let j=1;j<=4;j++) html += `<div class="axis">${j}</div>`;
  c.confusion.forEach((row,i) => {
    const total = row.reduce((a,b) => a+b,0);
    html += `<div class="axis">${i+1}</div>`;
    row.forEach((count,j) => {
      const share = total ? count/total : 0;
      const bg = i===j ? `rgba(77,104,61,${0.08+share*0.78})` : `rgba(185,130,76,${0.04+share*0.65})`;
      const label = `Actual ${i+1}, predicted ${j+1}: ${fmt(count)} records, ${pct(share)} of actual level ${i+1}`;
      html += `<div tabindex="0" class="matrix-cell ${i===j?'correct':''}" style="background:${bg};color:${i===j&&share>.65?'#fff':'#263b31'}" title="${label}" aria-label="${label}">${total?pct(share):'—'}<small>${fmt(count)}</small></div>`;
    });
  });
  $('confusion').innerHTML = html;
}
function renderRecall(c) {
  $('class-recall').innerHTML = [1,2,3,4].map((level) => {
    const v = c.classification_report[level];
    return `<div class="recall-row"><span class="level-tag" style="color:${colors[level-1]}">${level}</span><div><div class="bar-track"><i style="width:${100*v.recall}%;background:${colors[level-1]}"></i></div><div class="recall-meta">${fmt(v.support)} ACTUAL RECORDS · F1 ${v['f1-score'].toFixed(3)}</div></div><span class="recall-value">${v.support?pct(v.recall,0):'—'}</span></div>`;
  }).join('');
}
// Interactive results map (Leaflet). Each geo row: [state, lon*4, lat*4, records, correct,
// actual counts 1-4, predicted counts 1-4] for one 0.25° cell of the held-out test set.
let resultsMap=null, mapLayer=null, mapPoints=[], mapStale=true, mapWatched=false;
const accuracyColor=(a)=>a>=.9?'#3f6b4c':a>=.75?'#9bb36f':a>=.6?'#c38a41':'#ac5746';
function cellPopup(r) {
  const [s,lon,lat,n,ok]=r, act=r.slice(5,9), pred=r.slice(9,13);
  return `<div class="map-pop"><b>${esc(stateName(s))}</b><small>${(lat/4).toFixed(2)}°, ${(lon/4).toFixed(2)}° · 0.25° cell</small>`+
    `<div class="map-pop-kpi"><span>${fmt(n)}<small>incidents</small></span><span>${pct(ok/n,0)}<small>predicted correctly</small></span></div>`+
    `<table><thead><tr><th>Severity</th><th>Actual</th><th>Predicted</th></tr></thead><tbody>${[0,1,2,3].map(i=>`<tr><td>${badge(i+1)}</td><td>${fmt(act[i])}</td><td>${fmt(pred[i])}</td></tr>`).join('')}</tbody></table></div>`;
}
function createMap() {
  resultsMap=L.map('results-map',{preferCanvas:true,minZoom:3,maxZoom:18,maxBounds:[[5,-170],[72,-40]],zoomSnap:.5});
  // Esri tiles: free, no API key, attribution required. Satellite imagery with road and
  // place-name overlays (states, counties, cities); a plain street map is one click away.
  const esri='https://server.arcgisonline.com/ArcGIS/rest/services/';
  const tiles=(path,options)=>L.tileLayer(`${esri}${path}/MapServer/tile/{z}/{y}/{x}`,{maxZoom:18,maxNativeZoom:17,...options});
  // Roads and labels sit above the imagery (200) but below the result circles (400).
  resultsMap.createPane('labels'); resultsMap.getPane('labels').style.zIndex=350; resultsMap.getPane('labels').style.pointerEvents='none';
  const satellite=L.layerGroup([
    tiles('World_Imagery',{maxNativeZoom:18,attribution:'Imagery &copy; Esri, Maxar, Earthstar Geographics, USDA, USGS'}),
    tiles('Reference/World_Transportation',{pane:'labels',opacity:.9}),
    tiles('Reference/World_Boundaries_and_Places',{pane:'labels'})]);
  const streets=L.layerGroup([
    tiles('World_Street_Map',{maxNativeZoom:18,attribution:'Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors'})]);
  satellite.addTo(resultsMap);
  L.control.layers({'Satellite':satellite,'Street map':streets},null,{collapsed:false,position:'topright'}).addTo(resultsMap);
  L.control.scale({imperial:true,metric:true}).addTo(resultsMap);
  mapLayer=L.layerGroup().addTo(resultsMap);
}
function fitMap() {
  const bounds=mapPoints.length?L.latLngBounds(mapPoints.map(r=>[r[2]/4,r[1]/4])):L.latLngBounds([[24,-125],[50,-66]]);
  resultsMap.fitBounds(bounds,{padding:[24,24],maxZoom:9});
}
// Leaflet must first appear in a visible box, so the map is built and redrawn only when its
// page is on screen; while hidden, new data just marks it stale.
function drawMap() {
  if(!$('results-map').offsetWidth){mapStale=true;return;}
  if(!resultsMap)createMap(); else resultsMap.invalidateSize();
  mapLayer.clearLayers();
  const max=Math.max(1,...mapPoints.map(r=>r[3]));
  // Largest cells first, so small ones stay clickable on top.
  [...mapPoints].sort((a,b)=>b[3]-a[3]).forEach((r)=>{
    const n=r[3], acc=r[4]/n;
    L.circleMarker([r[2]/4,r[1]/4],{radius:3+Math.sqrt(n/max)*15,color:'#0b1a12',weight:.7,opacity:.8,fillColor:accuracyColor(acc),fillOpacity:.88})
      .bindTooltip(`${fmt(n)} incidents · ${pct(acc,0)} correct`,{direction:'top'})
      .bindPopup(()=>cellPopup(r),{maxWidth:260})
      .addTo(mapLayer);
  });
  fitMap(); mapStale=false;
}
function renderMap() {
  mapPoints = data.geo.filter((r) => state==='ALL'||r[0]===state);
  if(!window.L){$('map-offline').hidden=false;$('results-map').hidden=true;}
  else {
    if(!mapWatched) {
      mapWatched=true;
      new ResizeObserver(([entry])=>{
        if(!entry.contentRect.width)return;
        if(mapStale)drawMap(); else resultsMap.invalidateSize();
      }).observe($('results-map'));
      $('map-reset').addEventListener('click',()=>{if(resultsMap)fitMap();});
    }
    drawMap();
  }
  const top = Object.entries(data.cohorts).filter(([k])=>k!=='ALL').sort((a,b)=>b[1].rows-a[1].rows).slice(0,3);
  $('state-bars').innerHTML = top.map(([s,c])=>`<button class="state-bar" data-state="${esc(s)}" aria-label="Explore ${esc(stateName(s))}"><div>${esc(s)}<span>${pct(c.rows/data.stats.test_rows)}</span></div><div class="bar-track"><i style="width:${100*c.rows/top[0][1].rows}%"></i></div></button>`).join('');
}
function renderTimeline(c) {
  const rows = c.timeline, left=47,right=620,top=18,bottom=152;
  const maxShare = Math.max(.1,...rows.flatMap(r=>[r[2]/r[1],r[3]/r[1]]));
  const ceiling = Math.min(1,Math.ceil(maxShare*5)/5);
  const x=(i)=>left+i/Math.max(1,rows.length-1)*(right-left), y=(n)=>bottom-n/ceiling*(bottom-top);
  let html='';
  for(let i=0;i<=4;i++) {
    const v=i*ceiling/4, yy=y(v);
    html+=`<path d="M${left} ${yy}H${right}" stroke="#dde1d5" stroke-width=".7"/><text x="37" y="${yy+3}" text-anchor="end" font-size="8" fill="#727c70" font-family="monospace">${pct(v,0)}</text>`;
  }
  [2,3].forEach((col,k)=>{
    html+=`<path d="${rows.map((r,i)=>`${i?'L':'M'}${x(i).toFixed(2)},${y(r[col]/r[1]).toFixed(2)}`).join(' ')}" fill="none" stroke="${k?'#c38a41':'#3d5843'}" stroke-width="1.8" ${k?'stroke-dasharray="4 3"':''}/>`;
  });
  rows.forEach((r,i)=>{
    html+=`<circle cx="${x(i)}" cy="${y(r[2]/r[1])}" r="4" fill="transparent"><title>${r[0]} · ${fmt(r[1])} records · actual ${pct(r[2]/r[1])} · predicted ${pct(r[3]/r[1])}</title></circle>`;
    if(i===0||i===rows.length-1||(r[0].endsWith('-01')&&Number(r[0].slice(0,4))%2===0)) html+=`<text x="${x(i)}" y="175" text-anchor="middle" fill="#727c70" font-size="8" font-family="monospace">${r[0].slice(0,4)}</text>`;
  });
  $('timeline').innerHTML=html;
}

function matchingShards() {
  return data.manifest.filter((s)=>(state==='ALL'||s.state===state)&&(actual==='all'||s.actual===Number(actual))&&(outcome==='all'||s.correct===(outcome==='right')));
}
async function shardRows(path) {
  if(cache.has(path)) return cache.get(path);
  const rows=await getJSON(`data/${path}`);
  if(cache.size>=8) cache.delete(cache.keys().next().value);
  cache.set(path,rows);return rows;
}
async function renderRecords() {
  const request=++version, shards=matchingShards(), count=shards.reduce((s,r)=>s+r.count,0);
  page=Math.min(page,Math.max(0,Math.ceil(count/pageSize)-1));
  const begin=page*pageSize, end=Math.min(begin+pageSize,count);
  $('prev-page').disabled=page===0;$('next-page').disabled=end>=count;
  $('page-number').textContent=`${fmt(page+1)} / ${fmt(Math.max(1,Math.ceil(count/pageSize)))}`;
  $('filter-count').textContent=fmt(count);
  $('reset-filters').hidden=state==='ALL'&&actual==='all'&&outcome==='all';
  $('record-range').textContent=count?`${fmt(begin+1)}–${fmt(end)} of ${fmt(count)} records`:'No matching records';
  $('table-status').textContent='Loading predictions…'; $('records').style.opacity='.4';
  try {
    let pos=0;const pieces=[];
    for(const shard of shards) {
      if(pos<end&&pos+shard.count>begin) {
        const rows=await shardRows(shard.path);
        if(request!==version)return;
        pieces.push(...rows.slice(Math.max(0,begin-pos),Math.min(shard.count,end-pos)));
      }
      pos+=shard.count;if(pos>=end)break;
    }
    if(request!==version)return;
    currentRows=pieces;
    $('records').innerHTML=pieces.length?pieces.map((r,i)=>`<tr><td>${esc(r[0])}<small>${esc(r[1])}</small></td><td>${esc(r[2])}<small>${esc(stateName(r[3]))}</small></td><td>${badge(r[4])}</td><td>${badge(r[5])}</td><td><span class="score-cell">${pct(r[6])}</span><span class="mini-score"><i style="width:${100*r[6]}%"></i></span></td><td><span class="outcome ${r[4]===r[5]?'':'error'}">${r[4]===r[5]?'✓ Correct':'↗ Missed'}</span></td><td><button class="row-detail" data-row="${i}" aria-label="Inspect prediction ${esc(r[0])}">↗</button></td></tr>`).join(''):'<tr><td colspan="7">No predictions match these filters. Try another severity or state.</td></tr>';
    $('table-status').textContent='';
  } catch(error) {
    if(request===version){$('records').innerHTML='';$('table-status').textContent='Prediction records could not be loaded. Change a filter or reload to retry.';notify(error.message);}
  } finally {if(request===version)$('records').style.opacity='1';}
}
function setOutcome(next) {
  outcome=next;
  document.querySelectorAll('[data-outcome]').forEach(x=>{const on=x.dataset.outcome===next;x.classList.toggle('selected',on);x.setAttribute('aria-pressed',String(on));});
}
function setState(next) {
  state=next;$('state-filter').value=next;$('ledger-state').value=next;page=0;renderOverview();renderRecords();
}
function showDetails(row) {
  const r=currentRows[row];if(!r)return;
  $('detail-body').innerHTML=`<h2 id="detail-title" class="detail-title">${esc(r[0])}</h2><p class="detail-meta">${esc(r[2])}, ${esc(r[3])} · ${esc(r[1])} local</p><div class="detail-comparison"><div><small>ACTUAL IMPACT</small>${badge(r[4])}</div><div><small>PREDICTED IMPACT</small>${badge(r[5])}</div><div><small>OUTCOME</small><span class="outcome ${r[4]===r[5]?'':'error'}">${r[4]===r[5]?'Correct':'Misclassified'}</span></div></div><div class="eyebrow">SCORE BY CLASS</div><div class="detail-scores">${[1,2,3,4].map((level)=>`<div><span>Level ${level}</span><div class="bar-track"><i style="width:${100*r[6+level]}%;background:${colors[level-1]}"></i></div><span>${pct(r[6+level])}</span></div>`).join('')}</div><p class="detail-note">Weather group: ${esc(r[11])}<br>Coordinates: ${Number(r[12]).toFixed(4)}, ${Number(r[13]).toFixed(4)}<br><br>These are model scores, not calibrated confidence estimates. This incident was excluded from model fitting and selection.</p>`;
  $('detail-dialog').showModal();
}
function renderMethod() {
  const s=data.stats,c=data.cleaning;
  $('comparison-subtitle').textContent=`Same ${fmt(s.tuning_rows)} training and ${fmt(s.validation_rows)} validation records for every candidate.`;
  const sorted=[...s.validation_models].sort((a,b)=>b.macro_f1-a.macro_f1);
  $('model-comparison').innerHTML=sorted.map(r=>`<div class="model-row ${r.name===s.selected_model?'best':''}"><div>${esc(r.name.replace('LightGBM · ','LightGBM / '))}<small>${r.name===s.selected_model?'SELECTED · ':''}${r.seconds<60?`${r.seconds.toFixed(0)}s`:`${(r.seconds/60).toFixed(1)} min`} FIT + VALIDATION</small></div><div class="bar-track"><i style="width:${100*r.macro_f1}%"></i></div><strong>${r.macro_f1.toFixed(3)}</strong></div>`).join('');
  $('chosen-model').innerHTML=`<b>${esc(s.selected_model)}</b><br>Refit on all ${fmt(s.development_rows)} development records. ${s.selection.best_iteration?`${fmt(s.selection.best_iteration)} boosting rounds, frozen before test evaluation.`:''}`;
  const important=s.importance.filter(r=>r.macro_f1_drop>0).slice(0,7), max=important[0]?.macro_f1_drop||1;
  $('importance').innerHTML=important.map(r=>`<div class="importance-row"><span>${esc(r.feature.replaceAll('_',' '))}</span><div class="bar-track"><i style="width:${100*r.macro_f1_drop/max}%"></i></div><span>${r.macro_f1_drop.toFixed(3)}</span></div>`).join('');
  const steps=[[c.counts.input_rows,'raw records read from the source file'],[c.counts.repeated_event_rows_removed,'repeated time + location records removed'],[c.counts.weather_future,'raw rows with future weather observations masked'],[c.clean_rows,'clean, distinct event records retained'],[s.test_rows,'records held out for the final evaluation']];
  $('cleaning-steps').innerHTML=steps.map(([n,description])=>`<div class="audit-step"><span>${short(n)}</span><p>${esc(description)}</p></div>`).join('');
  const t=s.temporal;
  $('temporal-note').innerHTML=t.macro_f1!==undefined?`The time-based stress test scored <b>${t.macro_f1.toFixed(3)} macro F1</b>: trained on a development sample before 2022, evaluated on ${fmt(t.evaluation_rows)} development records from 2022 onward. This diagnostic highlights how reporting patterns change.`:'A chronological stress test was not available for this dataset.';
  $('limitations').innerHTML=s.limitations.map(x=>`<li>${esc(x)}</li>`).join('');
}
function csvCell(value) {
  let text=String(value??'');
  // Treat source text as text when opened in spreadsheet software.
  if(/^[=+@\t\r]/.test(text)||(/^-/ .test(text)&&!Number.isFinite(Number(text))))text="'"+text;
  return `"${text.replaceAll('"','""')}"`;
}
async function exportCSV() {
  if(exporting)return;
  const shards=matchingShards(), count=shards.reduce((n,s)=>n+s.count,0);
  if(!count){notify('No matching records to export.');return;}
  exporting=true;const button=$('export-csv');button.disabled=true;
  const filename=`impact-${state.toLowerCase()}-${actual}-${outcome}.csv`;
  let stream=null, parts=[];
  try {
    if(window.showSaveFilePicker) {
      try {
        const handle=await window.showSaveFilePicker({suggestedName:filename,types:[{description:'CSV predictions',accept:{'text/csv':['.csv']}}]});
        stream=await handle.createWritable();
      } catch(error) {
        // Embedded/private browsers may expose the API but disallow file-system access.
        if(!['SecurityError','NotAllowedError'].includes(error.name))throw error;
      }
    }
    const header=data.columns.map(csvCell).join(',')+'\r\n';
    if(stream)await stream.write(header);else parts.push(header);
    let done=0;
    for(let i=0;i<shards.length;i+=5) {
      const batches=await Promise.all(shards.slice(i,i+5).map(s=>getJSON(`data/${s.path}`)));
      for(const rows of batches) {
        const text=rows.map(row=>row.map(csvCell).join(',')).join('\r\n')+'\r\n';
        if(stream)await stream.write(text);else parts.push(text);
        done+=rows.length;
      }
      button.textContent=`Exporting ${Math.round(done/count*100)}%…`;
    }
    if(stream)await stream.close();
    else {
      const url=URL.createObjectURL(new Blob(parts,{type:'text/csv;charset=utf-8'}));
      const anchor=document.createElement('a');anchor.href=url;anchor.download=filename;anchor.click();setTimeout(()=>URL.revokeObjectURL(url),60_000);
    }
    notify(`Exported ${fmt(count)} predictions.`);
  } catch(error) {
    if(stream)await stream.abort().catch(()=>{});
    if(error.name!=='AbortError')notify(`Export failed: ${error.message}`);
  } finally {exporting=false;button.disabled=false;button.innerHTML='Export filtered CSV <span>↓</span>';}
}

async function init() {
  try {
    data=await getJSON('data/summary.json');
    if(!data.stats||!data.cohorts?.ALL||!Array.isArray(data.manifest))throw new Error('Results export is incomplete.');
    $('dataset-meta').textContent=`${fmt(data.cleaning.counts.input_rows)} SOURCE RECORDS / 4 CLASSES`;
    $('source-file').textContent=$('footer-source').textContent=data.cleaning.source_file||'an unnamed file';
    Object.keys(data.cohorts).filter(s=>s!=='ALL').sort((a,b)=>stateName(a).localeCompare(stateName(b))).forEach(s=>{
      for(const id of ['state-filter','ledger-state']){const option=document.createElement('option');option.value=s;option.textContent=stateName(s);$(id).append(option);}
    });
    renderOverview();renderMethod();
    await initJourney();
    document.querySelectorAll('.results-content').forEach(e=>{e.hidden=false;});$('loading').hidden=true;
    $('state-filter').addEventListener('change',e=>setState(e.target.value));
    $('ledger-state').addEventListener('change',e=>setState(e.target.value));
    $('reset-filters').addEventListener('click',()=>{actual='all';$('class-filter').value='all';setOutcome('all');setState('ALL');});
    $('reset-state').addEventListener('click',()=>setState('ALL'));
    $('state-bars').addEventListener('click',e=>{const b=e.target.closest('[data-state]');if(b)setState(b.dataset.state);});
    $('class-filter').addEventListener('change',e=>{actual=e.target.value;page=0;renderRecords();});
    document.querySelectorAll('[data-outcome]').forEach(b=>b.addEventListener('click',()=>{setOutcome(b.dataset.outcome);page=0;renderRecords();}));
    $('prev-page').addEventListener('click',()=>{if(page>0){page--;renderRecords();}});
    $('next-page').addEventListener('click',()=>{page++;renderRecords();});
    $('records').addEventListener('click',e=>{const b=e.target.closest('[data-row]');if(b)showDetails(Number(b.dataset.row));});
    $('close-dialog').addEventListener('click',()=>$('detail-dialog').close());
    $('detail-dialog').addEventListener('click',e=>{if(e.target===$('detail-dialog')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});
    $('export-csv').addEventListener('click',exportCSV);
    await renderRecords();
  } catch(error) {$('loading').hidden=true;$('fatal').hidden=false;console.error(error);}
}
// One section at a time: the URL hash picks it, the sidebar marks it, the page starts at its top.
const SECTIONS=['dataset','cleaning','features','split','pipeline','method','overview','predictions','try-model'];
let shown='dataset';
function showSection() {
  const wanted=location.hash.slice(1);
  if(SECTIONS.includes(wanted))shown=wanted;
  SECTIONS.forEach(id=>{$(id).hidden=id!==shown;});
  if(shown==='overview'&&mapWatched){if(mapStale)drawMap();else resultsMap.invalidateSize();}
  document.querySelectorAll('.nav-link').forEach(a=>{
    const on=a.hash===`#${shown}`;a.classList.toggle('active',on);
    if(on){a.setAttribute('aria-current','page');a.closest('details').open=true;}else a.removeAttribute('aria-current');
  });
  window.scrollTo({top:0,behavior:'instant'});
}
window.addEventListener('hashchange',showSection);
// Handle section links ourselves so the browser never animates a jump to an anchor.
document.addEventListener('click',e=>{
  const a=e.target.closest('a[href^="#"]');
  if(!a||e.ctrlKey||e.metaKey||e.shiftKey||!SECTIONS.includes(a.hash.slice(1)))return;
  e.preventDefault();
  if(location.hash!==a.hash)history.pushState(null,'',a.hash);
  showSection();
});
window.addEventListener('popstate',showSection);
// Start after every deferred companion script has defined its functions.
document.addEventListener('DOMContentLoaded',()=>{showSection();init();},{once:true});
