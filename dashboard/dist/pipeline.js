'use strict';
// Pipeline controls: upload a dataset, then run clean → train → test on the local server.
(() => {
  const el = (id) => document.getElementById(id);
  const steps = ['clean', 'train', 'test'];
  const labels = {clean: 'Cleaning', train: 'Training', test: 'Testing'};
  let poll = null, wasRunning = false, uploading = false;

  const size = (bytes) => bytes >= 1e9 ? `${(bytes / 1e9).toFixed(2)} GB` : `${(bytes / 1e6).toFixed(1)} MB`;
  const when = (iso) => iso ? new Date(iso).toLocaleString('en-US', {dateStyle: 'medium', timeStyle: 'short'}) : '';
  function toast(message) {
    const t = el('toast'); t.textContent = message; t.hidden = false;
    clearTimeout(toast.timer); toast.timer = setTimeout(() => { t.hidden = true; }, 6500);
  }
  function setBadge(id, text, kind) {
    const badge = el(id); badge.textContent = text; badge.className = `pipe-state ${kind || ''}`;
  }

  function render(status) {
    const {source, stages, job} = status;
    const running = job.running;
    el('source-info').innerHTML = source
      ? `<b></b><span>${size(source.bytes)} · uploaded ${when(source.uploaded)}</span>`
      : 'No dataset uploaded yet. Choose a US Accidents CSV, or a ZIP holding one.';
    if (source) el('source-info').querySelector('b').textContent = source.path.split(/[\\/]/).pop();
    setBadge('pipe-state-source', source ? 'Active' : 'Needed', source ? 'done' : '');
    el('upload-button').disabled = running || uploading;
    steps.forEach((stage, i) => {
      const ready = source && (i === 0 || stages[steps[i - 1]]);
      const active = running && job.stage === stage;
      const failed = !running && job.stage === stage && job.returncode && job.returncode !== 0;
      setBadge(`pipe-state-${stage}`, active ? 'Running…' : failed ? 'Failed' : stages[stage] ? 'Done' : ready ? 'Ready' : 'Waiting',
               active ? 'running' : failed ? 'failed' : stages[stage] ? 'done' : '');
      document.querySelector(`[data-step="${stage}"]`).classList.toggle('is-running', active);
      const button = document.querySelector(`[data-run="${stage}"]`);
      button.disabled = running || uploading || !ready;
      button.title = ready ? '' : `Finish step ${i + 1} first.`;
    });
    if (running) el('job-status').textContent = `${labels[job.stage]} since ${when(job.started)}. You can keep exploring; this panel updates by itself.`;
    else if (job.stage) el('job-status').textContent = job.returncode === 0
      ? `${labels[job.stage]} finished at ${when(job.finished)}.`
      : `${labels[job.stage]} failed. The log below shows the error.`;
    const log = el('job-log'), atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
    log.textContent = job.log.join('\n') || 'Logs from each step appear here.';
    if (atEnd) log.scrollTop = log.scrollHeight;
    // A finished test exports new results: reload everything from the new files.
    if (wasRunning && !running) {
      if (job.returncode === 0 && job.stage === 'test') { toast('Testing finished. Loading the new results…'); setTimeout(() => location.reload(), 1200); }
      else if (job.returncode === 0) toast(`${labels[job.stage]} finished. Run the next step.`);
      else toast(`${labels[job.stage]} failed. See the activity log.`);
    }
    wasRunning = running;
    clearTimeout(poll); poll = setTimeout(refresh, running ? 2000 : 15000);
  }

  async function refresh() {
    try {
      const response = await fetch('/api/pipeline', {cache: 'no-store'});
      if (!response.ok || !(response.headers.get('content-type') || '').includes('application/json')) throw new Error();
      el('pipeline-offline').hidden = true; el('pipeline-steps').hidden = false;
      render(await response.json());
    } catch {
      el('pipeline-offline').hidden = false;
      document.querySelectorAll('[data-run], #upload-button').forEach((b) => { b.disabled = true; });
    }
  }

  async function run(stage) {
    document.querySelectorAll('[data-run]').forEach((b) => { b.disabled = true; });
    const response = await fetch('/api/run', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({stage})});
    const body = await response.json().catch(() => ({}));
    if (!response.ok) toast(body.error || 'The step could not start.');
    refresh();
  }

  function upload(file) {
    if (!file) return;
    const bar = el('upload-progress');
    uploading = true; bar.hidden = false; bar.firstElementChild.style.width = '0%';
    el('upload-button').disabled = true;
    el('source-info').textContent = `Uploading ${file.name} (${size(file.size)})…`;
    // XMLHttpRequest reports upload progress; fetch does not.
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload');
    xhr.setRequestHeader('X-Filename', file.name);
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      bar.firstElementChild.style.width = `${(100 * e.loaded / e.total).toFixed(1)}%`;
      if (e.loaded === e.total) el('source-info').textContent = 'Checking the columns…';
    };
    xhr.onloadend = () => {
      uploading = false; bar.hidden = true; el('upload-input').value = '';
      let body = {}; try { body = JSON.parse(xhr.responseText); } catch {}
      toast(xhr.status === 200 ? `${file.name} is now the active dataset. Next: clean it.` : body.error || 'The upload failed.');
      refresh();
    };
    xhr.send(file);
  }

  document.addEventListener('DOMContentLoaded', () => {
    el('upload-button').addEventListener('click', () => el('upload-input').click());
    el('upload-input').addEventListener('change', (e) => upload(e.target.files[0]));
    document.querySelectorAll('[data-run]').forEach((b) => b.addEventListener('click', () => run(b.dataset.run)));
    refresh();
  }, {once: true});
})();
