"""Browser verification against the actual exported dashboard."""
import csv
import io
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'outputs' / 'reports' / 'qa'; ART.mkdir(parents=True, exist_ok=True)
summary = json.loads((ROOT / 'dashboard/dist/data/summary.json').read_text(encoding='utf-8'))

with sync_playwright() as p:
    browser = p.chromium.launch(channel='chrome', headless=True)
    context = browser.new_context(viewport={'width': 1440, 'height': 1100}, device_scale_factor=1)
    # Exercise the ordinary browser-download fallback without an OS picker.
    context.add_init_script("Object.defineProperty(window, 'showSaveFilePicker', {value: undefined, configurable: true});")
    page = context.new_page(); errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    response = page.goto('http://127.0.0.1:8765', wait_until='networkidle')
    assert response.status == 200
    page.locator('#records tr').first.wait_for()
    assert page.locator('#content').is_visible()
    assert page.locator('.matrix-cell').count() == 16
    assert page.locator('#macro').inner_text() == f"{summary['stats']['test']['macro_f1']:.3f}"
    assert page.locator('#records tr').count() == 10
    assert page.locator('#dataset-rows tr').count() == 10
    assert page.locator('#content > section').evaluate_all('(els) => els.map(e => e.id)') == ['dataset','cleaning','features','split','method','overview','predictions']
    assert '6,000' in page.locator('#sample-scope').inner_text()
    # Feature selection, filtering, pagination and sorting on the real preview.
    page.select_option('#dataset-state', 'CA')
    page.select_option('#dataset-severity', '2')
    assert page.locator('#dataset-rows tr').count() == 10
    assert page.locator('#dataset-rows tr td:nth-child(3)').all_text_contents() == ['CA']*10
    first_sample = page.locator('#dataset-rows tr').first.inner_text()
    page.click('#dataset-next')
    assert page.locator('#dataset-rows tr').first.inner_text() != first_sample
    page.select_option('#dataset-size', '25')
    assert page.locator('#dataset-rows tr').count() == 25
    page.click('.column-picker summary')
    page.click('#columns-all')
    assert page.locator('#dataset-head th').count() > 40
    page.click('#columns-default')
    assert page.locator('#dataset-head th').count() == 10
    with page.expect_download() as preview_download:
        page.click('#dataset-export')
    preview_download.value.save_as(ART / 'verified-preview.csv')
    sample_rows = list(csv.DictReader((ART/'verified-preview.csv').open(encoding='utf-8')))
    assert len(sample_rows) >= 25
    assert all(r['State']=='CA' and r['Severity']=='2' for r in sample_rows)
    assert len(sample_rows[0]) == 10
    page.click('#dataset-reset')
    page.select_option('#feature-filter','Temperature(F)')
    page.select_option('#feature-operator','gt')
    page.fill('#feature-value','10000')
    assert 'No records match' in page.locator('#dataset-rows').inner_text()
    assert page.locator('#dataset-export').is_disabled()
    page.click('#dataset-reset')
    page.select_option('#dataset-view','cleaned')
    assert page.locator('#dataset-rows tr').count() == 10
    page.select_option('#feature-filter','Temperature(F)')
    page.select_option('#feature-operator','missing')
    assert page.locator('#feature-value').is_disabled()
    page.click('#dataset-reset')
    page.select_option('#dataset-view','raw')
    page.click('[data-clean-stage="weather"]')
    assert 'Missing' in page.locator('#clean-tables').inner_text()
    assert 'Removed: earlier event retained' in page.locator('#duplicate-tables').inner_text()
    assert 'End_Time' in page.locator('#selection-tables').inner_text()
    page.click('[data-clean-stage="validation"]')
    page.locator('.column-picker').evaluate('(e) => e.open = false')
    page.evaluate("window.scrollTo({top:0,behavior:'instant'}); document.getElementById('toast').hidden = true")
    page.screenshot(path=str(ART / 'dashboard-desktop.png'), full_page=True)
    page.screenshot(path=str(ART / 'dashboard-top.png'))
    page.select_option('#state-filter', 'CA')
    page.wait_for_function("document.getElementById('table-status').textContent === ''")
    assert page.locator('#macro').inner_text() == f"{summary['cohorts']['CA']['macro_f1']:.3f}"
    assert 'California' in page.locator('#records tr').first.inner_text()
    page.click('[data-outcome="wrong"]'); page.select_option('#class-filter', '4')
    page.wait_for_function("document.getElementById('table-status').textContent === ''")
    assert page.locator('#records .outcome.error').count() == 10
    assert page.locator('#records tr td:nth-child(3) .sev-4').count() == 10
    first_id = page.locator('#records tr td').first.inner_text()
    page.click('#next-page')
    page.wait_for_function("document.getElementById('table-status').textContent === ''")
    assert page.locator('#records tr td').first.inner_text() != first_id
    page.locator('.row-detail').first.click()
    assert page.locator('#detail-dialog').is_visible()
    assert page.locator('.detail-scores>div').count() == 4
    page.keyboard.press('Escape')
    assert not page.locator('#detail-dialog').is_visible()
    # Export a small real cohort and validate the download, including filtering.
    page.select_option('#state-filter', 'ND')
    page.wait_for_function("document.getElementById('table-status').textContent === ''")
    expected = sum(s['count'] for s in summary['manifest'] if s['state']=='ND' and s['actual']==4 and not s['correct'])
    if expected:
        with page.expect_download(timeout=60000) as info:
            page.click('#export-csv')
        download = info.value; download.save_as(ART / 'verified-export.csv')
        rows = list(csv.DictReader((ART / 'verified-export.csv').open(encoding='utf-8')))
        assert len(rows) == expected
        assert all(r['State']=='ND' and r['actual']=='4' and r['predicted']!='4' for r in rows)
    # Empty cohort handled without NaN or stale records.
    zero = next((s for s in summary['cohorts'] if s!='ALL' and not any(x['state']==s and x['actual']==1 and not x['correct'] for x in summary['manifest'])), None)
    if zero:
        page.select_option('#state-filter', zero); page.select_option('#class-filter', '1')
        page.wait_for_function("document.getElementById('table-status').textContent === ''")
        assert 'No matching records' in page.locator('#record-range').inner_text()
    mobile = context.new_page()
    mobile.set_viewport_size({'width':390,'height':844})
    mobile.goto('http://127.0.0.1:8765',wait_until='networkidle')
    mobile.locator('#records tr').first.wait_for()
    assert mobile.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    assert mobile.locator('.sidebar nav').is_visible()
    assert mobile.locator('#dataset-rows tr').count() == 10
    mobile.click('.column-picker summary')
    mobile.click('#columns-all')
    assert mobile.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    mobile.click('#columns-default')
    mobile.click('.column-picker summary')
    mobile.evaluate("window.scrollTo({top:0,behavior:'instant'})")
    mobile.screenshot(path=str(ART / 'dashboard-mobile.png'),full_page=True)
    mobile.screenshot(path=str(ART / 'dashboard-mobile-top.png'))
    broken = context.new_page()
    broken.route('**/data/summary.json',lambda route:route.fulfill(status=503,body='Unavailable'))
    broken.goto('http://127.0.0.1:8765',wait_until='networkidle')
    assert broken.locator('#fatal').is_visible()
    assert not errors, errors
    browser.close()
print('PASS: desktop/mobile, actual metrics, filters, pagination, detail dialog, CSV export, failure state, no JavaScript errors')
