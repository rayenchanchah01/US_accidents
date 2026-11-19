# Prompt: supervisor requirements for the dashboard

Use this prompt to brief a developer or an AI assistant.

---

You are working on the US Accidents traffic-impact severity project (Python package `accidents/`, dashboard in `dashboard/dist/`, local server `serve_dashboard.py`). Today the pipeline runs only from the terminal (`python run_all.py --source <file>`), using one fixed CSV that lives on one laptop, and the dashboard only displays results that were exported ahead of time. The supervisor wants the dashboard to become the control panel for the whole project:

1. **Any dataset as input, from the dashboard.** The pipeline must not depend on a file in a specific folder (such as Downloads). The user picks a CSV (or a single-CSV ZIP) in the dashboard, and the pipeline runs on that file. The file must follow the US Accidents schema (`ID`, `Severity`, `Start_Time`, `Start_Lat`, `Start_Lng`, weather, road-feature and location columns). Validate the required columns and reject a file that is missing them, with a clear message.
2. **An "Upload CSV" button for new dataset versions.** When a newer release of the dataset comes out, the user uploads it in the dashboard, and it becomes the active dataset for the next run. Show which dataset is active (file name, size, upload time).
3. **Training and testing inside the dashboard, using buttons.** Add buttons for each pipeline stage: Prepare & clean → Train → Test / evaluate & export. Each button calls the existing Python functions (`prepare`, `train`, `export`, `export_story`, `report`). Jobs run in the background so the page never freezes. Show live status and a log, and allow only one job at a time. When a job finishes, the dashboard reloads its results.
4. **Everything is controlled from the dashboard.** The normal workflow (upload → clean → train → test → view results → try the model) needs no terminal commands. Only starting the server (`python serve_dashboard.py`) does.
5. **Dashboard layout.** In the left sidebar, group the sections into two categories:
   - **Dataset**: Dataset, Cleaning, Feature engineering, Split & encoding.
   - **Model training & testing**: Pipeline controls, Model comparison, Results, Prediction ledger, Try the model.
6. **Clean results filtering.** The filters on the results and prediction ledger should look tidy and consistent: aligned controls, clear labels, a visible count of matching rows, and a single reset action.
7. **Correctness.** Every number on the dashboard must come from the current run's exported files, with no stale or hard-coded values. The result must look correct at desktop and phone widths.

Keep the code minimal: use the Python standard library server that already exists, with no new web framework. Reuse the existing pipeline functions instead of rewriting them.
