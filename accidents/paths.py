"""Project locations, anchored to the source tree rather than the working directory.

Data holds prepared inputs, split assignments and predictions. Outputs holds
models, reports and run bookkeeping. The dashboard owns its deployable files.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
PROCESSED = DATA / 'processed'
SPLITS = DATA / 'splits'
PREDICTIONS = DATA / 'predictions'
OUTPUTS = ROOT / 'outputs'
MODELS = OUTPUTS / 'models'
REPORTS = OUTPUTS / 'reports'
CHECKPOINTS = OUTPUTS / 'checkpoints'
LOGS = OUTPUTS / 'logs'
ARCHIVE = OUTPUTS / 'archive'
DOCS = ROOT / 'docs'
PIPELINE = ROOT / 'pipeline'
NOTEBOOKS = PIPELINE  # Compatibility for scripts importing the old constant.
DASHBOARD = ROOT / 'dashboard' / 'dist'
