"""Run every notebook in order, top to bottom, and save the outputs inside them.

Usage (from the project folder):  python run_all.py
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

for path in sorted(Path("notebooks").glob("0*.ipynb")):
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": "notebooks"}}).execute()
    nbformat.write(nb, path)
    print("ran", path)
