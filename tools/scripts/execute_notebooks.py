"""Execute the canonical cleaning-and-evaluation notebook in pipeline/."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parents[2]
for filename in ['pipeline.ipynb']:
    path=root/'pipeline'/filename
    notebook=nbformat.read(path,as_version=4)
    NotebookClient(notebook,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(path.parent)}}).execute()
    nbformat.write(notebook,path)
    print('Executed',filename,'successfully',flush=True)
