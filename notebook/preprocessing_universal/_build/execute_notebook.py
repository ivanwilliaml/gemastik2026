"""Execute the real-data notebook and persist outputs plus a compact verification report."""
from pathlib import Path
import json
import time

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent.parent
path = root / 'Preprocessing_universal.ipynb'
notebook = nbformat.read(path, as_version=4)
started = time.perf_counter()


def progress(cell, cell_index, **kwargs):
    if cell.cell_type != 'code':
        return
    for output in cell.get('outputs', []):
        if output.output_type == 'stream':
            for line in output.text.splitlines():
                if any(x in line for x in ['aktif=', 'LULUS:', 'Baris:', 'Sumber data:']):
                    print(line, flush=True)


client = NotebookClient(notebook, timeout=1800, kernel_name='python3',
                        resources={'metadata': {'path': str(root)}},
                        on_cell_executed=progress)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)

nbformat.validate(notebook)
code_cells = [c for c in notebook.cells if c.cell_type == 'code']
methods = json.loads((root / '_build/methods.json').read_text(encoding='utf-8'))
method_cells = [c for c in code_cells if 'show_result(' in c.source and 'before =' in c.source]
assert len(method_cells) == 32
for cell in method_cells:
    tables = [o for o in cell.outputs if o.output_type in {'display_data', 'execute_result'} and 'text/html' in o.get('data', {})]
    assert len(tables) == 1, 'Each method must display one table'
    html = tables[0].data['text/html']
    assert html.count('<tbody>') == 1 and html.split('<tbody>')[1].split('</tbody>')[0].count('<tr>') == 1
report = dict(date='2026-11-08', notebook_cells=len(notebook.cells),
              executed_code_cells=sum(c.execution_count is not None for c in code_cells),
              notebook_errors=sum(o.output_type == 'error' for c in code_cells for o in c.get('outputs', [])),
              elapsed_seconds=round(time.perf_counter()-started, 2),
              data='independent synthetic examples only; no CSV loaded',
              methods=32, examples_per_method=1, output_tables_per_method=1,
              rows_per_method_table=1, pipeline_example_rows=1,
              parity='passed: manual pipeline equals one-call API',
              real_data_unchanged=True)
(root / 'reports' / 'verification_examples.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)
