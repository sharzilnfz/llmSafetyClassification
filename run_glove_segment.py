import json
import traceback
from pathlib import Path

root = Path(__file__).resolve().parent
nb_path = root / 'cse440_project.ipynb'
nb = json.loads(nb_path.read_text(encoding='utf-8'))
ns = {}
from IPython.display import display
ns['display'] = display

start = None
for i, cell in enumerate(nb.get('cells', [])):
    src = ''.join(cell.get('source', []))
    if 'Download & Extract GloVe' in src or 'glove_url' in src:
        start = i
        print(f'STARTING GLOVE SEGMENT AT CELL {i}')
        break

if start is None:
    raise RuntimeError('GloVe segment not found in notebook')

for i in range(start, len(nb.get('cells', []))):
    cell = nb['cells'][i]
    src = ''.join(cell.get('source', []))
    if cell.get('cell_type') != 'code' or not src.strip():
        continue
    print(f'=== EXECUTING CELL {i} ===')
    try:
        exec(compile(src, f'<cell {i}>', 'exec'), ns, ns)
        print(f'=== SUCCESS CELL {i} ===')
    except Exception as e:
        print(f'!!! ERROR IN CELL {i}: {type(e).__name__}: {e} !!!')
        traceback.print_exc()
        break

print('DONE RUNNING GLOVE SEGMENT')
