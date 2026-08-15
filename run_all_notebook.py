import json
import traceback
from pathlib import Path

root = Path(__file__).resolve().parent
nb_path = root / 'cse440_project.ipynb'
nb = json.loads(nb_path.read_text(encoding='utf-8'))
ns = {}
from IPython.display import display
ns['display'] = display

for i, cell in enumerate(nb.get('cells', [])):
    src = ''.join(cell.get('source', []))
    if cell.get('cell_type') != 'code' or not src.strip():
        continue
    print(f'=== EXECUTING CELL {i} ===')
    try:
        exec(compile(src, f'<cell {i}>', 'exec'), ns, ns)
    except Exception as e:
        print(f'!!! ERROR IN CELL {i} !!!')
        traceback.print_exc()
        raise

print('ALL CELLS EXECUTED SUCCESSFULLY')
