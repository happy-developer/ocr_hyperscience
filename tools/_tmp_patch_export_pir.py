import json
from pathlib import Path

path = Path('/home/fback/ocr_hyperscience/paddleocr_invoice_train_infer.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))

needle = '        f"Global.use_gpu={str(bool(use_gpu))}",\n'
insert = '        f"Global.use_gpu={str(bool(use_gpu))}",\n        "Global.export_with_pir=False",\n'
changed = False

for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if src.startswith("# Export des modeles d'inference") and needle in src:
        src2 = src.replace(needle, insert)
        if src2 != src:
            cell['source'] = [line + '\n' for line in src2.split('\n')]
            changed = True
        break

if not changed:
    raise RuntimeError('Failed to patch export_with_pir flag')

path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('Patched export_with_pir=False')
