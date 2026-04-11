import json
from pathlib import Path

nb = json.loads(Path('/home/fback/ocr_hyperscience/paddleocr_invoice_train_infer.ipynb').read_text(encoding='utf-8'))
for i,c in enumerate(nb.get('cells', [])):
    if c.get('cell_type')!='code':
        continue
    src=''.join(c.get('source', []))
    if src.startswith('# Export des modeles d\'inference'):
        print('EXPORT_CELL_INDEX', i)
        print(src)
        break
