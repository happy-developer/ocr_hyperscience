import json
from pathlib import Path

nb_path = Path('/home/fback/ocr_hyperscience/paddleocr_invoice_train_infer.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if not src.startswith('# Inference OCR brute sur un JPG facture'):
        continue

    src = src.replace("print('\n--- TEXTES EXTRAITS ---')", "print('\\n--- TEXTES EXTRAITS ---')")
    src = src.replace("raw_txt_out.write_text('\n'.join(all_texts), encoding='utf-8')", "raw_txt_out.write_text('\\n'.join(all_texts), encoding='utf-8')")
    src = src.replace("print('\nFichier texte sauvegarde:', raw_txt_out)", "print('\\nFichier texte sauvegarde:', raw_txt_out)")

    # Répare aussi les variantes cassées sur plusieurs lignes si présentes
    src = src.replace("print('\n--- TEXTES EXTRAITS ---')", "print('\\n--- TEXTES EXTRAITS ---')")
    src = src.replace("print('\nFichier texte sauvegarde:'", "print('\\nFichier texte sauvegarde:'")

    cell['source'] = [line + '\n' for line in src.split('\n')]
    break
else:
    raise RuntimeError('Inference cell not found')

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('Fixed newline escaping in inference cell')
