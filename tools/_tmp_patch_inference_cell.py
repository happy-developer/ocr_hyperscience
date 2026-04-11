import json
from pathlib import Path

nb_path = Path('/home/fback/ocr_hyperscience/paddleocr_invoice_train_infer.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

new_src = '''# Inference OCR brute sur un JPG facture (RapidOCR stable fallback)
from pathlib import Path

def _first_image_under(root: Path):
    exts = {'.jpg', '.jpeg', '.png'}
    if not root.exists():
        return None
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            return p
    return None

def resolve_test_image():
    candidates = []

    # 1) TEST_IMAGE (si défini en amont)
    if 'TEST_IMAGE' in globals() and TEST_IMAGE is not None:
        candidates.append(Path(TEST_IMAGE))

    # 2) chemins usuels Windows/WSL
    candidates += [
        Path('/mnt/c/Users/fback/Downloads/batch3-0501.jpg'),
        Path('/mnt/c/Users/fback/Downloads/batch3-0501.JPG'),
        Path('/mnt/c/Users/fback/Desktop/batch3-0501.jpg'),
        Path('/mnt/c/Users/fback/Documents/batch3-0501.jpg'),
    ]

    for p in candidates:
        if p.exists() and p.is_file():
            return p

    # 3) fallback dataset: première image de batch_2 si disponible en mémoire
    if 'batch2_images' in globals() and len(batch2_images) > 0:
        return Path(batch2_images[0])

    # 4) fallback local: scan BATCH2_ROOT si défini
    if 'BATCH2_ROOT' in globals() and BATCH2_ROOT.exists():
        p = _first_image_under(BATCH2_ROOT)
        if p is not None:
            return p

    # 5) fallback autonome: cache KaggleHub connu
    kaggle_batch2_root = (
        Path.home()
        / '.cache/kagglehub/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr'
    )
    if kaggle_batch2_root.exists():
        # ex: .../versions/3/batch_2
        for maybe in sorted(kaggle_batch2_root.rglob('batch_2')):
            p = _first_image_under(maybe)
            if p is not None:
                return p

    raise FileNotFoundError(
        'Aucune image de test trouvée. '
        'Définis TEST_IMAGE (ex: Path("/mnt/c/.../facture.jpg")) '
        'ou vérifie que batch_2 est bien présent.'
    )

TEST_IMAGE = resolve_test_image()
print('Image de test utilisée:', TEST_IMAGE)

print('Inference OCR engine: RapidOCR (onnxruntime, CPU)')
ocr = RapidOCR()
result, _ = ocr(str(TEST_IMAGE))

all_texts = []
all_scores = []
for item in result or []:
    if not item or len(item) < 3:
        continue
    text = str(item[1]).strip()
    score = float(item[2]) if item[2] is not None else np.nan
    if text:
        all_texts.append(text)
        all_scores.append(score)

print(f'Nombre de chaines extraites: {len(all_texts)}')
if len(all_scores) > 0:
    print(f'Confiance moyenne: {np.nanmean(all_scores):.4f}')

print('\\n--- TEXTES EXTRAITS ---')
for i, t in enumerate(all_texts[:200], 1):
    print(f'{i:03d}. {t}')

raw_txt_out = WORK_DIR / 'inference_raw_text.txt'
raw_txt_out.write_text('\\n'.join(all_texts), encoding='utf-8')
print('\\nFichier texte sauvegarde:', raw_txt_out)
'''

changed = False
for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if src.startswith('# Inference OCR brute sur un JPG facture'):
        cell['source'] = [line + '\n' for line in new_src.split('\n')]
        changed = True
        break

if not changed:
    raise RuntimeError('Inference cell not found')

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('Patched inference cell (standalone fallback incl. Kaggle cache)')
