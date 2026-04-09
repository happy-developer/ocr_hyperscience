from pathlib import Path

import nbformat as nbf


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# PaddleOCR - Training sur `batch_1` + validation qualitative sur `batch_2`

Ce notebook:
- telecharge le dataset Kaggle
- convertit les annotations CSV de `batch_1` en format PaddleOCR
- cree un split interne train/val depuis `batch_1` (supervise)
- entraine des modeles `det` et `rec`
- exporte les modeles d'inference
- lance une inference brute sur un JPG cible
- lance une validation qualitative non annotee sur les images de `batch_2`

Important: `batch_2` est traite comme jeu non annote (pas de metriques GT)."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# Dependances (a executer une fois si necessaire)
# Si Paddle n'est pas deja installe dans ton environnement, choisis UNE option:
# 1) GPU (adapter selon ta stack CUDA)
# !pip install -q paddlepaddle-gpu==2.6.2.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
# 2) CPU
# !pip install -q paddlepaddle==2.6.2

# Dependances projet
# !pip install -q kagglehub pandas numpy opencv-python pyyaml matplotlib tqdm paddleocr"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """import ast
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import kagglehub
import numpy as np
import pandas as pd
import paddle
from tqdm.auto import tqdm

print(f"Python: {sys.version.split()[0]}")
print(f"Paddle: {paddle.__version__}")
print(f"CUDA available: {paddle.is_compiled_with_cuda()}")
if paddle.is_compiled_with_cuda():
    print(f"CUDA devices: {paddle.device.cuda.device_count()}")"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# Configuration
DATASET_SLUG = "osamahosamabdellatif/high-quality-invoice-images-for-ocr"
TRAIN_BATCH_NAME = "batch_1"
UNLABELED_VAL_BATCH_NAME = "batch_2"

TRAIN_VAL_SPLIT = 0.9  # split supervise interne sur batch_1
RANDOM_SEED = 42
BATCH2_MAX_IMAGES = 300  # limite de validation qualitative sur batch_2

WORK_DIR = Path("./workspace_paddleocr_invoice").resolve()
PREP_DIR = WORK_DIR / "prepared_data"
RUNS_DIR = WORK_DIR / "runs"
EXPORT_DIR = WORK_DIR / "export"

# Parametres d'entrainement (ajuste selon ton GPU/temps)
DET_EPOCHS = 50
REC_EPOCHS = 50
USE_GPU = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0

TEST_IMAGE = Path("/mnt/c/Users/fback/Downloads/batch3-0501.jpg")

for p in [WORK_DIR, PREP_DIR, RUNS_DIR, EXPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("WORK_DIR:", WORK_DIR)
print("USE_GPU:", USE_GPU)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# Download dataset
dataset_path = Path(kagglehub.dataset_download(DATASET_SLUG)).resolve()
print("Dataset path:", dataset_path)

def print_tree(root: Path, max_depth: int = 3, max_entries: int = 120):
    root = root.resolve()
    count = 0
    for p in sorted(root.rglob("*")):
        depth = len(p.relative_to(root).parts)
        if depth > max_depth:
            continue
        print("  " * (depth - 1) + ("- " if depth > 0 else "") + p.name)
        count += 1
        if count >= max_entries:
            print("... (truncated)")
            break

print_tree(dataset_path, max_depth=3, max_entries=120)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# Helpers CSV -> format PaddleOCR
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def find_col(cols, candidates):
    nmap = {c: norm_col(c) for c in cols}
    for cand in candidates:
        ncand = norm_col(cand)
        for c, n in nmap.items():
            if n == ncand:
                return c
    for cand in candidates:
        ncand = norm_col(cand)
        for c, n in nmap.items():
            if ncand in n:
                return c
    return None

def find_batch_root(root: Path, batch_name: str) -> Path:
    candidates = [p for p in root.rglob(batch_name) if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"Impossible de trouver le dossier '{batch_name}' sous {root}")
    scored = []
    for p in candidates:
        csv_count = len(list(p.glob("*.csv")))
        img_count = sum(1 for f in p.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
        scored.append((csv_count, img_count, -len(p.parts), p))
    scored.sort(reverse=True)
    return scored[0][-1]

def build_image_lookup(batch_root: Path):
    lookup = {}
    for p in batch_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            rel = str(p.relative_to(batch_root)).replace("\\\\", "/").lower()
            lookup.setdefault(rel, p)
            lookup.setdefault(p.name.lower(), p)
    return lookup

def list_images_in_batch(dataset_root: Path, batch_name: str):
    batch_root = find_batch_root(dataset_root, batch_name)
    images = sorted([p for p in batch_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    return batch_root, images

def parse_points_from_row(row, cols):
    poly_col = find_col(cols, ["points", "polygon", "bbox", "box", "coordinates", "vertices"])
    if poly_col is not None and pd.notna(row.get(poly_col)):
        raw = row.get(poly_col)
        try:
            obj = ast.literal_eval(str(raw))
            if isinstance(obj, (list, tuple)):
                if len(obj) >= 4 and isinstance(obj[0], (list, tuple)):
                    return [[float(x), float(y)] for x, y in obj[:4]]
                if len(obj) >= 8 and isinstance(obj[0], (int, float, str)):
                    vals = [float(v) for v in obj[:8]]
                    return [[vals[0], vals[1]], [vals[2], vals[3]], [vals[4], vals[5]], [vals[6], vals[7]]]
        except Exception:
            pass
        nums = re.findall(r"-?\\d+(?:\\.\\d+)?", str(raw))
        if len(nums) >= 8:
            vals = [float(v) for v in nums[:8]]
            return [[vals[0], vals[1]], [vals[2], vals[3]], [vals[4], vals[5]], [vals[6], vals[7]]]

    x1 = find_col(cols, ["x1", "left_top_x", "tlx"])
    y1 = find_col(cols, ["y1", "left_top_y", "tly"])
    x2 = find_col(cols, ["x2", "right_top_x", "trx"])
    y2 = find_col(cols, ["y2", "right_top_y", "try"])
    x3 = find_col(cols, ["x3", "right_bottom_x", "brx"])
    y3 = find_col(cols, ["y3", "right_bottom_y", "bry"])
    x4 = find_col(cols, ["x4", "left_bottom_x", "blx"])
    y4 = find_col(cols, ["y4", "left_bottom_y", "bly"])
    if all(c is not None for c in [x1, y1, x2, y2, x3, y3, x4, y4]):
        return [
            [float(row[x1]), float(row[y1])],
            [float(row[x2]), float(row[y2])],
            [float(row[x3]), float(row[y3])],
            [float(row[x4]), float(row[y4])],
        ]

    xmin = find_col(cols, ["xmin", "x_min", "left"])
    ymin = find_col(cols, ["ymin", "y_min", "top"])
    xmax = find_col(cols, ["xmax", "x_max", "right"])
    ymax = find_col(cols, ["ymax", "y_max", "bottom"])
    if all(c is not None for c in [xmin, ymin, xmax, ymax]):
        x_min, y_min, x_max, y_max = float(row[xmin]), float(row[ymin]), float(row[xmax]), float(row[ymax])
        return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

    x = find_col(cols, ["x", "left"])
    y = find_col(cols, ["y", "top"])
    w = find_col(cols, ["w", "width"])
    h = find_col(cols, ["h", "height"])
    if all(c is not None for c in [x, y, w, h]):
        xx, yy, ww, hh = float(row[x]), float(row[y]), float(row[w]), float(row[h])
        return [[xx, yy], [xx + ww, yy], [xx + ww, yy + hh], [xx, yy + hh]]
    return None

def resolve_image_path(raw_value, batch_root: Path, image_lookup):
    if pd.isna(raw_value):
        return None
    raw = str(raw_value).strip()
    if not raw:
        return None
    candidates = [raw, raw.replace("\\\\", "/"), raw.lstrip("./"), Path(raw).name]
    for c in candidates:
        if c.lower() in image_lookup:
            return image_lookup[c.lower()]
    for c in candidates:
        p = (batch_root / c).resolve()
        if p.exists() and p.suffix.lower() in IMAGE_EXTS:
            return p
    stem = Path(raw).stem.lower()
    for _, p in image_lookup.items():
        if p.stem.lower() == stem:
            return p
    return None

def load_batch_annotations(dataset_root: Path, batch_name: str):
    batch_root = find_batch_root(dataset_root, batch_name)
    csv_files = sorted(batch_root.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Aucun CSV trouve dans {batch_root}")

    image_lookup = build_image_lookup(batch_root)
    image_col_candidates = ["image", "image_path", "img_path", "img", "file", "file_name", "filename", "path", "image_name"]
    text_col_candidates = ["text", "label", "transcription", "ocr_text", "word", "value", "content", "token"]

    by_image = defaultdict(list)
    report = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        cols = list(df.columns)
        image_col = find_col(cols, image_col_candidates)
        text_col = find_col(cols, text_col_candidates)

        if image_col is None or text_col is None:
            raise ValueError(
                f"CSV incompatible: {csv_path.name}\\n"
                f"Colonnes trouvees: {cols}\\n"
                f"Image col: {image_col} | Text col: {text_col}"
            )

        kept = 0
        for _, row in df.iterrows():
            img_path = resolve_image_path(row.get(image_col), batch_root, image_lookup)
            if img_path is None:
                continue
            text = str(row.get(text_col, "")).strip()
            if not text:
                continue
            points = parse_points_from_row(row, cols)
            if points is None:
                continue
            by_image[str(img_path)].append({"transcription": text, "points": points, "difficult": False})
            kept += 1

        report.append({"csv": str(csv_path), "rows": len(df), "kept_rows": kept, "columns": cols})

    return batch_root, by_image, report

def split_annotations_by_image(by_image: dict, train_ratio: float, seed: int):
    image_paths = sorted(by_image.keys())
    if len(image_paths) == 0:
        return {}, {}
    if len(image_paths) == 1:
        only = image_paths[0]
        return {only: by_image[only]}, {only: by_image[only]}

    rng = np.random.default_rng(seed)
    rng.shuffle(image_paths)

    train_n = int(round(len(image_paths) * train_ratio))
    train_n = max(1, min(len(image_paths) - 1, train_n))

    train_keys = set(image_paths[:train_n])
    val_keys = set(image_paths[train_n:])

    train_by = {k: by_image[k] for k in train_keys}
    val_by = {k: by_image[k] for k in val_keys}
    return train_by, val_by

def save_det_labels(by_image: dict, output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for img_path, anns in by_image.items():
            payload = json.dumps(anns, ensure_ascii=False)
            f.write(f"{img_path}\\t{payload}\\n")

def crop_by_quad(img, pts):
    pts = np.array(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        return None

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    rect = np.array([tl, tr, br, bl], dtype=np.float32)

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = int(max(width_a, width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = int(max(height_a, height_b))

    if max_w < 2 or max_h < 2:
        return None

    dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, m, (max_w, max_h), flags=cv2.INTER_CUBIC)

    if warped.shape[0] > warped.shape[1] * 1.5:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped

def save_rec_dataset(by_image: dict, rec_img_dir: Path, rec_label_file: Path, prefix: str):
    rec_img_dir.mkdir(parents=True, exist_ok=True)
    rec_label_file.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    sample_idx = 0
    for img_path, anns in tqdm(by_image.items(), desc=f"Generate rec crops ({prefix})"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        for ann in anns:
            text = str(ann.get("transcription", "")).strip()
            pts = ann.get("points")
            if not text or pts is None:
                continue
            crop = crop_by_quad(img, pts)
            if crop is None:
                continue
            out_name = f"{prefix}_{sample_idx:07d}.jpg"
            out_path = rec_img_dir / out_name
            cv2.imwrite(str(out_path), crop)
            safe_text = text.replace("\\t", " ").replace("\\n", " ")
            lines.append(f"{out_path}\\t{safe_text}\\n")
            sample_idx += 1

    with rec_label_file.open("w", encoding="utf-8") as f:
        f.writelines(lines)

    return len(lines)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# Construction des donnees supervisees depuis batch_1 + collecte batch_2 non annote
batch1_root, batch1_by_image, batch1_report = load_batch_annotations(dataset_path, TRAIN_BATCH_NAME)
train_by_image, val_by_image = split_annotations_by_image(batch1_by_image, TRAIN_VAL_SPLIT, RANDOM_SEED)

batch2_root, batch2_images = list_images_in_batch(dataset_path, UNLABELED_VAL_BATCH_NAME)

print("Batch1 root:", batch1_root)
print("Batch2 root:", batch2_root)
print("Batch1 images annotees (total):", len(batch1_by_image))
print("Train images annotees:", len(train_by_image))
print("Val images annotees (split interne):", len(val_by_image))
print("Batch2 images non annotees:", len(batch2_images))

print("\\nBatch1 CSV report:")
for r in batch1_report:
    print(f"- {Path(r['csv']).name}: rows={r['rows']}, kept={r['kept_rows']}")

if len(train_by_image) == 0 or len(val_by_image) == 0:
    raise RuntimeError("Split supervise vide. Verifie les annotations batch_1.")

train_det_label = PREP_DIR / "train_det.txt"
val_det_label = PREP_DIR / "val_det.txt"
save_det_labels(train_by_image, train_det_label)
save_det_labels(val_by_image, val_det_label)

train_rec_label = PREP_DIR / "train_rec.txt"
val_rec_label = PREP_DIR / "val_rec.txt"
train_rec_count = save_rec_dataset(train_by_image, PREP_DIR / "rec_images" / "train", train_rec_label, "train")
val_rec_count = save_rec_dataset(val_by_image, PREP_DIR / "rec_images" / "val", val_rec_label, "val")

batch2_list_file = PREP_DIR / "batch2_unlabeled_images.txt"
batch2_list_file.write_text("\\n".join(str(p) for p in batch2_images), encoding="utf-8")

print("\\nFichiers generes:")
print("-", train_det_label)
print("-", val_det_label)
print("-", train_rec_label, "(samples=", train_rec_count, ")")
print("-", val_rec_label, "(samples=", val_rec_count, ")")
print("-", batch2_list_file)

if train_rec_count == 0 or val_rec_count == 0:
    raise RuntimeError("Le dataset rec est vide apres crop. Verifie la qualite/format des bbox.")"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# Setup PaddleOCR repo (scripts d'entrainement officiels)
PADDLEOCR_REPO = WORK_DIR / "PaddleOCR"
if not PADDLEOCR_REPO.exists():
    subprocess.run(["git", "clone", "https://github.com/PaddlePaddle/PaddleOCR.git", str(PADDLEOCR_REPO)], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(PADDLEOCR_REPO / "requirements.txt")], check=True)
print("PaddleOCR repo:", PADDLEOCR_REPO)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# Selection automatique de configs stables
def pick_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

det_cfg_candidates = [
    PADDLEOCR_REPO / "configs/det/det_mv3_db.yml",
    PADDLEOCR_REPO / "configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml",
    PADDLEOCR_REPO / "configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml",
]
rec_cfg_candidates = [
    PADDLEOCR_REPO / "configs/rec/rec_mv3_none_bilstm_ctc.yml",
    PADDLEOCR_REPO / "configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml",
    PADDLEOCR_REPO / "configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml",
]

DET_CFG = pick_first_existing(det_cfg_candidates)
REC_CFG = pick_first_existing(rec_cfg_candidates)
if DET_CFG is None or REC_CFG is None:
    raise FileNotFoundError(f"Config introuvable. DET={DET_CFG}, REC={REC_CFG}")

print("DET_CFG:", DET_CFG)
print("REC_CFG:", REC_CFG)

dict_candidates = [
    PADDLEOCR_REPO / "ppocr/utils/en_dict.txt",
    PADDLEOCR_REPO / "ppocr/utils/dict/en_dict.txt",
]
CHAR_DICT = pick_first_existing(dict_candidates)
print("CHAR_DICT:", CHAR_DICT)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# Entrainement DET (val supervisee = split interne de batch_1)
DET_MODEL_DIR = RUNS_DIR / "det"
DET_MODEL_DIR.mkdir(parents=True, exist_ok=True)

cmd_det = [
    sys.executable,
    str(PADDLEOCR_REPO / "tools/train.py"),
    "-c", str(DET_CFG),
    "-o",
    f"Global.use_gpu={str(USE_GPU)}",
    f"Global.epoch_num={DET_EPOCHS}",
    f"Global.save_model_dir={DET_MODEL_DIR}",
    "Train.dataset.data_dir=/",
    f"Train.dataset.label_file_list=['{train_det_label}']",
    "Eval.dataset.data_dir=/",
    f"Eval.dataset.label_file_list=['{val_det_label}']",
]

print("Running:\\n", " ".join(cmd_det))
subprocess.run(cmd_det, check=True, cwd=PADDLEOCR_REPO)
print("DET training termine")"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# Entrainement REC (val supervisee = split interne de batch_1)
REC_MODEL_DIR = RUNS_DIR / "rec"
REC_MODEL_DIR.mkdir(parents=True, exist_ok=True)

cmd_rec = [
    sys.executable,
    str(PADDLEOCR_REPO / "tools/train.py"),
    "-c", str(REC_CFG),
    "-o",
    f"Global.use_gpu={str(USE_GPU)}",
    f"Global.epoch_num={REC_EPOCHS}",
    f"Global.save_model_dir={REC_MODEL_DIR}",
    "Train.dataset.data_dir=/",
    f"Train.dataset.label_file_list=['{train_rec_label}']",
    "Eval.dataset.data_dir=/",
    f"Eval.dataset.label_file_list=['{val_rec_label}']",
]
if CHAR_DICT is not None:
    cmd_rec += [f"Global.character_dict_path={CHAR_DICT}"]

print("Running:\\n", " ".join(cmd_rec))
subprocess.run(cmd_rec, check=True, cwd=PADDLEOCR_REPO)
print("REC training termine")"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# Export des modeles d'inference
def find_checkpoint_prefix(model_dir: Path):
    for pref in ["best_accuracy", "latest"]:
        if (model_dir / f"{pref}.pdparams").exists():
            return model_dir / pref
    pdparams = sorted(model_dir.glob("*.pdparams"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not pdparams:
        raise FileNotFoundError(f"Aucun checkpoint trouve dans {model_dir}")
    return pdparams[0].with_suffix("")

DET_CKPT = find_checkpoint_prefix(DET_MODEL_DIR)
REC_CKPT = find_checkpoint_prefix(REC_MODEL_DIR)

DET_INFER_DIR = EXPORT_DIR / "det_infer"
REC_INFER_DIR = EXPORT_DIR / "rec_infer"
DET_INFER_DIR.mkdir(parents=True, exist_ok=True)
REC_INFER_DIR.mkdir(parents=True, exist_ok=True)

cmd_export_det = [
    sys.executable, str(PADDLEOCR_REPO / "tools/export_model.py"),
    "-c", str(DET_CFG),
    "-o",
    f"Global.checkpoints={DET_CKPT}",
    f"Global.save_inference_dir={DET_INFER_DIR}",
]
cmd_export_rec = [
    sys.executable, str(PADDLEOCR_REPO / "tools/export_model.py"),
    "-c", str(REC_CFG),
    "-o",
    f"Global.checkpoints={REC_CKPT}",
    f"Global.save_inference_dir={REC_INFER_DIR}",
]
if CHAR_DICT is not None:
    cmd_export_rec += [f"Global.character_dict_path={CHAR_DICT}"]

subprocess.run(cmd_export_det, check=True, cwd=PADDLEOCR_REPO)
subprocess.run(cmd_export_rec, check=True, cwd=PADDLEOCR_REPO)

print("DET_INFER_DIR:", DET_INFER_DIR)
print("REC_INFER_DIR:", REC_INFER_DIR)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# Inference OCR brute sur un JPG facture
from paddleocr import PaddleOCR

if not TEST_IMAGE.exists():
    raise FileNotFoundError(f"Image de test introuvable: {TEST_IMAGE}")

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    device=("gpu:0" if USE_GPU else "cpu"),
    det_model_dir=str(DET_INFER_DIR),
    rec_model_dir=str(REC_INFER_DIR),
)

result = ocr.ocr(str(TEST_IMAGE), cls=True)

all_texts = []
all_scores = []
for page in result:
    for item in page:
        if not item or len(item) < 2:
            continue
        text = str(item[1][0]).strip()
        score = float(item[1][1]) if len(item[1]) > 1 else np.nan
        if text:
            all_texts.append(text)
            all_scores.append(score)

print(f"Nombre de chaines extraites: {len(all_texts)}")
if len(all_scores) > 0:
    print(f"Confiance moyenne: {np.nanmean(all_scores):.4f}")

print("\\n--- TEXTES EXTRAITS ---")
for i, t in enumerate(all_texts[:200], 1):
    print(f"{i:03d}. {t}")

raw_txt_out = WORK_DIR / "inference_raw_text.txt"
raw_txt_out.write_text("\\n".join(all_texts), encoding="utf-8")
print("\\nFichier texte sauvegarde:", raw_txt_out)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# Validation qualitative non annotee sur batch_2
if len(batch2_images) == 0:
    raise RuntimeError("Aucune image trouvee dans batch_2 pour validation qualitative.")

rows = []
max_images = min(BATCH2_MAX_IMAGES, len(batch2_images))
for img_path in tqdm(batch2_images[:max_images], desc="Batch2 qualitative validation"):
    res = ocr.ocr(str(img_path), cls=True)
    texts = []
    confs = []
    for page in res:
        for item in page:
            if not item or len(item) < 2:
                continue
            txt = str(item[1][0]).strip()
            if not txt:
                continue
            score = float(item[1][1]) if len(item[1]) > 1 else np.nan
            texts.append(txt)
            confs.append(score)

    rows.append(
        {
            "image_path": str(img_path),
            "n_texts": len(texts),
            "mean_conf": float(np.nanmean(confs)) if len(confs) > 0 else np.nan,
            "raw_text": " | ".join(texts),
        }
    )

val_df = pd.DataFrame(rows)
val_csv = WORK_DIR / "batch2_unlabeled_validation.csv"
val_df.to_csv(val_csv, index=False)

print("Validation qualitative batch_2 terminee")
print("Images traitees:", len(val_df))
print("Moyenne n_texts:", val_df["n_texts"].mean() if len(val_df) else 0)
print("Moyenne confidence:", val_df["mean_conf"].mean(skipna=True) if len(val_df) else float("nan"))
print("CSV resultat:", val_csv)
val_df.head(10)"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Notes
- `batch_2` est utilise pour validation qualitative uniquement (non annote).
- Les metriques supervisees d'entrainement sont calculees sur un split interne de `batch_1`.
- Si l'entrainement est trop long, baisse `DET_EPOCHS` et `REC_EPOCHS` (ex: 10/10).
- Tu peux changer `TEST_IMAGE` pour tester n'importe quel JPG de facture."""
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }

    output = Path("/home/fback/ocr_hyperscience/paddleocr_invoice_train_infer.ipynb")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook created: {output}")


if __name__ == "__main__":
    main()
