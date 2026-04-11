import json
from pathlib import Path

nb_path = Path('/home/fback/ocr_hyperscience/paddleocr_invoice_train_infer.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

new_src = '''# Export des modeles d'inference
import os
import subprocess
import sys
from pathlib import Path

# Bootstrap minimal pour permettre de relancer cette cellule sans rejouer tout le notebook
if "WORK_DIR" not in globals():
    WORK_DIR = Path("./workspace_paddleocr_invoice").resolve()
if "RUNS_DIR" not in globals():
    RUNS_DIR = WORK_DIR / "runs"
if "EXPORT_DIR" not in globals():
    EXPORT_DIR = WORK_DIR / "export"
if "PADDLEOCR_REPO" not in globals():
    PADDLEOCR_REPO = WORK_DIR / "PaddleOCR"
if "DET_MODEL_DIR" not in globals():
    DET_MODEL_DIR = RUNS_DIR / "det"
if "REC_MODEL_DIR" not in globals():
    REC_MODEL_DIR = RUNS_DIR / "rec"
if "DET_CFG" not in globals():
    DET_CFG = PADDLEOCR_REPO / "configs/det/det_mv3_db.yml"
if "REC_CFG" not in globals():
    REC_CFG = PADDLEOCR_REPO / "configs/rec/rec_mv3_none_bilstm_ctc.yml"
if "CHAR_DICT" not in globals():
    _default_dict = PADDLEOCR_REPO / "ppocr/utils/en_dict.txt"
    CHAR_DICT = _default_dict if _default_dict.exists() else None

for p in [EXPORT_DIR, DET_MODEL_DIR, REC_MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def find_checkpoint_prefix(model_dir: Path):
    for pref in ["best_accuracy", "latest"]:
        if (model_dir / f"{pref}.pdparams").exists():
            return model_dir / pref
    pdparams = sorted(model_dir.glob("*.pdparams"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not pdparams:
        raise FileNotFoundError(f"Aucun checkpoint trouve dans {model_dir}")
    return pdparams[0].with_suffix("")

def build_export_cmd(cfg_path: Path, ckpt_prefix: Path, infer_dir: Path, use_gpu: bool, extra_opts=None):
    cmd = [
        sys.executable,
        str(PADDLEOCR_REPO / "tools/export_model.py"),
        "-c",
        str(cfg_path),
        "-o",
        f"Global.checkpoints={ckpt_prefix}",
        f"Global.save_inference_dir={infer_dir}",
        f"Global.use_gpu={str(bool(use_gpu))}",
        "Global.export_with_pir=False",
    ]
    if extra_opts:
        cmd += list(extra_opts)
    return cmd

def build_cuda_env():
    env = os.environ.copy()

    cuda_home_local = (Path.home() / "ocr_hyperscience" / ".cuda_local").resolve()
    cuda_home_candidates = [
        cuda_home_local,
        Path("/usr/local/cuda"),
        Path("/usr/local/cuda-12.0"),
        Path("/usr/local/cuda-12.1"),
        Path("/usr/local/cuda-12.2"),
        Path("/usr/local/cuda-12.3"),
    ]

    selected_cuda_home = None
    for p in cuda_home_candidates:
        if p.exists():
            selected_cuda_home = p
            break

    if selected_cuda_home is not None:
        env["CUDA_HOME"] = str(selected_cuda_home)

    extra_lib_dirs = [
        "/usr/lib/wsl/lib",
        "/usr/lib/x86_64-linux-gnu",
    ]
    if selected_cuda_home is not None:
        extra_lib_dirs.append(str(selected_cuda_home / "lib64"))

    # Ajoute les libs CUDA/cuDNN packagées par pip (nvidia-*)
    py_lib = Path(sys.executable).resolve().parent.parent / "lib"
    site_candidates = sorted(py_lib.glob("python*/site-packages"))
    if site_candidates:
        site = site_candidates[0]
        nvidia_rel = [
            "nvidia/cudnn/lib",
            "nvidia/cublas/lib",
            "nvidia/cuda_runtime/lib",
            "nvidia/cufft/lib",
            "nvidia/curand/lib",
            "nvidia/cusolver/lib",
            "nvidia/cusparse/lib",
            "nvidia/nvjitlink/lib",
            "nvidia/nvtx/lib",
        ]
        for rel in nvidia_rel:
            p = site / rel
            if p.exists():
                extra_lib_dirs.append(str(p))

    existing = [p for p in extra_lib_dirs if Path(p).exists()]
    if existing:
        current = env.get("LD_LIBRARY_PATH", "")
        prefix = ":".join(existing)
        env["LD_LIBRARY_PATH"] = f"{prefix}:{current}" if current else prefix

    return env

def export_with_fallback(cfg_path: Path, ckpt_prefix: Path, infer_dir: Path, extra_opts=None):
    env_base = build_cuda_env()

    cmd_gpu = build_export_cmd(cfg_path, ckpt_prefix, infer_dir, use_gpu=True, extra_opts=extra_opts)
    print("Running export (GPU):\\n", " ".join(cmd_gpu))

    try:
        subprocess.run(cmd_gpu, check=True, cwd=PADDLEOCR_REPO, env=env_base)
        return "gpu"
    except subprocess.CalledProcessError as gpu_err:
        print("GPU export failed; retry on CPU...")
        print("GPU error:", gpu_err)

    cmd_cpu = build_export_cmd(cfg_path, ckpt_prefix, infer_dir, use_gpu=False, extra_opts=extra_opts)
    env_cpu = dict(env_base)
    env_cpu["CUDA_VISIBLE_DEVICES"] = ""
    print("Running export (CPU fallback):\\n", " ".join(cmd_cpu))
    subprocess.run(cmd_cpu, check=True, cwd=PADDLEOCR_REPO, env=env_cpu)
    return "cpu"

DET_CKPT = find_checkpoint_prefix(DET_MODEL_DIR)
REC_CKPT = find_checkpoint_prefix(REC_MODEL_DIR)

DET_INFER_DIR = EXPORT_DIR / "det_infer"
REC_INFER_DIR = EXPORT_DIR / "rec_infer"
DET_INFER_DIR.mkdir(parents=True, exist_ok=True)
REC_INFER_DIR.mkdir(parents=True, exist_ok=True)

rec_extra_opts = []
if CHAR_DICT is not None:
    rec_extra_opts.append(f"Global.character_dict_path={CHAR_DICT}")

det_mode = export_with_fallback(DET_CFG, DET_CKPT, DET_INFER_DIR)
rec_mode = export_with_fallback(REC_CFG, REC_CKPT, REC_INFER_DIR, extra_opts=rec_extra_opts)

print("DET_INFER_DIR:", DET_INFER_DIR, f"(exported via {det_mode})")
print("REC_INFER_DIR:", REC_INFER_DIR, f"(exported via {rec_mode})")
'''

for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if src.startswith("# Export des modeles d'inference"):
        cell['source'] = [line + '\n' for line in new_src.split('\n')]
        break
else:
    raise RuntimeError('Export cell not found')

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('Patched export cell (v4 standalone)')
