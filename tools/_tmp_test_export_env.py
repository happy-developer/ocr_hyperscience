import os
import subprocess
from pathlib import Path

root = Path('/home/fback/ocr_hyperscience')
repo = root / 'workspace_paddleocr_invoice' / 'PaddleOCR'
runs = root / 'workspace_paddleocr_invoice' / 'runs'
export = root / 'workspace_paddleocr_invoice' / 'export'

def find_prefix(model_dir: Path) -> Path:
    for pref in ('best_accuracy', 'latest'):
        if (model_dir / f'{pref}.pdparams').exists():
            return model_dir / pref
    pds = sorted(model_dir.glob('*.pdparams'), key=lambda x: x.stat().st_mtime, reverse=True)
    if not pds:
        raise FileNotFoundError(model_dir)
    return pds[0].with_suffix('')

def build_env():
    env = os.environ.copy()
    cuda_home_local = (Path.home() / 'ocr_hyperscience' / '.cuda_local').resolve()
    selected = cuda_home_local if cuda_home_local.exists() else Path('/usr/local/cuda')
    env['CUDA_HOME'] = str(selected)

    extra = ['/usr/lib/wsl/lib', '/usr/lib/x86_64-linux-gnu', str(selected / 'lib64')]
    py_lib = Path('/home/fback/ocr_hyperscience/.venv/bin/python').resolve().parent.parent / 'lib'
    site_candidates = sorted(py_lib.glob('python*/site-packages'))
    if site_candidates:
        site = site_candidates[0]
        for rel in [
            'nvidia/cudnn/lib',
            'nvidia/cublas/lib',
            'nvidia/cuda_runtime/lib',
            'nvidia/cufft/lib',
            'nvidia/curand/lib',
            'nvidia/cusolver/lib',
            'nvidia/cusparse/lib',
            'nvidia/nvjitlink/lib',
            'nvidia/nvtx/lib',
        ]:
            p = site / rel
            if p.exists():
                extra.append(str(p))

    existing = [p for p in extra if Path(p).exists()]
    current = env.get('LD_LIBRARY_PATH', '')
    prefix = ':'.join(existing)
    env['LD_LIBRARY_PATH'] = f'{prefix}:{current}' if current else prefix
    return env

env = build_env()
print('CUDA_HOME', env.get('CUDA_HOME'))

det_ckpt = find_prefix(runs / 'det')
rec_ckpt = find_prefix(runs / 'rec')

cmd_det = [
    '/home/fback/ocr_hyperscience/.venv/bin/python', str(repo / 'tools/export_model.py'),
    '-c', str(repo / 'configs/det/det_mv3_db.yml'),
    '-o',
    f'Global.checkpoints={det_ckpt}',
    f'Global.save_inference_dir={export / "det_infer"}',
    'Global.use_gpu=False',
    'Global.export_with_pir=False',
]

cmd_rec = [
    '/home/fback/ocr_hyperscience/.venv/bin/python', str(repo / 'tools/export_model.py'),
    '-c', str(repo / 'configs/rec/rec_mv3_none_bilstm_ctc.yml'),
    '-o',
    f'Global.checkpoints={rec_ckpt}',
    f'Global.save_inference_dir={export / "rec_infer"}',
    'Global.use_gpu=False',
    'Global.export_with_pir=False',
    f'Global.character_dict_path={repo / "ppocr/utils/en_dict.txt"}',
]

print('run det export...')
subprocess.run(cmd_det, check=True, cwd=repo, env=env)
print('run rec export...')
subprocess.run(cmd_rec, check=True, cwd=repo, env=env)
print('exports ok')
