"""Check that the released data loader produces exactly the samples of the
original training loader, with and without augmentation.

    python verification/check_data.py ORIG_CODE DATA_ROOT MANIFEST [N]

Both loaders are given identical random states before every sample (Python
random, NumPy's global generator, and the generator used for the MOLA cut,
which the original draws from fresh entropy), so augmented samples are
comparable too. N samples are drawn from each split.
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mesa.data import TileDataset  # noqa: E402

orig_code, root, manifest = sys.argv[1:4]
n_check = int(sys.argv[4]) if len(sys.argv) > 4 else 64
sys.path.insert(0, orig_code)
from mars_ctx_dataset import MarsCTXDEMDataset  # noqa: E402

incidence = str(Path(root) / 'ctx_overlapping_95pct_polar.json')
_default_rng = np.random.default_rng


def seed_everything(k):
    random.seed(k)
    np.random.seed(k)
    np.random.default_rng = lambda *a, **kw: _default_rng(10_000 + k)


def same(a, b):
    a = a.numpy() if torch.is_tensor(a) else np.asarray(a)
    b = b.numpy() if torch.is_tensor(b) else np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


failures = 0
for split, augment in (('val', False), ('train', True)):
    orig = MarsCTXDEMDataset(
        dataset_root=root, incidence_json=incidence, manifest_path=manifest, crop_size=384, augment=augment,
        split=split, shot_dir=str(Path(root) / 'shot'), input_mode='mola_conf', cutout_mode='trackaware',
        cutout_p=0.3, cutout_dilate_px=6, dist_dir=str(Path(root) / 'd1_shot'), conf_d0=2000.0, relief_cond=True)
    clean = TileDataset(root, manifest, split, incidence_json=incidence, augment=augment)
    assert len(orig) == len(clean), (len(orig), len(clean))
    assert all(a['dem_tile'] == b['dem_tile'] and a['ctx_tile'] == b['ctx_tile']
               for a, b in zip(orig.rows, clean.rows)), 'row order differs'
    idx = np.random.default_rng(1).choice(len(clean), size=min(n_check, len(clean)), replace=False)
    n_bad, n_cut = 0, 0
    for k, i in enumerate(idx):
        seed_everything(k)
        ctx, x, inc, dem, shot_m, shot_mask, mola, meta = orig[int(i)]
        seed_everything(k)
        s = clean[int(i)]
        checks = {
            'ctx': same(ctx, s['ctx']), 'x': same(x, s['x']), 'cond': same(inc, s['cond']),
            'dem': same(dem, s['dem']), 'mola': same(mola, s['mola']), 'shot_m': same(shot_m, s['shot_m']),
            'shot_mask': same(shot_mask, s['shot_mask']),
            'stats': all(meta[k_] == s[k_] for k_ in ('dem_mean', 'dem_std', 'mola_mean', 'mola_std')),
            'conf_mean': meta['conf_mean'] == s['conf_mean'],
        }
        n_cut += int(not torch.equal(x[0], mola[0]))
        if not all(checks.values()):
            n_bad += 1
            print(f'  {split} row {i}: differs in {[k_ for k_, v in checks.items() if not v]}')
    failures += n_bad
    print(f'{split} (augment={augment}): {len(idx)} samples compared, {n_bad} differ; '
          f'{n_cut} had the MOLA cut applied')
np.random.default_rng = _default_rng
print(f'DATA CHECK {"PASS" if failures == 0 else "FAIL"}')
