"""Convert the production checkpoint and check that the released network
reproduces the original network exactly.

    python verification/check_model.py ORIG_CODE ORIG_RUN_DIR OUT_CHECKPOINT [DATA_ROOT TEST_MANIFEST]

ORIG_CODE      directory with the original film_unet.py / implicit_solve.py /
               joint_shape_albedo.py (the run's own code)
ORIG_RUN_DIR   run directory with config.json and checkpoints/best.pt
OUT_CHECKPOINT where to write the released checkpoint (the original weights
               without the unused `mola_weight` scalar)

Compares the two networks on random inputs and, if a test manifest is given,
on real test samples, in float32 on the CPU (bitwise) and on the GPU.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mesa.model import load_model  # noqa: E402

orig_code, run_dir, out_ckpt = sys.argv[1:4]
sys.path.insert(0, orig_code)
from joint_shape_albedo import FiLMUNetDualHead  # noqa: E402

cfg = json.load(open(Path(run_dir) / 'config.json'))
orig = FiLMUNetDualHead(
    in_channels=int(cfg['in_channels']), out_channels=1, depths=cfg['depths'], dropout=cfg['dropout'],
    use_attention=cfg['use_attention'], skip_mode=cfg['skip_mode'], skip_blur_sigma=cfg['skip_blur_sigma'],
    sigma_lo=cfg['sigma_lo'], sigma_hi=cfg['sigma_hi'], conf_gamma=cfg['conf_gamma'], cg_iters=cfg['cg_iters'],
    cg_mu=cfg['cg_mu'], freeze_mu=cfg['freeze_mu'], grad_lambda=cfg['grad_lambda'],
    grad_cg_iters=cfg['grad_cg_iters'], trust_hidden=cfg['trust_hidden'], upsample_mode=cfg['upsample_mode'],
    film_condition_dim=cfg['film_condition_dim'], relief_film=cfg['relief_film'],
    relief_solve=cfg['relief_solve'], film_sun_off=cfg.get('film_sun_off', False))
ck = torch.load(Path(run_dir) / 'checkpoints' / 'best.pt', map_location='cpu', weights_only=False)
state = {k.removeprefix('_forward_module.').removeprefix('module.'): v for k, v in ck['model'].items()}
missing, unexpected = orig.load_state_dict(state, strict=False)
assert not missing and not unexpected, (missing, unexpected)
orig.eval()

released = {k: v for k, v in state.items() if k != 'mola_weight'}
print(f"dropped from the checkpoint: {sorted(set(state) - set(released))} "
      f"(value {float(state['mola_weight']):.4f}; the parameter is never used by the implicit fusion)")
torch.save({'model': released, 'epoch': ck.get('epoch'),
            'note': 'MESA DEM network, epoch 88'}, out_ckpt)
clean = load_model(out_ckpt)
n_orig = sum(p.numel() for p in orig.parameters())
n_clean = sum(p.numel() for p in clean.parameters())
print(f'parameters: original {n_orig:,}, released {n_clean:,}')


def compare(tag, ctx, x, cond, device):
    a, b = orig.to(device), clean.to(device)
    with torch.no_grad():
        za, aa = a(ctx.to(device), x.to(device), cond.to(device))
        zb, ab = b(ctx.to(device), x.to(device), cond.to(device))
    dz = (za - zb).abs().max().item()
    da = (aa - ab).abs().max().item()
    print(f'{tag} [{device}]: max |dz| = {dz:.3e}, max |d albedo| = {da:.3e}, '
          f'bitwise equal: {torch.equal(za, zb) and torch.equal(aa, ab)}')
    return dz, da


torch.manual_seed(0)
B = 2
ctx = torch.rand(B, 1, 384, 384)
x = torch.stack([torch.randn(B, 384, 384), torch.rand(B, 384, 384)], 1)
sun = torch.nn.functional.normalize(torch.randn(B, 3), dim=1)
sun[:, 2] = sun[:, 2].abs()
cond = torch.cat([sun, torch.randn(B, 1) * 0.5], 1)
results = [compare('random inputs', ctx, x, cond, 'cpu')]
if torch.cuda.is_available():
    results.append(compare('random inputs', ctx, x, cond, 'cuda'))

if len(sys.argv) > 5:
    from mesa.data import TileDataset
    import random
    root, manifest = sys.argv[4:6]
    ds = TileDataset(root, manifest, split=None,
                     incidence_json=str(Path(root) / 'ctx_overlapping_95pct_polar.json'))
    random.seed(0)
    idx = np.random.default_rng(0).choice(len(ds), size=8, replace=False)
    batch = [ds[int(i)] for i in idx]
    ctx = torch.stack([b['ctx'] for b in batch])
    x = torch.stack([b['x'] for b in batch])
    cond = torch.stack([b['cond'] for b in batch])
    results.append(compare('8 test samples', ctx, x, cond, 'cpu'))
    if torch.cuda.is_available():
        results.append(compare('8 test samples', ctx, x, cond, 'cuda'))

worst = max(max(r) for r in results)
print(f'MODEL CHECK {"PASS" if worst == 0.0 else "DIFF"}: largest difference {worst:.3e}')
