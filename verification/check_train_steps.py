"""Check that the released training loop reproduces the original training
loop step by step.

    python verification/check_train_steps.py ORIG_CODE DATA_ROOT MANIFEST [STEPS] [BATCH]

Both trainers start from the same weights (the original network's random
initialisation, copied into the released network), see the same batches
(identical random states for shuffling, cropping, flips, jitter, the MOLA cut
and dropout) and take STEPS optimiser steps on the CPU. The script compares
the loss of every step and every weight at the end.
"""
import itertools
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mesa.data import TileDataset  # noqa: E402
from mesa.loss import MesaLoss  # noqa: E402
from mesa.model import MESA  # noqa: E402
from mesa.train import train_one_epoch  # noqa: E402

orig_code, root, manifest = sys.argv[1:4]
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 4
batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 2
sys.path.insert(0, orig_code)
from joint_shape_albedo import FiLMUNetDualHead, JointShapeAlbedoLoss  # noqa: E402
from mars_ctx_dataset import MarsCTXDEMDataset  # noqa: E402
import train_mars_ctx  # noqa: E402

torch.use_deterministic_algorithms(True)
incidence = str(Path(root) / 'ctx_overlapping_95pct_polar.json')
_default_rng = np.random.default_rng


class Limited:
    """The first n batches of a loader."""
    def __init__(self, loader, n):
        self.loader, self.n = loader, n

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.n)

    def __len__(self):
        return self.n


class Recorder(torch.nn.Module):
    """Wraps a loss and records the value of every call."""
    def __init__(self, loss):
        super().__init__()
        self.loss, self.values = loss, []

    def forward(self, *a, **kw):
        out = self.loss(*a, **kw)
        self.values.append(out[0].item())
        return out


class FabricShim:
    device = torch.device('cpu')
    is_global_zero = True

    @staticmethod
    def backward(loss):
        loss.backward()


def reset_random_state():
    random.seed(7)
    np.random.seed(7)
    counter = itertools.count()
    np.random.default_rng = lambda *a, **kw: _default_rng(20_000 + next(counter))
    torch.manual_seed(7)


torch.manual_seed(0)
orig = FiLMUNetDualHead(in_channels=3, out_channels=1, depths=[64, 128, 256, 512], dropout=0.1,
                        use_attention=True, skip_mode='implicit', cg_iters=25, cg_mu=0.05,
                        relief_film=True, relief_solve=True)
clean = MESA()
clean.load_state_dict({k: v for k, v in orig.state_dict().items() if k != 'mola_weight'}, strict=True)

# original trainer
orig_ds = MarsCTXDEMDataset(
    dataset_root=root, incidence_json=incidence, manifest_path=manifest, crop_size=384, augment=True,
    split='train', shot_dir=str(Path(root) / 'shot'), input_mode='mola_conf', cutout_mode='trackaware',
    cutout_p=0.3, cutout_dilate_px=6, dist_dir=str(Path(root) / 'd1_shot'), conf_d0=2000.0, relief_cond=True)
orig_loss = Recorder(JointShapeAlbedoLoss(dem_weight=1.0, grad_weight=0.5, lap_weight=0.0, photo_weight=0.3,
                                          albedo_tv_weight=0.01, pixel_size=18.0, shot_weight=0.0,
                                          hp_weight=0.0, hp_sigma=2.0, footprint_radius_px=4))
orig_opt = torch.optim.AdamW(orig.parameters(), lr=1e-4, weight_decay=1e-5)
reset_random_state()
orig_loader = torch.utils.data.DataLoader(orig_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                                          num_workers=0, collate_fn=MarsCTXDEMDataset.collate_fn)
train_mars_ctx.train_epoch(FabricShim(), orig, Limited(orig_loader, steps), orig_opt, orig_loss, epoch=0)

# released trainer
clean_ds = TileDataset(root, manifest, 'train', incidence_json=incidence, augment=True)
clean_loss = Recorder(MesaLoss())
clean_opt = torch.optim.AdamW(clean.parameters(), lr=1e-4, weight_decay=1e-5)
reset_random_state()
clean_loader = torch.utils.data.DataLoader(clean_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                                           num_workers=0)
train_one_epoch(clean, Limited(clean_loader, steps), clean_opt, clean_loss, 'cpu')
np.random.default_rng = _default_rng

print('step   original loss        released loss        difference')
for k, (a, b) in enumerate(zip(orig_loss.values, clean_loss.values)):
    print(f'{k:4d}   {a:.12f}   {b:.12f}   {a - b:+.3e}')
worst = 0.0
osd = orig.state_dict()
for name, v in clean.state_dict().items():
    worst = max(worst, (osd[name].float() - v.float()).abs().max().item())
print(f'largest weight difference after {steps} steps: {worst:.3e}')
same_loss = orig_loss.values == clean_loss.values and len(clean_loss.values) == steps
print(f'TRAIN CHECK {"PASS" if same_loss and worst == 0.0 else "DIFF"}')
