"""Train MESA (Section 2.4 of the paper).

AdamW (lr 1e-4, weight decay 1e-5), batch 16, 100 epochs, cosine decay of the
learning rate to 1e-6 (stepped once per epoch), gradient norm clipped at 1,
single precision with TF32 matrix products. Each epoch is checkpointed; the
released model is the epoch with the lowest validation MAE in metres
(checkpoints/best.pt here).

    python -m mesa.train --root /data/mars_ctx_dem_v4_pedr \
        --manifest /data/mars_ctx_dem_v4_pedr/manifest_v4sT_ge10.csv \
        --incidence-json /data/mars_ctx_dem_v4_pedr/ctx_overlapping_95pct_polar.json \
        --out runs/mesa --seed 1

An interrupted run continues with --resume runs/mesa/checkpoints/epoch_NNN.pt.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import TileDataset
from .loss import MesaLoss
from .model import MESA


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_one_epoch(model, loader, optimizer, loss_fn, device, log_every=100):
    model.train()
    totals, n = {}, 0
    for i, b in enumerate(loader):
        b = to_device(b, device)
        optimizer.zero_grad()
        z, albedo = model(b['ctx'], b['x'], b['cond'])
        loss, parts = loss_fn(z, albedo, b['dem'], b['ctx'], b['cond'][:, :3],
                              b['dem_mean'].float(), b['dem_std'].float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        for k, v in dict(parts, loss=loss.item()).items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
        if i % log_every == 0:
            print(f'  [{i}/{len(loader)}] loss {loss.item():.4f}', flush=True)
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    """Mean loss over batches, and the MAE in metres (standardised L1 times the
    stereo DEM's standard deviation, averaged over samples)."""
    model.eval()
    loss_sum, n_batches, mae_sum, n = 0.0, 0, 0.0, 0
    for b in loader:
        b = to_device(b, device)
        z, albedo = model(b['ctx'], b['x'], b['cond'])
        loss, _ = loss_fn(z, albedo, b['dem'], b['ctx'], b['cond'][:, :3],
                          b['dem_mean'].float(), b['dem_std'].float())
        loss_sum += loss.item()
        n_batches += 1
        for i in range(z.shape[0]):
            mae_sum += F.l1_loss(z[i], b['dem'][i]).item() * float(b['dem_std'][i])
            n += 1
    return {'loss': loss_sum / max(n_batches, 1), 'mae_m': mae_sum / max(n, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, help='tile archive')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--incidence-json', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-5)
    ap.add_argument('--num-workers', type=int, default=16)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--resume', default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision('high')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out = Path(args.out)
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / 'config.json', 'w'), indent=2)

    common = dict(root=args.root, manifest=args.manifest, incidence_json=args.incidence_json)
    train_set = TileDataset(split='train', augment=True, **common)
    val_set = TileDataset(split='val', augment=False, **common)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = MESA().to(device)
    loss_fn = MesaLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                           eta_min=args.lr * 0.01)
    print(f'MESA: {sum(p.numel() for p in model.parameters()):,} parameters')

    start, history, best = 0, [], float('inf')
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start = state['epoch'] + 1
        if (out / 'history.json').exists():
            history = [h for h in json.load(open(out / 'history.json')) if h['epoch'] < start]
            best = min((h['val']['mae_m'] for h in history), default=best)

    for epoch in range(start, args.epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = validate(model, val_loader, loss_fn, device)
        scheduler.step()
        history.append({'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'],
                        'train': train_metrics, 'val': val_metrics})
        json.dump(history, open(out / 'history.json', 'w'), indent=1)
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'epoch': epoch}
        torch.save(state, out / 'checkpoints' / f'epoch_{epoch:03d}.pt')
        if val_metrics['mae_m'] < best:
            best = val_metrics['mae_m']
            torch.save(state, out / 'checkpoints' / 'best.pt')
        print(f"epoch {epoch}: train loss {train_metrics['loss']:.4f}, val loss {val_metrics['loss']:.4f}, "
              f"val MAE {val_metrics['mae_m']:.3f} m ({time.time() - t0:.0f} s)", flush=True)


if __name__ == '__main__':
    main()
