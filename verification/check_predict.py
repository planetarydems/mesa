"""Check that the released prediction reproduces the production frames.

    python verification/check_predict.py MODEL MOLA PEDR OUT_DIR PRODUCT_TAG BOXES_ROOT \
        BOX STEM IMAGE INC AZ [BOX STEM IMAGE INC AZ ...]

For each frame the released code predicts from the same registered image and
the result is compared, file by file, with the production frame
BOXES_ROOT/<BOX>/box_run/preds_<PRODUCT_TAG>/<STEM>/. Run it on the kind of
node the frame was predicted on (GPU or CPU, same thread count), since
floating-point results depend on the hardware.
"""
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mader.geo import MolaSampler  # noqa: E402
from mader.model import load_model  # noqa: E402
from mader.pedr import PedrDatabase  # noqa: E402
from mader.predict import predict_image  # noqa: E402

model_path, mola_path, pedr_path, out_dir, tag, boxes = sys.argv[1:7]
frames = [sys.argv[i:i + 5] for i in range(7, len(sys.argv), 5)]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device {device}, torch threads {torch.get_num_threads()}')
model = load_model(model_path, device)
mola, pedr = MolaSampler(mola_path), PedrDatabase(pedr_path)

worst = 0.0
for box, stem, image, inc, az in frames:
    t0 = time.time()
    out = Path(out_dir) / box / stem
    predict_image(image, float(inc), float(az), model, mola, pedr, str(out), device)
    ref = Path(boxes) / box / 'box_run' / f'preds_{tag}' / stem
    for name in ('pred_shot_anchor.tif', 'pred_feather.tif', 'ctx_on_grid.tif', 'mola_on_grid.tif'):
        with rasterio.open(out / name) as a, rasterio.open(ref / name) as b:
            grid = a.transform == b.transform and a.shape == b.shape and a.crs == b.crs
            x, y = a.read(1).astype(np.float64), b.read(1).astype(np.float64)
        nd_same = np.array_equal(x == -9999.0, y == -9999.0)
        m = (x != -9999.0) & (y != -9999.0)
        d = np.abs(x[m] - y[m]).max() if m.any() else 0.0
        if name.startswith('pred'):
            worst = max(worst, d)
        print(f'{box} {stem} {name}: same grid {grid}, same no-data mask {nd_same}, '
              f'max |released - production| {d:.3e} m, bitwise {np.array_equal(x, y)}')
    print(f'  ({time.time() - t0:.0f} s)', flush=True)
print(f'PREDICT CHECK {"PASS" if worst < 1e-3 else "DIFF"}: largest difference {worst:.3e} m')
