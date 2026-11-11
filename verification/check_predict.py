"""Compare the released prediction with stored production frames.

    [FOOTPRINT_M=150] [RELEASE1_RULES=1] python verification/check_predict.py MODEL MOLA PEDR OUT_DIR \
        PRODUCT_TAG BOXES_ROOT BOX STEM IMAGE INC AZ [BOX STEM IMAGE INC AZ ...]

FOOTPRINT_M sets the anchoring footprint (default anchor.FOOTPRINT_M, 168 m;
the frames of the v3 product and of release 1 were anchored with 150 m).
The v3 rules of mesa.predict (PEDR over all longitudes near the poles, the
minimum shot-distance mean) are applied unless RELEASE1_RULES=1.

For each frame the released code predicts from the same registered image and
the result is compared, file by file, with the production frame
BOXES_ROOT/<BOX>/box_run/preds_<PRODUCT_TAG>/<STEM>/. Run it on the kind of
node the frame was predicted on (GPU or CPU, same thread count), since
floating-point results depend on the hardware. Older products may use another
anchoring footprint; differences remain visible rather than changing the
comparison tolerance. See verification/README.md for the historical checks.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mesa.geo import MolaSampler  # noqa: E402
from mesa.model import load_model  # noqa: E402
from mesa.pedr import PedrDatabase  # noqa: E402
from mesa.predict import predict_image  # noqa: E402
from mesa.anchor import FOOTPRINT_M  # noqa: E402
from mesa.predict import MIN_CONF_MEAN  # noqa: E402

footprint = float(os.environ.get('FOOTPRINT_M', FOOTPRINT_M))
release1 = os.environ.get('RELEASE1_RULES', '0') == '1'

model_path, mola_path, pedr_path, out_dir, tag, boxes = sys.argv[1:7]
frames = [sys.argv[i:i + 5] for i in range(7, len(sys.argv), 5)]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device {device}, torch threads {torch.get_num_threads()}, footprint {footprint} m, '
      f'{"release-1" if release1 else "v3"} rules')
model = load_model(model_path, device)
print(f'model configuration {model.config}')
mola, pedr = MolaSampler(mola_path), PedrDatabase(pedr_path, full_lon_near_poles=not release1)

worst = 0.0
for box, stem, image, inc, az in frames:
    t0 = time.time()
    out = Path(out_dir) / box / stem
    predict_image(image, float(inc), float(az), model, mola, pedr, str(out), device, footprint,
                  0.0 if release1 else MIN_CONF_MEAN)
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
