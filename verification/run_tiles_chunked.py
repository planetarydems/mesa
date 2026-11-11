"""`mesa.evaluate tiles` split over several CPU processes (same rows, same
order, same output as one process).

    python verification/run_tiles_chunked.py MODEL ROOT MANIFESTS PAIRS OUT_CSV [CHUNKS] [THREADS] [FOOTPRINT_M]

MANIFESTS is comma-separated, as for `mesa.evaluate tiles`. Each of CHUNKS
processes (default 12) scores every CHUNKS-th row with THREADS PyTorch threads
(default 4); the rows are then written in the order of the manifests.
"""
import os
import sys
from multiprocessing import get_context
from pathlib import Path

model_path, root, manifests, pairs_file, out_csv = sys.argv[1:6]
chunks = int(sys.argv[6]) if len(sys.argv) > 6 else 12
threads = int(sys.argv[7]) if len(sys.argv) > 7 else 4
footprint = float(sys.argv[8]) if len(sys.argv) > 8 else None


def work(k):
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[v] = str(threads)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    torch.set_num_threads(threads)
    from mesa import evaluate as ev
    from mesa.anchor import FOOTPRINT_M
    from mesa.data import load_incidence
    from mesa.model import load_model
    model = load_model(model_path, 'cpu')
    incidence = load_incidence(str(Path(root) / 'ctx_overlapping_95pct_polar.json'))
    rows = ev.read_rows(manifests.split(','), set(open(pairs_file).read().split()))
    out = []
    for i in range(k, len(rows), chunks):
        r = ev.score_sample(model, rows[i], root, incidence, 'cpu', footprint or FOOTPRINT_M)
        out.append((i, r))
    return out


if __name__ == '__main__':
    with get_context('spawn').Pool(chunks) as pool:
        parts = pool.map(work, range(chunks))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mesa import evaluate as ev
    rows = [r for _, r in sorted((x for p in parts for x in p), key=lambda t: t[0]) if r is not None]
    ev.write_csv(out_csv, rows)
    print(f'{len(rows)} samples scored -> {out_csv}')
