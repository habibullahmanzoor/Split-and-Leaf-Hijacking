"""Repeat the two VFL tasks that were running when a Paillier OverflowError was seen,
to see whether the overflow reproduces (it depends on the random key)."""
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_vfl_converged as E


def one(spec):
    kind, args = spec[0], spec[1:]
    try:
        r = dict(tgt=E.task_target)[kind](*args)
        return spec, "ok", r
    except OverflowError as e:
        return spec, "OVERFLOW", str(e)


if __name__ == "__main__":
    specs = [("tgt", "credit_default", "variance", 0)] * 4 + [("tgt", "adult", "fixed", 4)] * 4
    with ProcessPoolExecutor(max_workers=8) as ex:
        for spec, status, r in ex.map(one, specs):
            print(spec, status, r, flush=True)
