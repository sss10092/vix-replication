from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import error_budget, notebook_k0_omission
from .synthetic import chain, skewed_smile, true_variance
from .vix import vix_from_chains


def demo():
    F, r = 5000.0, 0.045
    smile = skewed_smile(0.18)
    T1, T2 = 23 / 365, 37 / 365
    out = vix_from_chains({T1: chain(F, T1, r, smile, 5, tick=0.05), T2: chain(F, T2, r, smile, 5, tick=0.05)}, {T1: r, T2: r})
    v1, v2 = true_variance(smile, F, T1, r), true_variance(smile, F, T2, r)
    n30, n1, n2 = 30 * 1440, T1 * 525600, T2 * 525600
    truth = 100 * np.sqrt((T1 * v1 * (n2 - n30) / (n2 - n1) + T2 * v2 * (n30 - n1) / (n2 - n1)) * 525600 / n30)
    for k in ("near", "next"):
        e = out[k]
        print(f"{k:5s} T={e.T * 365:5.1f}d  F={e.F:8.2f}  K0={e.K0:7.1f}  strikes used={e.n_strikes:4d}  "
              f"sigma={100 * np.sqrt(e.sigma2):6.3f}")
    print(f"\n30-day index: {out['index']:.4f}   exact model-free value: {truth:.4f}   "
          f"error: {out['index'] - truth:+.4f} vol pts")
    k0 = notebook_k0_omission()
    print(f"\nNat-gas-like chain (55% ATM): exact {k0['true']:.2f}, CBOE method {k0['cboe_method']:.2f}, "
          f"original notebook without K0 {k0['notebook_without_K0']:.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vixcalc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo")
    sub.add_parser("budget")
    sp = sub.add_parser("sample")
    sp.add_argument("--out-dir", default="data/sample")
    c = sub.add_parser("cme")
    c.add_argument("--options", required=True)
    c.add_argument("--cmt", required=True)
    c.add_argument("--date", required=True)
    c.add_argument("--days", type=float, default=30)
    c.add_argument("--strike-scale", type=float, default=1.0)
    a = ap.parse_args(argv)
    if a.cmd == "demo":
        demo()
    elif a.cmd == "budget":
        print(error_budget().to_string(index=False))
    elif a.cmd == "sample":
        from .synthetic import cme_sample
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        opts, cmt = cme_sample()
        opts.to_csv(out / "ng_options_synthetic.csv", index=False)
        cmt.to_csv(out / "treasury_cmt_synthetic.csv", index=False)
        from .cme import cme_index
        res = cme_index(opts, cmt.melt(id_vars="Date", var_name="maturity", value_name="value")
                        .assign(Date=lambda x: pd.to_datetime(x["Date"])), "2020-11-12", 30)
        T1, T2 = res["near"].T, res["next"].T
        v1 = true_variance(opts.attrs["smile"], opts.attrs["F"], T1, opts.attrs["r"])
        v2 = true_variance(opts.attrs["smile"], opts.attrs["F"], T2, opts.attrs["r"])
        n30, n1, n2 = 30 * 1440, T1 * 525600, T2 * 525600
        truth = 100 * np.sqrt((T1 * v1 * (n2 - n30) / (n2 - n1) + T2 * v2 * (n30 - n1) / (n2 - n1)) * 525600 / n30)
        print(f"SYNTHETIC CME-format files, trade date 2020-11-12 -> {out}/")
        print(f"exact model-free 30-day value for this chain: {truth:.2f}  (vixcalc cme should come close)")
    else:
        from .cme import cme_index
        for path in (a.options, a.cmt):
            if not Path(path).exists():
                raise SystemExit(f"{path} not found. Put your CME settlement and Treasury CSVs in this folder, "
                                 "or run: vixcalc sample   and use the files in data/sample/")
        opts = pd.read_csv(a.options)
        cmt = pd.read_csv(a.cmt, parse_dates=["Date"])
        if "maturity" not in cmt.columns:
            cmt = cmt.melt(id_vars="Date", var_name="maturity", value_name="value")
        res = cme_index(opts, cmt, a.date, a.days, strike_scale=a.strike_scale)
        for k in ("near", "next"):
            e = res[k]
            print(f"{k}: contract {res[k + '_contract']:%Y-%m}  T={e.T * 365:.1f}d  F={e.F:.4f}  K0={e.K0}  "
                  f"strikes={e.n_strikes}  sigma={100 * np.sqrt(e.sigma2):.2f}")
        print(f"{a.days:.0f}-day index: {res['index']:.2f}")


if __name__ == "__main__":
    main()
