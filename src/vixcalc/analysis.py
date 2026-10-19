from __future__ import annotations

import numpy as np
import pandas as pd

from .synthetic import chain, skewed_smile, true_variance
from .vix import expiry_variance


def error_budget(F=5000.0, T=30 / 365, r=0.045, atm=0.18) -> pd.DataFrame:
    smile = skewed_smile(atm)
    truth = 100 * np.sqrt(true_variance(smile, F, T, r))
    rows = []
    for spacing in (5, 25, 50, 100):
        for k_lo in (0.3, 0.6, 0.8):
            e = expiry_variance(chain(F, T, r, smile, spacing, k_lo=k_lo, tick=0.05), T, r)
            rows.append({"strike_spacing": spacing, "lowest_put_K/F": k_lo, "n_strikes": e.n_strikes,
                         "index": 100 * np.sqrt(e.sigma2), "error_volpts": 100 * np.sqrt(e.sigma2) - truth})
    return pd.DataFrame(rows).round(4)


def notebook_k0_omission(F=3.0, T=60 / 365, r=0.001, atm=0.55, spacing=0.05) -> dict:
    smile = skewed_smile(atm, skew=0.1, curv=0.4)
    ch = chain(F, T, r, smile, spacing, k_lo=0.2, k_hi=3.0, tick=0.001)
    e = expiry_variance(ch, T, r)
    sel = e.strikes_used
    no_k0 = sel[sel["kind"] != "put/call avg"]
    K = no_k0["strike"].to_numpy()
    dK = np.empty_like(K)
    dK[1:-1] = (K[2:] - K[:-2]) / 2
    dK[0], dK[-1] = K[1] - K[0], K[-1] - K[-2]
    s2 = 2 / T * np.sum(dK / K**2 * np.exp(r * T) * no_k0["_px"].to_numpy()) - (e.F / e.K0 - 1) ** 2 / T
    truth = 100 * np.sqrt(true_variance(smile, F, T, r))
    return {"true": truth, "cboe_method": 100 * np.sqrt(e.sigma2), "notebook_without_K0": 100 * np.sqrt(s2)}
