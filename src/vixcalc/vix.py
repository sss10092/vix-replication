from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MINUTES_PER_YEAR = 525_600


@dataclass
class ExpiryVariance:
    T: float
    minutes: float
    F: float
    K0: float
    sigma2: float
    n_strikes: int
    strikes_used: pd.DataFrame


def _quote_price(df: pd.DataFrame) -> pd.Series:
    if {"bid", "ask"} <= set(df.columns):
        return 0.5 * (df["bid"] + df["ask"])
    return df["price"]


def _bid(df: pd.DataFrame) -> pd.Series:
    return df["bid"] if "bid" in df.columns else df["price"]


def forward_and_k0(chain: pd.DataFrame, T: float, r: float) -> tuple[float, float]:
    px = chain.assign(px=_quote_price(chain)).pivot_table(index="strike", columns="type", values="px", aggfunc="first")
    px = px.dropna()
    px = px[(px["C"] > 0) & (px["P"] > 0)]
    if px.empty:
        raise ValueError("no strike has both a call and a put quote")
    k_star = (px["C"] - px["P"]).abs().idxmin()
    F = k_star + np.exp(r * T) * (px.loc[k_star, "C"] - px.loc[k_star, "P"])
    strikes = np.sort(chain["strike"].unique())
    below = strikes[strikes <= F]
    if below.size == 0:
        raise ValueError("no strike at or below the forward")
    return float(F), float(below[-1])


def _walk(side: pd.DataFrame) -> pd.DataFrame:
    keep, zeros = [], 0
    for idx, row in side.iterrows():
        if row["_bid"] <= 0:
            zeros += 1
            if zeros >= 2:
                break
            continue
        zeros = 0
        keep.append(idx)
    return side.loc[keep]


def expiry_variance(chain: pd.DataFrame, T: float, r: float) -> ExpiryVariance:
    F, K0 = forward_and_k0(chain, T, r)
    c = chain.assign(_px=_quote_price(chain), _bid=_bid(chain))
    puts = c[(c["type"] == "P") & (c["strike"] < K0)].sort_values("strike", ascending=False)
    calls = c[(c["type"] == "C") & (c["strike"] > K0)].sort_values("strike")
    at_k0 = c[c["strike"] == K0]
    q_k0 = at_k0["_px"].mean()
    sel = pd.concat([
        _walk(puts)[["strike", "_px"]].assign(kind="put"),
        pd.DataFrame({"strike": [K0], "_px": [q_k0], "kind": ["put/call avg"]}),
        _walk(calls)[["strike", "_px"]].assign(kind="call"),
    ]).sort_values("strike").reset_index(drop=True)

    K = sel["strike"].to_numpy(float)
    dK = np.empty_like(K)
    dK[1:-1] = (K[2:] - K[:-2]) / 2
    dK[0] = K[1] - K[0]
    dK[-1] = K[-1] - K[-2]
    contrib = dK / K**2 * np.exp(r * T) * sel["_px"].to_numpy(float)
    sel["contribution"] = contrib
    sigma2 = 2 / T * contrib.sum() - (F / K0 - 1) ** 2 / T
    return ExpiryVariance(T, T * MINUTES_PER_YEAR, F, K0, float(sigma2), len(sel), sel)


def interpolate_index(near: ExpiryVariance, nxt: ExpiryVariance, target_days: float = 30) -> float:
    n_target = target_days * 1440
    n1, n2 = near.minutes, nxt.minutes
    if not (n1 <= n_target <= n2):
        raise ValueError(f"expiries ({n1 / 1440:.1f}d, {n2 / 1440:.1f}d) do not bracket {target_days}d")
    w1 = (n2 - n_target) / (n2 - n1)
    w2 = (n_target - n1) / (n2 - n1)
    var = (near.T * near.sigma2 * w1 + nxt.T * nxt.sigma2 * w2) * MINUTES_PER_YEAR / n_target
    return 100 * float(np.sqrt(var))


def vix_from_chains(chains: dict[float, pd.DataFrame], rates: dict[float, float], target_days: float = 30) -> dict:
    Ts = sorted(chains)
    tgt = target_days / 365
    near = max([t for t in Ts if t <= tgt], default=None)
    nxt = min([t for t in Ts if t >= tgt], default=None)
    if near is None or nxt is None:
        raise ValueError("need one expiry on each side of the target tenor")
    if near == nxt:
        e = expiry_variance(chains[near], near, rates[near])
        return {"index": 100 * np.sqrt(e.sigma2), "near": e, "next": e}
    e1 = expiry_variance(chains[near], near, rates[near])
    e2 = expiry_variance(chains[nxt], nxt, rates[nxt])
    return {"index": interpolate_index(e1, e2, target_days), "near": e1, "next": e2}
