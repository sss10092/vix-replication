from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def black(F, K, T, sig, r, is_call):
    d1 = (np.log(F / K) + 0.5 * sig**2 * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    df = np.exp(-r * T)
    return np.where(is_call, df * (F * norm.cdf(d1) - K * norm.cdf(d2)), df * (K * norm.cdf(-d2) - F * norm.cdf(-d1)))


def skewed_smile(atm: float, skew: float = -0.35, curv: float = 0.6):
    def f(K, F, T):
        x = np.log(np.asarray(K) / F) / np.sqrt(max(T, 1e-6))
        return np.maximum(atm + skew * atm * x + curv * atm * x**2 / (1 + np.abs(x)), 0.05)
    return f


def true_variance(smile, F, T, r, n: int = 400_001) -> float:
    x = np.linspace(np.log(1e-4), np.log(30.0), n)
    K = F * np.exp(x)
    q = black(F, K, T, smile(K, F, T), r, K >= F)
    integrand = q * np.exp(-x) / F
    return float(2 * np.exp(r * T) / T * np.trapezoid(integrand, x))


def chain(F, T, r, smile, spacing, k_lo=0.3, k_hi=2.0, tick=0.005, min_price=None) -> pd.DataFrame:
    K = np.arange(np.ceil(k_lo * F / spacing) * spacing, k_hi * F + spacing / 2, spacing)
    rows = []
    for typ in ("C", "P"):
        px = black(F, K, T, smile(K, F, T), r, typ == "C")
        mid = np.round(px / tick) * tick
        floor = tick if min_price is None else min_price
        bid = np.where(mid >= floor, mid - tick / 2, 0.0)
        ask = np.where(mid >= floor, mid + tick / 2, tick)
        rows.append(pd.DataFrame({"strike": K, "type": typ, "bid": np.maximum(bid, 0), "ask": ask}))
    return pd.concat(rows, ignore_index=True)


def cme_sample(trade_date: str = "2020-11-12", F: float = 2.9, atm: float = 0.55, n_contracts: int = 4,
               r_pct: float = 0.09) -> tuple[pd.DataFrame, pd.DataFrame]:
    from .cme import ng_option_expiry

    now = pd.Timestamp(f"{trade_date} 14:30")
    smile = skewed_smile(atm, skew=0.15, curv=0.3)
    first = pd.Timestamp(trade_date).replace(day=1)
    rows = []
    for i in range(1, n_contracts + 1):
        cm = first + pd.DateOffset(months=i)
        T = (ng_option_expiry(cm) - now).total_seconds() / 60 / 525_600
        K = np.round(np.arange(0.5, 8.0, 0.05), 2)
        vol = smile(K, F, T)
        for typ in ("call", "put"):
            px = np.round(black(F, K, T, vol, r_pct / 100, typ == "call"), 3)
            rows.append(pd.DataFrame({"tradeDate": trade_date, "futures-expirationDate": cm.strftime("%Y-%m-%d"),
                                      "options-strikePrice": K, "options-optiontype": typ, "options-priorSettle": px}))
    opts = pd.concat(rows, ignore_index=True)
    opts = opts[opts["options-priorSettle"] > 0]
    opts.attrs["smile"], opts.attrs["F"], opts.attrs["r"] = smile, F, r_pct / 100
    tenors = ["1 Mo", "2 Mo", "3 Mo", "6 Mo", "1 Yr", "2 Yr"]
    cmt = pd.DataFrame([{"Date": trade_date, **{t: r_pct for t in tenors}}])
    return opts, cmt
