from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from .vix import MINUTES_PER_YEAR, expiry_variance, interpolate_index

HOLIDAYS = USFederalHolidayCalendar().holidays("1990-01-01", "2060-12-31").values.astype("datetime64[D]")


def ng_option_expiry(contract_month: pd.Timestamp) -> pd.Timestamp:
    first = np.datetime64(pd.Timestamp(contract_month).replace(day=1).strftime("%Y-%m-%d"), "D")
    day = np.busday_offset(first, -4, roll="forward", holidays=HOLIDAYS)
    return pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=30)


def treasury_rate(cmt: pd.DataFrame, date: pd.Timestamp, T_years: float) -> float:
    day = cmt[cmt["Date"] <= date]
    day = day[day["Date"] == day["Date"].max()]
    months = day["maturity"].str.extract(r"(\d+)\s*(Mo|Yr)").astype({0: float})
    m = np.where(months[1] == "Yr", months[0] * 12, months[0])
    order = np.argsort(m)
    return float(np.interp(T_years * 12, m[order], day["value"].to_numpy()[order]) / 100)


def cme_chains(df: pd.DataFrame, trade_date: str, option_expiry: Callable = ng_option_expiry,
               asof_time: str = "14:30", strike_scale: float = 1.0) -> dict[pd.Timestamp, pd.DataFrame]:
    d = df[pd.to_datetime(df["tradeDate"]) == pd.Timestamp(trade_date)].copy()
    if d.empty:
        raise ValueError(f"no rows for tradeDate {trade_date}")
    d["strike"] = d["options-strikePrice"].astype(float) * strike_scale
    d["type"] = d["options-optiontype"].str[0].str.upper()
    d["price"] = d["options-priorSettle"].astype(float)
    out = {}
    for cm, g in d.groupby(pd.to_datetime(d["futures-expirationDate"])):
        out[cm] = g[["strike", "type", "price"]].reset_index(drop=True)
    return out


def check_units(chain: pd.DataFrame, T: float, r: float) -> None:
    px = chain.pivot_table(index="strike", columns="type", values="price", aggfunc="first").dropna()
    px = px[(px["C"] > 0) & (px["P"] > 0)]
    if len(px) < 3:
        return
    near = (px["C"] - px["P"]).abs().nsmallest(min(10, len(px))).index
    slope = np.polyfit(near.to_numpy(float), (px.loc[near, "C"] - px.loc[near, "P"]).to_numpy(), 1)[0]
    if not (0.5 < -slope / np.exp(-r * T) < 2.0):
        raise ValueError(f"put-call parity slope {slope:.4f} (expected ~{-np.exp(-r * T):.3f}): strikes and "
                         "settlement prices are probably in different units; set strike_scale")


def cme_index(df: pd.DataFrame, cmt: pd.DataFrame, trade_date: str, target_days: float = 30,
              option_expiry: Callable = ng_option_expiry, strike_scale: float = 1.0) -> dict:
    now = pd.Timestamp(f"{trade_date} 14:30")
    chains = cme_chains(df, trade_date, option_expiry, strike_scale=strike_scale)
    expiries = {}
    for cm, ch in chains.items():
        exp = option_expiry(cm)
        T = (exp - now).total_seconds() / 60 / MINUTES_PER_YEAR
        if T > 0:
            expiries[T] = (cm, ch)
    tgt = target_days / 365
    near = max([t for t in expiries if t <= tgt], default=None)
    nxt = min([t for t in expiries if t >= tgt], default=None)
    if near is None or nxt is None:
        raise ValueError(f"no pair of option expiries brackets {target_days} days "
                         f"(available: {sorted(round(t * 365, 1) for t in expiries)} days). "
                         "Extrapolating gives negative weights and can collapse the index.")
    res = {}
    for label, T in (("near", near), ("next", nxt)):
        r = treasury_rate(cmt, pd.Timestamp(trade_date), T)
        check_units(expiries[T][1], T, r)
        e = expiry_variance(expiries[T][1], T, r)
        res[label] = e
        res[f"{label}_contract"] = expiries[T][0]
    res["index"] = interpolate_index(res["near"], res["next"], target_days) if near != nxt else 100 * np.sqrt(res["near"].sigma2)
    return res
