import numpy as np
import pandas as pd
import pytest

from vixcalc.cme import cme_index, ng_option_expiry, treasury_rate
from vixcalc.synthetic import black, chain, skewed_smile, true_variance
from vixcalc.vix import expiry_variance, forward_and_k0, interpolate_index, vix_from_chains


def flat(v):
    return lambda K, F, T: np.full(np.shape(K), v)


@pytest.mark.parametrize("vol", [0.12, 0.25, 0.60])
def test_flat_smile_returns_its_volatility(vol):
    F, T, r = 5000.0, 30 / 365, 0.04
    e = expiry_variance(chain(F, T, r, flat(vol), 5, k_lo=0.05, k_hi=4.0, tick=0.01), T, r)
    assert abs(np.sqrt(e.sigma2) - vol) < 5e-4


def test_skewed_smile_matches_continuous_variance():
    F, r, sm = 5000.0, 0.045, skewed_smile(0.18)
    T = 30 / 365
    e = expiry_variance(chain(F, T, r, sm, 5, tick=0.05), T, r)
    assert abs(100 * np.sqrt(e.sigma2) - 100 * np.sqrt(true_variance(sm, F, T, r))) < 0.02


def test_forward_from_parity_and_k0_below_forward():
    F, T, r = 4987.3, 30 / 365, 0.05
    ch = chain(F, T, r, flat(0.2), 5, tick=0.01)
    F_hat, K0 = forward_and_k0(ch, T, r)
    assert abs(F_hat - F) < 0.05
    assert K0 <= F_hat < K0 + 5


def test_two_consecutive_zero_bids_stop_the_walk():
    F, T, r = 100.0, 30 / 365, 0.0
    ch = chain(F, T, r, flat(0.2), 1, k_lo=0.5, k_hi=1.5, tick=0.01)
    far = (ch["type"] == "P") & ch["strike"].isin([80.0, 79.0])
    ch.loc[far, "bid"] = 0.0
    used = expiry_variance(ch, T, r).strikes_used
    assert used["strike"].min() > 80.0


def test_interpolation_reproduces_a_flat_term_structure():
    F, r = 5000.0, 0.03
    chains = {t: chain(F, t, r, flat(0.2), 5, k_lo=0.05, k_hi=4.0, tick=0.01) for t in (20 / 365, 45 / 365)}
    assert abs(vix_from_chains(chains, {t: r for t in chains})["index"] - 20.0) < 0.05


def test_interpolation_refuses_to_extrapolate():
    F, r = 5000.0, 0.03
    e1 = expiry_variance(chain(F, 40 / 365, r, flat(0.2), 5, tick=0.01), 40 / 365, r)
    e2 = expiry_variance(chain(F, 70 / 365, r, flat(0.2), 5, tick=0.01), 70 / 365, r)
    with pytest.raises(ValueError):
        interpolate_index(e1, e2, target_days=90)


def _cme_frame(trade_date, contracts, F=3.0, vol=0.55, scale=1.0):
    rows = []
    now = pd.Timestamp(f"{trade_date} 14:30")
    for cm in contracts:
        T = (ng_option_expiry(pd.Timestamp(cm)) - now).total_seconds() / 60 / 525_600
        K = np.round(np.arange(0.5, 8.0, 0.05), 2)
        for typ in ("call", "put"):
            px = np.round(black(F, K, T, vol, 0.001, typ == "call"), 3)
            rows.append(pd.DataFrame({"tradeDate": trade_date, "futures-expirationDate": cm,
                                      "options-strikePrice": K / scale, "options-optiontype": typ,
                                      "options-priorSettle": px}))
    return pd.concat(rows, ignore_index=True)


def test_cme_adapter_recovers_flat_vol_and_detects_unit_mismatch():
    cmt = pd.DataFrame({"Date": pd.Timestamp("2020-11-12"), "maturity": ["1 Mo", "3 Mo", "1 Yr"], "value": [0.1, 0.1, 0.1]})
    df = _cme_frame("2020-11-12", ["2020-12-01", "2021-01-01", "2021-02-01"])
    assert abs(cme_index(df, cmt, "2020-11-12", target_days=30)["index"] - 55.0) < 0.3
    bad = _cme_frame("2020-11-12", ["2020-12-01", "2021-01-01", "2021-02-01"], scale=0.01)
    with pytest.raises(ValueError, match="units"):
        cme_index(bad, cmt, "2020-11-12", target_days=30)
    ok = cme_index(bad, cmt, "2020-11-12", target_days=30, strike_scale=0.01)
    assert abs(ok["index"] - 55.0) < 0.3


def test_treasury_interpolation():
    cmt = pd.DataFrame({"Date": pd.Timestamp("2020-11-12"), "maturity": ["1 Mo", "2 Mo", "1 Yr"], "value": [1.0, 2.0, 3.0]})
    assert abs(treasury_rate(cmt, pd.Timestamp("2020-11-12"), 1.5 / 12) - 0.015) < 1e-12


def test_sample_files_round_trip(tmp_path, capsys):
    from vixcalc.cli import main

    main(["sample", "--out-dir", str(tmp_path)])
    truth = float(capsys.readouterr().out.split("chain: ")[1].split()[0])
    main(["cme", "--options", str(tmp_path / "ng_options_synthetic.csv"), "--cmt", str(tmp_path / "treasury_cmt_synthetic.csv"),
          "--date", "2020-11-12", "--days", "30"])
    index = float(capsys.readouterr().out.strip().split()[-1])
    assert abs(index - truth) < 0.5
    with pytest.raises(SystemExit, match="not found"):
        main(["cme", "--options", str(tmp_path / "missing.csv"), "--cmt", "x.csv", "--date", "2020-11-12"])


def test_option_expiry_is_four_business_days_before_month():
    assert ng_option_expiry(pd.Timestamp("2020-12-01")) == pd.Timestamp("2020-11-24 14:30")
    assert ng_option_expiry(pd.Timestamp("2021-01-01")) == pd.Timestamp("2020-12-28 14:30")
