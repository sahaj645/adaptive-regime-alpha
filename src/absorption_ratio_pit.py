"""
absorption_ratio_pit.py -- institutional Absorption Ratio (Phase 4).
Kritzman Absorption Ratio on point-in-time GICS portfolio returns; method held
identical to the ETF version so only the cross-sectional universe changes.
Method: trailing equal-weighted covariance (252d default); eigvalsh PCA;
n=round(N/5) components; AR=sum(top-n)/sum(all); standardized shift
dAR=(MA15-MA_sw)/STD_sw with shift_window sw=252 (1y) decoupled from cov window.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import core

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT  = ROOT / "outputs"; OUT.mkdir(exist_ok=True, parents=True)


def absorption_ratio(rets, window=252, n_eig=None, ew_halflife=None, shift_window=252):
    """AR + standardized shift from a (dates x assets) return matrix. Causal.
    shift_window (default 252 = 1y) is decoupled from the covariance window."""
    rets = rets.dropna(how="all")
    N = rets.shape[1]
    n_eig = n_eig or max(1, round(N / 5))
    vals = rets.values
    if ew_halflife:
        w = 0.5 ** (np.arange(window)[::-1] / ew_halflife); w = w / w.sum()
    ar = np.full(len(rets), np.nan)
    for i in range(window, len(rets) + 1):
        W = np.nan_to_num(vals[i - window:i])
        if ew_halflife:
            Wm = W - (w[:, None] * W).sum(0)
            cov = (Wm * w[:, None]).T @ Wm
        else:
            cov = np.cov(W, rowvar=False)
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        ar[i - 1] = eig[:n_eig].sum() / eig.sum()
    ar = pd.Series(ar, index=rets.index, name="AR").dropna()
    shift = ((ar.rolling(15).mean() - ar.rolling(shift_window).mean()) / ar.rolling(shift_window).std()).rename("dAR")
    return pd.concat([ar, shift], axis=1).dropna()


def _fwd_diag(dar, spy_ret, thr=1.0, h=20):
    fwd = spy_ret.rolling(h).sum().shift(-h)
    idx = dar.dropna().index.intersection(fwd.dropna().index)
    hi = dar.loc[idx] > thr
    return fwd.loc[idx][hi].mean() * 100, fwd.loc[idx][~hi].mean() * 100, int(hi.sum())


if __name__ == "__main__":
    cap = pd.read_csv(PROC / "ig_returns_capwt.csv", index_col=0, parse_dates=True)
    eqw = pd.read_csv(PROC / "ig_returns_eqwt.csv", index_col=0, parse_dates=True)
    ar_cap = absorption_ratio(cap, 252, 5)
    ar_eq  = absorption_ratio(eqw, 252, 5)
    ar_cap.to_csv(PROC / "ar_pit_capwt.csv")
    ar_eq.to_csv(PROC / "ar_pit_eqwt.csv")
    _, spy, sec = core.load()
    etf = core.ar_features(sec)
    spy_ret = spy.pct_change()
    c = ar_cap.index.intersection(etf.index)
    print("AR level: PIT-cap mean=%.3f ETF mean=%.3f corr(level)=%.2f corr(shift)=%.2f" % (
        ar_cap.AR.mean(), etf.AR.mean(), ar_cap.AR.loc[c].corr(etf.AR.loc[c]), ar_cap.dAR.loc[c].corr(etf.dAR.loc[c])))
    print("\n[analysis only] fwd-20d SPY after dAR>+1sigma vs otherwise:")
    for nm, d in [("PIT-cap", ar_cap.dAR), ("PIT-eq", ar_eq.dAR), ("ETF", etf.dAR)]:
        hi, lo, n = _fwd_diag(d, spy_ret)
        print("   %-8s hi=%+.2f%% lo=%+.2f%% (hi days=%d)" % (nm, hi, lo, n))
    crises = {"Q4-2018": ("2018-10-01", "2018-12-24"), "COVID": ("2020-02-19", "2020-03-23"), "2022 bear": ("2022-01-03", "2022-10-12")}
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(spy.index, spy.values, color="black", lw=1); ax[0].set_yscale("log"); ax[0].set_title("SPY total return (log)")
    ax[1].plot(ar_cap.index, ar_cap.dAR, color="tab:purple", lw=1.0, label="PIT industry-group dAR (N=25, n=5)")
    ax[1].plot(etf.index, etf.dAR, color="tab:gray", lw=0.9, alpha=0.8, label="ETF sector dAR (N=10, n=2)")
    ax[1].axhline(1.0, color="red", ls="--", lw=0.8, alpha=0.5)
    ax[1].set_title("Standardized absorption shift: constituent vs ETF")
    for a0, b0 in crises.values():
        for a_ in ax:
            a_.axvspan(pd.Timestamp(a0), pd.Timestamp(b0), color="orange", alpha=0.15)
    ax[1].legend(loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "ar_pit_vs_etf.png", dpi=110)
    print("\nsaved plot -> outputs/ar_pit_vs_etf.png")
