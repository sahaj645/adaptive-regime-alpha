"""
Absorption Ratio  (Kritzman, Li, Page & Rigobon, 2010)
======================================================
A systemic-risk / market-fragility sensor for the Sarang regime project.

Idea in one line: measure how *concentrated* the co-movement of the sector
cross-section is. When a few common factors absorb most of the variance, the
market is tightly coupled and fragile; when variance is spread across many
factors, it is resilient. A *rising* absorption ratio is fragility building up.

Stage-4 design decisions (all causal -- no look-ahead):
  * Cross-section : the 11 SPDR sector ETFs, EXCLUDING XLC (Communication
                    Services) which only begins mid-2018. Using the 10 sectors
                    with full history keeps the cross-section a constant size
                    and the AR series free of discontinuities. (XLC was carved
                    out of Tech / Discretionary in 2018, so its variance is
                    already largely represented.)  ->  N = 10 assets.
  * Window        : trailing 252 trading days (~1y), equal-weighted. With only
                    10 assets this is very well-conditioned (252 >> 10). A
                    window shorter than the paper's 500d is justified here and
                    preserves our short sample so the COVID crash lands
                    out-of-sample. (We sensitivity-test 126/500 & exponential
                    weighting in Stage 8.)
  * Eigenvectors  : n = round(N/5) = 2  (the paper's convention).
  * Signal        : the AR level, plus the standardized shift
                    dAR_t = (mean AR last 15d - mean AR last 252d)
                            / std(AR last 252d)
                    A reading above ~ +1 means fragility is spiking.

Outputs: outputs/absorption_ratio.csv  and  a validation plot vs SPY.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data" / "etf_ohlcv.csv"
OUT    = ROOT / "outputs"; OUT.mkdir(parents=True, exist_ok=True)

WINDOW, N_EIG, EXCLUDE = 252, 2, ["XLC"]

# ----------------------------------------------------------------------------
# 1. Load sector-ETF prices -> daily returns  (Adj_Close = total return)
# ----------------------------------------------------------------------------
df  = pd.read_csv(DATA, parse_dates=["Date"])
sec = df[df.ETF_Group == "Sector ETF"]
px  = (sec.pivot(index="Date", columns="Ticker", values="Adj_Close")
          .sort_index()
          .drop(columns=[c for c in EXCLUDE if c in sec.Ticker.unique()]))
rets = px.pct_change().iloc[1:]
print(f"Cross-section: {list(px.columns)}  (N={px.shape[1]})")

# ----------------------------------------------------------------------------
# 2. Absorption ratio, computed on a trailing window each day (causal)
# ----------------------------------------------------------------------------
def absorption_ratio(window_returns: pd.DataFrame, n_eig: int) -> float:
    w = window_returns.dropna(axis=1, how="any")          # only fully-observed assets
    if w.shape[1] < n_eig + 1:
        return np.nan
    cov  = np.cov(w.values, rowvar=False)                 # equal-weighted sample cov
    eig  = np.sort(np.linalg.eigvalsh(cov))[::-1]         # eigenvalues, descending
    return float(eig[:n_eig].sum() / eig.sum())

ar = pd.Series(index=rets.index, dtype=float, name="AR")
for i in range(WINDOW, len(rets) + 1):
    ar.iloc[i - 1] = absorption_ratio(rets.iloc[i - WINDOW:i], N_EIG)
ar = ar.dropna()

# ----------------------------------------------------------------------------
# 3. Standardized shift (causal: trailing means / std of past AR only)
# ----------------------------------------------------------------------------
dar = (((ar.rolling(15).mean() - ar.rolling(WINDOW).mean())
        / ar.rolling(WINDOW).std())).rename("dAR")

out = pd.concat([ar, dar], axis=1).dropna()
out.to_csv(OUT / "absorption_ratio.csv")
print(f"AR  : {ar.index.min().date()} -> {ar.index.max().date()}  (n={len(ar)})")
print(f"dAR : starts {dar.dropna().index.min().date()}")
print(out.describe().round(3).to_string())

# ----------------------------------------------------------------------------
# 4. Validation plot vs SPY  (do the fragility spikes line up with drawdowns?)
# ----------------------------------------------------------------------------
spy = df[df.Ticker == "SPY"].set_index("Date")["Adj_Close"].sort_index()
crises = {"Q4-2018": ("2018-10-01", "2018-12-24"),
          "COVID":    ("2020-02-19", "2020-03-23"),
          "2022 bear":("2022-01-03", "2022-10-12")}

fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1]})
ax[0].plot(spy.index, spy.values, color="black", lw=1)
ax[0].set_yscale("log"); ax[0].set_title("SPY total return (log scale)")
ax[1].plot(ar.index, ar.values, color="tab:blue", lw=1.0, label="Absorption Ratio (top-2 / total var)")
axr = ax[1].twinx()
axr.plot(dar.index, dar.values, color="tab:red", lw=0.9, alpha=0.7, label="Standardized shift dAR")
axr.axhline(1.0, color="tab:red", ls="--", lw=0.8, alpha=0.6)
ax[1].set_title("Absorption Ratio (blue, left)  &  standardized shift dAR (red, right; dashed = +1 sigma)")
for name, (a0, b0) in crises.items():
    for a_ in ax:
        a_.axvspan(pd.Timestamp(a0), pd.Timestamp(b0), color="orange", alpha=0.18)
    ax[0].text(pd.Timestamp(a0), spy.max() * 0.78, name, fontsize=8)
ax[1].legend(loc="upper left", fontsize=8); axr.legend(loc="upper right", fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "absorption_ratio_validation.png", dpi=110)
print("saved plot -> outputs/absorption_ratio_validation.png")

# ----------------------------------------------------------------------------
# 5. Quick quantitative sanity check (ANALYSIS ONLY -- uses future returns,
#    never used by the trading signal): is forward SPY weaker after dAR spikes?
# ----------------------------------------------------------------------------
spy_ret = spy.pct_change()
fwd20   = spy_ret.rolling(20).sum().shift(-20)          # next-20d SPY return
idx     = dar.dropna().index.intersection(fwd20.dropna().index)
hi      = dar.loc[idx] > 1.0
print(f"\n[analysis only] forward 20d SPY return  |  dAR>+1sigma: "
      f"{fwd20.loc[idx][hi].mean()*100:+.2f}%   otherwise: "
      f"{fwd20.loc[idx][~hi].mean()*100:+.2f}%   (hi days={int(hi.sum())}, lo days={int((~hi).sum())})")
