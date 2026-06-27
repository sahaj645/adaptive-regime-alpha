"""
Statistical Jump Model  (Shu, Yu & Mulvey, 2024)
================================================
A regime classifier that reads the TIME-SERIES behaviour of SPY itself
(trend + downside risk) and labels each day bull or bear. Complements the
Absorption Ratio, which reads the cross-section.

How it works (what the 'jump penalty' buys us):
  It is k-means clustering on a feature vector, PLUS a penalty lambda charged
  every time the state path switches. Plain k-means would flip the regime on
  every noisy day; the jump penalty makes the model only switch when the
  evidence outweighs the cost -> persistent, tradeable regimes.

Features (4), derived ONLY from SPY returns, per the paper:
  1. EWM downside deviation,  halflife 10   (recent downside volatility)
  2. EWM Sortino ratio,       halflife 20   (short risk-adjusted trend)
  3. EWM Sortino ratio,       halflife 60   (medium risk-adjusted trend)
  4. EWMA return,             halflife 120  (slow trend)
Bear = high downside deviation, low Sortino, negative slow trend.

NOTE: this script fits IN-SAMPLE (sees the whole period) purely to check that
the features + model find economically sensible regimes. The real, no-look-
ahead, out-of-sample version is wired into the walk-forward in Stage 7.
Methodology mirrors the authors' `jumpmodels` package; implemented from
scratch here so every step is transparent.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "etf_ohlcv.csv"
OUT  = ROOT / "outputs"; OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. SPY returns -> 4 features
# ----------------------------------------------------------------------------
df  = pd.read_csv(DATA, parse_dates=["Date"])
spy = df[df.Ticker == "SPY"].set_index("Date")["Adj_Close"].sort_index()
r   = np.log(spy / spy.shift(1)).dropna()
dr  = r.clip(upper=0.0)                                   # downside returns only
ew_dd = lambda x, hl: np.sqrt((x ** 2).ewm(halflife=hl).mean())

feat = pd.DataFrame({
    "dd10":      ew_dd(dr, 10),
    "sortino20": r.ewm(halflife=20).mean() / (ew_dd(dr, 20) + 1e-8),
    "sortino60": r.ewm(halflife=60).mean() / (ew_dd(dr, 60) + 1e-8),
    "ewma120":   r.ewm(halflife=120).mean(),
}).dropna().iloc[150:]                                    # burn-in for the slow EWMA
dates = feat.index
X  = ((feat - feat.mean()) / feat.std()).values           # z-score (in-sample)
rr = r.loc[dates]

# ----------------------------------------------------------------------------
# 2. Jump model = k-means + jump penalty, fit by coordinate descent
# ----------------------------------------------------------------------------
def kmeans(X, K, seed, iters=100):
    rng = np.random.default_rng(seed)
    mu  = X[rng.choice(len(X), K, replace=False)]
    for _ in range(iters):
        lab = ((X[:, None] - mu[None]) ** 2).sum(2).argmin(1)
        new = np.vstack([X[lab == k].mean(0) if (lab == k).any() else mu[k] for k in range(K)])
        if np.allclose(new, mu): break
        mu = new
    return mu

def fit_path(X, mu, lam):
    """Optimal state path: min  sum_t ||x_t-mu_{s_t}||^2 + lam * #switches  (DP)."""
    T, K = len(X), len(mu)
    L = ((X[:, None] - mu[None]) ** 2).sum(2)             # (T,K) squared distances
    C = np.empty((T, K)); Bk = np.zeros((T, K), int); C[0] = L[0]
    for t in range(1, T):
        for k in range(K):
            prev = C[t - 1] + lam * (np.arange(K) != k)   # pay lam to switch
            j = prev.argmin(); Bk[t, k] = j; C[t, k] = L[t, k] + prev[j]
    s = np.empty(T, int); s[-1] = C[-1].argmin()
    for t in range(T - 2, -1, -1): s[t] = Bk[t + 1, s[t + 1]]
    return s

def objective(X, mu, s, lam):
    return ((X - mu[s]) ** 2).sum() + lam * (np.diff(s) != 0).sum()

def fit_jm(X, lam, K=2, seeds=range(8)):
    best = None
    for sd in seeds:
        mu = kmeans(X, K, sd); s = None
        for _ in range(40):
            s2 = fit_path(X, mu, lam)
            mu = np.vstack([X[s2 == k].mean(0) if (s2 == k).any() else mu[k] for k in range(K)])
            if s is not None and np.array_equal(s2, s): break
            s = s2
        ob = objective(X, mu, s, lam)
        if best is None or ob < best[0]: best = (ob, s.copy(), mu.copy())
    return best[1], best[2]

def label_bear(s, rr):
    """Bear = the state with the lower realized mean SPY return."""
    return 0 if rr[s == 0].mean() < rr[s == 1].mean() else 1

# ----------------------------------------------------------------------------
# 3. The jump penalty in action: how lambda controls persistence
# ----------------------------------------------------------------------------
ann = 252
print(f"{'lambda':>6} | {'switches':>8} | {'%bear':>5} | {'bull ann%':>9} | {'bear ann%':>9}")
for lam in [0, 10, 30, 50, 100]:
    s, mu = fit_jm(X, lam)
    bear = label_bear(s, rr); isb = (s == bear)
    print(f"{lam:6d} | {int((np.diff(s)!=0).sum()):8d} | {isb.mean()*100:4.0f}% | "
          f"{rr[~isb].mean()*ann*100:+8.1f} | {rr[isb].mean()*ann*100:+8.1f}")

# ----------------------------------------------------------------------------
# 4. Pick a persistence level and inspect / plot
# ----------------------------------------------------------------------------
LAM = 50
s, mu = fit_jm(X, LAM); bear = label_bear(s, rr); isb = (s == bear)
regime = pd.Series(np.where(isb, "bear", "bull"), index=dates, name="regime")
regime.to_csv(OUT / "jump_regimes_insample.csv")

dur = []  # regime durations
run = 1
for i in range(1, len(s)):
    if s[i] == s[i-1]: run += 1
    else: dur.append(run); run = 1
dur.append(run)
print(f"\nChosen lambda={LAM}:")
print(f"  time in bear: {isb.mean()*100:.0f}%   regimes: {len(dur)}   avg duration: {np.mean(dur):.0f} trading days")
print(f"  bull: ann return {rr[~isb].mean()*ann*100:+.1f}%, ann vol {rr[~isb].std()*np.sqrt(ann)*100:.1f}%")
print(f"  bear: ann return {rr[isb].mean()*ann*100:+.1f}%, ann vol {rr[isb].std()*np.sqrt(ann)*100:.1f}%")
print("  bear centroid (z-scored dd10, sortino20, sortino60, ewma120):",
      np.round(mu[bear], 2), " bull:", np.round(mu[1-bear], 2))

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(spy.index, spy.values, color="black", lw=1); ax.set_yscale("log")
start = None
ib = isb
for i in range(len(dates)):
    if ib[i] and start is None: start = dates[i]
    if start is not None and (not ib[i] or i == len(dates) - 1):
        ax.axvspan(start, dates[i], color="red", alpha=0.15); start = None
ax.set_title(f"SPY with jump-model BEAR regimes shaded (in-sample, lambda={LAM})")
plt.tight_layout(); plt.savefig(OUT / "jump_model_regimes.png", dpi=110)
print("saved plot -> outputs/jump_model_regimes.png")
