"""
core.py — shared, CAUSAL building blocks for the Sarang regime project.

Everything here is written so that any value at day t uses only data up to and
including day t. The walk-forward engine adds the second guarantee (every
fitted transform is trained on past data only) and the t+1 trade rule.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "etf_ohlcv.csv"


# ---------------------------------------------------------------- data
def load():
    df  = pd.read_csv(DATA, parse_dates=["Date"])
    spy = df[df.Ticker == "SPY"].set_index("Date")["Adj_Close"].sort_index()
    sec = (df[df.ETF_Group == "Sector ETF"]
           .pivot(index="Date", columns="Ticker", values="Adj_Close")
           .sort_index().drop(columns=["XLC"]))          # 10 full-history sectors
    return df, spy, sec


# ---------------------------------------------------------------- features
def jm_features(spy):
    """The 4 jump-model features from SPY returns (causal EWMAs)."""
    r  = np.log(spy / spy.shift(1)).dropna()
    dr = r.clip(upper=0.0)
    ewdd = lambda x, hl: np.sqrt((x ** 2).ewm(halflife=hl).mean())
    f = pd.DataFrame({
        "dd10":      ewdd(dr, 10),
        "sortino20": r.ewm(halflife=20).mean() / (ewdd(dr, 20) + 1e-8),
        "sortino60": r.ewm(halflife=60).mean() / (ewdd(dr, 60) + 1e-8),
        "ewma120":   r.ewm(halflife=120).mean(),
    }).dropna()
    return f, r


def absorption_ratio_series(sec_px, window=252, n_eig=2):
    rets = sec_px.pct_change().iloc[1:].values
    idx  = sec_px.pct_change().iloc[1:].index
    ar = np.full(len(rets), np.nan)
    for i in range(window, len(rets) + 1):
        w   = rets[i - window:i]
        cov = np.cov(w, rowvar=False)
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        ar[i - 1] = eig[:n_eig].sum() / eig.sum()
    return pd.Series(ar, index=idx, name="AR").dropna()


def ar_features(sec_px, window=252, n_eig=2):
    ar = absorption_ratio_series(sec_px, window, n_eig)
    shift = (ar.rolling(15).mean() - ar.rolling(window).mean()) / ar.rolling(window).std()
    return pd.DataFrame({"AR": ar, "dAR": shift.rename("dAR")}).dropna()


# ---------------------------------------------------------------- jump model
def _kmeans(X, K, seed, iters=100):
    rng = np.random.default_rng(seed)
    mu  = X[rng.choice(len(X), K, replace=False)]
    for _ in range(iters):
        lab = ((X[:, None] - mu[None]) ** 2).sum(2).argmin(1)
        new = np.vstack([X[lab == k].mean(0) if (lab == k).any() else mu[k] for k in range(K)])
        if np.allclose(new, mu): break
        mu = new
    return mu


def _dp_path(X, mu, lam):
    """Full DP (forward + backward). Uses the WHOLE sequence -> in-sample only."""
    T, K = len(X), len(mu)
    L = ((X[:, None] - mu[None]) ** 2).sum(2)
    C = np.empty((T, K)); Bk = np.zeros((T, K), int); C[0] = L[0]
    for t in range(1, T):
        for k in range(K):
            prev = C[t - 1] + lam * (np.arange(K) != k)
            j = prev.argmin(); Bk[t, k] = j; C[t, k] = L[t, k] + prev[j]
    s = np.empty(T, int); s[-1] = C[-1].argmin()
    for t in range(T - 2, -1, -1): s[t] = Bk[t + 1, s[t + 1]]
    return s


def fit_jump(X, lam, K=2, seeds=range(6)):
    """Fit centroids by coordinate descent (in-sample state path for training)."""
    best = None
    for sd in seeds:
        mu = _kmeans(X, K, sd); s = None
        for _ in range(40):
            s2 = _dp_path(X, mu, lam)
            mu = np.vstack([X[s2 == k].mean(0) if (s2 == k).any() else mu[k] for k in range(K)])
            if s is not None and np.array_equal(s2, s): break
            s = s2
        ob = ((X - mu[s]) ** 2).sum() + lam * (np.diff(s) != 0).sum()
        if best is None or ob < best[0]: best = (ob, s.copy(), mu.copy())
    return best[1], best[2]


def forward_filter(X, mu, lam):
    """
    CAUSAL online inference: the FORWARD pass of the DP only.
    C[t,k] = min cost to be in state k at t given data <= t; the filtered state
    is argmin_k C[t,k]. No backward pass -> never uses the future. This is what
    we use out-of-sample. (K=2 fast path.)
    """
    T, K = len(X), len(mu)
    L = ((X[:, None] - mu[None]) ** 2).sum(2)
    C = np.empty((T, K)); C[0] = L[0]
    state = np.empty(T, int); state[0] = C[0].argmin()
    for t in range(1, T):
        for k in range(K):
            C[t, k] = L[t, k] + min(C[t - 1, k], C[t - 1, 1 - k] + lam)
        state[t] = C[t].argmin()
    return state


def label_bear(states, r_aligned):
    """Bear = the state with the lower realised mean SPY return (on TRAIN)."""
    m = [r_aligned[states == k].mean() for k in range(2)]
    return int(np.nanargmin(m))
