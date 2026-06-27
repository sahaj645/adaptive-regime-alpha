"""Statistical Jump Model (Shu, Yu & Mulvey): k-means clustering plus a jump
penalty on state transitions, fit by coordinate descent. In-sample fitting uses
the full Viterbi path; out-of-sample inference uses the CAUSAL forward filter."""
from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np


def _kmeans(X: np.ndarray, k: int, seed: int, iters: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = ((X[:, None] - mu[None]) ** 2).sum(2).argmin(1)
        new = np.vstack([X[lab == c].mean(0) if (lab == c).any() else mu[c] for c in range(k)])
        if np.allclose(new, mu):
            break
        mu = new
    return mu


def _viterbi(X: np.ndarray, mu: np.ndarray, lam: float) -> np.ndarray:
    """Optimal state path: min sum ||x_t - mu_s||^2 + lam * #transitions."""
    T, K = len(X), len(mu)
    loss = ((X[:, None] - mu[None]) ** 2).sum(2)
    cost = np.empty((T, K)); back = np.zeros((T, K), int); cost[0] = loss[0]
    for t in range(1, T):
        for k in range(K):
            prev = cost[t - 1] + lam * (np.arange(K) != k)
            j = int(prev.argmin()); back[t, k] = j; cost[t, k] = loss[t, k] + prev[j]
    s = np.empty(T, int); s[-1] = int(cost[-1].argmin())
    for t in range(T - 2, -1, -1):
        s[t] = back[t + 1, s[t + 1]]
    return s


def fit(X: np.ndarray, lam: float, n_states: int = 2,
        seeds: Sequence[int] = (0, 1, 2)) -> Tuple[np.ndarray, np.ndarray]:
    """Fit centroids by coordinate descent; multi-start, keep the lowest objective."""
    best = None
    for seed in seeds:
        mu = _kmeans(X, n_states, seed); s = None
        for _ in range(40):
            s2 = _viterbi(X, mu, lam)
            mu = np.vstack([X[s2 == c].mean(0) if (s2 == c).any() else mu[c] for c in range(n_states)])
            if s is not None and np.array_equal(s2, s):
                break
            s = s2
        obj = ((X - mu[s]) ** 2).sum() + lam * (np.diff(s) != 0).sum()
        if best is None or obj < best[0]:
            best = (obj, s.copy(), mu.copy())
    return best[1], best[2]


def forward_filter(X: np.ndarray, mu: np.ndarray, lam: float) -> np.ndarray:
    """Causal online inference: the forward DP pass only. state_t uses data <= t."""
    T, K = len(X), len(mu)
    loss = ((X[:, None] - mu[None]) ** 2).sum(2)
    cost = np.empty((T, K)); cost[0] = loss[0]
    state = np.empty(T, int); state[0] = int(cost[0].argmin())
    for t in range(1, T):
        for k in range(K):
            cost[t, k] = loss[t, k] + min(cost[t - 1, k], cost[t - 1, 1 - k] + lam)
        state[t] = int(cost[t].argmin())
    return state


def bear_state(states: np.ndarray, returns: np.ndarray) -> int:
    """Label the unfavourable state by lower realized mean return (on TRAIN only)."""
    return int(np.nanargmin([returns[states == k].mean() for k in range(2)]))
