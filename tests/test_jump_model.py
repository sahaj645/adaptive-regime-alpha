import numpy as np
from models.jump_model import fit, forward_filter, bear_state


def test_recovers_two_persistent_regimes():
    rng = np.random.default_rng(0)
    blocks = []
    for _ in range(6):
        blocks += [rng.normal(0.5, 0.2, (100, 2)), rng.normal(-0.5, 1.0, (100, 2))]
    X = np.vstack(blocks)
    states, mu = fit(X, lam=50, n_states=2, seeds=(0, 1))
    assert len(np.unique(states)) == 2
    filtered = forward_filter(X, mu, 50)
    assert len(filtered) == len(X)
    returns = X[:, 0]
    assert bear_state(states, returns) in (0, 1)
