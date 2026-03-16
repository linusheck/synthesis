"""Analyze sampled policies from test.json.

Usage: python analyze_policies.py test.json
"""
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def load(path):
    with open(path) as f:
        d = json.load(f)
    X = np.array(d["X"])           # (n_states, n_features)
    Y = np.array(d["Y"])           # (n_samples, n_states)
    return X, Y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_state_entropy(Y, n_actions, reachable):
    """Shannon entropy of the action distribution at each state."""
    n_samples, n_states = Y.shape
    entropies = np.zeros(n_states)
    for s in np.where(reachable)[0]:
        counts = np.bincount(Y[:, s], minlength=n_actions).astype(float)
        p = counts / counts.sum()
        p = p[p > 0]
        entropies[s] = -np.sum(p * np.log2(p))
    return entropies[reachable]


def pairwise_hamming(Y, reachable, max_pairs=5000):
    """Sample pairwise Hamming distances (fraction of reachable states that differ)."""
    Y_r = Y[:, reachable]
    n_samples = Y_r.shape[0]
    idx = np.random.default_rng(0).choice(n_samples, size=(max_pairs, 2), replace=True)
    dists = np.mean(Y_r[idx[:, 0]] != Y_r[idx[:, 1]], axis=1)
    return dists


def chain_autocorrelation(Y, reachable, max_lag=50):
    """Fraction of reachable states that change between sample t and sample t+lag."""
    Y_r = Y[:, reachable]
    n_samples = Y_r.shape[0]
    lags = np.arange(1, min(max_lag + 1, n_samples))
    autocorr = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        diffs = np.mean(Y_r[:-lag] != Y_r[lag:], axis=1)
        autocorr[i] = np.mean(diffs)
    return lags, autocorr


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_all(X, Y, out_prefix="policy_analysis"):
    n_samples, n_states = Y.shape
    # mask: reachable in all samples (no -1 anywhere)
    reachable = np.all(Y >= 0, axis=0)
    n_reachable = reachable.sum()
    Y_r = Y[:, reachable]
    n_actions = Y_r.max() + 1
    n_unique = len(set(map(tuple, Y_r.tolist())))

    print(f"states:          {n_states} ({n_reachable} reachable in all samples)")
    print(f"features/state:  {X.shape[1]}")
    print(f"samples:         {n_samples}")
    print(f"unique policies: {n_unique}/{n_samples} ({n_unique/n_samples*100:.1f}%)")
    print(f"actions:         0..{n_actions-1}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Global action frequencies
    ax = axes[0, 0]
    counts = np.bincount(Y_r.ravel(), minlength=n_actions)
    ax.bar(range(n_actions), counts / counts.sum())
    ax.set_xlabel("Action")
    ax.set_ylabel("Frequency")
    ax.set_title("Global action distribution")

    # 2. Per-state entropy histogram
    ax = axes[0, 1]
    ent = per_state_entropy(Y_r, n_actions, np.ones(n_reachable, dtype=bool))
    ax.hist(ent, bins=50, edgecolor="black")
    ax.set_xlabel("Entropy (bits)")
    ax.set_ylabel("# states")
    ax.set_title(f"Per-state action entropy (mean={ent.mean():.2f})")

    # 3. Pairwise Hamming distance
    ax = axes[0, 2]
    dists = pairwise_hamming(Y, reachable)
    ax.hist(dists, bins=50, edgecolor="black")
    ax.set_xlabel("Hamming distance (fraction)")
    ax.set_ylabel("# pairs")
    ax.set_title(f"Pairwise Hamming (mean={dists.mean():.3f})")

    # 4. Chain autocorrelation
    ax = axes[1, 0]
    lags, autocorr = chain_autocorrelation(Y, reachable)
    ax.plot(lags, autocorr, marker=".", markersize=3)
    ax.set_xlabel("Lag (samples)")
    ax.set_ylabel("Mean Hamming distance")
    ax.set_title("Chain mixing (Hamming vs lag)")

    # 5. Per-state dominant action fraction
    ax = axes[1, 1]
    dominant_frac = np.zeros(n_reachable)
    for i, s in enumerate(np.where(reachable)[0]):
        counts = np.bincount(Y[:, s], minlength=n_actions)
        dominant_frac[i] = counts.max() / n_samples
    ax.hist(dominant_frac, bins=50, edgecolor="black")
    ax.set_xlabel("Dominant action fraction")
    ax.set_ylabel("# states")
    ax.set_title("Per-state action agreement")

    # 6. # unique actions per state
    ax = axes[1, 2]
    n_unique_per_state = np.array([len(set(Y_r[:, i])) for i in range(n_reachable)])
    ax.hist(n_unique_per_state, bins=range(1, n_actions + 2), edgecolor="black", align="left")
    ax.set_xlabel("# distinct actions observed")
    ax.set_ylabel("# states")
    ax.set_title("Action diversity per state")

    fig.suptitle(f"Policy sample analysis  (n={n_samples}, unique={n_unique})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=150)
    print(f"\nSaved {out_prefix}.png")
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test.json"
    X, Y = load(path)
    plot_all(X, Y)
