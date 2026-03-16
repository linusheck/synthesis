"""Train linear SVM on sampled policies and plot decision boundaries.

Usage:
    All policies:    python train_svm.py test.json
    Single policy:   python train_svm.py test.json --policy 0
    Skip plot:       python train_svm.py test.json --no-plot
"""
import json
import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def load(path):
    with open(path) as f:
        d = json.load(f)
    return np.array(d["X"]), np.array(d["Y"])


def train(X, y, label="policy"):
    """Train a linear SVM, print cross-val accuracy, return fitted model."""
    mask = y != -1
    X_f, y_f = X[mask], y[mask]

    clf = SVC(kernel="linear", decision_function_shape="ovr")
    min_class = min(np.bincount(y_f))
    cv = min(5, max(2, min_class))
    scores = cross_val_score(clf, X_f, y_f, cv=cv)
    print(f"{label}: acc={scores.mean():.3f} (+/- {scores.std():.3f})  [states={mask.sum()}, cv={cv}]")

    clf.fit(X_f, y_f)
    return clf, X_f, y_f


def _fmt_score(w, b, n_features):
    """Format a linear score function as a readable string."""
    terms = []
    for j in range(n_features):
        if abs(w[j]) < 1e-6:
            continue
        terms.append(f"{w[j]:+.3f} * f{j}")
    if not terms:
        return f"{b:+.3f}"
    return " ".join(terms) + f" {b:+.3f}"


def print_rules(clf, X):
    """Extract and print decision rules from a linear SVM.

    Uses clf.decision_function (OVR-shaped) so predicted action = argmax.
    Fits a linear model to each OVR score column to express rules as w.x + b.
    """
    classes = clf.classes_
    n_features = X.shape[1]

    # OVR decision scores: (n_samples, n_classes)
    scores = clf.decision_function(X)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    preds = classes[scores.argmax(axis=1)]
    predicted_actions = sorted(set(preds))

    # approximate each OVR score as a linear function of features
    from numpy.linalg import lstsq
    X_aug = np.column_stack([X, np.ones(len(X))])
    n_classes = len(classes)
    W = np.zeros((n_classes, n_features))
    b = np.zeros(n_classes)
    for i in range(n_classes):
        sol, _, _, _ = lstsq(X_aug, scores[:, i], rcond=None)
        W[i] = sol[:n_features]
        b[i] = sol[n_features]

    print("\nScore functions (predicted action = argmax score):")
    for i, c in enumerate(classes):
        print(f"  action {c}: {_fmt_score(W[i], b[i], n_features)}")

    # pairwise boundaries between actions that are both predicted somewhere
    print("\nDecision boundaries:")
    for idx_i, ci in enumerate(classes):
        if ci not in predicted_actions:
            continue
        for idx_j, cj in enumerate(classes):
            if cj not in predicted_actions or cj <= ci:
                continue

            dw = W[idx_i] - W[idx_j]
            db = b[idx_i] - b[idx_j]

            nonzero = [k for k in range(n_features) if abs(dw[k]) > 1e-6]
            if len(nonzero) == 0:
                winner = ci if db > 0 else cj
                print(f"  action {ci} vs {cj}: action {winner} always dominates")
            elif len(nonzero) == 1:
                k = nonzero[0]
                threshold = -db / dw[k]
                if dw[k] > 0:
                    print(f"  action {ci} vs {cj}: f{k} >= {threshold:.3f} -> action {ci}, else action {cj}")
                else:
                    print(f"  action {ci} vs {cj}: f{k} <= {threshold:.3f} -> action {ci}, else action {cj}")
            else:
                lhs = _fmt_score(dw, db, n_features)
                print(f"  action {ci} vs {cj}: {lhs} >= 0 -> action {ci}, else action {cj}")

    # summary: predicted action at each unique feature point
    unique_X, idx = np.unique(X, axis=0, return_index=True)
    unique_preds = preds[idx]
    print("\nPredicted action at each state:")
    for xi, pi in zip(unique_X, unique_preds):
        coords = ", ".join(f"f{j}={xi[j]:.0f}" for j in range(n_features))
        print(f"  ({coords}) -> action {pi}")

def plot_boundaries(clf, X, y, title, outpath):
    """Plot 2D decision boundaries with training points overlaid."""
    assert X.shape[1] == 2, f"plotting requires 2 features, got {X.shape[1]}"

    classes = sorted(clf.classes_)
    n_classes = len(classes)
    cmap_bg = ListedColormap(plt.cm.tab10.colors[:n_classes])
    cmap_pt = ListedColormap(plt.cm.tab10.colors[:n_classes])

    # mesh over feature space
    margin = 0.5
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    h = 0.02 * max(x_max - x_min, y_max - y_min)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_bg, levels=np.arange(min(classes) - 0.5, max(classes) + 1.5))
    ax.contour(xx, yy, Z, colors="k", linewidths=0.5, levels=np.arange(min(classes) - 0.5, max(classes) + 1.5))

    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_pt, edgecolors="k", s=60,
                         vmin=min(classes) - 0.5, vmax=max(classes) + 0.5)
    cbar = fig.colorbar(scatter, ax=ax, ticks=classes)
    cbar.set_label("action")

    # mark support vectors
    sv = clf.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=150, facecolors="none", edgecolors="red", linewidths=1.5, label="support vectors")

    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"saved plot to {outpath}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", default="test.json")
    parser.add_argument("--policy", type=int, default=None, help="train on a single policy")
    parser.add_argument("--no-plot", action="store_true", help="skip decision boundary plot")
    parser.add_argument("--output", default="svm_boundaries.png", help="plot output path")
    args = parser.parse_args()

    X, Y = load(args.path)

    if args.policy is not None:
        label = f"policy {args.policy}"
        clf, X_f, y_f = train(X, Y[args.policy], label=label)
    else:
        label = f"all {len(Y)} policies"
        X_all = np.tile(X, (len(Y), 1))
        y_all = Y.ravel()
        clf, X_f, y_f = train(X_all, y_all, label=label)

    print_rules(clf, X_f)

    if not args.no_plot and X.shape[1] == 2:
        plot_boundaries(clf, X_f, y_f, f"SVM decision boundaries ({label})", args.output)
    elif not args.no_plot:
        print(f"skipping plot: {X.shape[1]} features (need exactly 2)")


if __name__ == "__main__":
    main()
