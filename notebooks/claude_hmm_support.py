"""Build HMM initialization matrices from group-level relative weights.

You specify weights, not probabilities: "2 -> 0 is five times more likely than
2 -> 3". Rows are normalized at the end. Weights are relative within a row --
scaling a whole row changes nothing.
"""

import numpy as np


def row_normalize(weights):
    """Scale each row to sum to 1. Rows summing to zero become uniform."""
    W = np.asarray(weights, dtype=float)
    if np.any(W < 0):
        raise ValueError("Weights must be non-negative.")
    sums = W.sum(axis=1, keepdims=True)
    dead = sums.squeeze(-1) == 0
    if dead.any():
        W = W.copy()
        W[dead] = 1.0
        sums = W.sum(axis=1, keepdims=True)
    return W / sums


def _resolve_groups(groups, n_states):
    """Validate a name -> state-indices mapping and return name -> array."""
    index = {}
    seen = set()
    for name, members in groups.items():
        members = np.atleast_1d(np.asarray(members, dtype=int))
        if members.min() < 0 or members.max() >= n_states:
            raise ValueError(f"Group '{name}' has indices outside 0..{n_states - 1}.")
        overlap = seen & set(members.tolist())
        if overlap:
            raise ValueError(f"State(s) {sorted(overlap)} appear in more than one group.")
        seen.update(members.tolist())
        index[name] = members

    missing = set(range(n_states)) - seen
    if missing:
        raise ValueError(f"State(s) {sorted(missing)} are not in any group.")
    return index


def group_transition_matrix(n_states, groups, affinities, self_weight=None,
                            background=1.0):
    """Expand group-level relative weights into a row-stochastic matrix.

    Parameters
    ----------
    n_states : int
    groups : dict
        Group name -> list of state indices. Every state must appear exactly once.
    affinities : dict
        (from_group, to_group) -> relative weight. Directional: ('a','b') and
        ('b','a') are set independently. Unlisted pairs get `background`.
    self_weight : float, dict, or None
        Weight on the diagonal (a state staying put), overriding the group
        affinity for that one cell. A dict keys by group name. Higher values
        make states stickier; low values let states last a single timestep.
    background : float
        Weight for group pairs you did not mention.

    Returns
    -------
    (n_states, n_states) row-stochastic array.
    """
    index = _resolve_groups(groups, n_states)
    W = np.full((n_states, n_states), float(background))

    for (src, dst), weight in affinities.items():
        for name in (src, dst):
            if name not in index:
                raise KeyError(f"Unknown group '{name}' in affinities.")
        W[np.ix_(index[src], index[dst])] = float(weight)

    if self_weight is not None:
        if np.isscalar(self_weight):
            np.fill_diagonal(W, float(self_weight))
        else:
            for name, weight in self_weight.items():
                if name not in index:
                    raise KeyError(f"Unknown group '{name}' in self_weight.")
                W[index[name], index[name]] = float(weight)

    return row_normalize(W)


def group_start_prob(n_states, groups, weights, background=1.0):
    """Initial state distribution from group-level relative weights.

    Weight is split evenly among the states within each group.
    """
    index = _resolve_groups(groups, n_states)
    w = np.full(n_states, float(background))
    for name, weight in weights.items():
        if name not in index:
            raise KeyError(f"Unknown group '{name}' in weights.")
        members = index[name]
        w[members] = float(weight) / len(members)
    return w / w.sum()


def symmetrize(affinities):
    """Mirror each (a, b) weight onto (b, a), for relationships with no direction.

    Explicit reverse entries already in the dict are left alone.
    """
    out = dict(affinities)
    for (src, dst), weight in affinities.items():
        out.setdefault((dst, src), weight)
    return out


def describe(matrix, groups=None, decimals=3):
    """Print a labelled matrix. Faster than reading a raw array when tuning."""
    M = np.asarray(matrix)
    labels = [""] * len(M)
    if groups:
        for name, members in groups.items():
            for i in np.atleast_1d(members):
                labels[int(i)] = name
    width = max((len(l) for l in labels), default=0) + 4
    for i, row in enumerate(M):
        tag = f"{i}:{labels[i]}".ljust(width)
        print(tag, " ".join(f"{v:.{decimals}f}" for v in row))



"""Get hmmlearn's PoissonHMM to fit on sparse count data without nan blowups.

Two failure modes, both from the M-step:

  lambda -> 0   A feature column that is zero across every trial weighted into
                a state gives that state a rate of exactly 0. Any trial with a
                nonzero count there then has log-probability -inf. When all
                states hit zero in the same column, forward-backward
                normalizes -inf minus -inf and everything becomes nan.

  dead state    A state with no posterior mass divides 0 by 0.

diagnose_poisson_input finds the columns at risk before you fit.
GuardedPoissonHMM floors the rates after every M-step so neither can happen.
"""

import numpy as np


def diagnose_poisson_input(X, lengths=None, n_components=None, zero_thresh=0.95):
    """Check count data for the conditions that make PoissonHMM produce nan.

    Returns a dict. `risky_columns` are the ones most likely to drive a state's
    rate to exactly zero; `all_zero_columns` are guaranteed to.
    """
    X = np.asarray(X)
    report = {
        "shape": X.shape,
        "dtype": str(X.dtype),
        "has_nan": bool(np.isnan(X).any()),
        "has_inf": bool(np.isinf(X).any()),
        "has_negative": bool((X < 0).any()),
        "is_integer_valued": bool(np.all(X == np.floor(X))),
        "min": float(X.min()),
        "max": float(X.max()),
        "overall_zero_fraction": float(np.mean(X == 0)),
    }

    zero_frac = (X == 0).mean(axis=0)
    report["all_zero_columns"] = np.flatnonzero(zero_frac == 1.0).tolist()
    report["risky_columns"] = np.flatnonzero(
        (zero_frac >= zero_thresh) & (zero_frac < 1.0)
    ).tolist()
    report["max_column_zero_fraction"] = float(zero_frac.max())

    if lengths is not None:
        report["lengths_match"] = int(np.sum(lengths)) == len(X)
        report["n_sequences"] = len(lengths)
        report["shortest_sequence"] = int(np.min(lengths))

    if n_components is not None:
        report["n_components"] = n_components
        report["rows_per_state"] = len(X) / n_components

    return report


def summarize(report):
    """Print a diagnosis with the specific thing to change."""
    print(f"X: {report['shape']}  dtype={report['dtype']}  "
          f"range [{report['min']:g}, {report['max']:g}]")
    print(f"zeros: {report['overall_zero_fraction']:.1%} of all entries")

    problems = []
    if report["has_nan"] or report["has_inf"]:
        problems.append("X contains nan or inf -- fix this first.")
    if report["has_negative"]:
        problems.append("X contains negative values; Poisson requires counts >= 0.")
    if not report["is_integer_valued"]:
        problems.append(
            "X is not integer-valued. PoissonHMM assumes counts. If these are "
            "rates, multiply back to counts or switch models."
        )
    if report["all_zero_columns"]:
        problems.append(
            f"{len(report['all_zero_columns'])} column(s) are zero everywhere: "
            f"{report['all_zero_columns'][:8]}. Drop them -- they carry no "
            "information and force rates to exactly zero."
        )
    if report["risky_columns"]:
        problems.append(
            f"{len(report['risky_columns'])} column(s) are >=95% zeros. These "
            "are the likely source of zero rates. Use GuardedPoissonHMM."
        )
    if report.get("rows_per_state") is not None and report["rows_per_state"] < 20:
        problems.append(
            f"Only {report['rows_per_state']:.0f} rows per state. States will "
            "collapse; reduce n_components."
        )
    if report.get("lengths_match") is False:
        problems.append("sum(lengths) != len(X) -- sequences are being sliced wrong.")

    print()
    for p in problems:
        print(f"  - {p}")
    if not problems:
        print("  No structural problems found.")


def make_guarded_poisson_hmm(min_rate=1e-3, **kwargs):
    """PoissonHMM that floors emission rates after every M-step.

    The floor is the whole fix: a rate of 1e-3 against an observed count of 3
    gives a very negative but finite log-probability, which EM recovers from.
    A rate of exactly 0 gives -inf, which it does not.

    Also replaces nan rates left by states that received no posterior mass,
    so one dead state cannot poison the whole model.
    """
    from hmmlearn import hmm

    class GuardedPoissonHMM(hmm.PoissonHMM):
        def _do_mstep(self, stats):
            super()._do_mstep(stats)
            rates = np.asarray(self.lambdas_, dtype=float)
            bad = ~np.isfinite(rates)
            if bad.any():
                # A collapsed state keeps a usable rate rather than nan.
                rates[bad] = min_rate
            self.lambdas_ = np.maximum(rates, min_rate)

    return GuardedPoissonHMM(**kwargs)




"""Decompose lick rasters into a few spatial motifs, for use as HMM emissions.

Running an HMM directly on 450 track bins asks the emission model to treat
strongly correlated neighbouring bins as independent, which inflates the number
of states needed. NMF factors the raster into a small set of non-negative
spatial motifs plus per-trial loadings. The loadings are dense, low-dimensional,
non-negative, and keep positional information that zone sums discard -- a much
better-conditioned input for a state model.

Unlike hand-defined zone features, the motifs are learned from the data.
"""

import numpy as np
from sklearn.decomposition import NMF


def fit_lick_motifs(rasters, n_components=6, random_state=0, max_iter=500):
    """Factor (n_trials, n_bins) lick rates into motifs and per-trial loadings.

    Uses the Kullback-Leibler loss, which corresponds to a Poisson noise model
    -- the right choice for count-like data, and the same assumption your
    PoissonHMM emissions make.

    Returns (model, loadings, motifs) where loadings is (n_trials,
    n_components) and motifs is (n_components, n_bins).
    """
    X = np.asarray(rasters, dtype=float)
    if np.any(X < 0):
        raise ValueError("NMF requires non-negative input.")

    model = NMF(n_components=n_components, beta_loss="kullback-leibler",
                solver="mu", init="nndsvda", max_iter=max_iter,
                random_state=random_state)
    loadings = model.fit_transform(X)
    return model, loadings, model.components_


def reconstruction_quality(rasters, model, loadings):
    """Fraction of variance retained. Sweep n_components and take the knee."""
    X = np.asarray(rasters, dtype=float)
    approx = loadings @ model.components_
    resid = np.sum((X - approx) ** 2)
    total = np.sum((X - X.mean()) ** 2)
    return {"explained": 1.0 - resid / total,
            "reconstruction_err": float(model.reconstruction_err_)}


def loadings_to_counts(loadings, scale=100):
    """Discretize loadings for a Poisson emission model.

    Same trick you used on the raw rates, but applied to ~6 dimensions instead
    of 450. Keep `scale` modest: larger values sharpen the emission likelihood
    and progressively switch off the transition prior.
    """
    return np.rint(np.asarray(loadings) * scale).astype(int)