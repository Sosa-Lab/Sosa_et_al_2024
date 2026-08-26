from matplotlib.widgets import RectangleSelector, LassoSelector
from matplotlib.path import Path
import numpy as np
def add_selectors(scatter, linker, meta, click_radius_px=5):
    """Selection on `scatter`, wired into `linker` (all gestures send trial IDs):
      left-click        : point select
      left-drag         : box select
      right-drag (btn3) : lasso select
    Modifiers: plain = replace, shift = add, ctrl/cmd = remove. Empty plain click clears.
    Returns (rect_selector, lasso_selector) — keep BOTH alive.
    """
    ax = scatter.axes

    trial_idx_to_pos = {tid: i for i, tid in enumerate(meta['idx'].values)}

    def positions_to_ids(positions):
        return meta.iloc[np.asarray(positions, dtype=int)]['idx']

    def points_in_box(x0, x1, y0, y1):
        xy = scatter.get_offsets()
        x, y = xy[:, 0], xy[:, 1]
        inside = ((x >= min(x0, x1)) & (x <= max(x0, x1)) &
                  (y >= min(y0, y1)) & (y <= max(y0, y1)))
        return np.nonzero(inside)[0]

    def points_in_poly(verts):
        if len(verts) < 3:                       # too small to enclose anything
            return np.array([], dtype=int)
        inside = Path(verts).contains_points(scatter.get_offsets())
        return np.nonzero(inside)[0]

    # --- modifier tracking (shared by both selectors) -------------------
    mods = set()
    def on_key_press(event):
        if event.key in ('shift', 'control', 'cmd', 'super'):
            mods.add(event.key)
            
    def on_key_release(event):
        mods.discard(event.key)
    fig = ax.figure
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)

    

    def apply(positions):
        ids = positions_to_ids(list(map(int, positions)))
        if 'shift' in mods:
            linker.add_selected(ids)
        elif mods & {'control', 'cmd', 'super'}:
            linker.remove_selected_ids(ids)
        else:
            linker.set_selected(ids)

    # --- box + click (left button) --------------------------------------
    def on_box(eclick, erelease):
        dpx = np.hypot((erelease.x or 0) - (eclick.x or 0),
                       (erelease.y or 0) - (eclick.y or 0))
        if dpx < click_radius_px:                # point click
            hit, info = scatter.contains(eclick)
            if hit:
                apply(info['ind'])
            elif not mods:                       # empty click, no modifier -> clear
                linker.clear()
        else:                                    # box drag
            apply(points_in_box(eclick.xdata, erelease.xdata,
                                 eclick.ydata, erelease.ydata))

    rect_selector = RectangleSelector(
        ax, on_box,
        useblit=True, interactive=False, button=[1],
        spancoords='pixels', minspanx=0, minspany=0,
        props=dict(facecolor='0.6', edgecolor='0.3', alpha=0.2, fill=True),
        state_modifier_keys=dict(square='9', center='0'),
    )

    # --- lasso (right button) -------------------------------------------
    def on_lasso(verts):
        apply(points_in_poly(verts))

    lasso_selector = LassoSelector(ax, on_lasso, button=[3], useblit=True)

    return rect_selector, lasso_selector


"""Magnitude-preserving distances between binned lick profiles.

Unlike the Wasserstein approach, nothing here normalizes rows to sum to 1, so
per-bin lick magnitude is retained. All distances operate in absolute track
coordinates -- no warping to the reward zone.

Two options:

  kernel_distance_matrix  -- Gaussian-kernel quadratic distance (== MMD).
                             Recommended default. Tolerates spatial shifts via
                             a kernel width you set in track units.

  cumulative_l1_matrix    -- L1 between UNNORMALIZED cumulative profiles.
                             Cheapest, but weights magnitude differences by
                             track position; see the docstring caveat.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def _bin_positions(n_bins, bin_width):
    return np.arange(n_bins, dtype=float) * bin_width


 
def gaussian_bin_kernel(n_bins, bin_width=1.0, sigma=None, circular=False):
    """Similarity matrix between track bins, K_ij = exp(-d_ij^2 / 2 sigma^2).
 
    sigma is in track position units: the distance over which two bins are
    treated as roughly interchangeable. Defaults to 2 bin widths.
 
    circular=True wraps the track into a loop, so the last bin is adjacent to
    the first. Only appropriate if the end of the track really does lead back
    to the start (e.g. teleport in VR); otherwise it invents an adjacency that
    does not exist and will pull unrelated trials together.
    """
    if sigma is None:
        sigma = 2.0 * bin_width
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    pos = _bin_positions(n_bins, bin_width)
    dist = np.abs(pos[:, None] - pos[None, :])
    if circular:
        track_length = n_bins * bin_width
        dist = np.minimum(dist, track_length - dist)
    return np.exp(-dist ** 2 / (2.0 * sigma ** 2))



def kernel_feature_map(rates, bin_width=1.0, sigma=None, kernel=None):
    """Project profiles so that plain Euclidean distance equals the MMD.

    Since K is PSD, K = B B^T, and (x-y)^T K (x-y) = ||B^T x - B^T y||^2.
    Returning B^T x lets any Euclidean-metric tool (PCA, k-means, UMAP without
    a precomputed matrix) inherit the kernel's shift tolerance.
    """
    rates = np.asarray(rates, dtype=float)
    if rates.ndim != 2:
        raise ValueError(f"rates must be 2-D (trials x bins), got {rates.ndim}-D.")

    if kernel is None:
        kernel = gaussian_bin_kernel(rates.shape[1], bin_width, sigma)

    eigvals, eigvecs = np.linalg.eigh(kernel)
    eigvals = np.clip(eigvals, 0.0, None)  # kill tiny negative numerical noise
    B = eigvecs * np.sqrt(eigvals)
    return rates @ B


def kernel_distance_matrix(rates, bin_width=1.0, sigma=None, kernel=None,
                           squareform_out=True):
    """Gaussian-kernel quadratic distance (MMD) between lick profiles.

    Keeps per-bin magnitude intact. Distances are in lick-rate units.
    sigma=0-ish recovers plain Euclidean; large sigma makes all bins
    interchangeable and the distance approaches |total_i - total_j|.
    """
    features = kernel_feature_map(rates, bin_width, sigma, kernel)
    condensed = pdist(features, metric="euclidean")
    return (squareform(condensed) if squareform_out else condensed)


def cumulative_l1_matrix(rates, bin_width=1.0, squareform_out=True):
    """L1 distance between unnormalized cumulative lick profiles.

    This is the Wasserstein CDF trick with the normalization step removed, so
    magnitude survives. It is a proper metric and costs one pdist call.

    CAVEAT: a magnitude difference in bin k propagates through every later
    entry of the cumulative sum, so the same rate difference is penalized more
    when it occurs early on the track than late. If reward zones sit at
    different track positions across days, that positional bias is not
    something you want. Prefer kernel_distance_matrix unless you have checked
    that this does not matter for your comparison.
    """
    rates = np.asarray(rates, dtype=float)
    cdfs = np.cumsum(rates, axis=1)
    condensed = pdist(cdfs, metric="cityblock") * bin_width
    return (squareform(condensed) if squareform_out else condensed)

"""
####### !!!!!!!!!!!!!!!!!!! MINIMALLY REVIEWED CLAUDE OUTPUT !!!!!!!!!!!!!!! #####################3

Piecewise / segmented colormapping for 1D data.

A base colormap spans [vmin, vmax]. Optional sub-ranges override that mapping,
each with either its own colormap (rescaled to that sub-range) or a flat color.
"""

import numpy as np
import matplotlib as mpl
from matplotlib.colors import Colormap, ListedColormap, Normalize, to_rgba


def _resolve_shade(spec):
    """Interpret a user spec as a Colormap or a single RGBA tuple.

    Colormap names win over CSS color names when both exist (e.g. 'pink',
    'gray', 'hot'). Pass an RGB(A) tuple to force a flat color.
    """
    if isinstance(spec, Colormap):
        return spec
    if isinstance(spec, str):
        if spec in mpl.colormaps:
            return mpl.colormaps[spec]
        return to_rgba(spec)
    return to_rgba(spec)


def _unit_scale(values, lo, hi):
    """Scale values to [0, 1] over [lo, hi]. NaNs pass through."""
    if hi == lo:
        return np.where(np.isnan(values), np.nan, 0.5)
    return (values - lo) / (hi - lo)


def _pair_zones(alternate_value_limits, alternate_cmaps):
    """Validate and zip zone limits with their color specs."""
    if alternate_value_limits is None:
        return []

    zones = [tuple(sorted(float(v) for v in pair))
             for pair in alternate_value_limits]

    if alternate_cmaps is None:
        raise ValueError("alternate_value_limits requires alternate_cmaps.")

    # A bare string, Colormap, or RGB(A) tuple is broadcast to every zone.
    is_scalar_spec = (
        isinstance(alternate_cmaps, (str, Colormap))
        or (isinstance(alternate_cmaps, tuple)
            and len(alternate_cmaps) in (3, 4)
            and all(isinstance(c, (int, float)) for c in alternate_cmaps))
    )
    specs = [alternate_cmaps] * len(zones) if is_scalar_spec else list(alternate_cmaps)

    if len(specs) != len(zones):
        raise ValueError(
            f"alternate_cmaps has {len(specs)} entries but "
            f"alternate_value_limits has {len(zones)}."
        )
    return list(zip(zones, specs))


def data_to_colors(data, cmap="viridis", vmin=None, vmax=None,
                   alternate_value_limits=None, alternate_cmaps=None, alpha = 1):
    """Map values to RGBA using a base colormap plus overriding sub-ranges.

    Parameters
    ----------
    data : array-like of float or int
    cmap : str or Colormap
        Base colormap, normalized across [vmin, vmax].
    vmin, vmax : float, optional
        Defaults to the finite min/max of `data`.
    alternate_value_limits : sequence of (start, end), optional
        Value sub-ranges (inclusive) that get their own mapping.
    alternate_cmaps : sequence, str, Colormap, or RGB(A) tuple, optional
        Index-matched to `alternate_value_limits`. Each entry is either a
        colormap (rescaled to that zone's own limits) or a flat color.
        Later zones win where zones overlap.

    Returns
    -------
    (N, 4) float array of RGBA values. NaNs get the colormap's "bad" color.
    """
    values = np.asarray(data, dtype=float)
    vmin = np.nanmin(values) if vmin is None else float(vmin)
    vmax = np.nanmax(values) if vmax is None else float(vmax)

    base = _resolve_shade(cmap)
    if isinstance(base, Colormap):
        rgba = base(np.clip(_unit_scale(values, vmin, vmax), 0.0, 1.0))
    else:
        rgba = np.tile(base, (values.size, 1)).reshape(values.shape + (4,))

    for (lo, hi), spec in _pair_zones(alternate_value_limits, alternate_cmaps):
        mask = (values >= lo) & (values <= hi)
        if not mask.any():
            continue
        shade = _resolve_shade(spec)
        if isinstance(shade, Colormap):
            rgba[mask] = shade(np.clip(_unit_scale(values[mask], lo, hi), 0.0, 1.0))
        else:
            rgba[mask] = shade
    
    rgba[:,3] = alpha

    return rgba


def make_segmented_cmap(vmin, vmax, cmap="viridis", alternate_value_limits=None,
                        alternate_cmaps=None, n=1024, name="segmented"):
    """Bake the same mapping into a (ListedColormap, Normalize) pair.

    Use this when you need a real colormap object -- for `imshow`, `pcolormesh`,
    or a colorbar. Zone edges are quantized to n bins; raise n for sharp edges.
    """
    edges = np.linspace(vmin, vmax, n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    colors = data_to_colors(centers, cmap=cmap, vmin=vmin, vmax=vmax,
                            alternate_value_limits=alternate_value_limits,
                            alternate_cmaps=alternate_cmaps)
    return ListedColormap(colors, name=name), Normalize(vmin=vmin, vmax=vmax)