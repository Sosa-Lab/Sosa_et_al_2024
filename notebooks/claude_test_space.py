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