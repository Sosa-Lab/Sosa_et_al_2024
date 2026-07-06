from matplotlib.widgets import RadioButtons
from matplotlib.colors import Normalize, to_rgba

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def add_hover( scatter, meta, columns=None, ax=None):
    """Show metadata in a tooltip when hovering scatter points.

    scatter : the PathCollection returned by ax.scatter(...)
    meta    : DataFrame, one row per point, row order == point order
    columns : which columns to show (default: all)
    Returns the matplotlib connection id (disconnect with fig.canvas.mpl_disconnect).
    """
    ax = ax or scatter.axes
    fig = ax.figure
    columns = list(meta.columns) if columns is None else columns

    annot = ax.annotate(
        "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
        ha="left", va="bottom", fontsize=8, zorder=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffe0", ec="0.4", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="0.4"),
    )
    annot.set_visible(False)

    def text_for(i):
        row = meta.iloc[i]
        return "\n".join(f"{c}: {row[c]}" for c in columns)

    state = {"visible": False, "ind": None}   # remember last paint to skip redundant redraws

    def on_move(event):
        if event.inaxes is not ax:
            if state["visible"]:               # left the axes — hide once
                annot.set_visible(False)
                state.update(visible=False, ind=None)
                fig.canvas.draw_idle()
            return

        hit, info = scatter.contains(event)
        if hit:
            i = int(info["ind"][0])            # first point if several overlap
            if i != state["ind"]:              # only repaint when the point changes
                annot.xy = scatter.get_offsets()[i]
                annot.set_text(text_for(i))
                annot.set_visible(True)
                state.update(visible=True, ind=i)
                fig.canvas.draw_idle()
        elif state["visible"]:                 # moved off all points
            annot.set_visible(False)
            state.update(visible=False, ind=None)
            fig.canvas.draw_idle()

    return fig.canvas.mpl_connect("motion_notify_event", on_move)


#clade made this, too. worked without any changes using default values. 
def add_color_selector(scatter, meta, columns=None,
                       cmap_continuous='viridis', cmap_categorical='tab10',
                       after_recolor=None):
    """Recolor a scatter by a chosen metadata column via on-figure radio buttons.
    Assumes point order == meta row order (point i is row i)."""
    ax  = scatter.axes
    fig = ax.figure
    columns = list(meta.columns) if columns is None else columns
    state = {'cbar': None}

    def set_positions_to_facecolor(plt_idx, color):
        if isinstance(color, str):
            color = to_rgba(color)
        
        cur_colors = scatter.get_facecolors()

        if len(cur_colors) == 1 and len(meta)>1:
            cur_colors = np.repeat(cur_colors, len(meta), axis = 0)

        new_colors = cur_colors
        print(plt_idx)
        new_colors[plt_idx] = color
        scatter.set_facecolors(new_colors)

    def _clear_extras():
        if state['cbar'] is not None:
            state['cbar'].remove(); state['cbar'] = None
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    
    def set_continuous_cmap(values, col):
        v = np.asarray(values, float)
        scatter.set_array(v)
        scatter.set_cmap(cmap_continuous)
        scatter.set_norm(Normalize(np.nanmin(v), np.nanmax(v)))
        state['cbar'] = fig.colorbar(scatter,  ax=ax, label=col)
    
    def set_categorical_cmap(values, col):
        codes, uniques = pd.factorize(values)
        cmap = plt.get_cmap(cmap_categorical)
        scatter.set_array(None)                   # detach the scalar mappable << what does this actually do?
        scatter.set_facecolors(cmap(codes % cmap.N))
        
        handles = [Line2D([], [], marker='o', ls='', color=cmap(i % cmap.N),
                        label=str(u)) for i, u in enumerate(uniques) if not u == -1]

        if any(values == -1):
            set_positions_to_facecolor(np.where(values == -1)[0], 'black')
            handles = [Line2D([], [], marker='o', ls='', color='black',
                        label=str(-1))] + handles

        
        
        
        ax.legend(handles=handles, title=col, fontsize=8,
                loc='upper right', bbox_to_anchor=(-0.02, 1))

    def guess_is_continuous(values):
        numeric = (pd.api.types.is_numeric_dtype(values)
                    and not pd.api.types.is_bool_dtype(values))
        if not numeric:
            return False
        elif pd.api.types.is_integer_dtype(values) and len(np.unique(values)) / len(values) < 0.5:
    
            return False
        else:
            return True
        
    
    
    def recolor(col):
        _clear_extras()
        values = meta[col]

        
        continous = guess_is_continuous(values)
        if continous:                                   # continuous -> colormap
            set_continuous_cmap(values, col)
        else:                                         # categorical/bool -> discrete
            set_categorical_cmap(values, col)
        if after_recolor:                                                            # << not implemented
            after_recolor(col)                        # e.g. re-stamp selection alpha  << not implemented
        fig.canvas.draw_idle()

    fig.subplots_adjust(left=0.28)                    # make room on the left
    rax = fig.add_axes([0.02, 0.5, 0.1, 0.3])        # [left, bottom, w, h], fig coords
    rax.set_title('color by', fontsize=9)
    radio = RadioButtons(rax, columns)
    radio.on_clicked(recolor)

    recolor(columns[0])
    return radio        # keep this reference alive — see note



from matplotlib.widgets import RectangleSelector, LassoSelector
#
def add_selectors(scatter, linker, meta, click_radius_px=5):
    """Box-drag and point-click selection on `scatter`, wired into `linker`.

    Modifiers:  plain = replace, shift = add, ctrl/cmd = remove.
    Empty plain click clears. Point order must match linker.metadata row order.
    Returns the RectangleSelector — keep a reference to it (see note).
    """
    ax = scatter.axes
    trial_idx_to_pos = {tid: i for i, tid in enumerate(meta['idx'].values)}

    def ids_to_positions(ids):
        """Global IDs -> local positions within this View. skip IDs not in this view"""
        return np.array([trial_idx_to_pos[t] for t in ids if t in trial_idx_to_pos], dtype=int)

    def positions_to_ids(positions):
        """Local positions (e.g. from .contains) -> global IDs."""
        return meta.iloc[np.asarray(positions, dtype=int)]['idx']

    def points_in_box(x0, x1, y0, y1):
        xy = scatter.get_offsets()                  # (N, 2), data coords
        x, y = xy[:, 0], xy[:, 1]
        inside = ((x >= min(x0, x1)) & (x <= max(x0, x1)) &
                  (y >= min(y0, y1)) & (y <= max(y0, y1)))
        return np.nonzero(inside)[0]

    def apply(ids, key):
        ids = positions_to_ids(list(map(int, ids)))



        if key and 'shift' in key:
            linker.add_selected(ids)
        elif key and ('control' in key or 'cmd' in key or 'super' in key):
            linker.remove_selected_ids(ids)
        else:
            linker.set_selected(ids)

    def on_select(eclick, erelease):
        key = eclick.key                            # modifier held during the gesture
        dpx = np.hypot((erelease.x or 0) - (eclick.x or 0),
                       (erelease.y or 0) - (eclick.y or 0))
        if dpx < click_radius_px:                   # --- point click ---
            hit, info = scatter.contains(eclick)
            if hit:
                apply(info['ind'], key)
            elif not key:                           # click on empty space, no modifier
                linker.clear()
        else:                                       # --- box drag ---
            apply(points_in_box(eclick.xdata, erelease.xdata,
                                eclick.ydata, erelease.ydata), key)
    def points_in_poly(verts):
        if len(verts) < 3:                       # too small to enclose anything
            return np.array([], dtype=int)
        inside = Path(verts).contains_points(scatter.get_offsets())
        return np.nonzero(inside)[0]

    def on_lasso(verts):
        apply()
    
    selector = RectangleSelector(
        ax, on_select,
        useblit=True, interactive=False, button=[1],          # left button only
        spancoords='pixels', minspanx=0, minspany=0,          # 0 -> clicks still fire onselect
        props=dict(facecolor='0.6', edgecolor='0.3', alpha=0.2, fill=True),
        # free shift/ctrl from the selector's own square/center modes so we can use them:
        state_modifier_keys=dict(square='9', center='0'),
    )

    selector2 = LassoSelector(
        ax, on_select,
        useblit=True,  button=[3],          # left button only
                # 0 -> clicks still fire onselect
        props=dict(color='0.3', alpha=0.2, ),
        # free shift/ctrl from the selector's own square/center modes so we can use them:
       )
    
    
        

    return selector, selector2

class View:
    ''''''
    def __init__(self, data, metadata, plt_obj, sort_dict: dict = {'idx':'ascending'}):
        '''sort the current selection by metadata values. remember than in python 3, dicts are ordered!
           sort_dict --> order of column names and their sort order (ascending or descending)'''


        self.data = data
        self.plt_obj = plt_obj
        self.metadata = metadata
        self.trial_idx_to_pos = {tid: i for i, tid in enumerate(self.metadata['idx'].values)}
        self.id_selection = []

        self.sort_dict = self._convert_sort_dict(sort_dict)
        

    def ids_to_positions(self, ids):
        """Global IDs -> local positions within this View. skip IDs not in this view"""
        return np.array([self.trial_idx_to_pos[t] for t in ids if t in self.trial_idx_to_pos], dtype=int)

    def positions_to_ids(self, positions):
        """Local positions (e.g. from .contains) -> global IDs."""
        return self.metadata.iloc[np.asarray(positions, dtype=int)]['idx']
    
    def _convert_sort_dict(self, sort_dict):
        new_dict = {}
        for k, v in sort_dict.items():
            if v == 'descending':
                new_v = False
            elif v == 'ascending':
                new_v = True
            else:
                print(f'warning: sort_dict expects only "ascending" or "descending" order, but {v} was passed.\ndefaulting to ascending order')
                new_v = True
            new_dict[k] = new_v


        return new_dict
    
    def sort_by(self, sort_dict: dict|None  = None):
        '''sort the current selection by metadata values. remember than in python 3, dicts are ordered!
           sort_dict --> order of column names and their sort order (ascending or descending), eg:
           {'idx':'ascending'}'''
        if not sort_dict is None:
            self.sort_dict = self._convert_sort_dict(sort_dict)
        
        #turn dict into lists to pass to pd.DataFrame.sort_values
        col_list, order_list = [k for k, v in self.sort_dict.items()], [v for k, v in self.sort_dict.items()]

        #sort metdata
        self.metadata = self.metadata.sort_values(by = col_list, ascending = order_list)

        #re-get indices from these sorted vals
        self.id_selection = self.metadata.loc[self.metadata.idx.isin(self.id_selection), 'idx'].values

        #run the update func
        self.update(self.id_selection, None, None, None)
    
    def update(self, selected_ids, sel_meta, deselected, desel_meta):
        '''to be overriden by view'''


class ScatterView(View):

    def __init__(self, data, meta, plt_obj, 
                                alpha_on_select = 0.5, 
                                alpha_on_deselect = 0.05,
                                color_on_select = None):
        super().__init__(data, meta, plt_obj)
        self.alpha_on_select = alpha_on_select
        self.alpha_on_deselect = alpha_on_deselect
        self.meta = meta
        self.update_color = color_on_select
        

    #claude wrote, i reviewed and changed to utilize new trial_idx <-> pos system
    def update(self, selected_ids, sel_meta, deselected, desel_meta):
        """Selected points opaque, everything else faded."""
        n = len(self.plt_obj.get_offsets())          # points actually drawn
        alphas = np.full(n, self.alpha_on_deselect)  # default everyone faded
        self.id_selection = np.asarray(selected_ids, dtype=int)
        positions = self.ids_to_positions(self.id_selection)
        alphas[positions] = self.alpha_on_select           # lift the selected
        self.plt_obj.set_alpha(alphas)
        if not self.update_color is None:
            self.set_trial_ids_to_facecolor(self.id_selection, self.update_color)
        self.plt_obj.figure.canvas.draw_idle()

    def set_trial_ids_to_facecolor(self, ids, color):
        if isinstance(color, str):
            color = to_rgba(color)
        
        cur_colors = self.plt_obj.get_facecolors()

        if len(cur_colors) == 1 and len(self.metadata)>1:
            cur_colors = np.repeat(cur_colors, len(self.metadata), axis = 0)

        new_colors = cur_colors
        plt_idx = self.ids_to_positions(ids)
        new_colors[plt_idx] = color
        self.plt_obj.set_facecolors(new_colors)
    

    
    def set_trial_ids_to_edgecolor(self, ids, color):
        if isinstance(color, str):
            color = to_rgba(color)
        if len(cur_colors) == 1 and len(self.metadata>1):
            cur_colors = np.repeat(cur_colors, len(self.metadata), axis = 0)
        cur_colors = self.plt_obj.get_edgecolors()
        new_colors = cur_colors
        plt_idx = self.ids_to_positions(ids)
        new_colors[plt_idx] = color
        self.plt_obj.set_edgecolors(new_colors)


from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import filters

class LickRasterView(View):

    def __init__(self, licks, bins, metadata, ax):
        '''data should be a numpy array of licks of shape (trials, bins),
        and must match the metadata length'''
        super().__init__(licks, metadata, ax)
        
        self.licks = licks
        self.bins = bins
        self.smooth_sigma = 0.5
        # self.plot_step = np.percentile(self.licks.ravel()[self.licks>0.1], 99)*1.1 << in case i decide to change per animal                                                                            
        self.plot_step = 2.75
        self.ax = ax
        self.set_raster_colormap()
        self.plot_raster( self.metadata['idx'].values[:10])
        
        
        
    
    def set_raster_colormap(self, cm = None):
        if cm is None:
            cm = LinearSegmentedColormap.from_list('MgK', ['black', 'magenta'])
        elif isinstance(cm, list):
            cm = LinearSegmentedColormap.from_list('custom', cm)
        elif isinstance(cm, LinearSegmentedColormap):
            cm = cm
        else:
            raise TypeError('invalid CM type')

        self.colormap = cm
    
    def update(self, ids, sel_meta, deselected, desel_meta):
        self.ax.clear()
        
        # take in ids (meta['idx']), but call plot_raster with index for
        # lick data passed in
        self.id_selection = ids
        self.plot_raster(self.ids_to_positions(ids))

    def plot_raster(self, pos_ind):
        # print(pos_ind)
        if not self.smooth_sigma == None:
            licks = filters.gaussian_filter1d(self.licks, self.smooth_sigma, axis=1)
        


        #slice out just the lick trials we need by index
        licks = licks[pos_ind]

        #ill need to revist this in case I want to make it more flexible
        vals = self.metadata.iloc[pos_ind]['omit'].values
        vals = vals.astype(float)
        rstarts = self.metadata.iloc[pos_ind]['reward_zone_start'].values
        rends = self.metadata.iloc[pos_ind]['reward_zone_end'].values

        for i, ind in enumerate(np.arange(0, licks.shape[0], 1)):

            y_pos = i*self.plot_step
            # if vals is not None:
            self.ax.fill_between(self.bins, licks[ind, :] + y_pos, y2=y_pos, 
                                 color=self.colormap(vals[ind]), linewidth=.001)

            # else:
            #     self.ax.fill_between(self.bins, licks[ind, :] + y_pos, y2=i*y_pos,
                                # color='black', linewidth=.001)
                
            self.ax.set_ylabel(f"{self.metadata.iloc[ind]['day']}, {self.metadata.iloc[ind]['trial']}")
            self.ax.fill_betweenx(y = [y_pos,y_pos+self.plot_step],
                                  x1 = [rstarts[ind]], 
                                  x2 =  [rends[ind]], color = 'red', alpha = 0.25)
        
        self.ax.set_yticks([i*self.plot_step for i in range(len(licks))])
        self.ax.set_yticklabels([f"d: {row[0]}\nt: {row[1]}" for row in self.metadata.iloc[pos_ind][['day','trial']].values])



    

class TrialPlotLinker:
    '''class to hold Views and the associated shared indeces used to highlight particular data'''

    def __init__(self,
                 meta:pd.DataFrame,
                    shared_selected: list = None, 
                    views: list = None):
        
        self.metadata = meta
        self.selected = set(shared_selected or [])
        self.views = set(views or [])

    #claude
    @property
    def last_deselected(self):                       # always the complement
        return sorted(set(range(len(self.metadata))) - self.selected)
    
    def add_views(self, views:list ):
        self.views =( list(self.views) + list(set(views)))

    def add_view(self, view):
        self.views = list(self.views)
        self.views.append(view)
        self.views = set(self.views)
    
    #claude approach, replacing funcs I wrote with a diff strategy.
    # using sets in this way is great, though.
    # i didnt know you could do intersection and unions so easily
    def _apply(self, new_selected):             # single source of truth
        '''make new selection and make sure its a set of unique IDs'''
        new_selected = set(new_selected)
        newly_selected   = new_selected - self.selected
        newly_deselected = self.selected - new_selected
        self.selected = new_selected
        self._on_transition(newly_selected, newly_deselected)   # your hook
        self.update_views()

    #clade wrote, I reviewed vvvvv
    def set_selected(self, ids):
        self._apply(ids)
    
    def add_selected(self, ids):

        #fun trivia here. set() + set() is not allowed, python
        #wants set() bitwise-or set(), to separate the syntax
        #from concatenation (like, list() + list() ), since
        #sets have no order
        self.selected = set(ids) | self.selected
        self.update_views()

    def remove_selected_ids(self, ids): 
        self._apply(self.selected - set(ids))
    def select_all(self):               
        self._apply(range(len(self.metadata)))
    def clear(self):                    
        self._apply([])

    

    def _on_transition(self, newly_selected, newly_deselected):
        self.last_deseleted = newly_deselected
        pass   # override / fill in when you want the switch action

    def update_views(self):
        sel, desel = sorted(self.selected), sorted(self.last_deselected)
        for v in self.views:
            v.update(sel, self.metadata.iloc[sel],
                     desel, self.metadata.iloc[desel])

    # ^^^^^^^^^