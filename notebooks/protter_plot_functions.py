from matplotlib.widgets import RadioButtons
from matplotlib.colors import Normalize, to_rgba
from matplotlib.cm import ScalarMappable

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def add_default_rzone_lines_to_matshow(ax, color = 'white', alpha = 0.5):
    for loc in [80, 200, 320]:
        ax.vlines(x = [loc], ymin=0, ymax=449, color = color, alpha = alpha, linestyles= '--')
        ax.hlines(y = [loc], xmin = 0, xmax = 449, color = color, alpha = alpha,  linestyles= '--')

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

def plot_raster_from_licks( data, metadata_df, ax, bin_centers = np.arange(5,455,10), starting_selection = -1, 
                           colorby = None, cmap = None,
                           edge_colorby = None, edgecm = None):

    v = LickRasterView(data, bin_centers, metadata_df, ax, starting_selection = starting_selection, 
                       colorby=colorby, cmap = cmap,
                           edge_colorby = edge_colorby, edgecm = edgecm)
    return v


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
    rax = fig.add_axes([0.1, 0.5, 0.1, 0.3])        # [left, bottom, w, h], fig coords
    rax.set_title('color by', fontsize=9)
    radio = RadioButtons(rax, columns)
    radio.on_clicked(recolor)

    recolor(columns[0])
    return radio        # keep this reference alive — see note

from matplotlib.widgets import RectangleSelector, LassoSelector
#
class ColorSelector:
    def __init__(self, scatter, meta, columns=None,
                        cmap_continuous='viridis', cmap_categorical='tab10',
                        after_recolor=None) -> None:
        '''state-saving RadioSelector'''

        self.scatter = scatter
        self.meta = meta
        self.color_linker = ColorLinker(column_name='omit')
        self.additional_scatters = []
        self.cmap_continuous='viridis'
        self.cmap_categorical='tab10'
        self.after_recolor=[]
        
        self.radio_obj = self.add_color_selector(scatter, meta, columns, cmap_continuous, cmap_categorical)
        
        

    def add_after_recolor_call(self, func):
        '''add a new func to run after recoloring/button press'''
        self.after_recolor += [func]

    def get_color_linker(self):
        return self.color_linker
    
    #clade made this, too. worked without any changes using default values. 
    def add_color_selector(self, scatter, meta, columns=None,
                        cmap_continuous='viridis', cmap_categorical='tab10',
                        ):
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
            mappable = ScalarMappable(norm=Normalize(np.nanmin(v), np.nanmax(v)), cmap=cmap_continuous)
            scatter.set_cmap(cmap_continuous) #just to track the name of the cmap
            scatter.set_facecolors(mappable.to_rgba(v))
            state['cbar'] = fig.colorbar(scatter,  ax=ax, label=col)
            self.color_linker.set_continuous(mappable.to_rgba, col)
            
            
        
        def set_categorical_cmap(values, col):
            codes, uniques = pd.factorize(values)
            cmap = plt.get_cmap(cmap_categorical)
            scatter.set_array(None)                   # detach the scalar mappable << what does this actually do?
            scatter.set_facecolors(cmap(codes % cmap.N))
            
            categorical_cmap = {unique:cmap(i % cmap.N) for i, unique in enumerate(uniques)} #keep this to pass to other plot objects
           

            handles = [Line2D([], [], marker='o', ls='', color=cmap(i % cmap.N),
                            label=str(u)) for i, u in enumerate(uniques) if not u == -1]

            if any(values == -1):
                categorical_cmap[-1] = (0, 0, 0, 1)
                set_positions_to_facecolor(np.where(values == -1)[0], 'black')
                handles = [Line2D([], [], marker='o', ls='', color='black',
                            label=str(-1))] + handles

            self.color_linker.set_categorical(categorical_cmap, col)

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
            for func in self.after_recolor:                                                            # << not implemented
                func()                        # e.g. re-stamp selection alpha  << not implemented
            fig.canvas.draw_idle()

        fig.subplots_adjust(left=0.28)                    # make room on the left
        rax = fig.add_axes([0.1, 0.5, 0.1, 0.3])        # [left, bottom, w, h], fig coords
        rax.set_title('color by', fontsize=9)
        radio = RadioButtons(rax, columns)
        radio.on_clicked(recolor)

        recolor(columns[0])
        return radio        # keep this reference alive — see note

class ColorSelectorV2:
    def __init__(self, scatter, meta, columns=None,
                        cmap_continuous='viridis', cmap_categorical='tab10',
                        after_recolor=None) -> None:
        '''state-saving RadioSelector'''

        self.scatters = [scatter]
        self.meta = meta
        self.color_linker = ColorLinker(column_name='omit')
        
        self.cmap_continuous='viridis'
        self.cmap_categorical='tab10'
        self.after_recolor=[]
        
        self.radio_obj = self.add_color_selector(scatter, meta, columns, cmap_continuous, cmap_categorical)
        
        
    def add_additional_scatter_objects(self, plots):
        '''need to check if plots have the same shape, or else funny stuff can happen'''

        self.scatters += plots

    def add_after_recolor_call(self, func):
        '''add a new func to run after recoloring/button press'''
        self.after_recolor += [func]

    def get_color_linker(self):
        return self.color_linker
    
    #clade made this, too. worked without any changes using default values. 
    def add_color_selector(self, scatter, meta, columns=None,
                        cmap_continuous='viridis', cmap_categorical='tab10',
                        ):
        """Recolor a scatter by a chosen metadata column via on-figure radio buttons.
        Assumes point order == meta row order (point i is row i)."""
        ax  = self.scatters[0].axes
        fig = ax.figure
        columns = list(meta.columns) if columns is None else columns
        state = {'cbar': None}

        def set_positions_to_facecolor(plt_idx, color):
            if isinstance(color, str):
                color = to_rgba(color)
            
            cur_colors = self.scatters[0].get_facecolors()

            if len(cur_colors) == 1 and len(meta)>1:
                cur_colors = np.repeat(cur_colors, len(meta), axis = 0)

            new_colors = cur_colors
            print(plt_idx)
            new_colors[plt_idx] = color
            for scatter in self.scatters:
                scatter.set_facecolors(new_colors)

        def _clear_extras():
            if state['cbar'] is not None:
                state['cbar'].remove(); state['cbar'] = None
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        
        def set_continuous_cmap(values, col):
            v = np.asarray(values, float)
            mappable = ScalarMappable(norm=Normalize(np.nanmin(v), np.nanmax(v)), cmap=cmap_continuous)
            for scatter in self.scatters:
                scatter.set_array(v)
                
                scatter.set_cmap(cmap_continuous) #just to track the name of the cmap
                scatter.set_facecolors(mappable.to_rgba(v))
            state['cbar'] = fig.colorbar(scatter,  ax=ax, label=col)
            self.color_linker.set_continuous(mappable.to_rgba, col)
            
            
        
        def set_categorical_cmap(values, col):
            codes, uniques = pd.factorize(values)
            cmap = plt.get_cmap(cmap_categorical)
            for scatter in self.scatters:
                scatter.set_array(None)                   # detach the scalar mappable << what does this actually do?
                scatter.set_facecolors(cmap(codes % cmap.N))
            
            categorical_cmap = {unique:cmap(i % cmap.N) for i, unique in enumerate(uniques)} #keep this to pass to other plot objects
           

            handles = [Line2D([], [], marker='o', ls='', color=cmap(i % cmap.N),
                            label=str(u)) for i, u in enumerate(uniques) if not u == -1]

            if any(values == -1):
                categorical_cmap[-1] = (0, 0, 0, 1)
                set_positions_to_facecolor(np.where(values == -1)[0], 'black')
                handles = [Line2D([], [], marker='o', ls='', color='black',
                            label=str(-1))] + handles

            self.color_linker.set_categorical(categorical_cmap, col)

            for scatter in self.scatters:
                ax = scatter.axes
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
            for func in self.after_recolor:                                                            # << not implemented
                func()                        # e.g. re-stamp selection alpha  << not implemented
            fig.canvas.draw_idle()

        fig.subplots_adjust(left=0.28)                    # make room on the left
        rax = fig.add_axes([0.1, 0.5, 0.1, 0.3])        # [left, bottom, w, h], fig coords
        rax.set_title('color by', fontsize=9)
        radio = RadioButtons(rax, columns)
        radio.on_clicked(recolor)

        recolor(columns[0])
        return radio   

    
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
        """Global IDs -> local array indices ("positions") within this View. skip IDs not in this view"""
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
        self.update(self.id_selection)
    
    def update(self, selected_ids,):
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
        self.reveal = False #hold this flag to recolor deselected points temporarily
        self.cmap = None
        self._connect_reveal()
        

    #claude wrote, i reviewed and changed to utilize new trial_idx <-> pos system
    # def update(self, selected_ids,):
    #     """Selected points opaque, everything else faded."""
    #     n = len(self.plt_obj.get_offsets())          # points actually drawn
    #     alphas = np.full(n, self.alpha_on_deselect)  # default everyone faded
    #     self.id_selection = np.asarray(selected_ids, dtype=int)
    #     positions = self.ids_to_positions(self.id_selection)
    #     alphas[positions] = self.alpha_on_select           # lift the selected
    #     self.plt_obj.set_alpha(alphas)
    #     if not self.update_color is None:
    #         self.set_trial_ids_to_facecolor(self.id_selection, self.update_color)
    #     self.plt_obj.figure.canvas.draw_idle()

    def update(self, selected_ids):
        """Store the current selection and draw it (respecting the reveal toggle)."""

        if len(selected_ids) > 0:
            self.id_selection = np.asarray(selected_ids)
            self._render_alpha()
            if self.update_color is not None:
                self.set_trial_ids_to_facecolor(self.id_selection, self.update_color)
            self.plt_obj.figure.canvas.draw_idle()
        else:
            print('uhh, empty list of IDs passed...')


    # def link_colormaps(self, color_linker, link_face = False, link_edge = True):
    #         '''expects to be able to access a dict with  "cmap" and "col" keys'''
    #         self.color_from_metadata = True
    #         self.link_face = link_face
    #         self.link_edge = link_edge
    
    #         if link_face:
    #             self.face_color_linker = color_linker
    #             self.set_face_colormap_from_linker(color_linker, color_linker.column_name)
    #         if link_edge:
    #             self.edge_color_linker = color_linker
    #             self.set_edge_colormap_from_linker(color_linker, color_linker.column_name)

    # def set_face_colormap_from_linker(self, cm, colorby):
    #     if cm is None:
    #         cm = sns.color_palette("flare", as_cmap=True)
    #     elif isinstance(cm, list):
    #         cm = LinearSegmentedColormap.from_list('custom', cm)
    #     elif isinstance(cm, ColorLinker):
    #         cm = cm
        
    #     self.cmap = cm
    #     self.colorby = colorby

    
    # def set_edge_colormap_from_linker(self, cm, colorby):
    #         if cm is None:
    #             cm = sns.color_palette("flare", as_cmap=True)
    #         elif isinstance(cm, list):
    #             cm = LinearSegmentedColormap.from_list('custom', cm)
    #         elif isinstance(cm, ColorLinker):
    #             cm = cm
            
    #         self.edgecm = cm
    #         self.edge_colorby = colorby

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
    
    def _connect_reveal(self, keys=(' ', 'space')):
        '''add a function to use the space bar to reveal all points'''
        fig = self.plt_obj.figure

        def on_key_release(event):
            print('keypress')
            if event.key in keys and self.reveal:
                self.reveal = False
                self._render_alpha(); fig.canvas.draw_idle()
            elif event.key in keys:
                self.reveal = True
                self._render_alpha(); fig.canvas.draw_idle()
            else:
                print(f'{event.key}')

        self._reveal_cid_release = fig.canvas.mpl_connect("key_release_event", on_key_release)
        
    def _render_alpha(self):
        """Paint alpha from the stored selection. reveal=True lifts everyone to selected alpha."""
        n = len(self.plt_obj.get_offsets())
        if self.reveal:
            alphas = np.full(n, self.alpha_on_select)      # all points visible for re-selecting
        else:
            alphas = np.full(n, self.alpha_on_deselect)
            positions = self.ids_to_positions(self.id_selection)
            alphas[positions] = self.alpha_on_select
        self.plt_obj.set_alpha(alphas)


from matplotlib.colors import LinearSegmentedColormap, Colormap
from scipy.ndimage import filters

class LickRasterView(View):

    def __init__(self, licks, bins, metadata, ax, color_from_metadata = True, starting_selection = 10, 
                 colorby = None, cmap = None,
                 edge_colorby = None, edgecm = None):
        '''data should be a numpy array of licks of shape (trials, bins),
        and must match the metadata length.
        starting_selection = -1 to plot all passed data, [id1, id2...] to start with a selection, or int to 
        '''
        super().__init__(licks, metadata, ax)
        
        self.licks = licks
        self.bins = bins
        self.smooth_sigma = 0.5
        # self.plot_step = np.percentile(self.licks.ravel()[self.licks>0.1], 99)*1.1 << in case i decide to change per animal                                                                            
        self.plot_step = 2.75
        self.ax = ax

        self.linked_cmap_dict = None
        self.linked_cmap_obj = None
        self.color_from_metadata = color_from_metadata
        if color_from_metadata or not colorby is None or not edge_colorby is None:
            self.set_edge_colormap(edgecm = edgecm, edge_colorby=edge_colorby)
            self.set_face_colormap(cm = cmap, colorby=colorby)

        if not (isinstance(starting_selection, list) or isinstance(starting_selection, np.ndarray)):
            if starting_selection == -1:
                self.id_selection =self.metadata['idx'].values
                self.plot_raster( self.metadata['idx'].values)
            elif isinstance(starting_selection, int):
                self.id_selection =self.metadata['idx'].values[:starting_selection]
                self.plot_raster( self.metadata['idx'].values[:starting_selection])

            else:
                self.id_selection =self.metadata['idx'].values[:10]
                self.plot_raster( self.metadata['idx'].values[:10])
        else:
                self.id_selection = starting_selection
                self.plot_raster( self.id_selection)
        
        
        
        
    def link_colormaps(self, color_linker, link_face = False, link_edge = True):
        '''expects to be able to access a dict with  "cmap" and "col" keys'''
        

        self.color_from_metadata = True
        self.link_face = link_face
        self.link_edge = link_edge

        if link_face:
            self.face_color_linker = color_linker
            self.set_face_colormap(color_linker, color_linker.column_name)
        if link_edge:
            self.edge_color_linker = color_linker
            self.set_edge_colormap(color_linker, color_linker.column_name)
    
   

    def set_face_colormap(self, cm = None, colorby = None):
        colorby = colorby if not colorby is None else 'omit'
        if cm is None:
            cm = LinearSegmentedColormap.from_list('MgK', ['black', 'magenta'])
            cm.__setattr__('column_name', colorby)
        elif isinstance(cm, list) or isinstance(cm, np.ndarray):
            cm = LinearSegmentedColormap.from_list('custom', cm)
            cm.__setattr__('column_name', colorby)
        elif isinstance(cm, ColorLinker):
            cm = cm
        elif isinstance(cm, Colormap):
            cm = cm
        else:
            raise TypeError('invalid CM type')
        self.color_from_metadata = True
        self.colormap = cm
        self.colorby = colorby
    


    def set_edge_colormap(self, edgecm = None, edge_colorby = None):
        edge_colorby = edge_colorby if not edge_colorby is None else 'omit'

        if edgecm is None:
            edgecm = LinearSegmentedColormap.from_list('MgK', ['black', 'magenta'])
            edgecm.__setattr__('column_name', edge_colorby)
        elif isinstance(edgecm, list) or isinstance(edgecm, np.ndarray):
            edgecm = LinearSegmentedColormap.from_list('custom', edgecm)
            edgecm.__setattr__('column_name', edge_colorby)
        elif isinstance(edgecm, ColorLinker):
            edgecm = edgecm
        else:
            raise TypeError(f'invalid CM type: {edgecm}')
        self.color_from_metadata = True
        self.edgecm = edgecm
        self.edge_colorby = edge_colorby
    
    def update(self, ids = None):
        if ids is None:
            ids = self.id_selection
        if len(ids) > 0:
            self.ax.clear()
            
            # take in ids (meta['idx']), and call raster plot with IDs
            # lick data passed in. first, get order of idxs in case the metadata df has
            #been sorted (EG descending by day). 
            self.id_selection = self.metadata.loc[self.metadata.idx.isin(ids), 'idx'].values
        else:
            print('uhh, empy list of IDs passed')

        
        self.plot_raster(self.id_selection)
    
    def update_linked_colors(self):
        self.ax.clear()
        self.plot_raster(self.id_selection)

    def plot_raster(self, ids):
        pos_ind = self.ids_to_positions(ids)

        metadata_slicer = self.metadata.idx.isin(ids)

        if not self.smooth_sigma == None:
            licks = filters.gaussian_filter1d(self.licks, self.smooth_sigma, axis=1)
        


        #slice out just the lick trials we need by index
        licks = licks[pos_ind]

        if self.color_from_metadata:
            face_vals = self.metadata.loc[metadata_slicer][self.colorby].values.astype(float)

            if not self.edgecm is None:
                if isinstance(self.edgecm, ColorLinker):
                    self.edge_colorby = self.edgecm.column_name
                self.edge_vals = self.metadata.loc[metadata_slicer][self.edge_colorby].values



            if 'reward_zone_start' in self.metadata.columns:
                rstarts = self.metadata.loc[metadata_slicer]['reward_zone_start'].values
                rends = self.metadata.loc[metadata_slicer]['reward_zone_end'].values

        
        y_pos_list = []
        for i, ind in enumerate(np.arange(0, licks.shape[0], 1)):
            top_y = self.plot_step*licks.shape[0]
            y_pos = top_y - i*self.plot_step
            y_pos_list += [y_pos]
            if self.color_from_metadata == False:
                self.ax.fill_between(self.bins, licks[ind, :] + y_pos, y2=y_pos, 
                                color='black', linewidth=.001)
            
            else:

                if not self.edge_colorby is None:
                    self.ax.fill_between(self.bins, licks[ind, :] + y_pos, y2=y_pos, 
                                        color=self.colormap(face_vals[ind]), 
                                        edgecolor = self.edgecm(self.edge_vals[ind]),
                                                    linewidth=1)
                else:
                    
                    self.ax.fill_between(self.bins, licks[ind, :] + y_pos, y2=y_pos, 
                                    color=self.colormap(face_vals[ind]), linewidth=.001)


                #shaded reward zone
                self.ax.fill_betweenx(y = [y_pos,y_pos+self.plot_step],
                                    x1 = [rstarts[ind]], 
                                    x2 =  [rends[ind]], color = 'red', alpha = 0.25)
                
                if 'swap_zone_start' in self.metadata.columns:
                    sr_st = self.metadata[metadata_slicer].iloc[ind]['swap_zone_start']
                    sr_en = self.metadata[metadata_slicer].iloc[ind]['swap_zone_end']

                    if self.metadata[metadata_slicer].iloc[ind]['trial_type'] == 'post_swap':
                        color = 'orange'
                        self.ax.fill_betweenx(y = [y_pos,y_pos+self.plot_step],
                                    x1 = [sr_st], 
                                    x2 =  [sr_en], color = color, alpha = 0.25)
                    else:
                        color = 'gray'
                        self.ax.fill_betweenx(y = [y_pos,y_pos+self.plot_step],
                                    x1 = [sr_st], 
                                    x2 =  [sr_en], color = color, alpha = 0.25)

                    handles = [Line2D([], [], marker='o', ls='', color='orange',
                            label='past_rzone'),
                            Line2D([], [], marker='o', ls='', color='gray',
                            label='future_rzone')]
                    self.ax.legend(handles= handles, bbox_to_anchor = (0.8,0.9))
                    

        
        self.ax.set_yticks(y_pos_list)
        if 'day' in self.metadata.columns and 'trial' in self.metadata.columns:
                                                                                                                        
            self.ax.set_yticklabels([f"{row[1]}" for row in self.metadata.loc[metadata_slicer][['day','trial']].values]) 

class ColorLinker:

    def __init__(self, column_name):
        self.cmap = None
        self.column_name = column_name
        self.cmap_type = 'categorical'

    def __call__(self, X):
        '''take a val or list of vals and return colors'''
        
        return self.get_colors(X)

    def get_colors(self, X:list|np.ndarray|int|float):
        if np.ndim(X) == 0:
            
            if self.cmap_type == 'categorical':
                return self.cmap[X] 
            else:
                return self.cmap(X)
        
        else:
            if self.cmap_type == 'categorical':
                return [self.cmap[k] for k in X]
            else:
                return [self.cmap(value) for value in X]

    def set_categorical(self, mapping_dict, col_name):
        self.cmap_type = 'categorical'
        self.cmap = mapping_dict
        self.column_name = col_name

    def set_continuous(self, cmap , col_name):
        '''pass a colormap right from a scatter plot, for example'''
        self.cmap_type = 'continous'
        self.cmap = cmap
        self.column_name = col_name

    def add_color(self, key, color):
        '''add a single key/color combo'''
        if self.cmap_type == 'categorical':
            self.cmap[key] = color
        else:
            raise TypeError('cannot add a key/value color pair to a categorical color linker')
    

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
            v.update(sel)

    # ^^^^^^^^^