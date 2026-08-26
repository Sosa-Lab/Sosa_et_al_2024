from reward_relative.path_dict_seahorse import path_dictionary as path_dict
from reward_relative import utilities as ut
from reward_relative import plotUtils as pt
from reward_relative import spatial
from reward_relative import placeCellPlot
from reward_relative import dayData as dd
from reward_relative import behavior
from reward_relative import rewardAnalysis as ra
import pickle
# import dill

import numpy as np
import os 
import matplotlib.pyplot as plt
import TwoPUtils
import astropy
from TwoPUtils import spatial_analyses
from scipy.ndimage import filters
import pandas as pd



class TrialInfo:

    def __init__(self, session_dict):
        self.session_dict = session_dict
        dat = [self.parse_session(day, session) for day, session in session_dict.items()]
        
        self.lookup_df = dat[0]
        for dat_day in dat[1:]:
            self.lookup_df = pd.concat([self.lookup_df, dat_day] ) 
    
    def parse_session(self, day, session):
        animal = session.mouse
        rzones = behavior.get_reward_zones(session)[0]
        swap_zones = np.zeros((len(rzones), 2))
        omit_trials = ra.get_omission_trials(session)['trials']
        omit = np.zeros(shape = (len(rzones)), dtype = bool)
        omit[omit_trials] = True



        trials = np.arange(0, len(rzones), 1)

        idxs = [f'{animal}_{day}_{trial}' for trial in trials]

        if rzones[0,][0]<rzones[-1][0]:
            type_swap = 'swap_distal'
        elif rzones[0,0] == rzones[-1,0]:
            type_swap = 'stay'
        else:
            type_swap = 'swap_proximal'

        if 'swap' in type_swap:
            trial_type = ['pre_swap']*sum(rzones[:,0]==rzones[0,0]) 
            trial_type += ['post_swap']*sum(rzones[:,0]==rzones[-1,0])

            swap_zones[rzones[:,0] == rzones[0,0],:] = rzones[-1,:]
            swap_zones[rzones[:,0] == rzones[-1,0],:] = rzones[0,:]

        else:
            trial_type = ['stay']*len(rzones)


        return pd.DataFrame(data = {'idx' : idxs, 
                                        'day' : [day]*len(rzones), 
                                        'trial' :  trials, 
                                        'reward_zone_start': rzones[:,0], 
                                        'reward_zone_end': rzones[:,1],
                                        'swap_zone_start':swap_zones[:,0],
                                        'swap_zone_end':swap_zones[:,1],
                                        'trial_type':trial_type,
                                        'swap_type':type_swap, 
                                        'omit':omit} )

    def add_session(self, day, session):
        self.lookup_df = pd.concat([self.lookup_df, self.parse_session(day, session)])
    




def correct_licks(session, correction_thr = 0.3):
    # get rid of trials where lick sensor may have gotten stuck on
    licks = np.copy(session.vr_data['lick'].values)
    licks, error_count = behavior.correct_lick_sensor_error(
        licks, session.trial_start_inds, session.teleport_inds, correction_thr=correction_thr)
    
    licks_mat = spatial_analyses.trial_matrix(licks, 
                                            session.vr_data['pos']._values, 
                                            session.trial_start_inds,
                                            session.teleport_inds, impute_nans=False)
    
    return licks_mat

def smooth_trial_matrix(trial_mat, sigma = 0.5):
    '''pass a trial matrix (data, ?, position bins), and return a 1d smoothed version.
    taking default sigma from behavior.plot_norm_lick_raster'''
    data, occupancy, bin_edges, bin_centers = trial_mat
    smoothed_data = filters.gaussian_filter1d(data, sigma, axis=1)
    

    return (smoothed_data)


def generate_lick_metrics(session, correct_sensor_error = True, correction_thr=0.3,):
    if correct_sensor_error:
        # get rid of trials where lick sensor may have gotten stuck on
        licks = np.copy(session.vr_data['lick'].values)
        licks, error_count = behavior.correct_lick_sensor_error(
            licks, session.trial_start_inds, session.teleport_inds, correction_thr=0.3)
    
    
    else:
        licks = np.copy(session.vr_data['lick'].values)

    licks_mat = spatial_analyses.trial_matrix(licks, 
                                            session.vr_data['pos']._values, 
                                            session.trial_start_inds,
                                            session.teleport_inds, impute_nans=False)
        
    positions = np.asarray(licks_mat[-1])
    licks = licks_mat[0]


    com_vs_circvar = np.zeros((80,2))
    com_vs_circvar[:,0] = ut.center_of_mass(licks, coord = positions, axis = 1).reshape((80))

    reward_loc = behavior.get_reward_zones(session)[0]

    reward_start = reward_loc[:,0]

    com_vs_circvar[:,0] = reward_start - com_vs_circvar[:,0]

    com_vs_circvar[:,1] = astropy.stats.circstats.circvar(licks, axis = 1)
    return com_vs_circvar

def correlate_two_pop_vectors(pv1, pv2, pv1_name=None, pv2_name=None):
    '''take two pop vectors and their names, and return:
    corr_matrices --> list
    corr_names    --> list'''
    corr_mat = np.corrcoef(pv1, pv2)

    return [corr_mat[:len(pv1),:len(pv1)],  corr_mat[len(pv1):, len(pv1):], corr_mat[0:len(pv1), len(pv1):],], [f'{pv1_name}', f'{pv2_name}', f'{pv1_name}_v_{pv2_name}']


def recreate_trial_meta_df(hdf5_group):
    data_dict = {key:hdf5_group[key][:] for key in hdf5_group.keys()}
    for k,vals in data_dict.items():
        if vals.dtype == np.dtypes.ObjectDType:
            data_dict[k] = np.asarray([v.decode() for v in vals])
    return pd.DataFrame.from_dict(data_dict)

def generate_features_from_zone_list(licks, bin_centers, zone_objects, norm = False):
    '''take a list of zone objects and create features'''
    n_licks = licks.sum(axis = 1)
    features = {'n_licks':n_licks}
    features['start'] = get_licks_at_location(licks, [0, 30], bin_centers)['licks'].sum(axis = 1)
    features['end'] = get_licks_at_location(licks, [400, 450], bin_centers)['licks'].sum(axis = 1)
    for zone in zone_objects:
        features.update(get_licks_from_zone_object(licks, zone, bin_centers, include_name = True))

    features_df = pd.DataFrame.from_dict(features)
    if norm:
        features_df = features_df/n_licks[:,np.newaxis]
        features_df['n_licks'] = n_licks
    
    return features_df




def generate_cross_day_features(licks, meta, bin_centers, norm = False, reward_relative = True):
    '''if multiple days are passed, this could potentially reorder your df and licks, so both are passed back in the order
    they get modified'''
    dfs = []
    features = []
    data = []
    for animal in meta.animal.unique():
        ani_slice = meta.loc[meta.animal == animal]

        days = ani_slice.day.unique()
        # day_slicer = ani_slice.day == days[0]
        # day_slice = ani_slice.loc[day_slicer]
        # day_zone_obj = DayZones(day_slice)
        # feats = generate_zone_lick_features_from_class(licks[day_slicer], day_zone_obj, bin_centers, norm = norm, 
        #                                                 reward_relative=reward_relative)
        # features += [features]
        # dfs +=[ day_slice.copy()]
        # data+= [licks[day_slicer]]

       
        for day in days:
            day_slicer = (meta.day == day) & (meta.animal == animal)

            day_slice = meta.loc[day_slicer]
            day_licks = licks[day_slicer]

            day_zone_obj = DayZones(day_slice, post_zone_distance=15)

            day_feats = generate_zone_lick_features_from_class(day_licks, 
                                                            day_zone_obj, 
                                                            bin_centers, norm = norm,
                                                            reward_relative=reward_relative)

            features += [day_feats]
            dfs +=[ day_slice.copy()]
            data+= [day_licks]
    features = pd.concat(features)
    sorted_meta = pd.concat(dfs)
    sorted_licks = np.concatenate(data)

    return features, sorted_meta, sorted_licks


def get_pop_vec_similarity_across_array(pv_array, pv_comparitor = None):
    '''if comparing against self, compare trial to mean of all other trials. that is, leave out each 
        individual trial from the mean. '''
    if pv_comparitor is not None:
        
        corr_mats, corr_labels =correlate_two_pop_vectors(pv_comparitor, pv_array[0], pv1_name=f'_', pv2_name=f'_')
        corr_mat = corr_mats[-1]
        out = np.zeros(shape = (pv_array.shape[0], corr_mat.shape[0], corr_mat.shape[1]))
        out[0] = corr_mat
        for i, pv in enumerate(pv_array[1:]):
            corr_mats, corr_labels = correlate_two_pop_vectors(pv_comparitor, pv, pv1_name=f'_', pv2_name=f'_')
            out[i+1] = corr_mats[-1]

    else:
        
        out = np.zeros(shape = (pv_array.shape[0], pv_array.shape[1], pv_array.shape[1]))
        pv_ma_array = np.ma.array(pv_array, mask = False)
        for i in range(pv_array.shape[0]):
            pv_ma_array.mask[i] = True
            pv = pv_array[i]

            pv_comparitor = pv_ma_array.mean(axis = 0)
            corr_mats, corr_labels = correlate_two_pop_vectors(pv_comparitor, pv, pv1_name=f'_', pv2_name=f'_')
            out[i]= corr_mats[-1]
            pv_ma_array.mask[i] = False

    return out

def tm_zscore(tm):
    '''zscore across all trials of a trial matrix,
    assuming shape (trials, position, n_cells).
    
    data is reshaped to be of shape (trials*position, n_cells) to take
    mean and std, then these vectors (shape = ncells) are used to 
    calculate the z-score for each cells data
    '''
    #append trials along axis 0
    resh = tm.reshape(-1, tm.shape[-1])
    std = np.std(resh, axis = 0)
    mean = np.mean(resh, axis = 0)

    return (tm - mean) / std

def get_data_trials_by_idx(data, data_meta_df, idx_list):
    out =  data[data_meta_df.idx.isin(idx_list).values]
    if out.shape[0] == 1:
        out = out[0]
    return out

from dataclasses import dataclass

@dataclass
class Zone:

    name:'str'
    start: int
    end: int
    anticipatory_distance: int = 50
    post_zone_distance: int = 20

    def __post_init__(self):
        self.anticipatory_zone = [self.start-self.anticipatory_distance, self.start]
        self.rzone = [self.start, self.end]
        self.after_zone = [self.end, self.end+self.post_zone_distance]

rzone_start_to_name = {80:'A', 200:'B', 320:'C'}
rzone_dict = {'A':[80, 130], 'B':[200, 250], 'C':[320, 370]}


default_zones = [Zone(name = k,
                             start = v[0],
                             end = v[1],
                             anticipatory_distance = 50, 
                             post_zone_distance = 15)

                for k,v in rzone_dict.items()
                

                ]

class DayZones:


    def __init__(self, meta_slice, anticipatory_distance = 50, post_zone_distance = 20):
        self.rzone_dict = {'A':[80, 130], 'B':[200, 250], 'C':[320, 370]}
        self.meta = meta_slice
        rzone_1_start = self.meta.reward_zone_start.values[0]
        rzone_2_start = self.meta.reward_zone_start.values[-1]

        rzone_1_end = self.meta.reward_zone_end.values[0]
        rzone_2_end = self.meta.reward_zone_end.values[-1]
        
        self.pre_zone = Zone(name = rzone_start_to_name[rzone_1_start],
                             start = rzone_1_start,
                             end = rzone_1_end,
                             anticipatory_distance = anticipatory_distance, 
                             post_zone_distance = post_zone_distance)
        
        self.post_zone = Zone(name = rzone_start_to_name[rzone_2_start],
                             start = rzone_2_start,
                             end = rzone_2_end,
                             anticipatory_distance = anticipatory_distance, 
                             post_zone_distance = post_zone_distance)
        
        self.pre_trials = self.meta.reward_zone_start == rzone_1_start
        self.post_trials = self.meta.reward_zone_start == rzone_2_start

        self.unused_zone_name = [v for v in self.rzone_dict.keys() if not v in [self.pre_zone.name, self.post_zone.name]][0]
        self.unused_zone = Zone(name = self.unused_zone_name,
                                start = self.rzone_dict[self.unused_zone_name][0],
                                end = self.rzone_dict[self.unused_zone_name][1],
                                anticipatory_distance = anticipatory_distance, 
                                post_zone_distance = post_zone_distance)


def get_licks_from_zone_object(licks, zone_object, bin_centers, include_name = False):

    if include_name:
        name = zone_object.name
    else:
        name = ''
    
    
    return {f'{name}_at_zone':get_licks_at_location(licks, zone_object.rzone, bin_centers)['licks'].sum(axis = 1),
            f'{name}_anticipatory_zone':get_licks_at_location(licks, zone_object.anticipatory_zone, bin_centers)['licks'].sum(axis = 1),
            f'{name}_after_zone':get_licks_at_location(licks, zone_object.after_zone, bin_centers)['licks'].sum(axis = 1),}



def generate_zone_lick_features_from_class(licks, day_zone_obj, bin_centers, norm = False, reward_relative = True):
    n_licks = licks.sum(axis = 1)
    pre_zone_licks = get_licks_from_zone_object(licks, day_zone_obj.pre_zone, bin_centers)
    post_zone_licks = get_licks_from_zone_object(licks, day_zone_obj.post_zone, bin_centers)
    other_zone_licks = get_licks_from_zone_object(licks, day_zone_obj.unused_zone, bin_centers)

    features = {'n_licks':n_licks}
    if reward_relative:
        for key in pre_zone_licks.keys():
            feat = pre_zone_licks[key].copy() #this copy killed me. i was modifying the orig. ndarray
            feat[day_zone_obj.post_trials] = post_zone_licks[key][day_zone_obj.post_trials]
            features[key] = feat

            other_feat = np.zeros_like(feat)

            other_feat[day_zone_obj.post_trials] = pre_zone_licks[key][day_zone_obj.post_trials]
            other_feat[day_zone_obj.pre_trials] = post_zone_licks[key][day_zone_obj.pre_trials]

            features[key+'_other'] = other_feat

            features[key+'_third_zone'] = other_zone_licks[key]
    else:
        for key in pre_zone_licks.keys():
            feat = pre_zone_licks[key].copy() #this copy killed me. i was modifying the orig. ndarray
            
            features[key + '_pre'] = feat

            feat = post_zone_licks[key].copy() #this copy killed me. i was modifying the orig. ndarray
                        
            features[key + '_post'] = feat

            features[key+'_third_zone'] = other_zone_licks[key]

    
    features['start'] = get_licks_at_location(licks, [0, 30], bin_centers)['licks'].sum(axis = 1)
    features['end'] = get_licks_at_location(licks, [400, 450], bin_centers)['licks'].sum(axis = 1)


    features_df = pd.DataFrame.from_dict(features)
    if norm:
        features_df = features_df/n_licks[:,np.newaxis]
        features_df['n_licks'] = n_licks
    
    return features_df

def get_licks_at_location(licks, loc, bins):
    '''loc = [start, end]'''
    st, en = np.searchsorted(bins, loc)
    return {'licks':licks[:,st:en], 'bin_centers':bins[st:en]}

def get_unique_reward_zones_from_meta(df):
    '''take a meta dataframe and return an np_array of shape (n_zones, (start, end))'''
    rstarts = df.reward_zone_start.astype(int).unique()
    rends = df.reward_zone_end.astype(int).unique()
    if len(rstarts) != len(rends):
        raise ValueError('different number of unique reward_zone starts and reward_zone ends')
    else:
        out = np.zeros((len(rstarts), 2), dtype=int)
        out[:,0] = rstarts
        out[:,1] = rends
        return out


def load_sqlite(fpath, downcast = True):
    conn = sql.connect(fpath)
    df = pd.read_sql("SELECT * FROM data", conn)
    conn.close()
    if downcast:




        # downcast integers
        int_cols = df.select_dtypes(include='integer').columns
        df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast='integer')

        # downcast floats
        float_cols = df.select_dtypes(include='float').columns
        df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast='float')

    return df

def get_sqlite_fpath(sess):
    return os.path.join(sess['basedir_VR'], sess['vr_filename'])


def get_reward_zones_from_sqlite_file(fpath,
                                 reward_zone_center_col = 'currrewardcenter',
                                 reward_zone_width = 50):
    ''' Get reward zone positions and labels for a given session based on sqlite file.
    :param sess: session class

    :return: rz_coords: 2 x N array of zone [start, stop] positions by N trials
             rz_labels: 1 x N array of task-relevant zone label (i.e. 'A') by N trials'''


    df = load_sqlite(fpath)

    trials = df.drop_duplicates('trialnum')
    rzone_centers = trials[reward_zone_center_col].values

    rz_coords = np.repeat(rzone_centers[:,np.newaxis],2, axis = 1)

    rz_coords[:,0] = rz_coords[:,0] - reward_zone_width / 2
    rz_coords[:,1] = rz_coords[:,1] + reward_zone_width / 2


    rz_labels = trials['currrewardzone']

    return rz_coords, rz_labels.values