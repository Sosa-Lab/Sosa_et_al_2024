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
        dat = []
        for day, session in self.session_dict.items():

            rzones = behavior.get_reward_zones(session)[0]
            swap_zones = np.zeros((len(rzones), 2))
            omit_trials = ra.get_omission_trials(session)['trials']
            omit = np.zeros(shape = (len(rzones)), dtype = bool)
            omit[omit_trials] = True

            if len(dat) == 0:
                start = 0
            else:
                start = max(dat[-1].idx)+1

            idxs = np.arange(len(rzones), dtype = int)+start

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


            dat.append(pd.DataFrame(data = {'idx' : idxs, 
                                            'day' : [day]*len(rzones), 
                                            'trial' :  np.arange(0, len(rzones), 1), 
                                            'reward_zone_start': rzones[:,0], 
                                            'reward_zone_end': rzones[:,1],
                                            'swap_zone_start':swap_zones[:,0],
                                            'swap_zone_end':swap_zones[:,1],
                                            'trial_type':trial_type,
                                            'swap_type':type_swap, 
                                            'omit':omit} ))
        self.lookup_df = dat[0]
        for dat_day in dat[1:]:
            self.lookup_df = pd.concat([self.lookup_df, dat_day] ) 
    
    def add_session(self, day, session):
        rzones = behavior.get_reward_zones(session)[0]
        swap_zones = np.zeros((len(rzones), 2))
        omit_trials = ra.get_omission_trials(session)['trials']
        omit = np.zeros(shape = (len(rzones)), dtype = bool)
        omit[omit_trials] = True

        start = self.lookup_df.idx.max()+1

        idxs = np.arange(len(rzones), dtype = int)+start

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


        self.lookup_df = pd.concat([self.lookup_df, pd.DataFrame(data = {'idx' : idxs, 
                                        'day' : [day]*len(rzones), 
                                        'trial' :  np.arange(0, len(rzones), 1), 
                                        'reward_zone_start': rzones[:,0], 
                                        'reward_zone_end': rzones[:,1],
                                        'swap_zone_start':swap_zones[:,0],
                                        'swap_zone_end':swap_zones[:,1],
                                        'trial_type':trial_type,
                                        'swap_type':type_swap, 
                                        'omit':omit} )])
    




def correct_licks(session, correction_thr = 0.3):
    # get rid of trials where lick sensor may have gotten stuck on
    licks = np.copy(session.vr_data['lick'].values)
    licks, error_count = behavior.correct_lick_sensor_error(
        licks, session.trial_start_inds, session.teleport_inds, correction_thr=0.3)
    
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

def correlate_two_pop_vectors(pv1, pv2, pv1_name, pv2_name):
    '''take two pop vectors and their names, and return:
    corr_matrices --> list
    corr_names    --> list'''
    corr_mat = np.corrcoef(pv1, pv2)

    return [corr_mat[:450,:450],  corr_mat[450:, 450:], corr_mat[0:450, 450:],], [f'{pv1_name}', f'{pv2_name}', f'{pv1_name}_v_{pv2_name}']
