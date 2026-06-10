from reward_relative.path_dict_seahorse import path_dictionary as path_dict
from reward_relative import utilities as ut
from reward_relative import plotUtils as pt
from reward_relative import spatial
from reward_relative import placeCellPlot
from reward_relative import dayData as dd
from reward_relative import behavior
import pickle
# import dill

import numpy as np
import os 
import matplotlib.pyplot as plt
import TwoPUtils
import astropy
from TwoPUtils import spatial_analyses



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