from dataclasses import dataclass, field
import numpy as np
import TwoPUtils

@dataclass
class ResultDB:
    animal: str #animal_ID





@dataclass
class LickMetrics:
    sess: TwoPUtils.sess.Session
    trials: np.ndarray
    lick_COM: np.ndarray
    lick_circ_var: np.ndarray
    params: dict



    def __post_init__(self):
        self.animal_id = self.sess.animal
        self.omit_trials = np.logical_not(self.sess['isreward'])
        
