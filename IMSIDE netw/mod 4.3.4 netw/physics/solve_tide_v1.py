# =============================================================================
# module to solve the subtidal salinity balance in a general estuarine network
# model includes tides, with vertical advection, those are all taken into account 
# in the subtidal depth-averaged balance, but not in the the depth-perturbed balance
# at the junctions a boundary layer correction is applied, and in that manner salinity
# matches also at the tidal timescale. 
# =============================================================================

#import libraries
import numpy as np
import scipy as sp          #do fits
from scipy import optimize   , interpolate 
import time                              #measure time of operation
import matplotlib.pyplot as plt         


def run_model_onlytide(self):
    # =============================================================================
    # code to run the subtidal salintiy model
    # =============================================================================
    #print('Start the salinity calculation')

    #preparations
    self.tide_calc()
    self.indices()
    #self.subtidal_module()
    #self.prep_junc()
    
    # =============================================================================
    # first run the equilibrium simulation for the start of the simulation
    # =============================================================================
    #self.out = model_ti(self)
    self.out = np.zeros(self.ch_inds[self.ch_keys[-1]]['totx'][-1] * self.M)

















