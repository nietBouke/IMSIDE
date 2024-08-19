# =============================================================================
# Time-dependent parts in the NR algoritm
# 
# =============================================================================

import numpy as np


def sol_inertia(self, key, ans_o, ans_n, dt):
    
    so = np.zeros(self.ch_inds[key]['di3'][-1]*self.M)
    inds = self.ch_inds[key].copy()

    so[inds['xn_m']]   = 1/dt     * ans_n[inds['xn_m']]   - 1/dt     * ans_o[inds['xn_m']]    
    so[inds['xnr_mj']] = 1/(2*dt) * ans_n[inds['xnr_mj']] - 1/(2*dt) * ans_o[inds['xnr_mj']]

    return so

def sol_inertia_o(self, key, ans_o, dt):
    
    so = np.zeros(self.ch_inds[key]['di3'][-1]*self.M)
    inds = self.ch_inds[key].copy()

    so[inds['xn_m']]   = - 1/dt     * ans_o[inds['xn_m']]    
    so[inds['xnr_mj']] = - 1/(2*dt) * ans_o[inds['xnr_mj']]

    return so

def sol_inertia_n(self, key, ans_n, dt):
    
    so = np.zeros(self.ch_inds[key]['di3'][-1]*self.M)
    inds = self.ch_inds[key].copy()

    so[inds['xn_m']]   = 1/dt     * ans_n[inds['xn_m']]  
    so[inds['xnr_mj']] = 1/(2*dt) * ans_n[inds['xnr_mj']] 

    return so


def jac_inertia(self, key, dt):
    
    jac = np.zeros((self.ch_inds[key]['di3'][-1]*self.M,self.ch_inds[key]['di3'][-1]*self.M))
    inds = self.ch_inds[key].copy()

    jac[inds['xn_m'],inds['xn_m']]     =  1/dt
    jac[inds['xnr_mj'],inds['xnr_mj']] =  1/(2*dt) 

    return jac







