# =============================================================================
# Time-dependent parts in the solution vector
# 
# =============================================================================

import numpy as np

def solu_timepart(self, ans_o, ans_n, indi, t):
    so = np.zeros(self.di3[-1]*self.M)

    so[indi['xn_m']]   = 1/self.dt[t]*ans_n[indi['xn_m']] - 1/self.dt[t]*ans_o[indi['xn_m']]    
    so[indi['xnr_mj']] = 1/(2*self.dt[t])*ans_n[indi['xnr_mj']] - 1/(2*self.dt[t])*ans_o[indi['xnr_mj']]

    return so

def jaco_timepart(self, ans_n, indi, t):
    jac = np.zeros((self.di3[-1]*self.M,self.di3[-1]*self.M))

    jac[indi['xn_m'],indi['xn_m']]     =  1/self.dt[t] 
    jac[indi['xnr_mj'],indi['xnr_mj']] =  1/(2*self.dt[t]) 

    return jac







