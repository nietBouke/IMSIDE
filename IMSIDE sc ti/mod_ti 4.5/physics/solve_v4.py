# ============================================================================
# Solve here the equations for salinity
# with Newton-Raphson
# =============================================================================
import numpy as np
import scipy as sp
from scipy import optimize   , interpolate 
import time


def NewtonRaphson(self, init, version):
    
    sss=init
    #'''
    
    #do the first iteration
    #subtidal part
    sol_tot = self.solu_subtidal(sss, self.st_all , self.ii_all , version) \
            + self.solu_bnd_subtidal(sss, self.st_all , self.ii_all , version) 
            
    jac_tot = self.jaco_subtidal(sss, self.st_all , self.ii_all , version) \
            + self.jaco_bnd_subtidal(sss, self.st_all , self.ii_all , version)
            
    #tidal part 
    for i in range(len(self.tid_comp)):
        tid_set = self.tid_sets[self.tid_comp[i]]
        tid_geg = self.tid_gegs[self.tid_comp[i]]
        tid_inp = self.tidal_salinity(sss, tid_set, tid_geg)
        
        #add to solution vector
        sol_tot += self.solu_tidal(sss, tid_inp, tid_geg, self.ii_all , version)
        sol_tot += self.solu_bnd_tidal(sss, tid_inp, tid_geg, self.ii_all , version)
        #add to jacobian
        jac_tot += self.jaco_tidal(sss, tid_geg, self.ii_all , version)
        jac_tot += self.jaco_bnd_tidal(sss, tid_geg, self.ii_all , version)

    sss_n = init - sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(jac_tot),sol_tot)  # this is faster then np.linalg.solve (at least for a sufficiently large matrix)
    
    ''' 
    
    jac_tot = self.jaco_subtidal(sss, self.st_all , self.ii_all , version) \
            + self.jaco_bnd_subtidal(sss, self.st_all , self.ii_all , version)
            
    for i in range(len(self.tid_comp)):
        tid_set = self.tid_sets[self.tid_comp[i]]
        tid_geg = self.tid_gegs[self.tid_comp[i]]
        tid_inp = self.tidal_salinity(sss, tid_set, tid_geg)

        #add to jacobian
        jac_tot += self.jaco_tidal(sss, tid_inp, tid_geg, self.ii_all , version)
        jac_tot += self.jaco_bnd_tidal(sss, tid_inp, tid_geg, self.ii_all , version)

    def sol_func(ans):
        sol_tot = self.solu_subtidal(ans, self.st_all , self.ii_all , version) \
                + self.solu_bnd_subtidal(ans, self.st_all , self.ii_all , version) 

        for i in range(len(self.tid_comp)):
            tid_set = self.tid_sets[self.tid_comp[i]]
            tid_geg = self.tid_gegs[self.tid_comp[i]]
            tid_inp = self.tidal_salinity(ans, tid_set, tid_geg)
            
            #add to solution vector
            sol_tot += self.solu_tidal(ans, tid_inp, tid_geg, self.ii_all , version)
            sol_tot += self.solu_bnd_tidal(ans, tid_inp, tid_geg, self.ii_all , version)

        return sol_tot
        
    # =============================================================================
    # check jacobian. 
    # =============================================================================
    def jac_alg3(sol_h, ans, e = 1e-5): #numeriek berekende jacobiaan
        #sol_h is oplossingfunctie
        #ans is de oplossingsvector
        
        jac = np.zeros((len(ans),len(ans)))
        for i2 in range(len(ans)):
            ans_hpe,ans_hme = ans.copy(),ans.copy()
            ans_hpe[i2] = ans_hpe[i2] + e
            ans_hme[i2] = ans_hme[i2] - e
            
            jac[:,i2] = (sol_h(ans_hpe) - sol_h(ans_hme) )/(2*e)
            print(i2/len(ans))
            
        
        return jac

    jac_num = jac_alg3(sol_func, sss)
    
    print()
    print('difference is' , np.max(np.abs(jac_num-jac_tot)))
    
    #
    #for i in range(len(jac_num)):
    #    print(i,np.max(np.abs(jac_num[:,i])) , np.max(np.abs(jac_tot[:,i])))
    
            
    #print()
    #print('difference is' , np.max(np.abs(jac_num-jac_tot)))
    #print()
    #print(jac_num[36:39,45:55])
    #print(jac_tot[36:39,45:55])
    
    return
    #'''

    t=1
    print('That was iteration step ', t)
    
    # do the rest of the iterations
    while np.max(np.abs(sss-sss_n))>10e-6: #check whether the algoritm has converged
        sss = sss_n #update
        #new iteration
        
        #subtidal part
        sol_tot = self.solu_subtidal(sss, self.st_all , self.ii_all , version) \
                + self.solu_bnd_subtidal(sss, self.st_all , self.ii_all , version) 
                
        jac_tot = self.jaco_subtidal(sss, self.st_all , self.ii_all , version) \
                + self.jaco_bnd_subtidal(sss, self.st_all , self.ii_all , version)
                
        #tidal part 
        for i in range(len(self.tid_comp)):
            tid_set = self.tid_sets[self.tid_comp[i]]
            tid_geg = self.tid_gegs[self.tid_comp[i]]
            tid_inp = self.tidal_salinity(sss, tid_set, tid_geg)
            
            #add to solution vector
            sol_tot += self.solu_tidal(sss, tid_inp, tid_geg, self.ii_all , version)
            sol_tot += self.solu_bnd_tidal(sss, tid_inp, tid_geg, self.ii_all , version)
            #add to jacobian
            jac_tot += self.jaco_tidal(sss, tid_geg, self.ii_all , version)
            jac_tot += self.jaco_bnd_tidal(sss, tid_geg, self.ii_all , version)

        sss_n = sss - sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(jac_tot),sol_tot)  # this is faster then np.linalg.solve (at least for a sufficiently large matrix)
        
        #plot
        #import matplotlib.pyplot as plt
        #plt.plot(sss_n.reshape( self.di3[-1] , self.M)[:,0])
        #plt.show()
        
        
        t=1+t
        print('That was iteration step ', t)
  
        if t>=20: break
      
    return sss_n
    

def solve_eqs(self, version):
    tijd = time.time()
    
    # =============================================================================
    # initialize some stuff, was first in the runfile but placed it now here.     
    # =============================================================================
    #load functions
    self.ii_all = self.indices()

    #subtidal
    self.st_all = self.subtidal_module()

    #tidal
    self.tid_gegs, self.tid_sets = {} , {}
    for i in range(len(self.tid_comp)):
        self.tid_sets[self.tid_comp[i]] = {'tid_comp': self.tid_comp[i], 'tid_per': self.tid_per[i], 'a_tide': self.a_tide[i], 'p_tide': self.p_tide[i],
                   'Av_ti': self.Av_ti[i], 'Kv_ti': self.Kv_ti[i], 'sf_ti': self.sf_ti[i], 'rr_ti': self.rr_ti[i], 'Kh_ti': self.Kh_ti[i]}
        self.tid_gegs[self.tid_comp[i]] = self.tidal_module(self.tid_sets[self.tid_comp[i]])
        
    #print(self.tid_gegs)
        
    # =============================================================================
    # first solve the subtidal equations
    # =============================================================================
    init = np.zeros(self.M * self.di3[-1])
    
    out_notide = NewtonRaphson(self, init, 'A')
    out_tide   = NewtonRaphson(self, out_notide, version)
    #    out_tide = NewtonRaphson(self, init, version)
    
    print('Doing the simulation takes ', time.time()-tijd, ' seconds')
        
    return out_tide
    
