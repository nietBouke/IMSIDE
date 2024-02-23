# ============================================================================
# Solve here the equations for salinity
# with Newton-Raphson
# =============================================================================
import numpy as np
import scipy as sp
from scipy import optimize   , interpolate 
import time
import copy

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
  
        if t>=20: 
            print('No solution found with normal parameters;')
            return [None]
      
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
                   'Av_ti': self.Av_ti[i], 'cv_ti': self.cv_ti[i], 'Kv_ti': self.Kv_ti[i], 'Sc_ti': self.Sc_ti[i], 'sf_ti': self.sf_ti[i], 'rr_ti': self.rr_ti[i], 'Kh_ti': self.Kh_ti[i]}
        self.tid_gegs[self.tid_comp[i]] = self.tidal_module(self.tid_sets[self.tid_comp[i]])
        
        
    # =============================================================================
    # first solve the subtidal equations
    # =============================================================================
    init = np.zeros(self.M * self.di3[-1])
    
    #out_notide = NewtonRaphson(self, init, 'A')
    #out_tide   = NewtonRaphson(self, out_notide, version)
    out_tide = NewtonRaphson(self, init, version)
    
    if out_tide[0] == None:
        print('Try parameter variation')
        
        n_guess = 6 #the number of iteration steps
        Kf_start = 10 #the factor with what to multiply the mixing 
        Kfac = np.linspace(Kf_start,1,n_guess)
        
        if self.choice_diffusivityh_st == 'constant': Kh0 = copy.deepcopy(self.Kh_st)             
        if self.choice_diffusivityh_st == 'cub':      ch0 = copy.deepcopy(self.ch_st)          

        #do the procedure again, with a beter approximation
        for sim in range(n_guess): 

            if self.choice_diffusivityh_st == 'constant': self.Kh_st = Kh0 * Kfac[sim]
            if self.choice_diffusivityh_st == 'cub':      self.ch_st = ch0 * Kfac[sim]

            #recalculate subtidal indices for this to have effect
            self.st_all = self.subtidal_module()
    
            #do the simulation
            if sim ==0 : out = NewtonRaphson(self, init, version)
            else: out = NewtonRaphson(self,out, version)
                
            if out[0] == None: #if this also not works, stop the calculation
                raise Exception("ABORT CALCULATION: Also with increased Kh no answer has been found. Check your input and think about \
                         if the model has a physical solution. If you think it has, you might wanna try increasing Kf_start or n_guess")   
                         
            print('Step ', sim, ' of ',n_guess-1,' is finished')
            
        out_tide = out.copy()
    
    print('Doing the simulation takes ', time.time()-tijd, ' seconds')
        
    return out_tide
    
