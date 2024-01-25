# =============================================================================
# Solve here the equations for salinity
# with Newton-Raphson
# =============================================================================
import numpy as np
import scipy as sp
from scipy import optimize   , interpolate 
import time

def NewtonRaphson_ti(self, init, version, t):
    
    sss=init
    
    #do the first iteration
    #subtidal part
    sol_tot = self.solu_subtidal(sss, self.st_all , self.ii_all , version, t) \
            + self.solu_bnd_subtidal(sss, self.st_all , self.ii_all , version, t) 
            
    jac_tot = self.jaco_subtidal(sss, self.st_all , self.ii_all , version, t) \
            + self.jaco_bnd_subtidal(sss, self.st_all , self.ii_all , version, t)
            
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

    it=1
    print('That was iteration step ', it)
    
    # do the rest of the iterations
    while np.max(np.abs(sss-sss_n))>10e-6: #check whether the algoritm has converged
        sss = sss_n #update
        #new iteration
        
        #subtidal part
        sol_tot = self.solu_subtidal(sss, self.st_all , self.ii_all , version, t) \
                + self.solu_bnd_subtidal(sss, self.st_all , self.ii_all , version, t) 
                
        jac_tot = self.jaco_subtidal(sss, self.st_all , self.ii_all , version, t) \
                + self.jaco_bnd_subtidal(sss, self.st_all , self.ii_all , version, t)
                
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
        
        it=1+it
        print('That was iteration step ', it)
  
        if it>=20: break
      
    return sss_n

    

def NewtonRaphson_td(self, init, version):
    sss_o, sss_n = init, init #initialize
    sss_save = [] 
    
    #the time loop
    for t2 in range(self.T):         
        
        #do the first iteration
        #subtidal part
        sol_tot = (1-self.theta) * self.solu_subtidal(sss_o, self.st_all , self.ii_all , version, t2) \
                + self.theta * self.solu_subtidal(sss_n, self.st_all , self.ii_all , version, t2) \
                + self.solu_bnd_subtidal(sss_n, self.st_all , self.ii_all , version, t2) \
                + self.solu_timepart(sss_o, sss_n, self.ii_all, t2)
                
        jac_tot = self.theta * self.jaco_subtidal(sss_n, self.st_all , self.ii_all , version, t2) \
                + self.jaco_bnd_subtidal(sss_n, self.st_all , self.ii_all , version, t2) \
                + self.jaco_timepart(sss_n, self.ii_all, t2)
                
                
                
        #tidal part 
        for i in range(len(self.tid_comp)):
            tid_set = self.tid_sets[self.tid_comp[i]]
            tid_geg = self.tid_gegs[self.tid_comp[i]]
            tid_inp_o = self.tidal_salinity(sss_o, tid_set, tid_geg)
            tid_inp_n = self.tidal_salinity(sss_n, tid_set, tid_geg)
            
            
            #add to solution vector
            sol_tot += (1-self.theta) * self.solu_tidal(sss_o, tid_inp_o, tid_geg, self.ii_all , version)
            sol_tot += self.theta * self.solu_tidal(sss_n, tid_inp_n, tid_geg, self.ii_all , version)
            sol_tot += self.solu_bnd_tidal(sss_n, tid_inp_n, tid_geg, self.ii_all , version)

            #add to jacobian
            jac_tot += self.theta * self.jaco_tidal(sss_n, tid_geg, self.ii_all , version)
            jac_tot += self.jaco_bnd_tidal(sss_n, tid_geg, self.ii_all , version)      
                
                
        #do iteration step
        sss_i =sss_n - sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(jac_tot),sol_tot)  #solve the matrix equation and use this for Newton Rapshon
        it=1 #count the number of iterations

        while np.max(np.abs(sss_i-sss_n))>1e-6: #check whether the algoritm has converged
            sss_n = sss_i #update
            
            #build solution vector and jacobian matrix
            sol_tot = (1-self.theta) * self.solu_subtidal(sss_o, self.st_all , self.ii_all , version, t2) \
                    + self.theta * self.solu_subtidal(sss_n, self.st_all , self.ii_all , version, t2) \
                    + self.solu_bnd_subtidal(sss_n, self.st_all , self.ii_all , version, t2) \
                    + self.solu_timepart(sss_o, sss_n, self.ii_all, t2)
                    
            jac_tot = self.theta * self.jaco_subtidal(sss_n, self.st_all , self.ii_all , version, t2) \
                    + self.jaco_bnd_subtidal(sss_n, self.st_all , self.ii_all , version, t2) \
                    + self.jaco_timepart(sss_n, self.ii_all, t2)
                    
                    
                    
            #tidal part 
            for i in range(len(self.tid_comp)):
                tid_set = self.tid_sets[self.tid_comp[i]]
                tid_geg = self.tid_gegs[self.tid_comp[i]]
                #tid_inp_o = self.tidal_salinity(sss_o, tid_set, tid_geg)
                tid_inp_n = self.tidal_salinity(sss_n, tid_set, tid_geg)
                
                
                #add to solution vector
                sol_tot += (1-self.theta) * self.solu_tidal(sss_o, tid_inp_o, tid_geg, self.ii_all , version)
                sol_tot += self.theta * self.solu_tidal(sss_n, tid_inp_n, tid_geg, self.ii_all , version)
                sol_tot += self.solu_bnd_tidal(sss_n, tid_inp_n, tid_geg, self.ii_all , version)
    
                #add to jacobian
                jac_tot += self.theta * self.jaco_tidal(sss_n, tid_geg, self.ii_all , version)
                jac_tot += self.jaco_bnd_tidal(sss_n, tid_geg, self.ii_all , version)      
                
                
            #do iteration step
            sss_i = sss_n - sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(jac_tot),sol_tot)  #solve the matrix equation and use this for Newton Rapshon
            it    = 1+it #one more iteration has been done
        
            if it>=10: break #if it does not converge within 10 iteration steps, something is wrong
        
        
        if it<10: #if salinity is above its minimum value and we have had less than 10 iteration steps, the time step was succesfull and we can continue
            print('Timestep', t2+1 , 'of total of', self.T, 'timesteps is finished. ') 
                
        else: #if no convergence or too low salinity, are we going to use a smaller time step
            sss_i = solve_ats(sss_o, version, t2)
            print('Timestep', t2+1 , 'of total of', self.T, 'timesteps is finished. ') 

        sss_o, sss_n =sss_i, sss_i #update for next timestep
        sss_save.append(sss_n)
           
    return sss_save 


def solve_ats(init, version, t):
    # =============================================================================
    # adaptive time step
    # =============================================================================
    reps = 2
    out_ats = [['NO SOLUTION']]
    import settings
    from core_v4      import mod1c_g4

    while out_ats[0][0] == 'NO SOLUTION' :
        
        #repeat the forcing parameters
        forc_pars_ats = list(settings.forc_pars)
        
        forc_pars_ats[0] = (reps , np.repeat(settings.forc_pars[0][1][t] , reps) / reps) #timestep
        forc_pars_ats[1] = (np.repeat(settings.forc_pars[1][0][t] , reps) , np.repeat(settings.forc_pars[1][1][t] , reps) , np.repeat(settings.forc_pars[1][2][t] , reps)) #Q, soc, sri
        forc_pars_ats = tuple(forc_pars_ats)
                
        #define temporary object
        run_ats = mod1c_g4(settings.constants, settings.phys_pars, settings.geo_pars, forc_pars_ats)
    
        #load functions
        run_ats.ii_all = run_ats.indices()
    
        #subtidal
        run_ats.st_all = run_ats.subtidal_module()
    
        #tidal
        run_ats.tid_gegs, run_ats.tid_sets = {} , {}
        for i in range(len(run_ats.tid_comp)):
            run_ats.tid_sets[run_ats.tid_comp[i]] = {'tid_comp': run_ats.tid_comp[i], 'tid_per': run_ats.tid_per[i], 'a_tide': run_ats.a_tide[i], 'p_tide': run_ats.p_tide[i],
                       'Av_ti': run_ats.Av_ti[i], 'Kv_ti': run_ats.Kv_ti[i], 'sf_ti': run_ats.sf_ti[i], 'rr_ti': run_ats.rr_ti[i], 'Kh_ti': run_ats.Kh_ti[i]}
            run_ats.tid_gegs[run_ats.tid_comp[i]] = run_ats.tidal_module(run_ats.tid_sets[run_ats.tid_comp[i]])
        
        #solve equations
        out_ats = NewtonRaphson_ats(run_ats, init, version)
        
        
        if reps > 65: 
            raise Exception('Also adaptive time step did not find a solution. You have a problem sir')
            
        reps = reps * 2 

    return out_ats[-1] 


def NewtonRaphson_ats(self, init, version):
    # =============================================================================
    # adaptive time step
    # =============================================================================
    sss_o, sss_n = init, init #initialize
    sss_save = [] 
    
    #the time loop
    for t2 in range(self.T): 
        #do the first iteration
        #subtidal part
        sol_tot = (1-self.theta) * self.solu_subtidal(sss_o, self.st_all , self.ii_all , version, t2) \
                + self.theta * self.solu_subtidal(sss_n, self.st_all , self.ii_all , version, t2) \
                + self.solu_bnd_subtidal(sss_n, self.st_all , self.ii_all , version, t2) \
                + self.solu_timepart(sss_o, sss_n, self.ii_all, t2)
                
        jac_tot = self.theta * self.jaco_subtidal(sss_n, self.st_all , self.ii_all , version, t2) \
                + self.jaco_bnd_subtidal(sss_n, self.st_all , self.ii_all , version, t2) \
                + self.jaco_timepart(sss_n, self.ii_all, t2)
                
                
                
        #tidal part 
        for i in range(len(self.tid_comp)):
            tid_set = self.tid_sets[self.tid_comp[i]]
            tid_geg = self.tid_gegs[self.tid_comp[i]]
            tid_inp_o = self.tidal_salinity(sss_o, tid_set, tid_geg)
            tid_inp_n = self.tidal_salinity(sss_n, tid_set, tid_geg)
            
            
            #add to solution vector
            sol_tot += (1-self.theta) * self.solu_tidal(sss_o, tid_inp_o, tid_geg, self.ii_all , version)
            sol_tot += self.theta * self.solu_tidal(sss_n, tid_inp_n, tid_geg, self.ii_all , version)
            sol_tot += self.solu_bnd_tidal(sss_n, tid_inp_n, tid_geg, self.ii_all , version)

            #add to jacobian
            jac_tot += self.theta * self.jaco_tidal(sss_n, tid_geg, self.ii_all , version)
            jac_tot += self.jaco_bnd_tidal(sss_n, tid_geg, self.ii_all , version)      
                
                
        #do iteration step
        sss_i =sss_n - sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(jac_tot),sol_tot)  #solve the matrix equation and use this for Newton Rapshon
        it=1 #count the number of iterations

        while np.max(np.abs(sss_i-sss_n))>1e-6: #check whether the algoritm has converged
            sss_n = sss_i #update
            
            #build solution vector and jacobian matrix
            sol_tot = (1-self.theta) * self.solu_subtidal(sss_o, self.st_all , self.ii_all , version, t2) \
                    + self.theta * self.solu_subtidal(sss_n, self.st_all , self.ii_all , version, t2) \
                    + self.solu_bnd_subtidal(sss_n, self.st_all , self.ii_all , version, t2) \
                    + self.solu_timepart(sss_o, sss_n, self.ii_all, t2)
                    
            jac_tot = self.theta * self.jaco_subtidal(sss_n, self.st_all , self.ii_all , version, t2) \
                    + self.jaco_bnd_subtidal(sss_n, self.st_all , self.ii_all , version, t2) \
                    + self.jaco_timepart(sss_n, self.ii_all, t2)
                    
                    
                    
            #tidal part 
            for i in range(len(self.tid_comp)):
                tid_set = self.tid_sets[self.tid_comp[i]]
                tid_geg = self.tid_gegs[self.tid_comp[i]]
                #tid_inp_o = self.tidal_salinity(sss_o, tid_set, tid_geg)
                tid_inp_n = self.tidal_salinity(sss_n, tid_set, tid_geg)
                
                
                #add to solution vector
                sol_tot += (1-self.theta) * self.solu_tidal(sss_o, tid_inp_o, tid_geg, self.ii_all , version)
                sol_tot += self.theta * self.solu_tidal(sss_n, tid_inp_n, tid_geg, self.ii_all , version)
                sol_tot += self.solu_bnd_tidal(sss_n, tid_inp_n, tid_geg, self.ii_all , version)
    
                #add to jacobian
                jac_tot += self.theta * self.jaco_tidal(sss_n, tid_geg, self.ii_all , version)
                jac_tot += self.jaco_bnd_tidal(sss_n, tid_geg, self.ii_all , version)
        
            #do iteration step
            sss_i =sss_n - sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(jac_tot),sol_tot)  #solve the matrix equation and use this for Newton Rapshon
            it=1+it #one more iteration has been done
        
            if it>=10: break #if it does not converge within 10 iteration steps, something is wrong
        
        
        if it<10: #if salinity is above its minimum value and we have had less than 10 iteration steps, the time step was succesfull and we can continue
            print('Timestep', t2+1 , 'of total of ', self.T, ' adaptive timesteps is finished. ') 
                
        else: #if no convergence or too low salinity, are we going to use a smaller time step
            #print('Adaptive time step did not find a solution, please decrease the time step  ') 
            return [['NO SOLUTION']]
        
        sss_o, sss_n =sss_i, sss_i #update for next timestep
        sss_save.append(sss_n)
           
    return sss_save 



def solve_eqs(self, version):
    tijd = time.time()
    
    # =============================================================================
    #     #load functions
    # =============================================================================
    self.ii_all = self.indices()

    #subtidal
    self.st_all = self.subtidal_module()

    #tidal
    self.tid_gegs, self.tid_sets = {} , {}
    for i in range(len(self.tid_comp)):
        self.tid_sets[self.tid_comp[i]] = {'tid_comp': self.tid_comp[i], 'tid_per': self.tid_per[i], 'a_tide': self.a_tide[i], 'p_tide': self.p_tide[i],
                   'Av_ti': self.Av_ti[i], 'Kv_ti': self.Kv_ti[i], 'sf_ti': self.sf_ti[i], 'rr_ti': self.rr_ti[i], 'Kh_ti': self.Kh_ti[i]}
        self.tid_gegs[self.tid_comp[i]] = self.tidal_module(self.tid_sets[self.tid_comp[i]])
        
    # =============================================================================
    # first solve the subtidal equations
    # =============================================================================
    init = np.zeros(self.M * self.di3[-1])
    
    out_notide = NewtonRaphson_ti(self, init, 'A', 0)
    out_tide   = NewtonRaphson_ti(self, out_notide, version, 0)
    out_time   = NewtonRaphson_td(self, out_tide, version)
    
    print('Doing the simulation takes ', time.time()-tijd, ' seconds')
        
    return out_time
    




