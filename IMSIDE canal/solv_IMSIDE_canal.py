# =============================================================================
# Solve equations for IMSIDE canal
# =============================================================================
import numpy as np
import scipy as sp
from tqdm import trange


def do_time_integration(self, init, crit = 1e-6, itmax = 10):
    # =============================================================================
    # integration of the set of equations in time. 
    # Uses central differences horizontally, spectral method in vertical, 
    # Crank-Nicolson for time integration, Newton-Rapshon as root finding
    # =============================================================================

    # sss_o, sss_n = init*self.soc[0]/self.soc_sca, init*self.soc[0]/self.soc_sca #initialize
    sss_o, sss_n = init, init  #initialize
    sss_save = [] 

    #for t2 in range(self.T): #do the integration in time    - without tqdm library
    for t2 in trange(self.T): #do the integration in time
        #do the first iteration step of the Newton Raphson
        jaco, solu = self.build_jac(sss_n, t2) , self.build_sol(sss_n, sss_o , t2) #calculation of jacobian and solution vector
        sss_i =sss_n - sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(jaco),solu)  #solve the matrix equation and use this for Newton Rapshon
        it=1 #count the number of iterations

        # do the rest of the iterations
        while np.max(np.abs(sss_i-sss_n))>crit: #check whether the algoritm has converged
            sss_n = sss_i #update
            
            jaco, solu = self.build_jac(sss_n, t2) , self.build_sol(sss_n, sss_o, t2) #calculation of jacobian and solution vector
            sss_i =sss_n - sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(jaco),solu)  #solve the matrix equation and use this for Newton Rapshon
            it+=1 #one more iteration has been done

            if it>=itmax: 
                break #if it does not converge within 10 iteration steps, something is wrong
                raise Exception('Newton-Raphson algoritm does not converge. Please check your input values. Too coarse resolution?')
        
        #print('Timestep', t2+1 , 'of total of', T, 'timesteps is finished. ') 

        sss_o, sss_n =sss_i, sss_i #update for next timestep
        sss_save.append(sss_n) #, salt_mm.append([np.min(s),np.max(s)])
        
    self.sss_save = sss_save
    return 

def run_func(self, init, setup = False):
    # =============================================================================
    # function to run the model
    # =============================================================================
    #prepare
    self.build_indices()
    self.build_coeffs()
    
    #do time integration
    if setup: self.do_time_integration(np.zeros(self.di[-1]*self.M))
    else: self.do_time_integration(init)
    
    
    

