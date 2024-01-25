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


def prep_jac_ti(self, pars_Q):
    # =============================================================================
    # build the non-changing part (over an iteration) of the jacobian for the NR algoritm
    # for the time-independent model 
    # =============================================================================
    
    #initialise
    jac_tot = np.zeros((self.ch_inds[self.ch_keys[-1]]['totx'][-1] * self.M,self.ch_inds[self.ch_keys[-1]]['totx'][-1] * self.M))
    
    #for all channels 
    def func_ch_soljac(key):
        # =============================================================================
        # Jacobian matrix
        # =============================================================================
        #run subtidal module
        ch_jac_st = self.jac_subtidal_fix(key, pars_Q)
        #run tidal module
        ch_jac_ti = self.jac_tidal_fix(key)
        #run boundaries
        ch_jac_bnd = self.jac_bound_fix(key, pars_Q)

        ch_jac = ch_jac_st + ch_jac_ti + ch_jac_bnd

        return key, ch_jac

    #run for all the channels
    chs = list(map(func_ch_soljac, self.ch_keys))
    #try paralellisation
    #chs = Parallel(n_jobs=3)(delayed(func_ch_soljac)(key) for key in self.ch_keys)
    #chs = dask.compute(*[dask.delayed(func_ch_soljac)(key) for key in self.ch_keys])

    #add to a large matrix
    for k in range(len(self.ch_keys)):
        key = chs[k][0]
        jac_tot[self.ch_inds[key]['totx'][0]*self.M:self.ch_inds[key]['totx'][-1]*self.M , self.ch_inds[key]['totx'][0]*self.M:self.ch_inds[key]['totx'][-1]*self.M] += chs[k][1]

    #run junctions
    jac_tot += self.jac_junc_tot_fix(self.junc_gegs, pars_Q)
    
    return jac_tot



def build_all_ti(self, ans, pars_Q, pars_s):
    # =============================================================================
    # build the solution vector and corresponding jacobian for the NR algoritm
    # for the time-independent model 
    # =============================================================================
    
    #initialise
    tid_inps = {}
    sol_tot = np.zeros(self.ch_inds[self.ch_keys[-1]]['totx'][-1] * self.M)
    jac_tot = np.zeros((self.ch_inds[self.ch_keys[-1]]['totx'][-1] * self.M,self.ch_inds[self.ch_keys[-1]]['totx'][-1] * self.M))

    
    #for all channels 
    def func_ch_soljac(key):
        # =============================================================================
        # Solution vector
        # =============================================================================
        #select relevant part of answer vector
        ch_ans = ans[self.ch_inds[key]['totx'][0]*self.M:self.ch_inds[key]['totx'][-1]*self.M]
        #prepare tides
        tid_inp = self.tidal_salinity(key,ch_ans)

        #run subtidal module
        ch_sol_st = self.sol_subtidal(key, ch_ans, pars_Q)
        #run tidal module
        ch_sol_ti = self.sol_tidal(key,tid_inp)
        #run boundaries
        ch_sol_bnd = self.sol_bound(key, ch_ans ,tid_inp, pars_Q, pars_s) 

        ch_sol = ch_sol_st + ch_sol_ti + ch_sol_bnd

        # =============================================================================
        # Jacobian matrix
        # =============================================================================
        #run subtidal module
        ch_jac_st = self.jac_subtidal_vary(key, ch_ans, pars_Q)
        #run tidal module
        ch_jac_ti = self.jac_tidal_vary(key)
        #run boundaries
        ch_jac_bnd = self.jac_bound_vary(key, ch_ans ,tid_inp, pars_Q)

        ch_jac = ch_jac_st + ch_jac_ti + ch_jac_bnd

        return key, tid_inp, ch_sol, ch_jac

    
    #initialise the large matrix 
    
    #run for all the channels
    chs = list(map(func_ch_soljac, self.ch_keys))
    #try paralellisation
    #chs = Parallel(n_jobs=3)(delayed(func_ch_soljac)(key) for key in self.ch_keys)
    #chs = dask.compute(*[dask.delayed(func_ch_soljac)(key) for key in self.ch_keys])

    #add to a large matrix
    for k in range(len(self.ch_keys)):
        key = chs[k][0]
        tid_inps[key] = chs[k][1] #save tides for junctions
        sol_tot[self.ch_inds[key]['totx'][0]*self.M:self.ch_inds[key]['totx'][-1]*self.M] += chs[k][2]
        jac_tot[self.ch_inds[key]['totx'][0]*self.M:self.ch_inds[key]['totx'][-1]*self.M , self.ch_inds[key]['totx'][0]*self.M:self.ch_inds[key]['totx'][-1]*self.M] += chs[k][3]

    #run junctions
    sol_tot += self.sol_junc_tot(ans, self.junc_gegs, tid_inps, pars_Q)
    jac_tot += self.jac_junc_tot_vary(ans, self.junc_gegs, tid_inps, pars_Q)
    
    return sol_tot, jac_tot




def build_all_n_td(self, ans_n, pars_Q, pars_s, dt):
    # =============================================================================
    # build the solution vector and corresponding jacobian for the NR algoritm
    # for the time-independent model 
    # =============================================================================
        
    #initialise
    tid_inps = {}
    sol_tot = np.zeros(self.ch_inds[self.ch_keys[-1]]['totx'][-1] * self.M)
    jac_tot = np.zeros((self.ch_inds[self.ch_keys[-1]]['totx'][-1] * self.M,self.ch_inds[self.ch_keys[-1]]['totx'][-1] * self.M))

    #for all channels 
    def func_ch_soljac(key):
        # =============================================================================
        # Solution vector
        # =============================================================================
        #select relevant part of ans_nwer vector
        ch_ans_n = ans_n[self.ch_inds[key]['totx'][0]*self.M:self.ch_inds[key]['totx'][-1]*self.M]
        #prepare tides
        tid_inp = self.tidal_salinity(key,ch_ans_n)

        #run subtidal module
        ch_sol_st = self.sol_subtidal(key, ch_ans_n, pars_Q)
        #run tidal module
        ch_sol_ti = self.sol_tidal(key,tid_inp)
        #run boundaries
        ch_sol_bnd = self.sol_bound(key, ch_ans_n ,tid_inp, pars_Q, pars_s) 
        #run inertia
        ch_sols_in = self.sol_inertia_n(key, ch_ans_n, dt)
        
        ch_sol = ch_sol_st + ch_sol_ti + ch_sol_bnd + ch_sols_in

        # =============================================================================
        # Jacobian matrix
        # =============================================================================
        #run subtidal module
        ch_jac_st = self.jac_subtidal_vary(key, ch_ans_n, pars_Q)
        #run tidal module
        ch_jac_ti = self.jac_tidal_vary(key)
        #run boundaries
        ch_jac_bnd = self.jac_bound_vary(key, ch_ans_n ,tid_inp, pars_Q)
        #run inertia - we could place this somewhere else, but it is fast anyway. 
        ch_jacs_in = self.jac_inertia(key, dt)
        
        ch_jac = ch_jac_st + ch_jac_ti + ch_jac_bnd + ch_jacs_in

        return key, tid_inp, ch_sol, ch_jac

    
    #initialise the large matrix 
    
    #run for all the channels
    chs = list(map(func_ch_soljac, self.ch_keys))
    #try paralellisation
    #chs = Parallel(n_jobs=3)(delayed(func_ch_soljac)(key) for key in self.ch_keys)
    #chs = dask.compute(*[dask.delayed(func_ch_soljac)(key) for key in self.ch_keys])

    #add to a large matrix
    for k in range(len(self.ch_keys)):
        key = chs[k][0]
        tid_inps[key] = chs[k][1] #save tides for junctions
        sol_tot[self.ch_inds[key]['totx'][0]*self.M:self.ch_inds[key]['totx'][-1]*self.M] += chs[k][2]
        jac_tot[self.ch_inds[key]['totx'][0]*self.M:self.ch_inds[key]['totx'][-1]*self.M , self.ch_inds[key]['totx'][0]*self.M:self.ch_inds[key]['totx'][-1]*self.M] += chs[k][3]

    #run junctions
    sol_tot += self.sol_junc_tot(ans_n, self.junc_gegs, tid_inps, pars_Q)
    jac_tot += self.jac_junc_tot_vary(ans_n, self.junc_gegs, tid_inps, pars_Q)
    
    return sol_tot, jac_tot


def build_sol_o_td(self, ans_o, pars_Q, pars_s, dt):
    # =============================================================================
    # build the solution vector and corresponding jacobian for the NR algoritm
    # for the time-independent model 
    # =============================================================================
        
    #decompose for different channels
    ch_ans_o , tid_inp_o = {} , {} 
    ch_sols_st_o, ch_sols_ti_o , ch_sols_in = {} , {} , {}
    
    for key in self.ch_keys: 
        # =============================================================================
        # Solution vector
        # =============================================================================
        #select relevant part of answer vector
        ch_ans_o[key] = ans_o[self.ch_inds[key]['totx'][0]*self.M:self.ch_inds[key]['totx'][-1]*self.M]
        #prepare tides
        tid_inp_o[key] = self.tidal_salinity(key, ch_ans_o[key])

        #run subtidal module
        ch_sols_st_o[key] = self.sol_subtidal(key, ch_ans_o[key], pars_Q)
        #run tidal module
        ch_sols_ti_o[key] = self.sol_tidal(key, tid_inp_o[key])
        #run inertia
        ch_sols_in[key] = self.sol_inertia_o(key, ch_ans_o[key], dt)
        
    #run junctions
    sol_tot = np.zeros(self.ch_inds[self.ch_keys[-1]]['totx'][-1] * self.M)
    
    #build complete solution vector and jacobian
    for key in self.ch_keys: 
        #solution vector        
        sol_tot[self.ch_inds[key]['totx'][0]*self.M:self.ch_inds[key]['totx'][-1]*self.M] += (1-self.theta) * (ch_sols_st_o[key] + ch_sols_ti_o[key]) + ch_sols_in[key]
                    
    return sol_tot 


def NR_ti(self, init, pars_Q, pars_s, prt = False):#, init_method = 'from_notide'):
    # =============================================================================
    # run the model, i.e. solve Newton-Raphson algoritm
    # =============================================================================
    #start
    sss = init
    jac_prep = prep_jac_ti(self, pars_Q)

    #do the first iteration step
    solu,jaco = build_all_ti(self, sss, pars_Q, pars_s)
    jaco +=jac_prep
    sss_n = sss - sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(jaco),solu)  # this is faster then np.linalg.solve (at least for a sufficiently large matrix) 
    
    it=1
    if prt==True: print('That was iteration step ', it)

    # do the rest of the iterations
    while np.max(np.abs(sss-sss_n))>10e-6: #check whether the algoritm has converged
        sss = sss_n.copy() #update
        #sss = sss_n.copy()*0.5 + sss * 0.5 #update
        solu,jaco = build_all_ti(self, sss, pars_Q, pars_s)
        jaco+=jac_prep
        sss_n = sss - sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(jaco),solu)  # this is faster then np.linalg.solve (at least for a sufficiently large matrix)

        #plotting
        '''
        plt.plot(sss.reshape((int(len(sss)/self.M),self.M))[:,0])#-sss.reshape((int(len(sss)/self.M),self.M))[:,0])
        plt.plot(sss.reshape((int(len(sss)/self.M),self.M))[:,1])
        plt.plot(sss.reshape((int(len(sss)/self.M),self.M))[:,2])
        plt.show()
        '''
        it+=1
        if prt==True: print('That was iteration step ', it)
    
        if it>=20: break
    
    if it<20:
        print('The time-independent algoritm has converged \n')  
    else:
        print('ERROR: no convergence')
        return [None]

    return sss


def model_ti(self):
    # =============================================================================
    # run the time-independent model 
    # =============================================================================
    
    tijd = time.time()

    #quantities for first timestep    
    Qr_inp = self.Qriv[:,0]  if len(self.Qriv)  >0 else []
    Qw_inp = self.Qweir[:,0] if len(self.Qweir) >0 else []
    Qh_inp = self.Qhar[:,0]  if len(self.Qhar)  >0 else []
    ns_inp = self.n_sea[:,0]  if len(self.n_sea)  >0 else []
    Qnow = self.Qdist_calc((Qr_inp , Qw_inp , Qh_inp, ns_inp))
    
    sr_inp = self.sri[:,0] if len(self.sri) >0 else []
    sw_inp = self.swe[:,0] if len(self.swe) >0 else []
    so_inp = self.soc[:,0] if len(self.soc) >0 else []
    snow = (sr_inp , sw_inp , so_inp)
    
    #spin up from subtidal model. Lets hope that does the job. 
    init_all = np.zeros(self.ch_inds[self.ch_keys[-1]]['totx'][-1] * self.M)
   
    out = NR_ti(self, init_all, Qnow, snow)
    #if this did not work, spin up from other state. 
    if out[0] == None:
        n_guess = 5 #the number of iteration steps
        Kf_start = 5 #the factor with what to multiply the mixing 
        Kfac = np.linspace(Kf_start,1,n_guess)

        #do the procedure again, with a beter approximation
        for sim in range(n_guess): 
            #choose a higher value for Kh
            count=0
            for key in self.ch_keys: 
                #formulation depends on how Kh is formulated. - for now we have one formulation 
                    
                self.ch_pars[key]['Kh'] = self.Kh_st * Kfac[sim] + np.zeros(self.ch_pars[key]['di'][-1]) #horizontal mixing coefficient
                #add the increase of Kh in the adjacent sea domain
                if self.ch_gegs[key]['loc x=-L'][0] == 's' : self.ch_pars[key]['Kh'][:self.ch_pars[key]['di'][1]] = self.ch_pars[key]['Kh'][self.ch_pars[key]['di'][1]] \
                                                                * self.ch_pars[key]['b'][:self.ch_pars[key]['di'][1]]/self.ch_pars[key]['b'][self.ch_pars[key]['di'][1]]
                if self.ch_gegs[key]['loc x=0'][0] == 's': self.ch_pars[key]['Kh'][self.ch_pars[key]['di'][-2]:] = self.ch_pars[key]['Kh'][self.ch_pars[key]['di'][-2]] \
                                                                * self.ch_pars[key]['b'][self.ch_pars[key]['di'][-2]:]/self.ch_pars[key]['b'][self.ch_pars[key]['di'][-2]]
                #recalculate subtidal indices for this to have effect
                self.subtidal_module()

                count+=1

            #do the simulation
            if sim ==0 : out = NR_ti(self, init_all, Qnow, snow)
            else: out = NR_ti(self,out, Qnow, snow)
                
            if out[0] == None: #if this also not works, stop the calculation
                raise Exception("ABORT CALCULATION: Also with increased Kh no answer has been found. Check your input and think about \
                         if the model has a physical solution. If you think it has, you might wanna try increasing Kf_start or n_guess")   
                         
            print('Step ', sim, ' of ',n_guess-1,' is finished')


    print('The time-independent simulation (initialisation) takes ', time.time() - tijd ,' seconds')
    
    return out



def NR_td(self, ini, pars_Q, pars_s,dt):#, init_method = 'from_notide'):
    # =============================================================================
    # run the model, i.e. solve Newton-Raphson algoritm
    # =============================================================================
    #start
    sss_o, sss_n = ini, ini #initialize
    sol_old = build_sol_o_td(self, sss_o, pars_Q, pars_s, dt)
    jac_prep = prep_jac_ti(self, pars_Q)

    #do the first iteration step
    solu,jaco = build_all_n_td(self, sss_n, pars_Q, pars_s, dt)    
    solu += sol_old
    jaco += jac_prep
    sss_i = sss_n - sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(jaco),solu)  # this is faster then np.linalg.solve (at least for a sufficiently large matrix) 

    it=1
    #print('That was iteration step ', it)

    # do the rest of the iterations
    while np.max(np.abs(sss_i-sss_n))>10e-6: #check whether the algoritm has converged
        sss_n = sss_i.copy() #update
        #sss = sss_n.copy()*0.5 + sss * 0.5 #update
        solu,jaco = build_all_n_td(self, sss_n, pars_Q, pars_s, dt)    
        solu += sol_old 
        jaco += jac_prep
        sss_i = sss_n - sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(jaco),solu)  # this is faster then np.linalg.solve (at least for a sufficiently large matrix)

        #plotting
        '''
        plt.title(it)
        plt.plot(sss_i.reshape((int(len(sss_i)/self.M),self.M))[:,0])#-sss_i.reshape((int(len(sss_i)/self.M),self.M))[:,0])
        plt.plot(sss_i.reshape((int(len(sss_i)/self.M),self.M))[:,1])
        plt.plot(sss_i.reshape((int(len(sss_i)/self.M),self.M))[:,2])
        plt.show()
        #'''
        it+=1
        #print('That was iteration step ', it)
    
        if it>=10: break
    
    if it>=10:
        raise Exception("ABORT CALCULATION: No solution found. Probably large steps in discharge. Try running with e.g. a smoother discharge, finer grid, smaller timestep. ")   


    return sss_n


def model_td(self, init):
    # =============================================================================
    # run the time-dependent model 
    # =============================================================================
    
    tijd = time.time()
    out = [init]
    
    for tt in range(1,self.T):
        #quantities for this timestep 
        Qr_inp = self.Qriv[:,tt]  if len(self.Qriv)  >0 else []
        Qw_inp = self.Qweir[:,tt] if len(self.Qweir) >0 else []
        Qh_inp = self.Qhar[:,tt]  if len(self.Qhar)  >0 else []
        ns_inp = self.n_sea[:,tt]  if len(self.n_sea)  >0 else []
        Qnow   = self.Qdist_calc((Qr_inp , Qw_inp , Qh_inp, ns_inp))
        
        sr_inp = self.sri[:,tt] if len(self.sri) >0 else []
        sw_inp = self.swe[:,tt] if len(self.swe) >0 else []
        so_inp = self.soc[:,tt] if len(self.soc) >0 else []
        snow = (sr_inp , sw_inp , so_inp)
        dt   = self.dt[tt]
        #run this timestep
        temp = NR_td(self, out[-1], Qnow, snow, dt)
        out.append(temp)
        print('Timestep ', tt+1, ' of a total of ',self.T,'is finished')
    
    return np.array(out)

def run_model(self):
    # =============================================================================
    # code to run the subtidal salintiy model
    # =============================================================================
    print('Start the salinity calculation')
    
    tijd2 = time.time()
    #preparations
    self.tide_calc()
    self.indices()
    self.subtidal_module()
    self.prep_junc()
    # =============================================================================
    # first run the equilibrium simulation for the start of the simulation
    # =============================================================================
    inita = model_ti(self)
    #save
    #self.init_save = inita.copy()
    
    # =============================================================================
    # then run the time-dependent model 
    # =============================================================================
    self.out = model_td(self, inita)
    
    print('The total simualation time is ' , time.time()-tijd2, ' seconds')
















