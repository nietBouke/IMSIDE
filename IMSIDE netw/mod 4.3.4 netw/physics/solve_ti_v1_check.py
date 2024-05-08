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
from tqdm import trange

def sol_build(self, ans, pars_Q, pars_s):
    # =============================================================================
    # build the solution vector and corresponding jacobian for the NR algoritm
    # for the time-independent model 
    # =============================================================================
    #initialise
    tid_inps = {}
    sol_tot = np.zeros(self.ch_inds[self.ch_keys[-1]]['totx'][-1] * self.M)
    
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

        return key, tid_inp, ch_sol

    
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

    #run junctions
    sol_tot += self.sol_junc_tot(ans, self.junc_gegs, tid_inps, pars_Q)
    
    return sol_tot
     
      
def jac_build(self, ans, pars_Q, pars_s):
    # =============================================================================
    # build the solution vector and corresponding jacobian for the NR algoritm
    # for the time-independent model 
    # =============================================================================
    
    #initialise
    tid_inps = {}
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

        # =============================================================================
        # Jacobian matrix
        # =============================================================================
        #run subtidal module
        ch_jac_st = (self.jac_subtidal_vary(key, ch_ans, pars_Q) + self.jac_subtidal_fix(key, pars_Q))
        #run tidal module
        ch_jac_ti = (self.jac_tidal_vary(key) + self.jac_tidal_fix(key))
        #run boundaries
        ch_jac_bnd = (self.jac_bound_vary(key, ch_ans ,tid_inp, pars_Q) + self.jac_bound_fix(key, pars_Q))

        ch_jac = ch_jac_st + ch_jac_ti + ch_jac_bnd

        return key, tid_inp, ch_jac

    
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
        jac_tot[self.ch_inds[key]['totx'][0]*self.M:self.ch_inds[key]['totx'][-1]*self.M , self.ch_inds[key]['totx'][0]*self.M:self.ch_inds[key]['totx'][-1]*self.M] += chs[k][2]

    #run junctions
    jac_tot += (self.jac_junc_tot_vary(ans, self.junc_gegs, tid_inps, pars_Q) + self.jac_junc_tot_fix(self.junc_gegs, pars_Q))
    
    return jac_tot
            


def check_ti(self, init, prt = False):
    
    sss = init
    #quantities for the entire simulation
    Qnow = self.Qdist_calc((self.Qriv, self.Qweir, self.Qhar, self.n_sea))
    snow = (self.sri , self.swe , self.soc)
    
    # =============================================================================
    # run the model
    # =============================================================================
    #do the first iteration step
    jac_ana = jac_build(self, sss, Qnow, snow)
    
    
    #sss_n = sss - sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(jaco),solu)  # this is faster then np.linalg.solve (at least for a sufficiently large matrix) 
    
    it=1
    #print('That was iteration step ', it)
    #return

    #check jacobian numerically 
    def sol_func(ans):
        return sol_build(self, ans, Qnow, snow)
        
    # =============================================================================
    # check jacobian. 
    # =============================================================================
    def jac_alg3(sol_h, ans, e = 1e-3): #numeriek berekende jacobiaan
        #sol_h is oplossingfunctie
        #ans is de oplossingsvector
        
        jac = np.zeros((len(ans),len(ans)))
        for i2 in trange(len(ans)):
            ans_hpe,ans_hme = ans.copy(),ans.copy()
            ans_hpe[i2] = ans_hpe[i2] + e
            ans_hme[i2] = ans_hme[i2] - e
            
            jac[:,i2] = (sol_h(ans_hpe) - sol_h(ans_hme) )/(2*e)
            if prt == True: print((i2+1)/len(ans))            
        
        return jac

    jac_num = jac_alg3(sol_func, sss)
    
    # print()
    # print('The difference is ', np.max(np.abs(jac_num-jac_ana)))
    # print()
    print('The maximum of the numerical jacobian is',np.max(np.abs(jac_num)))
    
    # for i in range(len(jac_num)):
    #     print(i, np.max(np.abs(jac_ana[i])), np.max(np.abs(jac_num[i])))
    #     print(i, np.max(np.abs(jac_ana[:,i])), np.max(np.abs(jac_num[:,i])))
    
    #print()
    #print(jac_ana[168])
    #print(jac_num[168])
    #print()
    #print(jac_ana[169])
    #print(jac_num[169])
    #print()
    #print(jac_ana[227])
    #print(jac_num[227])
    
    #print()
    #print(jac_ana[42] - jac_num[42])
    #print(jac_ana[44] - jac_num[44])

    # print()
    # print(jac_ana[120:125,115:120])
    # print()
    # print(jac_num[120:125,115:120])
    # print()
    
    print()
    print('The difference is ', np.max(np.abs(jac_num-jac_ana)))
    #print('The relative difference is ', np.nanmax(np.abs(1-jac_num/jac_ana)))
    print()
    return



    # do the rest of the iterations
    while np.max(np.abs(sss-sss_n))>10e-6: #check whether the algoritm has converged
        sss = sss_n.copy() #update
        #sss = sss_n.copy()*0.5 + sss * 0.5 #update
        solu,jaco = build_all(self, sss)
        sss_n = sss - sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(jaco),solu)  # this is faster then np.linalg.solve (at least for a sufficiently large matrix)
        
        #plotting
        plt.plot(sss.reshape((int(len(sss)/self.M),self.M))[:,0])#-sss.reshape((int(len(sss)/self.M),self.M))[:,0])
        plt.plot(sss.reshape((int(len(sss)/self.M),self.M))[:,1])
        plt.plot(sss.reshape((int(len(sss)/self.M),self.M))[:,2])
        plt.show()
        
        it+=1
        print('That was iteration step ', it)
    
        if it>=20: break
    
    if it<20:
        print('The algoritm has converged \n')  
    else:
        print('ERROR: no convergence')
        return [[None]]
    '''
    # =============================================================================
    # return output
    # a bit more complicated here because we have the boundary layer terms in the ans vector
    # =============================================================================
    out_ss = np.array([]) 
    out_Bs = {}
    
    for key in self.ch_keys:
        #subtract normal output
        out_ss = np.concatenate([out_ss, sss[ch_inds[key]['s_intot'][0]:ch_inds[key]['s_intot'][1]]])    
        #subtract output for tidal matching at junctions. bit ugly with the try statements but it seems the best for now. 
        res = []
        try:  
            temp = sss[ch_inds[key]['B_intot']['loc x=-L'][0]:ch_inds[key]['B_intot']['loc x=-L'][1]]
            res.append(temp[:self.M]+1j*temp[self.M:])        
        except:  res.append(np.zeros(self.M)+np.nan)
        try: 
            temp = sss[ch_inds[key]['B_intot']['loc x=0' ][0]:ch_inds[key]['B_intot']['loc x=0' ][1]]
            res.append(temp[:self.M]+1j*temp[self.M:])
        except: res.append(np.zeros(self.M)+np.nan)
        
        out_Bs[key] = np.array(res)

    return out_ss, out_Bs, sss
    '''




def run_check(self, init_all = [None]):
    # =============================================================================
    # code to run the subtidal salintiy model
    # =============================================================================
    print('Start the checks')
    tijd = time.time()

    #preparations
    self.indices()
    self.prep_junc()
    self.tide_calc()
    self.subtidal_module()
    

    #spin up from subtidal model. Lets hope that does the job. 
    if init_all[0] == None: out = check_ti(self, np.zeros(self.ch_inds[self.ch_keys[-1]]['totx'][-1] * self.M))
    else:  out = check_ti(self, init_all)    
    
    return 



















