# =============================================================================
# In this file we calculate the indices for allocation of the terms in the 
# solution vector and jacobian. 
# =============================================================================

#import libraries
import numpy as np
import scipy as sp          #do fits
from scipy import optimize   , interpolate 
import time                              #measure time of operation
import matplotlib.pyplot as plt         


def indices(self):
    # =============================================================================
    # function to calculate the indices for allocation of the terms in the jacobian and solution vectors
    # =============================================================================
    ch_inds = {} # indices for solution vector/jacobian
    temp = 0
    
    for key in self.ch_keys:    
        ch_inds[key] = {}
    
        # =============================================================================
        #  Build indices for jacobian and solution vector
        # =============================================================================
        di2 = np.zeros((len(self.ch_pars[key]['di'])-2)*2)
        for i in range(1,len(self.ch_pars[key]['di'])-1):
            di2[i*2-2] = self.ch_pars[key]['di'][i]-1
            di2[i*2-1] = self.ch_pars[key]['di'][i]
        ch_inds[key]['di2'] = np.array(di2, dtype=int)
        
        if len(ch_inds[key]['di2']) ==0: 
            ch_inds[key]['di2_x=-L'] = np.array([], dtype=int)
            ch_inds[key]['di2_x=0']  = np.array([], dtype=int)
        elif len(ch_inds[key]['di2']) < 3:
            ch_inds[key]['di2_x=-L'] = np.array([di2[1]], dtype=int)
            ch_inds[key]['di2_x=0']  = np.array([di2[0]], dtype=int)
        else: 
            ch_inds[key]['di2_x=-L'] = np.array(di2, dtype=int).reshape((self.ch_pars[key]['n_seg']-1,2))[:,1]
            ch_inds[key]['di2_x=0']  = np.array(di2, dtype=int).reshape((self.ch_pars[key]['n_seg']-1,2))[:,0]
            
        di3 = []
        for i in range(len(self.ch_pars[key]['di'])): di3.append(self.ch_pars[key]['di'][i]+4*i)
        ch_inds[key]['di3'] = np.array(di3, dtype=int)
        
        #first the old ones, for the 'parameters, ie C1a etc'
        ch_inds[key]['x'] = np.delete(np.arange(self.ch_pars[key]['di'][-1]),ch_inds[key]['di2'])[1:-1] # x coorself.dinates for the points which are not on a aboundary    
        ch_inds[key]['xr'] = ch_inds[key]['x'].repeat(self.N) # x for self.N values, mostly i in old code
        ch_inds[key]['xr2'] = ch_inds[key]['xr'].repeat(self.N)

        ch_inds[key]['j1'] = np.tile(np.arange(self.N),self.ch_pars[key]['di'][-1]-2-(len(self.ch_pars[key]['di'][1:-1])*2))
        ch_inds[key]['j12'] = ch_inds[key]['j1'].repeat(self.N)
        ch_inds[key]['k1'] = np.tile(ch_inds[key]['j1'],self.N)
    
        #then for the new locations of life.   
        lenx = len(ch_inds[key]['x'])

        ch_inds[key]['ix_del']  = np.sort(np.concatenate([[0,1,2],ch_inds[key]['di3'][1:-1]-3,ch_inds[key]['di3'][1:-1]-2,ch_inds[key]['di3'][1:-1]-1,ch_inds[key]['di3'][1:-1],\
                                    ch_inds[key]['di3'][1:-1]+1,ch_inds[key]['di3'][1:-1]+2,[ch_inds[key]['di3'][-1]-3,ch_inds[key]['di3'][-1]-2,ch_inds[key]['di3'][-1]-1]]))
        ch_inds[key]['ix_delm'] = (ch_inds[key]['ix_del']*self.M+np.arange(self.M)[:,np.newaxis]).T.flatten()
        ch_inds[key]['xn']      = np.delete(np.arange(ch_inds[key]['di3'][-1]),ch_inds[key]['ix_del'])
        ch_inds[key]['xn_m']    = ch_inds[key]['xn']*self.M
        ch_inds[key]['xnp_m']   = (ch_inds[key]['xn']+1)*self.M
        ch_inds[key]['xnm_m']   = (ch_inds[key]['xn']-1)*self.M
        ch_inds[key]['xnr']     = ch_inds[key]['xn'].repeat(self.N)
        
        ch_inds[key]['xnr_m']   = ch_inds[key]['xnr']*self.M 
        ch_inds[key]['xnrm_m']  = (ch_inds[key]['xnr']-1)*self.M
        ch_inds[key]['xnrp_m']  = (ch_inds[key]['xnr']+1)*self.M
        ch_inds[key]['xnr_mj']  = ch_inds[key]['xnr_m']+np.tile(np.arange(1,self.M),lenx)
        ch_inds[key]['xnrp_mj'] = ch_inds[key]['xnrp_m']+np.tile(np.arange(1,self.M),lenx)
        ch_inds[key]['xnrm_mj'] = ch_inds[key]['xnrm_m']+np.tile(np.arange(1,self.M),lenx)
        
        ch_inds[key]['xnr_mj2'] =np.repeat(ch_inds[key]['xnr_m'],self.N)+np.tile(np.arange(1,self.M),lenx).repeat(self.N)
    
        ch_inds[key]['xnr_mk']  = np.repeat(ch_inds[key]['xnr_m'],self.N)+np.tile(np.tile(np.arange(1,self.M),lenx),self.N)
        ch_inds[key]['xnrp_mk'] = np.repeat(ch_inds[key]['xnrp_m'],self.N)+np.tile(np.tile(np.arange(1,self.M),lenx),self.N)
        ch_inds[key]['xnrm_mk'] = np.repeat(ch_inds[key]['xnrm_m'],self.N)+np.tile(np.tile(np.arange(1,self.M),lenx),self.N)
       
        ch_inds[key]['xnr_m2']  = ch_inds[key]['xnr_m'].repeat(self.N)
        ch_inds[key]['xnrp_m2'] = ch_inds[key]['xnrp_m'].repeat(self.N)
        ch_inds[key]['xnrm_m2'] = ch_inds[key]['xnrm_m'].repeat(self.N)
        
        
        #for sum 
        ch_inds[key]['xnr_mj3']  = ch_inds[key]['xnr_mj'].reshape((lenx,self.N))
        ch_inds[key]['xnrm_mj3'] = ch_inds[key]['xnrm_mj'].reshape((lenx,self.N))
        ch_inds[key]['xnrp_mj3'] = ch_inds[key]['xnrp_mj'].reshape((lenx,self.N))

        ch_inds[key]['xnr_mj4']  = ch_inds[key]['xnr_mj3'].T.repeat(self.N).reshape((self.N,lenx*self.N)).T
        ch_inds[key]['xnrm_mj4'] = ch_inds[key]['xnrm_mj3'].T.repeat(self.N).reshape((self.N,lenx*self.N)).T
        ch_inds[key]['xnrp_mj4'] = ch_inds[key]['xnrp_mj3'].T.repeat(self.N).reshape((self.N,lenx*self.N)).T


        
        #indices of correction
        ch_inds[key]['ibc_rgt'] = ( ch_inds[key]['di3'][1:-1]  *self.M+np.arange(2*self.M)[:,np.newaxis]).T
        ch_inds[key]['ibc_lft'] = ((ch_inds[key]['di3'][1:-1]-2)*self.M+np.arange(2*self.M)[:,np.newaxis]).T
        ch_inds[key]['ibc_rgt2'] = np.repeat(ch_inds[key]['ibc_rgt'][:,:self.M],self.M)
        ch_inds[key]['ibc_lft2'] = np.repeat(ch_inds[key]['ibc_lft'][:,:self.M],self.M)

        
        #indicies of inner boundary
        ch_inds[key]['ib_rgt'] = ((ch_inds[key]['di3'][1:-1]+2)*self.M+np.arange(self.M)[:,np.newaxis]).T.flatten()
        ch_inds[key]['ib_lft'] = ((ch_inds[key]['di3'][1:-1]-3)*self.M+np.arange(self.M)[:,np.newaxis]).T.flatten()
    
    
        # size of the total parts
        ch_inds[key]['totx'] = ch_inds[key]['di3'] + temp
        temp += ch_inds[key]['di3'][-1]

        #locations of boundaries in total
        ch_inds[key]['ob_x=-L_bcR'] = (ch_inds[key]['totx'][0]  + 0) * self.M + np.arange(self.M)
        ch_inds[key]['ob_x=-L_bcI'] = (ch_inds[key]['totx'][0]  + 1) * self.M + np.arange(self.M)
        ch_inds[key]['ob_x=-L_s0'] = (ch_inds[key]['totx'][0]  + 2) * self.M + np.arange(self.M)
        ch_inds[key]['ob_x=-L_s1'] = (ch_inds[key]['totx'][0]  + 3) * self.M + np.arange(self.M)
        ch_inds[key]['ob_x=-L_s2'] = (ch_inds[key]['totx'][0]  + 4) * self.M + np.arange(self.M)
        ch_inds[key]['ob_x=-L_s3'] = (ch_inds[key]['totx'][0]  + 5) * self.M + np.arange(self.M)
        ch_inds[key]['ob_x=0_bcI'] = (ch_inds[key]['totx'][-1] - 1) * self.M + np.arange(self.M) 
        ch_inds[key]['ob_x=0_bcR'] = (ch_inds[key]['totx'][-1] - 2) * self.M + np.arange(self.M) 
        ch_inds[key]['ob_x=0_s_1'] = (ch_inds[key]['totx'][-1] - 3) * self.M + np.arange(self.M) 
        ch_inds[key]['ob_x=0_s_2'] = (ch_inds[key]['totx'][-1] - 4) * self.M + np.arange(self.M) 
        ch_inds[key]['ob_x=0_s_3'] = (ch_inds[key]['totx'][-1] - 5) * self.M + np.arange(self.M) 
        ch_inds[key]['ob_x=0_s_4'] = (ch_inds[key]['totx'][-1] - 6) * self.M + np.arange(self.M) 
            
        #indices for reconstruction at the end
        ch_inds[key]['ibc']  = np.sort(np.concatenate([[0,1],ch_inds[key]['di3'][1:-1]-2,ch_inds[key]['di3'][1:-1]-1,ch_inds[key]['di3'][1:-1],\
                                    ch_inds[key]['di3'][1:-1]+1,[ch_inds[key]['di3'][-1]-2,ch_inds[key]['di3'][-1]-1]]))
        ch_inds[key]['iss']      = np.delete(np.arange(ch_inds[key]['di3'][-1]),ch_inds[key]['ibc'])
        ch_inds[key]['isb'] = ch_inds[key]['iss']*self.M
        ch_inds[key]['isn'] = (ch_inds[key]['iss']).repeat(self.N)*self.M + np.tile(np.arange(1,self.M),len(ch_inds[key]['isb']))
        
        #locations of boundary layer correction for inner boundaries 
        ch_inds[key]['ib_x=-L_bcR'] = ( ch_inds[key]['di3'][1:-1]   *self.M+np.arange(self.M)[:,np.newaxis])
        ch_inds[key]['ib_x=-L_bcI'] = ( ch_inds[key]['di3'][1:-1]   *self.M+np.arange(self.M,2*self.M)[:,np.newaxis])
        ch_inds[key]['ib_x=0_bcR'] = ((ch_inds[key]['di3'][1:-1]-2)*self.M+np.arange(self.M)[:,np.newaxis])
        ch_inds[key]['ib_x=0_bcI'] = ((ch_inds[key]['di3'][1:-1]-2)*self.M+np.arange(self.M,2*self.M)[:,np.newaxis])
            
        #taken over from single channel code
        ch_inds[key]['bnl_rgt'] = ( ch_inds[key]['di3'][1:-1]  *self.M+np.arange(2*self.M)[:,np.newaxis]).T
        ch_inds[key]['bnl_lft'] = ((ch_inds[key]['di3'][1:-1]-2)*self.M+np.arange(2*self.M)[:,np.newaxis]).T

                                                          
        #ch_inds[key]['bnd2_rgt'] = ((ch_inds[key]['di3'][1:-1]+2)*self.M+np.arange(1,self.M)[:,np.newaxis]).T.flatten()
        #ch_inds[key]['bnd2_lft'] = ((ch_inds[key]['di3'][1:-1]-3)*self.M+np.arange(1,self.M)[:,np.newaxis]).T.flatten()
        
        #Indices of inner boundary in old locaitons
        #ch_inds[key]['idl_lft'] = np.repeat(self.ch_pars[key]['di'][ :-2],self.M)
        #ch_inds[key]['idl_rgt'] = np.repeat(self.ch_pars[key]['di'][1:-1],self.M)
          
        #ch_inds[key]['idl2_lft'] = np.repeat(self.ch_pars[key]['di'][ :-2],self.N)
        #ch_inds[key]['idl2_rgt'] = np.repeat(self.ch_pars[key]['di'][1:-1],self.N)
    
        #indices for salinities at inner boundaries    - from a local perspective 
        ch_inds[key]['i_sb_1'] = (ch_inds[key]['di3'][1:-1]-3) * self.M
        ch_inds[key]['i_sb_2'] = (ch_inds[key]['di3'][1:-1]-4) * self.M
        ch_inds[key]['i_sb_3'] = (ch_inds[key]['di3'][1:-1]-5) * self.M
        ch_inds[key]['i_sb_4'] = (ch_inds[key]['di3'][1:-1]-6) * self.M
        ch_inds[key]['i_sb0']  = (ch_inds[key]['di3'][1:-1]+2) * self.M
        ch_inds[key]['i_sb1']  = (ch_inds[key]['di3'][1:-1]+3) * self.M
        ch_inds[key]['i_sb2']  = (ch_inds[key]['di3'][1:-1]+4) * self.M
        ch_inds[key]['i_sb3']  = (ch_inds[key]['di3'][1:-1]+5) * self.M

        ch_inds[key]['i_sa_1'] = ((ch_inds[key]['di3'][1:-1]-3)[:,np.newaxis] * self.M + np.arange(self.M)).flatten()
        ch_inds[key]['i_sa_2'] = ((ch_inds[key]['di3'][1:-1]-4)[:,np.newaxis] * self.M + np.arange(self.M)).flatten()
        ch_inds[key]['i_sa_3'] = ((ch_inds[key]['di3'][1:-1]-5)[:,np.newaxis] * self.M + np.arange(self.M)).flatten()
        ch_inds[key]['i_sa0']  = ((ch_inds[key]['di3'][1:-1]+2)[:,np.newaxis] * self.M + np.arange(self.M)).flatten()
        ch_inds[key]['i_sa1']  = ((ch_inds[key]['di3'][1:-1]+3)[:,np.newaxis] * self.M + np.arange(self.M)).flatten()
        ch_inds[key]['i_sa2']  = ((ch_inds[key]['di3'][1:-1]+4)[:,np.newaxis] * self.M + np.arange(self.M)).flatten()
        
        ch_inds[key]['i_sn_1'] = (ch_inds[key]['di3'][1:-1]-3)[:,np.newaxis] * self.M + np.arange(1,self.M)
        ch_inds[key]['i_sn_2'] = (ch_inds[key]['di3'][1:-1]-4)[:,np.newaxis] * self.M + np.arange(1,self.M)
        ch_inds[key]['i_sn_3'] = (ch_inds[key]['di3'][1:-1]-5)[:,np.newaxis] * self.M + np.arange(1,self.M)
        ch_inds[key]['i_sn0']  = (ch_inds[key]['di3'][1:-1]+2)[:,np.newaxis] * self.M + np.arange(1,self.M)
        ch_inds[key]['i_sn1']  = (ch_inds[key]['di3'][1:-1]+3)[:,np.newaxis] * self.M + np.arange(1,self.M)
        ch_inds[key]['i_sn2']  = (ch_inds[key]['di3'][1:-1]+4)[:,np.newaxis] * self.M + np.arange(1,self.M)
        
        ch_inds[key]['i_p_1']  = np.repeat(self.ch_pars[key]['di'][1:-1]-1,self.M)
        ch_inds[key]['i_p0']  = np.repeat(self.ch_pars[key]['di'][1:-1],self.M)

              
        ch_inds[key]['ih_lft'] = self.ch_pars[key]['di'][1:-1]
        ch_inds[key]['ih_rgt'] = self.ch_pars[key]['di'][1:-1]-1
        #ch_inds[key]['ih2_lft'] = np.arange(1,self.ndom)
        #ch_inds[key]['ih2_rgt'] = np.arange(self.ndom-1)    
        
        #repeated stuff
        ch_inds[key]['i_sb_1_rep'] = np.repeat(ch_inds[key]['i_sb_1'],self.N).reshape((len(ch_inds[key]['i_sb_1']),self.N))
        ch_inds[key]['i_sb_2_rep'] = np.repeat(ch_inds[key]['i_sb_2'],self.N).reshape((len(ch_inds[key]['i_sb_1']),self.N))
        ch_inds[key]['i_sb_3_rep'] = np.repeat(ch_inds[key]['i_sb_3'],self.N).reshape((len(ch_inds[key]['i_sb_1']),self.N))

        ch_inds[key]['i_sb0_rep'] = np.repeat(ch_inds[key]['i_sb0'],self.N).reshape((len(ch_inds[key]['i_sb0']),self.N))
        ch_inds[key]['i_sb1_rep'] = np.repeat(ch_inds[key]['i_sb1'],self.N).reshape((len(ch_inds[key]['i_sb1']),self.N))
        ch_inds[key]['i_sb2_rep'] = np.repeat(ch_inds[key]['i_sb2'],self.N).reshape((len(ch_inds[key]['i_sb2']),self.N))
        
        ch_inds[key]['i_sb_1_rep2'] = np.repeat(ch_inds[key]['i_sb_1'],self.M).reshape((len(ch_inds[key]['i_sb_1']),self.M))

        ch_inds[key]['i_sb0_rep3'] = np.repeat(ch_inds[key]['i_sb0'],self.M**2)
        ch_inds[key]['i_sb1_rep3'] = np.repeat(ch_inds[key]['i_sb1'],self.M**2)
        ch_inds[key]['i_sb2_rep3'] = np.repeat(ch_inds[key]['i_sb2'],self.M**2)
        ch_inds[key]['i_sb3_rep3'] = np.repeat(ch_inds[key]['i_sb3'],self.M**2)
        
        ch_inds[key]['i_sb_1_rep3'] = np.repeat(ch_inds[key]['i_sb_1'],self.M**2)
        ch_inds[key]['i_sb_2_rep3'] = np.repeat(ch_inds[key]['i_sb_2'],self.M**2)
        ch_inds[key]['i_sb_3_rep3'] = np.repeat(ch_inds[key]['i_sb_3'],self.M**2)
        ch_inds[key]['i_sb_4_rep3'] = np.repeat(ch_inds[key]['i_sb_4'],self.M**2)
        
        ch_inds[key]['ibc_rgt_rep3'] = np.tile(ch_inds[key]['ibc_rgt'][:,:self.M],self.M).flatten()
        ch_inds[key]['ibc_lft_rep3'] = np.tile(ch_inds[key]['ibc_lft'][:,:self.M],self.M).flatten()
        ch_inds[key]['ibc_rgt_rep3b'] = np.tile(ch_inds[key]['ibc_rgt'][:,self.M:],self.M).flatten()
        ch_inds[key]['ibc_lft_rep3b'] = np.tile(ch_inds[key]['ibc_lft'][:,self.M:],self.M).flatten()
        
        
        
        
    self.ch_inds= ch_inds








