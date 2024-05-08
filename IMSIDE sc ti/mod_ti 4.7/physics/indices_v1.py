# =============================================================================
# build here the inself.dices for the self.Newton Raphson solution vector and jacobian
# =============================================================================

import numpy as np


def indices(self):
    
    indi = {}
    
    # =============================================================================
    #  Build inself.dices for jacobian and solution vector
    # =============================================================================
    self.di2 = np.zeros((len(self.di)-2)*2)
    for i in range(1,len(self.di)-1):
        self.di2[i*2-2] = self.di[i]-1
        self.di2[i*2-1] = self.di[i]
    self.di2 = np.array(self.di2, dtype=int)
        
    self.di3 = []
    for i in range(len(self.di)): self.di3.append(self.di[i]+4*i)
    self.di3 = np.array(self.di3, dtype=int)
    
    #first the old ones, for the 'parameters, ie C1a etc'
    indi['x'] = np.delete(np.arange(self.di[-1]),self.di2)[1:-1] # x coorself.dinates for the points which are not on a aboundary    
    indi['xr'] = indi['x'].repeat(self.N) # x for self.N values, mostly i in old code
    indi['xr2'] = indi['xr'].repeat(self.N)
    
    indi['j1'] = np.tile(np.arange(self.N),self.di[-1]-2-(len(self.di[1:-1])*2))
    indi['j12'] = indi['j1'].repeat(self.N)
    indi['k1'] = np.tile(indi['j1'],self.N)

    #then for the new locations of life. 
    
    indi['ix_del']  = np.sort(np.concatenate([[0,1,2],self.di3[1:-1]-3,self.di3[1:-1]-2,self.di3[1:-1]-1,self.di3[1:-1],self.di3[1:-1]+1,self.di3[1:-1]+2,[self.di3[-1]-3,self.di3[-1]-2,self.di3[-1]-1]]))
    indi['ix_delm'] = (indi['ix_del']*self.M+np.arange(self.M)[:,np.newaxis]).T.flatten()
    indi['xn']      = np.delete(np.arange(self.di3[-1]),indi['ix_del'])
    indi['xn_m']    = indi['xn']*self.M
    indi['xnp_m']   = (indi['xn']+1)*self.M
    indi['xnm_m']   = (indi['xn']-1)*self.M
    indi['xnr']     = indi['xn'].repeat(self.N)
    
    indi['xnr_m']   = indi['xnr']*self.M 
    indi['xnrm_m']  = (indi['xnr']-1)*self.M
    indi['xnrp_m']  = (indi['xnr']+1)*self.M
    indi['xnr_mj']  = indi['xnr_m']+np.tile(np.arange(1,self.M),len(indi['xn']))
    indi['xnrp_mj'] = indi['xnrp_m']+np.tile(np.arange(1,self.M),len(indi['xn']))
    indi['xnrm_mj'] = indi['xnrm_m']+np.tile(np.arange(1,self.M),len(indi['xn']))
    
    indi['xnr_mj2'] =np.repeat(indi['xnr_m'],self.N)+np.tile(np.arange(1,self.M),len(indi['xn'])).repeat(self.N)

    indi['xnr_mk']  = np.repeat(indi['xnr_m'],self.N)+np.tile(np.tile(np.arange(1,self.M),len(indi['xn'])),self.N)
    indi['xnrp_mk'] = np.repeat(indi['xnrp_m'],self.N)+np.tile(np.tile(np.arange(1,self.M),len(indi['xn'])),self.N)
    indi['xnrm_mk'] = np.repeat(indi['xnrm_m'],self.N)+np.tile(np.tile(np.arange(1,self.M),len(indi['xn'])),self.N)
   
    indi['xnr_m2']  = indi['xnr_m'].repeat(self.N)
    indi['xnrp_m2'] = indi['xnrp_m'].repeat(self.N)
    indi['xnrm_m2'] = indi['xnrm_m'].repeat(self.N)
    

    #indices of correction
    indi['bnl_lft'] = ( self.di3[1:-1]  *self.M+np.arange(2*self.M)[:,np.newaxis]).T
    indi['bnl_rgt'] = ((self.di3[1:-1]-2)*self.M+np.arange(2*self.M)[:,np.newaxis]).T
    indi['bnl_bnd'] = np.concatenate([self.di3[0]*self.M+np.arange(2*self.M) , (self.di3[-1]-2)*self.M+np.arange(2*self.M)])   
    
    #indicies of boundary
    indi['bnd_lft'] = ((self.di3[1:-1]+2)*self.M+np.arange(self.M)[:,np.newaxis]).T.flatten()
    indi['bnd_rgt'] = ((self.di3[1:-1]-3)*self.M+np.arange(self.M)[:,np.newaxis]).T.flatten()
   
    indi['bnd2_lft'] = ((self.di3[1:-1]+2)*self.M+np.arange(1,self.M)[:,np.newaxis]).T.flatten()
    indi['bnd2_rgt'] = ((self.di3[1:-1]-3)*self.M+np.arange(1,self.M)[:,np.newaxis]).T.flatten()
    
    #Indices of boundary in old locaitons
    indi['idl_lft'] = np.repeat(self.di[ :-2],self.M)
    indi['idl_rgt'] = np.repeat(self.di[1:-1],self.M)
      
    indi['idl2_lft'] = np.repeat(self.di[ :-2],self.N)
    indi['idl2_rgt'] = np.repeat(self.di[1:-1],self.N)
   
   
    #indices for salinities at boundaries     
    indi['i_s_1'] = (self.di3[1:-1]-3) * self.M #rgt
    indi['i_s_2'] = (self.di3[1:-1]-4) * self.M
    indi['i_s_3'] = (self.di3[1:-1]-5) * self.M
    indi['i_s_4'] = (self.di3[1:-1]-6) * self.M
    indi['i_s0']  = (self.di3[1:-1]+2) * self.M #lft
    indi['i_s1']  = (self.di3[1:-1]+3) * self.M
    indi['i_s2']  = (self.di3[1:-1]+4) * self.M
    indi['i_s3']  = (self.di3[1:-1]+5) * self.M
    
    #I am a bit surprised those do not change. 
    indi['ih_lft'] = self.di[1:-1]
    indi['ih_rgt'] =  self.di[1:-1]-1
    indi['ih2_lft'] = np.arange(1,self.ndom)
    indi['ih2_rgt'] = np.arange(self.ndom-1)    

    '''
    #indices of correction
    indi['bnl_lft'] = ( self.di3[1:-1]   *self.M + np.arange(2*self.M)[:,np.newaxis]).T
    indi['bnl_rgt'] = ((self.di3[1:-1]-2)*self.M + np.arange(2*self.M)[:,np.newaxis]).T
    indi['bnl_bnd'] = np.concatenate([self.di3[0]*self.M+np.arange(2*self.M) , (self.di3[-1]-2)*self.M+np.arange(2*self.M)])   
    
    #indicies of boundary
    indi['bnd_rgt'] = ((self.di3[1:-1]+2)*self.M+np.arange(self.M)[:,np.newaxis]).T.flatten()
    indi['bnd_lft'] = ((self.di3[1:-1]-3)*self.M+np.arange(self.M)[:,np.newaxis]).T.flatten()

    indi['bnd2_rgt'] = ((self.di3[1:-1]+2)*self.M+np.arange(1,self.M)[:,np.newaxis]).T.flatten()
    indi['bnd2_lft'] = ((self.di3[1:-1]-3)*self.M+np.arange(1,self.M)[:,np.newaxis]).T.flatten()
    
    #Indices of boundary in old locaitons
    indi['idl_lft'] = np.repeat(self.di[ :-2],self.M)
    indi['idl_rgt'] = np.repeat(self.di[1:-1],self.M)
      
    indi['idl2_lft'] = np.repeat(self.di[ :-2],self.N)
    indi['idl2_rgt'] = np.repeat(self.di[1:-1],self.N)


    #indices for salinities at boundaries     
    indi['i_s_1'] = (self.di3[1:-1]-3) * self.M
    indi['i_s_2'] = (self.di3[1:-1]-4) * self.M
    indi['i_s_3'] = (self.di3[1:-1]-5) * self.M
    indi['i_s_4'] = (self.di3[1:-1]-6) * self.M
    indi['i_s0']  = (self.di3[1:-1]+2) * self.M
    indi['i_s1']  = (self.di3[1:-1]+3) * self.M
    indi['i_s2']  = (self.di3[1:-1]+4) * self.M
    indi['i_s3']  = (self.di3[1:-1]+5) * self.M
    
    indi['ih_lft'] = self.di[1:-1]
    indi['ih_rgt'] =  self.di[1:-1]-1
    indi['ih2_lft'] = np.arange(1,self.ndom)
    indi['ih2_rgt'] = np.arange(self.ndom-1)    
    #indi['ih3_lft'] = [indi['ih_lft'][i] + np.arange(1,len(x_lft[i])+1) for i in range(self.ndom-1)]
    #indi['ih3_rgt'] = [indi['ih_rgt'][i] + np.arange(-len(x_rgt[i]),0) for i in range(self.ndom-1)]
    '''


    
    return indi



