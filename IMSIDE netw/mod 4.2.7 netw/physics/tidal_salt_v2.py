# =============================================================================
# File to calculate the effects of tides on the salinity
# TODO: check minus signs of e.g. the boundary layer correction 
# =============================================================================

import numpy as np
import time as time
from physics.tide_funcs5 import sti

# =============================================================================
# From here, I will define functions, to be used in the subtidal module, to calculate the effect of the tides
# =============================================================================

def conv_ans(self, key, zout): 
    # =============================================================================
    # function to convert the answer vector to salinities: sn, dsndx, dsbdx, dsbdx2
    # =============================================================================
    di_here = self.ch_pars[key]['di']
    di3_here = self.ch_inds[key]['di3']
    dxn_here = self.ch_pars[key]['dln']*self.Lsc

    s9 = np.zeros(di_here[-1]*self.M)+np.nan
    for i in range(self.ch_pars[key]['n_seg']): s9[di_here[i]*self.M:di_here[i+1]*self.M] = zout[(di3_here[i]+2)*self.M:(di3_here[i+1]-2)*self.M]
    ss = s9.reshape(di_here[-1],self.M)
    
    #depth averaged salinity and the fourier modes
    sb = self.soc_sca*ss[:,0]
    sn = self.soc_sca*ss[:,1:].T
    #derivatives
    dsbdx, dsbdx2, dsndx = np.zeros(di_here[-1]), np.zeros(di_here[-1]), np.zeros((self.N,di_here[-1]))
    for dom in range(self.ch_pars[key]['n_seg']):
        dsbdx[di_here[dom]] = (-3*sb[di_here[dom]] + 4*sb[di_here[dom]+1] - sb[di_here[dom]+2] )/(2*dxn_here[dom])
        dsbdx[di_here[dom]+1:di_here[dom+1]-1] = (sb[di_here[dom]+2:di_here[dom+1]] - sb[di_here[dom]:di_here[dom+1]-2])/(2*dxn_here[dom])
        dsbdx[di_here[dom+1]-1] = (sb[di_here[dom+1]-3] - 4*sb[di_here[dom+1]-2] + 3*sb[di_here[dom+1]-1] )/(2*dxn_here[dom])
   
        dsbdx2[di_here[dom]] = (2*sb[di_here[dom]] - 5*sb[di_here[dom]+1] +4*sb[di_here[dom]+2] -1*sb[di_here[dom]+3] )/(dxn_here[dom]**2)
        dsbdx2[di_here[dom]+1:di_here[dom+1]-1] = (sb[di_here[dom]+2:di_here[dom+1]] - 2*sb[di_here[dom]+1:di_here[dom+1]-1] + sb[di_here[dom]:di_here[dom+1]-2])/(dxn_here[dom]**2)
        dsbdx2[di_here[dom+1]-1] = (-sb[di_here[dom+1]-4] + 4*sb[di_here[dom+1]-3] - 5*sb[di_here[dom+1]-2] + 2*sb[di_here[dom+1]-1] )/(dxn_here[dom]**2)
   
        dsndx[:,di_here[dom]] = (-3*sn[:,di_here[dom]] + 4*sn[:,di_here[dom]+1] - sn[:,di_here[dom]+2] )/(2*dxn_here[dom])
        dsndx[:,di_here[dom]+1:di_here[dom+1]-1] = (sn[:,di_here[dom]+2:di_here[dom+1]] - sn[:,di_here[dom]:di_here[dom+1]-2])/(2*dxn_here[dom])
        dsndx[:,di_here[dom+1]-1] = (sn[:,di_here[dom+1]-3] - 4*sn[:,di_here[dom+1]-2] + 3*sn[:,di_here[dom+1]-1] )/(2*dxn_here[dom])
   
    #save
    d_ans = (sn[:,:,np.newaxis],dsndx[:,:,np.newaxis],dsbdx[np.newaxis,:,np.newaxis],dsbdx2[np.newaxis,:,np.newaxis])

    return d_ans

def tidal_salinity(self,key,zout):
    # =============================================================================
    # This function calculates tidal salinity from the answer vector       
    # =============================================================================
    #convert raw input
    sn,dsndx,dsbdx,dsbdx2 = conv_ans(self, key, zout)
    di_here = self.ch_pars[key]['di']
    nph = self.nn*np.pi/self.ch_gegs[key]['H']
    # =============================================================================
    # Now the tidal salinty and the derivatives   
    # =============================================================================
    #coefficients of the tidal salinity
    c1 = - self.ch_pars[key]['Kv_ti']/(1j*self.omega)
    c2, c3, c4 = np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex)
    for dom in range(self.ch_pars[key]['n_seg']): 
        c2[:,di_here[dom]:di_here[dom+1]] = self.go2 * self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]] * dsbdx[:,di_here[dom]:di_here[dom+1]]
        c3[:,di_here[dom]:di_here[dom+1]] = - nph * sn[:,di_here[dom]:di_here[dom+1]] * self.ch_pars[key]['eta'][:,di_here[dom]:di_here[dom+1]]
        c4[:,di_here[dom]:di_here[dom+1]] = - nph * sn[:,di_here[dom]:di_here[dom+1]] * self.go2 \
            * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom])
            
    #tidal salinity
    st = sti((c2,c3,c4),(self.ch_pars[key]['c2c'],self.ch_pars[key]['c3c'],self.ch_pars[key]['c4c']))    
    stb = np.mean(st,1)[:,np.newaxis] #kan in theorie analytisch
    stp = st-stb #kan in theorie analytisch

    # =============================================================================
    # derivatives
    # =============================================================================
    #coefficients for derivative
    dc2dx, dc3dx, dc4dx = np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex)
    for dom in range(self.ch_pars[key]['n_seg']): 
        dc2dx[:,di_here[dom]:di_here[dom+1]] = self.go2 * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]]*dsbdx[:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]*dsbdx2[:,di_here[dom]:di_here[dom+1]])
        dc3dx[:,di_here[dom]:di_here[dom+1]] = - nph * (self.ch_pars[key]['eta'][:,di_here[dom]:di_here[dom+1]]*dsndx[:,di_here[dom]:di_here[dom+1]] + sn[:,di_here[dom]:di_here[dom+1]]*self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]])
        dc4dx[:,di_here[dom]:di_here[dom+1]] = - nph * self.go2 * (dsndx[:,di_here[dom]:di_here[dom+1]] * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom]) \
                                                                                                             + sn[:,di_here[dom]:di_here[dom+1]] * (self.ch_pars[key]['detadx3'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom]))
    
    dstidx = (self.ch_pars[key]['c2c'] * dc2dx + (self.ch_pars[key]['c3c'] * dc3dx).sum(0) + (self.ch_pars[key]['c4c'] * dc4dx).sum(0))[0]
    dstibdx = np.mean(dstidx,1)[:,np.newaxis] #kan in theorie analytisch
    dstipdx = dstidx-dstibdx #kan in theorie analytisch
           
    #derivatives to z 
    dstdz = sti((c2,c3,c4),(self.ch_pars[key]['c2c_z'],self.ch_pars[key]['c3c_z'],self.ch_pars[key]['c4c_z']))          
    
    # =============================================================================
    # boundary layer correction     
    # =============================================================================
    stc_xL , stc_xLi , stc_x_xLi = [] , [] , []
    stcp_x_xLi, stc_z_xLi = [] , []
    
    stc_x0, stc_x0i , stc_x_x0i = [] , [] , []
    stcp_x_x0i, stc_z_x0i = [] , []
    
    for dom in range(self.ch_pars[key]['n_seg']):
        
        # =============================================================================
        # first at x=-L
        # =============================================================================
        #salinity at the boundary
        stc   = np.sum((zout[self.ch_inds[key]['di3'][dom]*self.M : (self.ch_inds[key]['di3'][dom]+1)*self.M] + 1j*zout[(self.ch_inds[key]['di3'][dom]+1)*self.M : (self.ch_inds[key]['di3'][dom]+2)*self.M])[:,np.newaxis] \
                       * np.cos(np.pi*self.m0[:,np.newaxis]*self.z_nd) , axis=0) * self.soc_sca
        stcp  = np.sum((zout[self.ch_inds[key]['di3'][dom]*self.M +1 : (self.ch_inds[key]['di3'][dom]+1)*self.M] + 1j*zout[(self.ch_inds[key]['di3'][dom]+1)*self.M +1: (self.ch_inds[key]['di3'][dom]+2)*self.M])[:,np.newaxis] \
                       * np.cos(np.pi*self.m0[1:,np.newaxis]*self.z_nd) , axis=0) * self.soc_sca
        stc_z = np.sum((zout[self.ch_inds[key]['di3'][dom]*self.M : (self.ch_inds[key]['di3'][dom]+1)*self.M] + 1j*zout[(self.ch_inds[key]['di3'][dom]+1)*self.M : (self.ch_inds[key]['di3'][dom]+2)*self.M])[:,np.newaxis] \
                       * -self.m0[:,np.newaxis]*np.pi/self.ch_gegs[key]['H'] * np.sin(np.pi*self.m0[:,np.newaxis]*self.z_nd) , axis=0) * self.soc_sca
        #calculate x 
        x_all = np.arange(di_here[dom+1]-di_here[dom])*self.ch_gegs[key]['dx'][dom] 
        x_bnd = x_all[np.where(x_all<(-np.sqrt(self.epsL)*np.log(self.tol)))[0]]
        if x_bnd[-1] >= self.ch_gegs[key]['L'][dom]: print('ERROR: boundary layer too large. Can be solved...')
        #salinity in inner domain
        stci = np.exp(-x_bnd[:,np.newaxis]/np.sqrt(self.epsL)) * stc
        stcpi= np.exp(-x_bnd[:,np.newaxis]/np.sqrt(self.epsL)) * stcp
        stc_zi= np.exp(-x_bnd[:,np.newaxis]/np.sqrt(self.epsL)) * stc_z

        #save
        stc_xL.append(stc)
        stc_xLi.append(stci)
        stc_x_xLi.append(stci * -1/np.sqrt(self.epsL))
        stcp_x_xLi.append(stcpi * -1/np.sqrt(self.epsL))
        stc_z_xLi.append(stc_zi)

        
        # =============================================================================
        # then at x=0        
        # =============================================================================
        #salinity at the boundary
        stc = np.sum((zout[(self.ch_inds[key]['di3'][dom+1]-2)*self.M : (self.ch_inds[key]['di3'][dom+1]-1)*self.M] + 1j*zout[(self.ch_inds[key]['di3'][dom+1]-1)*self.M : (self.ch_inds[key]['di3'][dom+1]-0)*self.M])[:,np.newaxis] \
                     * np.cos(np.pi*self.m0[:,np.newaxis]*self.z_nd) , axis=0) * self.soc_sca
        stcp = np.sum((zout[(self.ch_inds[key]['di3'][dom+1]-2)*self.M +1 : (self.ch_inds[key]['di3'][dom+1]-1)*self.M] + 1j*zout[(self.ch_inds[key]['di3'][dom+1]-1)*self.M +1 : (self.ch_inds[key]['di3'][dom+1]-0)*self.M])[:,np.newaxis] \
                      * np.cos(np.pi*self.m0[1:,np.newaxis]*self.z_nd) , axis=0) * self.soc_sca
        stc_z = np.sum((zout[(self.ch_inds[key]['di3'][dom+1]-2)*self.M : (self.ch_inds[key]['di3'][dom+1]-1)*self.M] + 1j*zout[(self.ch_inds[key]['di3'][dom+1]-1)*self.M : (self.ch_inds[key]['di3'][dom+1]-0)*self.M])[:,np.newaxis] \
                     * -self.m0[:,np.newaxis]*np.pi/self.ch_gegs[key]['H'] * np.sin(np.pi*self.m0[:,np.newaxis]*self.z_nd) , axis=0) * self.soc_sca
        
        #calculate x 
        x_all = np.arange(di_here[dom]-di_here[dom+1],1)*self.ch_gegs[key]['dx'][dom] 
        x_bnd = x_all[np.where(x_all>(np.sqrt(self.epsL)*np.log(self.tol)))[0]]
        if -x_bnd[0] > self.ch_gegs[key]['L'][dom]: print('ERROR: boundary layer too large. Can be solved...', key, -x_bnd[0] , self.ch_gegs[key]['L'][dom] )
        #salinity in inner domain
        stci = np.exp(x_bnd[:,np.newaxis]/np.sqrt(self.epsL)) * stc
        stcpi = np.exp(x_bnd[:,np.newaxis]/np.sqrt(self.epsL)) * stcp
        stc_zi = np.exp(x_bnd[:,np.newaxis]/np.sqrt(self.epsL)) * stc_z

        #save
        stc_x0.append(stc)
        stc_x0i.append(stci)
        stc_x_x0i.append(stci * 1/np.sqrt(self.epsL))
        stcp_x_x0i.append(stcpi * 1/np.sqrt(self.epsL))
        stc_z_x0i.append(stc_zi)
        

    
    #save variables which I need - not in object, since they change during the calculation
    save = { 'st': st , 
            'stb': stb ,
            'stp': stp ,
            
            'dstidx' : dstidx ,
            'dstibdx': dstibdx ,
            'dstipdx': dstipdx ,
    
            'dstdz': dstdz , 
            
            #rom here it is boundary layer correctoin
            
            'stc_x=-L' : stc_xL  ,
            'stci_x=-L': stc_xLi ,
            'stci_x_x=-L': stc_x_xLi ,
            
            'stcpi_x_x=-L': stcp_x_xLi ,
            'stc_zi_x=-L': stc_z_xLi ,

            'stc_x=0'  : stc_x0  ,
            'stci_x=0' : stc_x0i ,
            'stci_x_x=0' : stc_x_x0i ,
                        
            'stcpi_x_x=0': stcp_x_x0i ,
            'stc_zi_x=0': stc_z_x0i ,

            }
    
    return save 




    

def func_sol_int(self, tid_here, key):  
    # =============================================================================
    # This function calculates the contribution of the tides to the 
    # depth-averaged solution vector
    # =============================================================================
    #load from inp
    st , dstidx = tid_here
    
    # =============================================================================
    # Terms for in equation
    # =============================================================================
    dfdx = np.mean(-1/4 * np.real(self.ch_pars[key]['ut'][0]*np.conj(dstidx) + np.conj(self.ch_pars[key]['ut'][0])*dstidx + self.ch_pars[key]['dutdx'][0]*np.conj(st) + np.conj(self.ch_pars[key]['dutdx'][0])*st) , axis=1)
    flux = np.mean(-1/4 * np.real(self.ch_pars[key]['ut'][0]*np.conj(st) + np.conj(self.ch_pars[key]['ut'][0])*st) , axis=1) #part for dbdx
    tot = (dfdx + flux /self.ch_pars[key]['bex']) * self.Lsc/self.soc_sca

    return -tot


def func_jac_int(self, key):
    # =============================================================================
    # Contributions to Jacobian for the tidal salt in the depth-averaged solution vector
    # =============================================================================
    #load inp
    di_here = self.ch_pars[key]['di']
    dsdc2_here  = self.ch_pars[key]['c2c']#[:,di_here[dom]:di_here[dom+1]]
    dsdc3_here  = self.ch_pars[key]['c3c']#[:,di_here[dom]:di_here[dom+1]]
    dsdc4_here  = self.ch_pars[key]['c4c']#[:,di_here[dom]:di_here[dom+1]]
    
    #create empty vectors
    dT_dsb1,dT_dsb0,dT_dsb_1 = np.zeros(di_here[-1]) , np.zeros(di_here[-1]) , np.zeros(di_here[-1])
    dT_dsn1,dT_dsn0,dT_dsn_1 = np.zeros((self.N,di_here[-1])) , np.zeros((self.N,di_here[-1])) , np.zeros((self.N,di_here[-1]))
    
    #for every domain a calculation
    for dom in range(self.ch_pars[key]['n_seg']):
        #select parameters
        dx_here     = self.ch_gegs[key]['dx'][dom]
        bn_here     = self.ch_pars[key]['bn'][dom] 
        nph         = self.nn*np.pi/self.ch_gegs[key]['H']
        ut_here     = self.ch_pars[key]['ut'][:,di_here[dom]:di_here[dom+1]]
        dutdx_here  = self.ch_pars[key]['dutdx'][:,di_here[dom]:di_here[dom+1]]
        eta_here    = self.ch_pars[key]['eta'][:,di_here[dom]:di_here[dom+1]]
        detadx_here = self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]
        detadx2_here= self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]]
        detadx3_here= self.ch_pars[key]['detadx3'][:,di_here[dom]:di_here[dom+1]]

        
        #calculate contributions 
        dT_dsb1[di_here[dom]:di_here[dom+1]] = 1/4*self.Lsc * np.real(dutdx_here * np.conj( dsdc2_here * self.go2 * detadx_here * 1/(2*dx_here) ) 
                               + np.conj(dutdx_here) * dsdc2_here * self.go2 * detadx_here * 1/(2*dx_here)
                               + ut_here * np.conj(dsdc2_here * self.go2 * (detadx2_here/(2*dx_here) + detadx_here/(dx_here**2)))
                               + np.conj(ut_here)*(dsdc2_here * self.go2 * (detadx2_here/(2*dx_here) + detadx_here/(dx_here**2))) 
                               + ut_here/bn_here * np.conj( dsdc2_here * self.go2 * detadx_here * 1/(2*dx_here) ) 
                               + np.conj(ut_here)/bn_here * dsdc2_here * self.go2 * detadx_here * 1/(2*dx_here)
                               ).mean(2)
        
        dT_dsb0[di_here[dom]:di_here[dom+1]] = 1/4*self.Lsc * np.real( ut_here * np.conj(dsdc2_here * -2 * self.go2 * detadx_here/(dx_here**2)) 
                               +np.conj(ut_here)*(dsdc2_here * -2 * self.go2 * detadx_here/(dx_here**2)) ).mean(2) 
        
        dT_dsb_1[di_here[dom]:di_here[dom+1]] = 1/4*self.Lsc * np.real(dutdx_here * np.conj( dsdc2_here *-self.go2 * detadx_here * 1/(2*dx_here) ) 
                                                            + np.conj(dutdx_here)* dsdc2_here *-self.go2 * detadx_here * 1/(2*dx_here)
                                                            + ut_here * np.conj(dsdc2_here * self.go2 * (-detadx2_here/(2*dx_here) + detadx_here/(dx_here**2)))
                                                            + np.conj(ut_here)*(dsdc2_here * self.go2 * (-detadx2_here/(2*dx_here) + detadx_here/(dx_here**2))) 
                                                            + ut_here/bn_here * np.conj( dsdc2_here *-self.go2 * detadx_here * 1/(2*dx_here) ) 
                                                            + np.conj(ut_here/bn_here) * dsdc2_here *-self.go2 * detadx_here * 1/(2*dx_here)
                                                            ).mean(2)


        dT_dsn1[:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * np.real( ut_here * np.conj(dsdc3_here * -nph * eta_here/(2*dx_here) + dsdc4_here * -nph * self.go2 /(2*dx_here) * (detadx2_here + detadx_here/bn_here)) 
                                                      + np.conj(ut_here)*(dsdc3_here * -nph * eta_here/(2*dx_here) + dsdc4_here * -nph * self.go2 /(2*dx_here) * (detadx2_here + detadx_here/bn_here)) ).mean(2)
        
        dT_dsn0[:,di_here[dom]:di_here[dom+1]] = 1/4*self.Lsc * np.real(  dutdx_here * np.conj( dsdc3_here * - nph*eta_here + dsdc4_here * - nph * self.go2 * (detadx2_here + detadx_here/bn_here) ) 
                                                    + np.conj(dutdx_here)*( dsdc3_here * - nph*eta_here + dsdc4_here * - nph * self.go2 * (detadx2_here + detadx_here/bn_here) ) 
                                                    + ut_here * np.conj(dsdc3_here * (- nph * detadx_here) + dsdc4_here * (-nph * self.go2 * (detadx3_here + detadx2_here/bn_here)))
                                                    + np.conj(ut_here)*(dsdc3_here * (- nph * detadx_here) + dsdc4_here * (-nph * self.go2 * (detadx3_here + detadx2_here/bn_here)))
                                                    + ut_here/bn_here * np.conj( dsdc3_here * - nph*eta_here + dsdc4_here * - nph * self.go2 * (detadx2_here + detadx_here/bn_here) ) 
                                                    + np.conj(ut_here)/bn_here*( dsdc3_here * - nph*eta_here + dsdc4_here * - nph * self.go2 * (detadx2_here + detadx_here/bn_here) ) 
                                                    ).mean(2)
        
        dT_dsn_1[:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * np.real( ut_here * np.conj(dsdc3_here * nph * eta_here/(2*dx_here) + dsdc4_here * nph * self.go2 /(2*dx_here) * (detadx2_here + detadx_here/bn_here)) 
                                 + np.conj(ut_here)*(dsdc3_here * nph * eta_here/(2*dx_here) + dsdc4_here * nph * self.go2 /(2*dx_here) * (detadx2_here + detadx_here/bn_here)) ).mean(2)

    #I added here thwat we divide by soc. this is different as the single channel case. Needs a check maybe....
    #the sb are checked and correct, the sn are also checked but not understood why the minus sign change 

    return dT_dsb_1, dT_dsb0, dT_dsb1 , dT_dsn_1.T, dT_dsn0.T, dT_dsn1.T


def func_sol_intz(self,tid_here,key): 
    # =============================================================================
    # This function calculates the contribution of the tides to the 
    # depth-perturbed solution vector
    # =============================================================================
    #load from inp
    dstidx , dstipdx , dstdz = tid_here
    
    #do the calculations
    term1 = self.Lsc/self.soc_sca * (-1/4*np.real(self.ch_pars[key]['utp']*np.conj(dstidx) + np.conj(self.ch_pars[key]['utp'])*dstidx)  * np.cos(self.nn*np.pi*self.z_nd)).mean(2)
    term2 = self.Lsc/self.soc_sca * (-1/4*np.real(self.ch_pars[key]['utb']*np.conj(dstipdx) + np.conj(self.ch_pars[key]['utb'])*dstipdx)* np.cos(self.nn*np.pi*self.z_nd)).mean(2)
    term3 = self.Lsc/self.soc_sca * (-1/4*np.real(self.ch_pars[key]['wt'] * np.conj(dstdz) + np.conj(self.ch_pars[key]['wt']) * dstdz)  * np.cos(self.nn*np.pi*self.z_nd)).mean(2)

    return term1, term2, term3

def func_jac_intz(self,key): 
    # =============================================================================
    # Associated Jacobian for the contribution of the tide to the subtidal stratification
    # =============================================================================
    #prepare
    di_here = self.ch_pars[key]['di']
    dsdc2_here   = self.ch_pars[key]['c2c']#[:,di_here[dom]:di_here[dom+1]]
    dsdc3_here   = self.ch_pars[key]['c3c']#[:,di_here[dom]:di_here[dom+1]]
    dsdc4_here   = self.ch_pars[key]['c4c']#[:,di_here[dom]:di_here[dom+1]]
    dspdc2_here  = self.ch_pars[key]['c2pc']#[:,di_here[dom]:di_here[dom+1]]
    dspdc3_here  = self.ch_pars[key]['c3pc']#[:,di_here[dom]:di_here[dom+1]]
    dspdc4_here  = self.ch_pars[key]['c4pc']#[:,di_here[dom]:di_here[dom+1]]
    dstdz_dc2_here = self.ch_pars[key]['c2c_z']#[:,di_here[dom]:di_here[dom+1]]
    dstdz_dc3_here = self.ch_pars[key]['c3c_z']#[:,di_here[dom]:di_here[dom+1]]
    dstdz_dc4_here = self.ch_pars[key]['c4c_z']#[:,di_here[dom]:di_here[dom+1]]
        
    dt1_dsb1,dt1_dsb0,dt1_dsb_1 = np.zeros((self.N,di_here[-1])) , np.zeros((self.N,di_here[-1])) , np.zeros((self.N,di_here[-1])) 
    dt1_dsn1,dt1_dsn0,dt1_dsn_1 = np.zeros((self.N,self.N,di_here[-1])) , np.zeros((self.N,self.N,di_here[-1])) , np.zeros((self.N,self.N,di_here[-1]))
    dt2_dsb1,dt2_dsb0,dt2_dsb_1 = np.zeros((self.N,di_here[-1])) , np.zeros((self.N,di_here[-1])) , np.zeros((self.N,di_here[-1])) 
    dt2_dsn1,dt2_dsn0,dt2_dsn_1 = np.zeros((self.N,self.N,di_here[-1])) , np.zeros((self.N,self.N,di_here[-1])) , np.zeros((self.N,self.N,di_here[-1]))
    dt3_dsb1,dt3_dsb_1,dt3_dsn0 = np.zeros((self.N,di_here[-1])) , np.zeros((self.N,di_here[-1])) , np.zeros((self.N,self.N,di_here[-1])) 
      
    # =============================================================================
    # the terms in the Jacobian
    # =============================================================================        
    for dom in range(self.ch_pars[key]['n_seg']):     
        utp_here     = self.ch_pars[key]['utp'][:,di_here[dom]:di_here[dom+1]]
        utb_here     = self.ch_pars[key]['utb'][:,di_here[dom]:di_here[dom+1]]
        wt_here      = self.ch_pars[key]['wt'][:,di_here[dom]:di_here[dom+1]]
        eta_here     = self.ch_pars[key]['eta'][:,di_here[dom]:di_here[dom+1]]
        detadx_here  = self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]
        detadx2_here = self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]]
        detadx3_here = self.ch_pars[key]['detadx3'][:,di_here[dom]:di_here[dom+1]]
        dx_here      = self.ch_gegs[key]['dx'][dom]
        nph          = self.nn*np.pi/self.ch_gegs[key]['H']
        zlist        = self.ch_pars[key]['zlist']


        #term 1
        dt1_dsb1[:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * (np.real(utp_here * np.conj( dsdc2_here * self.go2 * (detadx2_here/(2*dx_here) + detadx_here/(dx_here**2)) ) 
                               + np.conj(utp_here) * dsdc2_here * self.go2 * (detadx2_here/(2*dx_here) + detadx_here/(dx_here**2))) * np.cos(nph*zlist)).mean(2)
        
        dt1_dsb0[:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * (np.real( utp_here * np.conj(dsdc2_here * -2 * self.go2 * detadx_here/(dx_here**2)) 
                               + np.conj(utp_here) * (dsdc2_here * -2 * self.go2 * detadx_here/(dx_here**2))) * np.cos(nph*zlist)).mean(2)
    
        dt1_dsb_1[:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * (np.real(utp_here * np.conj(dsdc2_here * self.go2 * (-detadx2_here/(2*dx_here) + detadx_here/(dx_here**2)))
                                + np.conj(utp_here) * dsdc2_here * self.go2 * (-detadx2_here/(2*dx_here) + detadx_here/(dx_here**2))) * np.cos(nph*zlist)).mean(2)
                                                           
        dt1_dsn1[:,:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * (np.real( utp_here * np.conj(dsdc3_here * -nph * eta_here/(2*dx_here) + dsdc4_here * - nph * self.go2 /(2*dx_here) * (detadx2_here + detadx_here/self.ch_pars[key]['bn'][dom])) 
                                + np.conj(utp_here)*(dsdc3_here * - nph * eta_here/(2*dx_here) + dsdc4_here * -nph * self.go2 /(2*dx_here) * (detadx2_here + detadx_here/self.ch_pars[key]['bn'][dom])))[:,np.newaxis,:,:] * np.cos(nph*zlist) ).mean(-1)
    
        dt1_dsn0[:,:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * (np.real(    utp_here * np.conj(dsdc3_here * (- nph * detadx_here) + dsdc4_here * (-nph * self.go2 * (detadx3_here + detadx2_here/self.ch_pars[key]['bn'][dom])))
                                                                  + np.conj(utp_here) * (dsdc3_here * (- nph * detadx_here) + dsdc4_here * (-nph * self.go2 * (detadx3_here + detadx2_here/self.ch_pars[key]['bn'][dom]))))[:,np.newaxis,:,:] * np.cos(nph*zlist)).mean(-1)
    
        dt1_dsn_1[:,:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * (np.real( utp_here * np.conj(dsdc3_here * -nph * eta_here/(2*dx_here) + dsdc4_here * -nph * self.go2 /(2*dx_here) * (detadx2_here + detadx_here/self.ch_pars[key]['bn'][dom])) 
                                + np.conj(utp_here)*(dsdc3_here * - nph * eta_here/(2*dx_here) + dsdc4_here * -nph * self.go2 /(2*dx_here) * (detadx2_here + detadx_here/self.ch_pars[key]['bn'][dom])) )[:,np.newaxis,:,:] * np.cos(nph*zlist) ).mean(-1) *-1 
        #term 2
        dt2_dsb1[:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * (np.real(utb_here * np.conj( dspdc2_here * self.go2 * (detadx2_here/(2*dx_here) + detadx_here/(dx_here**2)) ) 
                               + np.conj(utb_here) * dspdc2_here * self.go2 * (detadx2_here/(2*dx_here) + detadx_here/(dx_here**2))) * np.cos(nph*zlist)).mean(2)
        
        dt2_dsb0[:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * (np.real( utb_here * np.conj(dspdc2_here * -2 * self.go2 * detadx_here/(dx_here**2)) 
                               + np.conj(utb_here) * (dspdc2_here * -2 * self.go2 * detadx_here/(dx_here**2))) * np.cos(nph*zlist)).mean(2)
    
        dt2_dsb_1[:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * (np.real(utb_here * np.conj(dspdc2_here * self.go2 * (-detadx2_here/(2*dx_here) + detadx_here/(dx_here**2)))
                                + np.conj(utb_here) * dspdc2_here * self.go2 * (-detadx2_here/(2*dx_here) + detadx_here/(dx_here**2))) * np.cos(nph*zlist)).mean(2)
                                                           
        dt2_dsn1[:,:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * (np.real( utb_here * np.conj(dspdc3_here * -nph * eta_here/(2*dx_here) + dspdc4_here * -nph * self.go2 /(2*dx_here) * (detadx2_here + detadx_here/self.ch_pars[key]['bn'][dom])) 
                                + np.conj(utb_here)*(dspdc3_here * -nph * eta_here/(2*dx_here) + dspdc4_here * -nph * self.go2 /(2*dx_here) * (detadx2_here + detadx_here/self.ch_pars[key]['bn'][dom])))[:,np.newaxis,:,:] * np.cos(nph*zlist) ).mean(-1)
    
        dt2_dsn0[:,:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * (np.real(    utb_here * np.conj(dspdc3_here * (- nph * detadx_here) + dspdc4_here * (-nph * self.go2 * (detadx3_here + detadx2_here/self.ch_pars[key]['bn'][dom])))
                                                                  + np.conj(utb_here) * (dspdc3_here * (- nph * detadx_here) + dspdc4_here * (-nph * self.go2 * (detadx3_here + detadx2_here/self.ch_pars[key]['bn'][dom]))))[:,np.newaxis,:,:] * np.cos(nph*zlist)).mean(-1)
    
        dt2_dsn_1[:,:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * (np.real( utb_here * np.conj(dspdc3_here * -nph * eta_here/(2*dx_here) + dspdc4_here * -nph * self.go2 /(2*dx_here) * (detadx2_here + detadx_here/self.ch_pars[key]['bn'][dom])) 
                                + np.conj(utb_here)*(dspdc3_here * -nph * eta_here/(2*dx_here) + dspdc4_here * -nph * self.go2 /(2*dx_here) * (detadx2_here + detadx_here/self.ch_pars[key]['bn'][dom])) )[:,np.newaxis,:,:] * np.cos(nph*zlist) ).mean(-1) *-1 
        
        #term 3
        dt3_dsb1[:,di_here[dom]:di_here[dom+1]]  = -1/4*self.Lsc * (np.real(wt_here * np.conj(dstdz_dc2_here * self.go2 * detadx_here * 1/(2*dx_here)) + np.conj(wt_here) * (dstdz_dc2_here * self.go2 * detadx_here * 1/(2*dx_here)))* np.cos(nph*zlist) ).mean(-1)
        dt3_dsb_1[:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * (np.real(wt_here * np.conj(dstdz_dc2_here * self.go2 * detadx_here *-1/(2*dx_here)) + np.conj(wt_here) * (dstdz_dc2_here * self.go2 * detadx_here *-1/(2*dx_here)))* np.cos(nph*zlist) ).mean(-1)
        dt3_dsn0[:,:,di_here[dom]:di_here[dom+1]]= -1/4*self.Lsc * (np.real(wt_here * np.conj(- nph * eta_here * dstdz_dc3_here - nph * self.go2 * (detadx2_here + detadx_here/self.ch_pars[key]['bn'][dom]) * dstdz_dc4_here ) \
                                        + np.conj(wt_here) * (- nph * eta_here * dstdz_dc3_here - nph * self.go2 * (detadx2_here + detadx_here/self.ch_pars[key]['bn'][dom]) * dstdz_dc4_here   ))[:,np.newaxis,:,:] * np.cos(nph*zlist) ).mean(-1)
        
    return (dt1_dsb_1,dt1_dsb0,dt1_dsb1 , dt1_dsn_1,dt1_dsn0,dt1_dsn1) , (dt2_dsb_1,dt2_dsb0,dt2_dsb1 , dt2_dsn_1,dt2_dsn0,dt2_dsn1) , (dt3_dsb_1,dt3_dsb1,dt3_dsn0)
    #return (dt1_dsb_1*0,dt1_dsb0*0,dt1_dsb1*0 , dt1_dsn_1*0,dt1_dsn0*0,dt1_dsn1*0) , (dt2_dsb_1,dt2_dsb0,dt2_dsb1 , dt2_dsn_1,dt2_dsn0,dt2_dsn1) , (dt3_dsb_1,dt3_dsb1,dt3_dsn0)


def func_sol_ic(self,tid_here,key): 
    # =============================================================================
    # Function to calculate the contribution of the boundary layer correction to the inner domain
    # =============================================================================
    #prepare
    tid_stci_xL, tid_stci_x_xL , tid_stcpi_x_xL, tid_stc_zi_xL , tid_stci_x0 , tid_stci_x_x0 , tid_stcpi_x_x0, tid_stc_zi_x0 = tid_here
    
    #empty solution lists
    da_xL = []
    da_x0 = []
    dv_xL = []
    dv_x0 = []
    
    for dom in range(self.ch_pars[key]['n_seg']):
        #at x=-L
                
        #preparations
        len_xL  = len(tid_stci_xL[dom])
        ut_xL   = self.ch_pars[key]['ut'][0,self.ch_pars[key]['di'][dom] : self.ch_pars[key]['di'][dom] + len_xL]
        utb_xL  = self.ch_pars[key]['utb'][0,self.ch_pars[key]['di'][dom] : self.ch_pars[key]['di'][dom] + len_xL]
        utp_xL  = self.ch_pars[key]['utp'][0,self.ch_pars[key]['di'][dom] : self.ch_pars[key]['di'][dom] + len_xL]
        wt_xL   = self.ch_pars[key]['wt'][0,self.ch_pars[key]['di'][dom] : self.ch_pars[key]['di'][dom] + len_xL]
        ut_x_xL = self.ch_pars[key]['dutdx'][0,self.ch_pars[key]['di'][dom]  : self.ch_pars[key]['di'][dom] + len_xL]
                
        #depth - averaged
        da_xL.append( np.mean(1/4 * np.real(ut_xL*np.conj(tid_stci_x_xL [dom]) + np.conj(ut_xL) * tid_stci_x_xL [dom] \
                                    + ut_x_xL*np.conj(tid_stci_xL[dom]) + np.conj(ut_x_xL)*tid_stci_xL[dom] \
                                    + (ut_xL*np.conj(tid_stci_xL[dom]) + np.conj(ut_xL) * tid_stci_xL[dom])/self.ch_pars[key]['bex'][self.ch_pars[key]['di'][dom] : self.ch_pars[key]['di'][dom] + len_xL,np.newaxis])  , -1)* self.Lsc/self.soc_sca )

        # depth-varying
        t1 = np.mean(1/4 * np.real(utp_xL * np.conj(tid_stci_x_xL [dom]) + np.conj(utp_xL) * tid_stci_x_xL [dom] )[np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc_sca
        t2 = np.mean(1/4 * np.real(utb_xL * np.conj(tid_stcpi_x_xL[dom])+ np.conj(utb_xL) * tid_stcpi_x_xL[dom])[np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc_sca
        t3 = np.mean(1/4 * np.real(wt_xL  * np.conj(tid_stc_zi_xL[dom]) + np.conj(wt_xL)  * tid_stc_zi_xL[dom] )[np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc_sca
        
        dv_xL.append(t1+t2+t3)
        
        #at x=0
        
        #preparations
        len_x0  = len(tid_stci_x0[dom])
        ut_x0   = self.ch_pars[key]['ut'][0,self.ch_pars[key]['di'][dom+1] - len_x0 : self.ch_pars[key]['di'][dom+1] ]
        utb_x0  = self.ch_pars[key]['utb'][0,self.ch_pars[key]['di'][dom+1] - len_x0 : self.ch_pars[key]['di'][dom+1] ]
        utp_x0  = self.ch_pars[key]['utp'][0,self.ch_pars[key]['di'][dom+1] - len_x0 : self.ch_pars[key]['di'][dom+1] ]
        wt_x0   = self.ch_pars[key]['wt'][0,self.ch_pars[key]['di'][dom+1] - len_x0 : self.ch_pars[key]['di'][dom+1] ]
        ut_x_x0 = self.ch_pars[key]['dutdx'][0,self.ch_pars[key]['di'][dom+1] - len_x0 : self.ch_pars[key]['di'][dom+1] ]
        
        #depth - averaged
        da_x0.append( np.mean(1/4 * np.real(ut_x0*np.conj(tid_stci_x_x0[dom]) + np.conj(ut_x0) * tid_stci_x_x0[dom] \
                                    + ut_x_x0*np.conj(tid_stci_x0[dom]) + np.conj(ut_x_x0)*tid_stci_x0[dom] \
                                    + (ut_x0*np.conj(tid_stci_x0[dom]) + np.conj(ut_x0) * tid_stci_x0[dom])/self.ch_pars[key]['bex'][self.ch_pars[key]['di'][dom+1] - len_x0 : self.ch_pars[key]['di'][dom+1],np.newaxis])  , -1)* self.Lsc/self.soc_sca )

        # depth-varying
        t1 = np.mean(1/4 * np.real(utp_x0 * np.conj(tid_stci_x_x0[dom]) + np.conj(utp_x0) * tid_stci_x_x0[dom] )[np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc_sca
        t2 = np.mean(1/4 * np.real(utb_x0 * np.conj(tid_stcpi_x_x0[dom])+ np.conj(utb_x0) * tid_stcpi_x_x0[dom])[np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc_sca
        t3 = np.mean(1/4 * np.real(wt_x0  * np.conj(tid_stc_zi_x0[dom]) + np.conj(wt_x0)  * tid_stc_zi_x0[dom] )[np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc_sca
        
        dv_x0.append(t1+t2+t3)

    return da_xL , dv_xL , da_x0 , dv_x0
    


def func_jac_ic(self,tid_here,key): 
    # =============================================================================
    # Function to calculate the contribution of the boundary layer correction to the
    # jacobian of the inner domain
    # =============================================================================
    #prepare
    tid_dstci_xL, tid_dstci_x_xL , tid_dstcpi_x_xL, tid_dstc_zi_xL , tid_dstci_x0 , tid_dstci_x_x0 , tid_dstcpi_x_x0, tid_dstc_zi_x0 = tid_here

    #empty lists
    daR_xL, daI_xL = [] , []
    daR_x0, daI_x0 = [] , []
    dvR_xL, dvI_xL = [] , []
    dvR_x0, dvI_x0 = [] , []
    
    for dom in range(self.ch_pars[key]['n_seg']):
        #at x=-L
        
        #preparations        
        len_xL  = len(tid_dstci_xL[dom][0])
        ut_xL   = self.ch_pars[key]['ut'][0,self.ch_pars[key]['di'][dom] : self.ch_pars[key]['di'][dom] + len_xL]
        utb_xL  = self.ch_pars[key]['utb'][0,self.ch_pars[key]['di'][dom] : self.ch_pars[key]['di'][dom] + len_xL]
        utp_xL  = self.ch_pars[key]['utp'][0,self.ch_pars[key]['di'][dom] : self.ch_pars[key]['di'][dom] + len_xL]
        wt_xL   = self.ch_pars[key]['wt'][0,self.ch_pars[key]['di'][dom] : self.ch_pars[key]['di'][dom] + len_xL]
        ut_x_xL = self.ch_pars[key]['dutdx'][0,self.ch_pars[key]['di'][dom]  : self.ch_pars[key]['di'][dom] + len_xL]

        #depth - averaged
        daR_xL.append( np.mean(1/4 * (2*np.real(ut_xL) * tid_dstci_x_xL[dom] + 2*np.real(ut_x_xL) * tid_dstci_xL[dom] \
                             + (2*np.real(ut_xL) * tid_dstci_xL[dom])/self.ch_pars[key]['bex'][self.ch_pars[key]['di'][dom] : self.ch_pars[key]['di'][dom] + len_xL, np.newaxis])  , -1)* self.Lsc/self.soc_sca)
        
        daI_xL.append( np.mean(1/4 * (2*np.imag(ut_xL) * tid_dstci_x_xL[dom] + 2*np.imag(ut_x_xL) * tid_dstci_xL[dom] \
                      + (2*np.imag(ut_xL) * tid_dstci_xL[dom])/self.ch_pars[key]['bex'][self.ch_pars[key]['di'][dom] : self.ch_pars[key]['di'][dom] + len_xL, np.newaxis])  , -1)* self.Lsc/self.soc_sca)
        
        
        #depth-varying
        dt1 = np.mean(1/4 * (2*np.real(utp_xL) * tid_dstci_x_xL[dom])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc_sca
        dt2 = np.mean(1/4 * (2*np.real(utb_xL) * tid_dstcpi_x_xL[dom])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc_sca
        dt3 = np.mean(1/4 * (2*np.real(wt_xL)  * tid_dstc_zi_xL[dom])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc_sca
        
        dvR_xL.append(dt1+dt2+dt3)
                
        dt1 = np.mean(1/4 * (2*np.imag(utp_xL) * tid_dstci_x_xL[dom])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc_sca
        dt2 = np.mean(1/4 * (2*np.imag(utb_xL) * tid_dstcpi_x_xL[dom])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc_sca
        dt3 = np.mean(1/4 * (2*np.imag(wt_xL)  * tid_dstc_zi_xL[dom])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd) , -1) * self.Lsc/self.soc_sca
        
        dvI_xL.append(dt1+dt2+dt3)
        
        
        #at x=0
        
        #preparations
        len_x0  = len(tid_dstci_x0[dom][0])
        ut_x0   = self.ch_pars[key]['ut'][0,self.ch_pars[key]['di'][dom+1] - len_x0 : self.ch_pars[key]['di'][dom+1] ]
        utb_x0  = self.ch_pars[key]['utb'][0,self.ch_pars[key]['di'][dom+1] - len_x0 : self.ch_pars[key]['di'][dom+1] ]
        utp_x0  = self.ch_pars[key]['utp'][0,self.ch_pars[key]['di'][dom+1] - len_x0 : self.ch_pars[key]['di'][dom+1] ]
        wt_x0   = self.ch_pars[key]['wt'][0,self.ch_pars[key]['di'][dom+1] - len_x0 : self.ch_pars[key]['di'][dom+1] ]
        ut_x_x0 = self.ch_pars[key]['dutdx'][0,self.ch_pars[key]['di'][dom+1] - len_x0 : self.ch_pars[key]['di'][dom+1] ]
        
        #depth - averaged
        daR_x0.append( np.mean(1/4 * (2*np.real(ut_x0) * tid_dstci_x_x0[dom] + 2*np.real(ut_x_x0) * tid_dstci_x0[dom] \
                                     + (2*np.real(ut_x0) * tid_dstci_x0[dom])/self.ch_pars[key]['bex'][self.ch_pars[key]['di'][dom+1] - len_x0 : self.ch_pars[key]['di'][dom+1],np.newaxis])  , -1)* self.Lsc/self.soc_sca)
        daI_x0.append( np.mean(1/4 * (2*np.imag(ut_x0) * tid_dstci_x_x0[dom] + 2*np.imag(ut_x_x0) * tid_dstci_x0[dom] \
                             + (2*np.imag(ut_x0) * tid_dstci_x0[dom])/self.ch_pars[key]['bex'][self.ch_pars[key]['di'][dom+1] - len_x0 : self.ch_pars[key]['di'][dom+1],np.newaxis])  , -1)* self.Lsc/self.soc_sca)

        #depth-varying
        dt1 = np.mean(1/4 * (2*np.real(utp_x0) * tid_dstci_x_x0[dom])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc_sca
        dt2 = np.mean(1/4 * (2*np.real(utb_x0) * tid_dstcpi_x_x0[dom])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc_sca
        dt3 = np.mean(1/4 * (2*np.real(wt_x0)  * tid_dstc_zi_x0[dom])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc_sca
        
        dvR_x0.append(dt1+dt2+dt3)
                
        dt1 = np.mean(1/4 * (2*np.imag(utp_x0) * tid_dstci_x_x0[dom])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc_sca
        dt2 = np.mean(1/4 * (2*np.imag(utb_x0) * tid_dstcpi_x_x0[dom])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc_sca
        dt3 = np.mean(1/4 * (2*np.imag(wt_x0)  * tid_dstc_zi_x0[dom])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd) , -1) * self.Lsc/self.soc_sca
        
        dvI_x0.append(dt1+dt2+dt3)
   
    return daR_xL, daI_xL, dvR_xL, dvI_xL, daR_x0, daI_x0, dvR_x0, dvI_x0
    

def func_st_after(key,sn,dsndx,dsbdx,dsbdx2,bb):       
    # =============================================================================
    # Function to calculate total transport and salinity
    # for plotting purposes afterwards
    # DOES NOT WORK NOW 
    # =============================================================================
    di_here = self.ch_pars[key]['di']
    zlist = np.linspace(-self.ch_gegs[key]['H'],0,self.nz)[np.newaxis,np.newaxis,:]

    c1 = - self.ch_pars[key]['Kv_ti']/(1j*self.omega)
    c2, c3, c4 = np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex)
    for dom in range(len(self.ch_pars[key]['nxn'])): 
        c2[:,di_here[dom]:di_here[dom+1]] = self.go2 * self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]] * dsbdx[:,di_here[dom]:di_here[dom+1]]
        c3[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * sn[:,di_here[dom]:di_here[dom+1]] * self.ch_pars[key]['eta'][:,di_here[dom]:di_here[dom+1]]
        c4[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * sn[:,di_here[dom]:di_here[dom+1]] * self.go2 \
            * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom])
   
    st = sti(zlist,(c1,c2,c3,c4),(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))    
    
    #calculate corrected salinity
    st_cor = st.copy()
    sigma = np.linspace(-1,0,self.nz)
    if self.ch_gegs[key]['loc x=-L'][0] == 'j': 
        px_here = self.ch_outp[key]['px']-self.ch_outp[key]['px'][0]
        cor_dom = np.exp(-px_here/np.sqrt(self.Kh_tide/self.omega))[:,np.newaxis] *  np.sum(bb[0][:,np.newaxis] * np.cos(np.pi*self.m0[:,np.newaxis]*sigma) , axis=0)
        st_cor += self.soc_sca * cor_dom

    if self.ch_gegs[key]['loc x=0'][0] == 'j': 
        px_here = self.ch_outp[key]['px']
        cor_dom = np.exp(px_here/np.sqrt(self.Kh_tide/self.omega))[:,np.newaxis] *  np.sum(bb[1][:,np.newaxis] * np.cos(np.pi*self.m0[:,np.newaxis]*sigma) , axis=0)
        st_cor += self.soc_sca * cor_dom
        
    return st_cor


def sol_tidal(self, key, tid_inp):
    # =============================================================================
    # Here all the functions above are run, and everything is put in the right place in the
    # solution vecotr. 
    # =============================================================================
    
    # =============================================================================
    # part to add to the solution vector
    # =============================================================================
    so = np.zeros(self.ch_inds[key]['di3'][-1]*self.M)
    #local variables, for shorter notation
    inds = self.ch_inds[key].copy()
    
    #run tidal module
    tid_here = (tid_inp['st'] , tid_inp['dstidx'])    
    tid_otp = func_sol_int(self, tid_here, key)
    #add the results of the tidal module to the solution vector - depth-averaged
    so[inds['xn_m']] = tid_otp[inds['x']]    
    
    #tides in vertical balance.
    tid_here = (tid_inp['dstidx'], tid_inp['dstipdx'], tid_inp['dstdz'])    
    t1_sp, t2_sp, t3_sp = func_sol_intz(self,tid_here,key)    
    so[inds['xnr_mj']] +=  t1_sp[inds['j1'],inds['xr']] + t2_sp[inds['j1'],inds['xr']] + t3_sp[inds['j1'],inds['xr']]
    
    # =============================================================================
    # add the correction due to the boundary layer in the internal part. 
    # =============================================================================
    tid_here = ( tid_inp['stci_x=-L'], tid_inp['stci_x_x=-L'], tid_inp['stcpi_x_x=-L'], tid_inp['stc_zi_x=-L'] ,
                tid_inp['stci_x=0'], tid_inp['stci_x_x=0'], tid_inp['stcpi_x_x=0'], tid_inp['stc_zi_x=0'] )
    tid_cor = func_sol_ic(self,tid_here,key)
    
    for dom in range(self.ch_pars[key]['n_seg']):        
        #indices eigenlijk ergens anders vandaan, maarja je moet wat.

        #x=-L
        if len(tid_cor[0][dom]) > 1:  
            #prepare
            istart = self.ch_inds[key]['di3'][dom]+2+1
            istop  = self.ch_inds[key]['di3'][dom]+2+len(tid_cor[0][dom])
            
            #depth-averaged
            so[np.arange(istart , istop)*self.M] += tid_cor[0][dom][1:]
            #depth-varying
            so[self.mm[:,np.newaxis] + np.arange(istart , istop)*self.M] += tid_cor[1][dom][:,1:]
            
        #x=0
        if len(tid_cor[2][dom]) > 1:  
            istart = self.ch_inds[key]['di3'][dom+1]-2-len(tid_cor[2][dom])
            istop  = self.ch_inds[key]['di3'][dom+1]-2-1
            #depth-averaged
            so[np.arange(istart , istop)*self.M] += tid_cor[2][dom][:-1]
            #depth-varying
            so[self.mm[:,np.newaxis] + np.arange(istart , istop)*self.M] += tid_cor[3][dom][:,:-1]
    
    return so


def jac_tidal_fix(self, key):
    # =============================================================================
    # contribution to the jacobian by the tidal dynamics
    # =============================================================================
    #prepare
    jac = np.zeros((self.ch_inds[key]['di3'][-1]*self.M,self.ch_inds[key]['di3'][-1]*self.M))
    inds = self.ch_inds[key].copy()
    # ============================================================================
    #  depth-averaged
    # =============================================================================
    #contribution by tides, lets hope it works this way.
    tid_raw = func_jac_int(self,key)
    
    #left
    jac[inds['xn_m'], inds['xnm_m']] += tid_raw[0][inds['x']]
    jac[inds['xnr_m'], inds['xnrm_mj']] += tid_raw[5][inds['xr'],inds['j1']]
    #center
    jac[inds['xn_m'],inds['xn_m']] += tid_raw[1][inds['x']]
    jac[inds['xnr_m'], inds['xnr_mj']] += tid_raw[4][inds['xr'],inds['j1']] 
    #right
    jac[inds['xn_m'], inds['xnp_m']] += tid_raw[2][inds['x']]
    jac[inds['xnr_m'], inds['xnrp_mj']] += tid_raw[3][inds['xr'],inds['j1']]
    
    # =============================================================================
    # add tides to the vertical balance
    # =============================================================================
    NRv = func_jac_intz(self,key)
    t1a,t1b,t1c,t1d,t1e,t1f = NRv[0]
    t2a,t2b,t2c,t2d,t2e,t2f = NRv[1]
    t3a,t3c,t3e = NRv[2]
    
    jac[inds['xnr_mj'],inds['xnrm_m']] +=  (t1a[inds['j1'],inds['xr']]+t2a[inds['j1'],inds['xr']]+t3a[inds['j1'],inds['xr']])
    jac[inds['xnr_mj'],inds['xnr_m']]  +=  (t1b[inds['j1'],inds['xr']]+t2b[inds['j1'],inds['xr']])
    jac[inds['xnr_mj'],inds['xnrp_m']] +=  (t1c[inds['j1'],inds['xr']]+t2c[inds['j1'],inds['xr']]+t3c[inds['j1'],inds['xr']])
    
    jac[inds['xnr_mj2'],inds['xnrm_mk']] += (t1d[inds['k1'],inds['j12'],inds['xr2']]+t2d[inds['k1'],inds['j12'],inds['xr2']])
    jac[inds['xnr_mj2'], inds['xnr_mk']] += (t1e[inds['k1'],inds['j12'],inds['xr2']]+t2e[inds['k1'],inds['j12'],inds['xr2']]+t3e[inds['k1'],inds['j12'],inds['xr2']])
    jac[inds['xnr_mj2'],inds['xnrp_mk']] += (t1f[inds['k1'],inds['j12'],inds['xr2']]+t2f[inds['k1'],inds['j12'],inds['xr2']])
    
    # =============================================================================
    # Add boundary layer in the internal part.     
    # =============================================================================
    tid_here = ( self.ch_pars[key]['dstci_x=-L'][0], self.ch_pars[key]['dstci_x_x=-L'][0], self.ch_pars[key]['dstcpi_x_x=-L'][0], self.ch_pars[key]['dstc_zi_x=-L'][0] ,
                self.ch_pars[key]['dstci_x=0'][0], self.ch_pars[key]['dstci_x_x=0'][0], self.ch_pars[key]['dstcpi_x_x=0'][0], self.ch_pars[key]['dstc_zi_x=0'][0] )  
    
    tid_cor = func_jac_ic(self,tid_here,key)
    
    for dom in range(self.ch_pars[key]['n_seg']):        
        #x=-L
        len_xL = tid_cor[0][dom].shape[1]
        #define a lot of indices
        istart = self.ch_inds[key]['di3'][dom]+2+1
        istop  = self.ch_inds[key]['di3'][dom]+2+len_xL
        ireal = np.arange(self.ch_inds[key]['di3'][dom]*self.M, (self.ch_inds[key]['di3'][dom]+1)*self.M)
        iimag = np.arange((self.ch_inds[key]['di3'][dom]+1)*self.M, (self.ch_inds[key]['di3'][dom]+2)*self.M)
        
        inds_j1 = np.tile(np.arange(istart, istop)*self.M,self.M).reshape((self.M, istop-istart))
        inds_j2R = np.repeat(ireal,istop-istart).reshape((self.M, istop-istart))
        inds_j2I = np.repeat(iimag,istop-istart).reshape((self.M, istop-istart))
        
        inds_j1v = np.tile(self.mm[:,np.newaxis] + np.arange(istart , istop)*self.M , self.M).T.reshape((self.M,self.N,istop-istart))
        inds_j2Rv = np.repeat(ireal,(istop-istart)*self.N).reshape((self.M,self.N,istop-istart))
        inds_j2Iv = np.repeat(iimag,(istop-istart)*self.N).reshape((self.M,self.N,istop-istart))

        if len_xL > 1 :            
            #depth-averaged
            jac[inds_j1, inds_j2R] += tid_cor[0][dom][:,1:]
            jac[inds_j1, inds_j2I] += tid_cor[1][dom][:,1:]
            
            #depth-varying
            jac[inds_j1v, inds_j2Rv] += tid_cor[2][dom][:,:,1:]
            jac[inds_j1v, inds_j2Iv] += tid_cor[3][dom][:,:,1:]
            

        #x=0
        len_x0 = tid_cor[4][dom].shape[1]
        #define a lot of indices
        istart = self.ch_inds[key]['di3'][dom+1]-2-len_x0
        istop  = self.ch_inds[key]['di3'][dom+1]-2-1
        ireal = np.arange((self.ch_inds[key]['di3'][dom+1]-2)*self.M, (self.ch_inds[key]['di3'][dom+1]-1)*self.M)
        iimag = np.arange((self.ch_inds[key]['di3'][dom+1]-1)*self.M, (self.ch_inds[key]['di3'][dom+1]  )*self.M)

        inds_j1 = np.tile(np.arange(istart, istop)*self.M,self.M).reshape((self.M, istop-istart))
        inds_j2R = np.repeat(ireal,istop-istart).reshape((self.M, istop-istart))
        inds_j2I = np.repeat(iimag,istop-istart).reshape((self.M, istop-istart))
        
        inds_j1v = np.tile(self.mm[:,np.newaxis] + np.arange(istart , istop)*self.M , self.M).T.reshape((self.M,self.N,istop-istart))
        inds_j2Rv = np.repeat(ireal,(istop-istart)*self.N).reshape((self.M,self.N,istop-istart))
        inds_j2Iv = np.repeat(iimag,(istop-istart)*self.N).reshape((self.M,self.N,istop-istart))
        
        if len_x0 > 1 :
            #depth-averaged
            jac[inds_j1, inds_j2R] += tid_cor[4][dom][:,:-1]
            jac[inds_j1, inds_j2I] += tid_cor[5][dom][:,:-1]
            
            #depth-varying
            jac[inds_j1v, inds_j2Rv] += tid_cor[6][dom][:,:,:-1]
            jac[inds_j1v, inds_j2Iv] += tid_cor[7][dom][:,:,:-1]
    
    
    return jac



def jac_tidal_vary(self, key):
    # =============================================================================
    # contribution to the jacobian by the tidal dynamics
    # =============================================================================
    #prepare
    jac = np.zeros((self.ch_inds[key]['di3'][-1]*self.M,self.ch_inds[key]['di3'][-1]*self.M))
  

    return jac







