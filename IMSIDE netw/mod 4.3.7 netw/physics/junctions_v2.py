# =============================================================================
# file where functions are defined which calculate the conditions at the junctions
# of the channels in a channel network
# =============================================================================

import numpy as np

def prep_junc(self):
    # =============================================================================
    # do some preparations for the junctions, so we do not need to repeat this. 
    # Mostly gathering of indices
    # TODO: maybe place here also the tidal things. 
    # =============================================================================
    ju_geg = {}
    
    for j in range(1,self.n_j+1):
        jh = 'j'+str(j)
        ju_geg[jh] = {}
        
        ju_geg[jh]['key'] = []
        ju_geg[jh]['loc'] = []
        ju_geg[jh]['i0']  = []
        ju_geg[jh]['i1']  = []
        ju_geg[jh]['i2']  = []    
        ju_geg[jh]['i3']  = []    
        ju_geg[jh]['obcR']  = []    
        ju_geg[jh]['obcI']  = []    
        
        H3ch = []
            
        #find the associated data
        for key in self.ch_keys:   
            if self.ch_gegs[key]['loc x=-L'] == jh:  
                ju_geg[jh]['key'].append(key)
                ju_geg[jh]['loc'].append('loc x=-L')
                ju_geg[jh]['i0'].append(self.ch_inds[key]['ob_x=-L_s0'])
                ju_geg[jh]['i1'].append(self.ch_inds[key]['ob_x=-L_s1'])
                ju_geg[jh]['i2'].append(self.ch_inds[key]['ob_x=-L_s2'])
                ju_geg[jh]['i3'].append(self.ch_inds[key]['ob_x=-L_s3'])
                ju_geg[jh]['obcR'].append(self.ch_inds[key]['ob_x=-L_bcR'])
                ju_geg[jh]['obcI'].append(self.ch_inds[key]['ob_x=-L_bcI'])
                H3ch.append(self.ch_gegs[key]['Hn'][0])        
        
            elif self.ch_gegs[key]['loc x=0'] == jh: 
                ju_geg[jh]['key'].append(key)
                ju_geg[jh]['loc'].append('loc x=0')
                ju_geg[jh]['i0'].append(self.ch_inds[key]['ob_x=0_s_1'])
                ju_geg[jh]['i1'].append(self.ch_inds[key]['ob_x=0_s_2'])
                ju_geg[jh]['i2'].append(self.ch_inds[key]['ob_x=0_s_3'])
                ju_geg[jh]['i3'].append(self.ch_inds[key]['ob_x=0_s_4'])
                ju_geg[jh]['obcR'].append(self.ch_inds[key]['ob_x=0_bcR'])
                ju_geg[jh]['obcI'].append(self.ch_inds[key]['ob_x=0_bcI'])
                H3ch.append(self.ch_gegs[key]['Hn'][-1])        

        ju_geg[jh]['Ha'] = np.mean(H3ch)
        
    self.junc_gegs = ju_geg
    #return ju_geg


def func_sol_Tbnd(self, tid_inp, key):   
    # =============================================================================
    # Tidal transport at boundaries, to add to solution vector 
    # =============================================================================
    #load from inp
    st = tid_inp['st'][[0,-1]]
    
    #depth-averaged transport
    flux = self.ch_pars[key]['H'][[0,-1]]*self.ch_pars[key]['b'][[0,-1]] * np.mean(1/4 * np.real(self.ch_pars[key]['ut'][:,[0,-1]]*np.conj(st) + np.conj(self.ch_pars[key]['ut'][:,[0,-1]])*st) , axis=2)[0] / self.soc_sca 

    return flux 

def func_jac_Tbnd(self, key):  
    # =============================================================================
    # Tidal ransport at boundaries, to add to jacobian 
    # =============================================================================
    #prepare
    dsdc2 = self.ch_pars[key]['c2c'][:,[0,-1]]
    dsdc3 = self.ch_pars[key]['c3c'][:,[0,-1]]
    dsdc4 = self.ch_pars[key]['c4c'][:,[0,-1]]
    
    # =============================================================================
    # jacobian, derivatives, 8 terms
    # =============================================================================
    #local variables
    dx_here     = self.ch_gegs[key]['dx'][[0,-1]]
    bn_here     = self.ch_pars[key]['bn'][[0,-1]]
    ut_here     = self.ch_pars[key]['ut'][:,[0,-1]]
    dutdx_here  = self.ch_pars[key]['dutdx'][:,[0,-1]]
    eta_here    = self.ch_pars[key]['eta'][:,[0,-1]]
    detadx_here = self.ch_pars[key]['detadx'][:,[0,-1]]
    detadx2_here= self.ch_pars[key]['detadx2'][:,[0,-1]]
    nph         = self.nn*np.pi/self.ch_pars[key]['H'][[0,-1],np.newaxis]

    #depth-averaged
    #derivatives for x=-L
    dT0_dsb0 = 1/4* self.ch_pars[key]['H'][0]*self.ch_pars[key]['b'][0] * np.real( ut_here[:,0] * np.conj( dsdc2 * self.go2 * detadx_here[:,0] * -3/(2*dx_here[0]) ) + np.conj(ut_here[:,0]) * dsdc2 * self.go2 * detadx_here[:,0] * -3/(2*dx_here[0]) ).mean(2)
    dT0_dsb1 = 1/4* self.ch_pars[key]['H'][0]*self.ch_pars[key]['b'][0] * np.real( ut_here[:,0] * np.conj( dsdc2 * self.go2 * detadx_here[:,0] *  4/(2*dx_here[0]) ) + np.conj(ut_here[:,0]) * dsdc2 * self.go2 * detadx_here[:,0] *  4/(2*dx_here[0]) ).mean(2)
    dT0_dsb2 = 1/4* self.ch_pars[key]['H'][0]*self.ch_pars[key]['b'][0] * np.real( ut_here[:,0] * np.conj( dsdc2 * self.go2 * detadx_here[:,0] * -1/(2*dx_here[0]) ) + np.conj(ut_here[:,0]) * dsdc2 * self.go2 * detadx_here[:,0] * -1/(2*dx_here[0]) ).mean(2)
    dT0_dsn0 = 1/4* self.ch_pars[key]['H'][0]*self.ch_pars[key]['b'][0] * np.real( ut_here[:,0] * np.conj( dsdc3 * - nph*eta_here[:,0] + dsdc4 * - nph * self.go2 * (detadx2_here[:,0] + detadx_here[:,0]/bn_here[0]) ) 
                                                           + np.conj(ut_here[:,0]) * ( dsdc3 * - nph*eta_here[:,0] + dsdc4 * - nph * self.go2 * (detadx2_here[:,0] + detadx_here[:,0]/bn_here[0]) ) ).mean(2) 
    
    #derivatives for x=0
    dT_1_dsb_1 = 1/4* self.ch_pars[key]['H'][-1]*self.ch_pars[key]['b'][-1] * np.real( ut_here[:,1] * np.conj( dsdc2 * self.go2 * detadx_here[:,1] *  3/(2*dx_here[1]) ) + np.conj(ut_here[:,1]) * dsdc2 * self.go2 * detadx_here[:,1] *  3/(2*dx_here[1]) ).mean(2)
    dT_1_dsb_2 = 1/4* self.ch_pars[key]['H'][-1]*self.ch_pars[key]['b'][-1] * np.real( ut_here[:,1] * np.conj( dsdc2 * self.go2 * detadx_here[:,1] * -4/(2*dx_here[1]) ) + np.conj(ut_here[:,1]) * dsdc2 * self.go2 * detadx_here[:,1] * -4/(2*dx_here[1]) ).mean(2)
    dT_1_dsb_3 = 1/4* self.ch_pars[key]['H'][-1]*self.ch_pars[key]['b'][-1] * np.real( ut_here[:,1] * np.conj( dsdc2 * self.go2 * detadx_here[:,1] *  1/(2*dx_here[1]) ) + np.conj(ut_here[:,1]) * dsdc2 * self.go2 * detadx_here[:,1] *  1/(2*dx_here[1]) ).mean(2)
    dT_1_dsn_1 = 1/4* self.ch_pars[key]['H'][-1]*self.ch_pars[key]['b'][-1] * np.real( ut_here[:,1] * np.conj( dsdc3 * - nph*eta_here[:,1] + dsdc4 * - nph * self.go2 * (detadx2_here[:,1] + detadx_here[:,1]/bn_here[1]) ) 
                                                           + np.conj(ut_here[:,1]) * ( dsdc3 * - nph*eta_here[:,1] + dsdc4 * - nph * self.go2 * (detadx2_here[:,1] + detadx_here[:,1]/bn_here[1]) ) ).mean(2) 
    
    return dT0_dsb0[0,0],dT0_dsb1[0,0],dT0_dsb2[0,0],dT0_dsn0[:,0] , dT_1_dsb_1[0,0],dT_1_dsb_2[0,0],dT_1_dsb_3[0,0],dT_1_dsn_1[:,0]


def func_sol_Tblc(self, zout, key):   
    # =============================================================================
    # Solution for contribution to tidal transport at boundaries due to boundary correction
    # =============================================================================
    
    #load from inp
    stc = np.array([np.sum((zout[self.ch_inds[key]['ob_x=-L_bcR']] + 1j*zout[self.ch_inds[key]['ob_x=-L_bcI']])[:,np.newaxis] * np.cos(np.pi*self.m0[:,np.newaxis]*self.z_nd) , axis=0),
           np.sum((zout[self.ch_inds[key]['ob_x=0_bcR']] + 1j*zout[self.ch_inds[key]['ob_x=0_bcI']])[:,np.newaxis] * np.cos(np.pi*self.m0[:,np.newaxis]*self.z_nd) , axis=0)])[np.newaxis,:,:]
    
    #depth-averaged transport 
    flux = self.ch_pars[key]['H'][[0,-1]]*self.ch_pars[key]['b'][[0,-1]] * np.mean(1/4 * np.real(self.ch_pars[key]['ut'][:,[0,-1]]*np.conj(stc) + np.conj(self.ch_pars[key]['ut'][:,[0,-1]])*stc) , axis=2)[0] 

    return flux


def func_jac_Tblc(self, key):   
    # =============================================================================
    # Jacobian for contribution to tidal transport at boundaries due to boundary correction
    # =============================================================================
    dT_dBR = self.ch_pars[key]['H'][[0,-1]]*self.ch_pars[key]['b'][[0,-1]] * np.mean(1/4 * 2*np.real(self.ch_pars[key]['ut'][:,[0,-1]])*np.cos(self.npzh) , axis=2) #waarschijnlijk kan dit analytisch
    dT_dBI = self.ch_pars[key]['H'][[0,-1]]*self.ch_pars[key]['b'][[0,-1]] * np.mean(1/4 * 2*np.imag(self.ch_pars[key]['ut'][:,[0,-1]])*np.cos(self.npzh) , axis=2) #waarschijnlijk kan dit analytisch
    
    return dT_dBR, dT_dBI


def func_sol_Tblcz(self, zout, key):   
    # =============================================================================
    # Solution for contribution to tidal transport at boundaries due to boundary correction at depth
    # =============================================================================
    
    #load from inp
    stc = np.array([np.sum((zout[self.ch_inds[key]['ob_x=-L_bcR']] + 1j*zout[self.ch_inds[key]['ob_x=-L_bcI']])[:,np.newaxis] * np.cos(np.pi*self.m0[:,np.newaxis]*self.z_nd) , axis=0),
           np.sum((zout[self.ch_inds[key]['ob_x=0_bcR']] + 1j*zout[self.ch_inds[key]['ob_x=0_bcI']])[:,np.newaxis] * np.cos(np.pi*self.m0[:,np.newaxis]*self.z_nd) , axis=0)])[np.newaxis,:,:]
    
    #depth-perturbed transport 
    flux = self.ch_pars[key]['H'][[0,-1]]*self.ch_pars[key]['b'][[0,-1]] * np.mean(1/4 * np.real(self.ch_pars[key]['ut'][:,[0,-1]]*np.conj(stc) + np.conj(self.ch_pars[key]['ut'][:,[0,-1]])*stc)* np.cos(self.npzh)[1:] , axis=2)

    return flux

def func_jac_Tblcz(self, key):   
    # =============================================================================
    # Jacobian for contribution to tidal transport at boundaries due to boundary correction
    # =============================================================================
    #depth-perturbed transport 
    dT_dBR = self.ch_pars[key]['H'][[0,-1]]*self.ch_pars[key]['b'][[0,-1]] * np.mean(1/4 * 2*(np.real(self.ch_pars[key]['ut'][:,[0,-1]])*np.cos(self.npzh))[:,np.newaxis]*np.cos(self.npzh)[1:] , axis=-1) #waarschijnlijk kan dit analytisch
    dT_dBI = self.ch_pars[key]['H'][[0,-1]]*self.ch_pars[key]['b'][[0,-1]] * np.mean(1/4 * 2*(np.imag(self.ch_pars[key]['ut'][:,[0,-1]])*np.cos(self.npzh))[:,np.newaxis]*np.cos(self.npzh)[1:] , axis=-1)  #waarschijnlijk kan dit analytisch
    
    return dT_dBR, dT_dBI


def func_sol_Tbndz(self, tid_inp, key):   
    # =============================================================================
    # Tidal transport at boundaries in vertical, to add to solution vector 
    # =============================================================================
    #load from inp
    st   = tid_inp['st']
    nph  = self.nn*np.pi/self.ch_pars[key]['H']

    #transport at vertical levels
    flux_z = self.ch_pars[key]['H'][[0,-1]]*self.ch_pars[key]['b'][[0,-1]] * np.mean(1/4 * np.real(self.ch_pars[key]['ut'][:,[0,-1]]*np.conj(st[[0,-1]]) + np.conj(self.ch_pars[key]['ut'][:,[0,-1]])*st[[0,-1]]) * np.cos(self.npzh[1:]) , axis=2) / self.soc_sca   

    return flux_z

def func_jac_Tbndz(self, key):  
    # =============================================================================
    # Transport at boundaries, to add to jacobian 
    # =============================================================================
    #prepare
    dsdc2 = self.ch_pars[key]['c2c'][:,[0,-1]]
    dsdc3 = self.ch_pars[key]['c3c'][:,[0,-1]]
    dsdc4 = self.ch_pars[key]['c4c'][:,[0,-1]]
    
    # =============================================================================
    # jacobian, derivatives, 8 terms
    # =============================================================================
    #local variables
    dx_here     = self.ch_gegs[key]['dx'][[0,-1]]
    bn_here     = self.ch_pars[key]['bn'][[0,-1]]
    ut_here     = self.ch_pars[key]['ut'][:,[0,-1]]
    dutdx_here  = self.ch_pars[key]['dutdx'][:,[0,-1]]
    eta_here    = self.ch_pars[key]['eta'][:,[0,-1]]
    detadx_here = self.ch_pars[key]['detadx'][:,[0,-1]]
    detadx2_here= self.ch_pars[key]['detadx2'][:,[0,-1]]
    nph         = self.nn*np.pi/self.ch_pars[key]['H'][[0,-1],np.newaxis]

    #z levels
    #derivatives for x=-L
    dTz0_dsb0 = 1/4* self.ch_pars[key]['H'][0]*self.ch_pars[key]['b'][0] * np.mean(np.real( ut_here[:,0] * np.conj( dsdc2 * self.go2 * detadx_here[:,0] * -3/(2*dx_here[0]) ) + np.conj(ut_here[:,0]) * dsdc2 * self.go2 * detadx_here[:,0] * -3/(2*dx_here[0]) ) * np.cos(self.npzh[1:]), axis=-1)
    dTz0_dsb1 = 1/4* self.ch_pars[key]['H'][0]*self.ch_pars[key]['b'][0] * np.mean(np.real( ut_here[:,0] * np.conj( dsdc2 * self.go2 * detadx_here[:,0] *  4/(2*dx_here[0]) ) + np.conj(ut_here[:,0]) * dsdc2 * self.go2 * detadx_here[:,0] *  4/(2*dx_here[0]) ) * np.cos(self.npzh[1:]), axis=-1)
    dTz0_dsb2 = 1/4* self.ch_pars[key]['H'][0]*self.ch_pars[key]['b'][0] * np.mean(np.real( ut_here[:,0] * np.conj( dsdc2 * self.go2 * detadx_here[:,0] * -1/(2*dx_here[0]) ) + np.conj(ut_here[:,0]) * dsdc2 * self.go2 * detadx_here[:,0] * -1/(2*dx_here[0]) ) * np.cos(self.npzh[1:]), axis=-1)
    dTz0_dsn0 = 1/4* self.ch_pars[key]['H'][0]*self.ch_pars[key]['b'][0] * np.mean(np.real( ut_here[:,0] * np.conj( dsdc3 * - nph*eta_here[:,0] + dsdc4 * - nph * self.go2 * (detadx2_here[:,0] + detadx_here[:,0]/bn_here[0]) ) 
                                                           + np.conj(ut_here[:,0]) * ( dsdc3 * - nph*eta_here[:,0] + dsdc4 * - nph * self.go2 * (detadx2_here[:,0] + detadx_here[:,0]/bn_here[0]) ) )[:,np.newaxis,:,:] * np.cos(self.npzh[1:]) , axis = -1)

    #derivatives for x=0
    dTz_1_dsb_1 = 1/4* self.ch_pars[key]['H'][-1]*self.ch_pars[key]['b'][-1] * np.mean(np.real( ut_here[:,1] * np.conj( dsdc2 * self.go2 * detadx_here[:,1] *  3/(2*dx_here[1]) ) + np.conj(ut_here[:,1]) * dsdc2 * self.go2 * detadx_here[:,1] *  3/(2*dx_here[1]) ) * np.cos(self.npzh[1:]) , axis=-1)
    dTz_1_dsb_2 = 1/4* self.ch_pars[key]['H'][-1]*self.ch_pars[key]['b'][-1] * np.mean(np.real( ut_here[:,1] * np.conj( dsdc2 * self.go2 * detadx_here[:,1] * -4/(2*dx_here[1]) ) + np.conj(ut_here[:,1]) * dsdc2 * self.go2 * detadx_here[:,1] * -4/(2*dx_here[1]) ) * np.cos(self.npzh[1:]) , axis=-1)
    dTz_1_dsb_3 = 1/4* self.ch_pars[key]['H'][-1]*self.ch_pars[key]['b'][-1] * np.mean(np.real( ut_here[:,1] * np.conj( dsdc2 * self.go2 * detadx_here[:,1] *  1/(2*dx_here[1]) ) + np.conj(ut_here[:,1]) * dsdc2 * self.go2 * detadx_here[:,1] *  1/(2*dx_here[1]) ) * np.cos(self.npzh[1:]) , axis=-1)
    dTz_1_dsn_1 = 1/4* self.ch_pars[key]['H'][-1]*self.ch_pars[key]['b'][-1] * np.mean(np.real( ut_here[:,1] * np.conj( dsdc3 * - nph*eta_here[:,1] + dsdc4 * - nph * self.go2 * (detadx2_here[:,1] + detadx_here[:,1]/bn_here[1]) ) 
                                                           + np.conj(ut_here[:,1]) * ( dsdc3 * - nph*eta_here[:,1] + dsdc4 * - nph * self.go2 * (detadx2_here[:,1] + detadx_here[:,1]/bn_here[1]) ) )[:,np.newaxis,:,:] * np.cos(self.npzh[1:]) , axis=-1)

    return dTz0_dsb0[:,0],dTz0_dsb1[:,0],dTz0_dsb2[:,0],dTz0_dsn0[:,:,0] , dTz_1_dsb_1[:,0],dTz_1_dsb_2[:,0],dTz_1_dsb_3[:,0],dTz_1_dsn_1[:,:,0]




def func_sol_blc(self, zout, ju_geg, tid_inp):
    # =============================================================================
    # Solution vector for the matching conditions for the boundary layer correction 
    # =============================================================================
    
    #prepare
    res_3ch = []
    
    #calculate relevant quantities for all three channels connected to the junction
    for ch in range(3): 
        #prepare
        xi_C=-1 if ju_geg['loc'][ch] == 'loc x=0' else 0
        key_here = ju_geg['key'][ch]
        
        # =============================================================================
        # calculate (not corrected) tidal salinity
        # =============================================================================
        #tidal salinity and gradient
        st_C    = tid_inp[key_here]['st'][xi_C]
        dstdx_C = tid_inp[key_here]['dstidx'][xi_C]
        #the normalized tidal salinity, complex number
        P = st_C/self.soc_sca
        dPdx = dstdx_C /self.soc_sca*self.Lsc
                
        # =============================================================================
        # calculate salinity correction
        # =============================================================================
        Pc_Re = zout[ju_geg['obcR'][ch]]
        Pc_Im = zout[ju_geg['obcI'][ch]]
        Bm_Re = np.sum(Pc_Re[:,np.newaxis] * np.cos(np.pi*self.m0[:,np.newaxis]*self.z_nd) , axis=0) #real part of salinity correction
        Bm_Im = np.sum(Pc_Im[:,np.newaxis] * np.cos(np.pi*self.m0[:,np.newaxis]*self.z_nd) , axis=0) #imaginary part of salinity correction
            
        #save this
        res_3ch.append([np.real(P),np.imag(P),np.real(dPdx),np.imag(dPdx),Bm_Re,Bm_Im])
    
    # =============================================================================
    # calculate the matching conditions for the solution vector
    # =============================================================================
    #real parts C1 and C2 equal
    sol_pt1 = ((res_3ch[0][0]+res_3ch[0][4]) - (res_3ch[1][0]+res_3ch[1][4]))[self.z_inds]
    #imaginairy parts C1 and C2 equal
    sol_pt2 = ((res_3ch[0][1]+res_3ch[0][5]) - (res_3ch[1][1]+res_3ch[1][5]))[self.z_inds]
    #real parts C2 and C3 equal
    sol_pt3 = ((res_3ch[1][0]+res_3ch[1][4]) - (res_3ch[2][0]+res_3ch[2][4]))[self.z_inds]
    #imaginairy parts C2 and C3 equal
    sol_pt4 = ((res_3ch[1][1]+res_3ch[1][5]) - (res_3ch[2][1]+res_3ch[2][5]))[self.z_inds]
    
    #(diffusive) transport in tidal cycle conserved
    sol_pt5 = 0
    sol_pt6 = 0
    for ch in range(3):#transport for the channels
        key_here = ju_geg['key'][ch]
        if ju_geg['loc'][ch] == 'loc x=-L':
            sol_pt5 += -self.ch_pars[key_here]['H'][0]*self.ch_pars[key_here]['b'][0]*self.ch_pars[key_here]['eps'][0] * ( res_3ch[ch][2] + res_3ch[ch][4]/np.sqrt(self.ch_pars[key_here]['eps'][0]) )[self.z_inds]
            sol_pt6 += -self.ch_pars[key_here]['H'][0]*self.ch_pars[key_here]['b'][0]*self.ch_pars[key_here]['eps'][0] * ( res_3ch[ch][3] + res_3ch[ch][5]/np.sqrt(self.ch_pars[key_here]['eps'][0]) )[self.z_inds]
        elif ju_geg['loc'][ch] == 'loc x=0':
            sol_pt5 += self.ch_pars[key_here]['H'][-1]*self.ch_pars[key_here]['b'][-1]*self.ch_pars[key_here]['eps'][-1] * ( res_3ch[ch][2] - res_3ch[ch][4]/np.sqrt(self.ch_pars[key_here]['eps'][-1]) )[self.z_inds]
            sol_pt6 += self.ch_pars[key_here]['H'][-1]*self.ch_pars[key_here]['b'][-1]*self.ch_pars[key_here]['eps'][-1] * ( res_3ch[ch][3] - res_3ch[ch][5]/np.sqrt(self.ch_pars[key_here]['eps'][-1]) )[self.z_inds]
        else: print('ERROR')

    #return the results 
    return sol_pt1,sol_pt2,sol_pt3,sol_pt4,sol_pt5,sol_pt6


def func_jac_blc(self, ju_geg):
    # =============================================================================
    # Jacobian associated with the matching conditions for the boundary layer correction 
    # =============================================================================
    #prepare
    out = []
    
    for ch in range(3): #for all the 3 channels adjacent to the junction
        #prepare
        xi_C=-1 if ju_geg['loc'][ch] == 'loc x=0' else 0
        key_here = ju_geg['key'][ch]
        
        #prepare
        dsdc2 = self.ch_pars[key_here]['c2c'][:,xi_C,np.newaxis]
        dsdc3 = self.ch_pars[key_here]['c3c'][:,xi_C,np.newaxis]
        dsdc4 = self.ch_pars[key_here]['c4c'][:,xi_C,np.newaxis]
        nph   = self.nn*np.pi/self.ch_pars[key_here]['H'][xi_C,np.newaxis]
        
        #local parameters
        dx_here      = self.ch_gegs[key_here]['dx'][xi_C]
        bn_here      = self.ch_pars[key_here]['bn'][xi_C]
        eta_here     = self.ch_pars[key_here]['eta'][:,xi_C]
        detadx_here  = self.ch_pars[key_here]['detadx'][:,xi_C]
        detadx2_here = self.ch_pars[key_here]['detadx2'][:,xi_C]
        detadx3_here = self.ch_pars[key_here]['detadx3'][:,xi_C]
             
        # =============================================================================
        # derivatives for st and dstdx
        # =============================================================================
        if ju_geg['loc'][ch] == 'loc x=-L':          
            dst0_dsb0 = dsdc2 * self.go2 * detadx_here * -3/(2*dx_here)
            dst0_dsb1 = dsdc2 * self.go2 * detadx_here *  4/(2*dx_here) 
            dst0_dsb2 = dsdc2 * self.go2 * detadx_here * -1/(2*dx_here) 
            dst0_dsn0 = (dsdc3 * - nph*eta_here + dsdc4 * - nph * self.go2 * (detadx2_here + detadx_here/bn_here))
                        
            leng = self.ch_pars[key_here]['H'][0]*self.ch_pars[key_here]['b'][0]*self.Lsc
            
            dst0_x_dsb0 = leng*dsdc2 * self.go2 * (detadx2_here * -3/(2*dx_here) + detadx_here *  2/(dx_here**2) )
            dst0_x_dsb1 = leng*dsdc2 * self.go2 * (detadx2_here *  4/(2*dx_here) + detadx_here * -5/(dx_here**2) )
            dst0_x_dsb2 = leng*dsdc2 * self.go2 * (detadx2_here * -1/(2*dx_here) + detadx_here *  4/(dx_here**2) )
            dst0_x_dsb3 = leng*dsdc2 * self.go2 *  detadx_here * -1/(dx_here**2)
            dst0_x_dsn0 = leng*(dsdc3 * - nph * detadx_here + dsdc4 * -nph * self.go2 * (detadx3_here + detadx2_here/bn_here) \
                        + (dsdc3 * -nph *eta_here + dsdc4 * -nph *  self.go2 * (detadx2_here + detadx_here/bn_here) ) * -3/(2*dx_here))
            dst0_x_dsn1 = leng*(dsdc3 * -nph *eta_here + dsdc4 * -nph *  self.go2 * (detadx2_here + detadx_here/bn_here) ) *  4/(2*dx_here)
            dst0_x_dsn2 = leng*(dsdc3 * -nph *eta_here + dsdc4 * -nph *  self.go2 * (detadx2_here + detadx_here/bn_here) ) * -1/(2*dx_here)
            
            #print(dst0_dsb0.shape , dst0_dsb1.shape,dst0_dsb2.shape,dst0_dsn0.shape, 'LOOK IN DETAIL WHAT GOES ON HERE.' )
            #print(detadx_here.shape , dx_here.shape , dsdc2.shape)
            
            out.append(((np.real(dst0_dsb0[0,0]), np.real(dst0_dsb1[0,0]), np.real(dst0_dsb2[0,0]),
                    np.real(dst0_dsn0[:,0]),
                    np.real(dst0_x_dsb0[0,0]),np.real(dst0_x_dsb1[0,0]),np.real(dst0_x_dsb2[0,0]),np.real(dst0_x_dsb3[0,0]),
                    np.real(dst0_x_dsn0[:,0]),np.real(dst0_x_dsn1[:,0]),np.real(dst0_x_dsn2[:,0])) , \
                   (np.imag(dst0_dsb0[0,0]), np.imag(dst0_dsb1[0,0]), np.imag(dst0_dsb2[0,0]),
                    np.imag(dst0_dsn0[:,0]),
                    np.imag(dst0_x_dsb0[0,0]),np.imag(dst0_x_dsb1[0,0]),np.imag(dst0_x_dsb2[0,0]),np.imag(dst0_x_dsb3[0,0]),
                    np.imag(dst0_x_dsn0[:,0]),np.imag(dst0_x_dsn1[:,0]),np.imag(dst0_x_dsn2[:,0])) 
                    , self.ch_pars[key_here]['H'][0]*self.ch_pars[key_here]['b'][0]))

        
        elif ju_geg['loc'][ch] == 'loc x=0':
            dst_1_dsb_1 = dsdc2 * self.go2 * detadx_here *  3/(2*dx_here) 
            dst_1_dsb_2 = dsdc2 * self.go2 * detadx_here * -4/(2*dx_here) 
            dst_1_dsb_3 = dsdc2 * self.go2 * detadx_here *  1/(2*dx_here) 
            dst_1_dsn_1 = (dsdc3 * - nph*eta_here + dsdc4 * - nph * self.go2 * (detadx2_here + detadx_here/bn_here)) 
            
            leng = self.ch_pars[key_here]['H'][-1]*self.ch_pars[key_here]['b'][-1] * self.Lsc
            
            dst_1_x_dsb_1 = leng*dsdc2 * self.go2 * (detadx2_here *  3/(2*dx_here) + detadx_here *  2/(dx_here**2) )
            dst_1_x_dsb_2 = leng*dsdc2 * self.go2 * (detadx2_here * -4/(2*dx_here) + detadx_here * -5/(dx_here**2) )
            dst_1_x_dsb_3 = leng*dsdc2 * self.go2 * (detadx2_here *  1/(2*dx_here) + detadx_here *  4/(dx_here**2) )
            dst_1_x_dsb_4 = leng*dsdc2 * self.go2 *  detadx_here * -1/(dx_here**2)
            dst_1_x_dsn_1 = leng*(dsdc3 * - nph * detadx_here + dsdc4 * -nph * self.go2 * (detadx3_here + detadx2_here/bn_here) \
                          + (dsdc3 * -nph *eta_here + dsdc4 * -nph *  self.go2 * (detadx2_here + detadx_here/bn_here) ) *  3/(2*dx_here))
            dst_1_x_dsn_2 = leng*(dsdc3 * -nph *eta_here + dsdc4 * -nph *  self.go2 * (detadx2_here + detadx_here/bn_here) ) * -4/(2*dx_here)
            dst_1_x_dsn_3 = leng*(dsdc3 * -nph *eta_here + dsdc4 * -nph *  self.go2 * (detadx2_here + detadx_here/bn_here) ) *  1/(2*dx_here)
            
            out.append(( (np.real(dst_1_dsb_1[0,0]), np.real(dst_1_dsb_2[0,0]), np.real(dst_1_dsb_3[0,0]),
                np.real(dst_1_dsn_1[:,0]),
                np.real(dst_1_x_dsb_1[0,0]), np.real(dst_1_x_dsb_2[0,0]), np.real(dst_1_x_dsb_3[0,0]), np.real(dst_1_x_dsb_4[0,0]),
                np.real(dst_1_x_dsn_1[:,0]), np.real(dst_1_x_dsn_2[:,0]), np.real(dst_1_x_dsn_3[:,0])) ,\
               (np.imag(dst_1_dsb_1[0,0]), np.imag(dst_1_dsb_2[0,0]), np.imag(dst_1_dsb_3[0,0]),
                np.imag(dst_1_dsn_1[:,0]),
                np.imag(dst_1_x_dsb_1[0,0]), np.imag(dst_1_x_dsb_2[0,0]), np.imag(dst_1_x_dsb_3[0,0]), np.imag(dst_1_x_dsb_4[0,0]),
                np.imag(dst_1_x_dsn_1[:,0]), np.imag(dst_1_x_dsn_2[:,0]), np.imag(dst_1_x_dsn_3[:,0])) 
                       , self.ch_pars[key_here]['H'][-1]*self.ch_pars[key_here]['b'][-1]))

    return out


def sol_junc_tot(self, ans, junc_gegs, tid_inp, pars_Q):
    # =============================================================================
    # all the conditions at the junction, to be put in the subtidal solution procedure
    # =============================================================================
    so = np.zeros(self.ch_inds[self.ch_keys[-1]]['totx'][-1] * self.M)
    
    for j in range(1,self.n_j+1):
        
        ju_here = junc_gegs['j'+str(j)]
        keys_now = ju_here['key']
        locs_now = ju_here['loc']
        
        #eerst: 0-1 gelijk
        so[ju_here['i0'][0]] = ans[ju_here['i0'][0]] - ans[ju_here['i0'][1]]

        #second: 1-2 equal
        so[ju_here['i0'][1]] = ans[ju_here['i0'][1]] - ans[ju_here['i0'][2]]
        
        #third: transport continuous
        
        #for depth-averaged transport
        temp = 0
        for i in range(3): #calculate contributions from channels seperately 
            #prepare
            key_here = keys_now[i]
            Ttid = func_sol_Tbnd(self, tid_inp[key_here], key_here) #tidal contribution
            Ttic = func_sol_Tblc(self, ans, key_here) #boundary layer correction gives transport
            pars = self.ch_parja[key_here]

            if locs_now[i] == 'loc x=-L':
                temp = temp - (pars['C13a_x=-L']*pars_Q[key_here]*ans[ju_here['i0'][i][0]]
                               + np.sum(ans[ju_here['i0'][i][1:]]*(pars['C13b_x=-L']*pars_Q[key_here] + pars['C13c_x=-L'] * (-3*ans[ju_here['i0'][i][0]]+4*ans[ju_here['i1'][i][0]]-ans[ju_here['i2'][i][0]])/(2*self.ch_pars[key_here]['dl'][0]) )) 
                               + pars['C13d_x=-L']*(-3*ans[ju_here['i0'][i][0]]+4*ans[ju_here['i1'][i][0]]-ans[ju_here['i2'][i][0]])/(2*self.ch_pars[key_here]['dl'][0]) + Ttid[0] + Ttic[0] )
            
            elif locs_now[i] == 'loc x=0':
                temp = temp + (pars['C13a_x=0']*pars_Q[key_here]*ans[ju_here['i0'][i][0]]
                               + np.sum(ans[ju_here['i0'][i][1:]] *(pars['C13b_x=0']*pars_Q[key_here] + pars['C13c_x=0'] * (ans[ju_here['i2'][i][0]] -4*ans[ju_here['i1'][i][0]] +3*ans[ju_here['i0'][i][0]] )/(2*self.ch_pars[key_here]['dl'][-1]) ))
                               + pars['C13d_x=0'] * (ans[ju_here['i2'][i][0]] -4*ans[ju_here['i1'][i][0]] +3*ans[ju_here['i0'][i][0]] )/(2*self.ch_pars[key_here]['dl'][-1]) + Ttid[-1] + Ttic[-1] )
            else: print('ERROR')
            
        so[ju_here['i0'][2][0]] = temp #add to the solution vector

        
        #calculate for transport at depth, with tidal contribution   
        for k in range(1,self.M):
            temp = 0
            for i in range(3): #calculate contributions from channels seperately 
                #prepare
                key_here = keys_now[i]
                Ttid = func_sol_Tbndz(self, tid_inp[key_here], key_here)
                Ttic = func_sol_Tblcz(self, ans, key_here) #boundary layer correction gives transport
                pars = self.ch_parja[key_here]

                if locs_now[i] == 'loc x=-L':
                    temp = temp - (pars['C12a_x=-L'] * (-3*ans[ju_here['i0'][i][k]]+4*ans[ju_here['i1'][i][k]]-ans[ju_here['i2'][i][k]])/(2*self.ch_pars[key_here]['dl'][0])
                                   + pars['C12b_x=-L'][k-1] * ans[ju_here['i0'][i][k]] * (-3*ans[ju_here['i0'][i][0]]+4*ans[ju_here['i1'][i][0]]-ans[ju_here['i2'][i][0]])/(2*self.ch_pars[key_here]['dl'][0])  
                                   + np.sum(pars['C12c_x=-L'][k-1]*ans[ju_here['i0'][i][1:]]) * (-3*ans[ju_here['i0'][i][0]]+4*ans[ju_here['i1'][i][0]]-ans[ju_here['i2'][i][0]])/(2*self.ch_pars[key_here]['dl'][0]) 
                                   + pars['C12d_x=-L'][k-1]*pars_Q[key_here]*ans[ju_here['i0'][i][k]]
                                   + np.sum(pars['C12e_x=-L'][k-1]*ans[ju_here['i0'][i][1:]])*pars_Q[key_here]
                                   + Ttid[k-1,0] + Ttic[k-1,0]
                                   )
                
                elif locs_now[i] == 'loc x=0' :
                    temp = temp + (pars['C12a_x=0'] * (ans[ju_here['i2'][i][k]]-4*ans[ju_here['i1'][i][k]]+3*ans[ju_here['i0'][i][k]])/(2*self.ch_pars[key_here]['dl'][-1])  
                                   + pars['C12b_x=0'][k-1] * ans[ju_here['i0'][i][k]] * (ans[ju_here['i2'][i][0]]-4*ans[ju_here['i1'][i][0]]+3*ans[ju_here['i0'][i][0]])/(2*self.ch_pars[key_here]['dl'][-1])  
                                   + np.sum(pars['C12c_x=0'][k-1]*ans[ju_here['i0'][i][1:]]) * (ans[ju_here['i2'][i][0]]-4*ans[ju_here['i1'][i][0]]+3*ans[ju_here['i0'][i][0]])/(2*self.ch_pars[key_here]['dl'][-1])  
                                   + pars['C12d_x=0'][k-1]*pars_Q[key_here]*ans[ju_here['i0'][i][k]]
                                   + np.sum(pars['C12e_x=0'][k-1]*ans[ju_here['i0'][i][1:]])*pars_Q[key_here]
                                   + Ttid[k-1,-1] + Ttic[k-1,-1] 
                                   )
                
                else: print("ERROR")
                
            so[ju_here['i0'][2][k]] = temp #add to the solution vector
            

        # =============================================================================
        # boundary layer correction 
        # =============================================================================
        '''
        for i in range(3):
            if locs_now[i] == 'loc x=-L':
                so[self.ch_inds[keys_now[i]]['ob_x=-L_bcR']] += ans[self.ch_inds[keys_now[i]]['ob_x=-L_bcR']]
                so[self.ch_inds[keys_now[i]]['ob_x=-L_bcI']] += ans[self.ch_inds[keys_now[i]]['ob_x=-L_bcI']] 
            elif locs_now[i] == 'loc x=0':
                so[self.ch_inds[keys_now[i]]['ob_x=0_bcR']] += ans[self.ch_inds[keys_now[i]]['ob_x=0_bcR']]
                so[self.ch_inds[keys_now[i]]['ob_x=0_bcI']] += ans[self.ch_inds[keys_now[i]]['ob_x=0_bcI']]
            else: print('Error!')
        '''
        cor_so = func_sol_blc(self, ans, ju_here, tid_inp)

        for i in range(3):
            if locs_now[i] == 'loc x=-L':
                so[self.ch_inds[keys_now[i]]['ob_x=-L_bcR']] += cor_so[0+2*i]
                so[self.ch_inds[keys_now[i]]['ob_x=-L_bcI']] += cor_so[1+2*i] 
            elif locs_now[i] == 'loc x=0':
                so[self.ch_inds[keys_now[i]]['ob_x=0_bcR']] += cor_so[0+2*i]
                so[self.ch_inds[keys_now[i]]['ob_x=0_bcI']] += cor_so[1+2*i]
            else: print('Error!')
        #'''
        
    return so


 


def jac_junc_tot_fix(self, junc_gegs, pars_Q):
    # =============================================================================
    # The jacobian matrix for the jucntions
    # =============================================================================
    
    #create empty vector
    jac = np.zeros((self.ch_inds[self.ch_keys[-1]]['totx'][-1] * self.M,self.ch_inds[self.ch_keys[-1]]['totx'][-1] * self.M))
    
    for j in range(1,self.n_j+1):
        
        ju_here = junc_gegs['j'+str(j)]
        keys_now = ju_here['key']
        locs_now = ju_here['loc']
        
        #eerst: 0-1 gelijk
        jac[ju_here['i0'][0],ju_here['i0'][0]] =  1
        jac[ju_here['i0'][0],ju_here['i0'][1]] = -1

        #second: 1-2 equal
        jac[ju_here['i0'][1],ju_here['i0'][1]] =  1
        jac[ju_here['i0'][1],ju_here['i0'][2]] = -1
        
        #third:transport
        
        #for depth-averaged transport
        for i in range(3): #calculate contributions from channels seperately 
            #prepare    
            key_here = keys_now[i]
            Ttid = func_jac_Tbnd(self, key_here)
            Ttic = np.array(func_jac_Tblc(self, key_here)) # boundary layer correction - 
            pars = self.ch_parja[key_here]
            
            if locs_now[i] == 'loc x=-L':
                jac[ju_here['i0'][2][0],ju_here['i0'][i][0]] = -(pars['C13a_x=-L']*pars_Q[key_here] -3/(2*self.ch_pars[key_here]['dl'][0])  * pars['C13d_x=-L']  + Ttid[0]) 
                jac[ju_here['i0'][2][0],ju_here['i1'][i][0]] = -(4/(2*self.ch_pars[key_here]['dl'][0])  * pars['C13d_x=-L'] + Ttid[1]) 
                jac[ju_here['i0'][2][0],ju_here['i2'][i][0]] = -(-1/(2*self.ch_pars[key_here]['dl'][0])  * pars['C13d_x=-L']  + Ttid[2]) 
                jac[ju_here['i0'][2][0],ju_here['i0'][i][self.mm]] = -(pars['C13b_x=-L']*pars_Q[key_here] + Ttid[3]) 
                jac[ju_here['i0'][2][0],ju_here['obcR'][i]]  = -Ttic[0][:,0]
                jac[ju_here['i0'][2][0],ju_here['obcI'][i]]  = -Ttic[1][:,0]
                
            elif locs_now[i] == 'loc x=0':
                jac[ju_here['i0'][2][0],ju_here['i0'][i][0]] = pars['C13a_x=0']*pars_Q[key_here] +3/(2*self.ch_pars[key_here]['dl'][-1]) * pars['C13d_x=0'] + Ttid[4]
                jac[ju_here['i0'][2][0],ju_here['i1'][i][0]] =-4/(2*self.ch_pars[key_here]['dl'][-1]) * pars['C13d_x=0'] + Ttid[5]
                jac[ju_here['i0'][2][0],ju_here['i2'][i][0]] = 1/(2*self.ch_pars[key_here]['dl'][-1]) * pars['C13d_x=0'] + Ttid[6]
                jac[ju_here['i0'][2][0],ju_here['i0'][i][self.mm]] = pars['C13b_x=0']*pars_Q[key_here] + Ttid[7]
                jac[ju_here['i0'][2][0],ju_here['obcR'][i]] = Ttic[0][:,1]
                jac[ju_here['i0'][2][0],ju_here['obcI'][i]] = Ttic[1][:,1]
                
            else: print('ERROR')
        
        #transport at every vertical level , with tidal contriubtion
        for k in range(1,self.M):
            for i in range(3): #calculate contributions from channels seperately 
                #prepare
                key_here = keys_now[i]
                Ttid = func_jac_Tbndz(self, key_here)
                Ttic = np.array(func_jac_Tblcz(self, key_here)) #boundary layer correction gives transport
                pars = self.ch_parja[key_here]
                
                if locs_now[i] == 'loc x=-L':
                    jac[ju_here['i0'][2][k],ju_here['i0'][i][0]] = - (Ttid[0][k-1])
                    jac[ju_here['i0'][2][k],ju_here['i1'][i][0]] = - Ttid[1][k-1]
                    jac[ju_here['i0'][2][k],ju_here['i2'][i][0]] = - Ttid[2][k-1]
                    jac[ju_here['i0'][2][k],ju_here['i0'][i][k]] = - (-3/(2*self.ch_pars[key_here]['dl'][0]) * pars['C12a_x=-L'] + pars['C12d_x=-L'][k-1]*pars_Q[key_here])
                    jac[ju_here['i0'][2][k],ju_here['i1'][i][k]] = - ( 4/(2*self.ch_pars[key_here]['dl'][0]) * pars['C12a_x=-L'])
                    jac[ju_here['i0'][2][k],ju_here['i2'][i][k]] = - (-1/(2*self.ch_pars[key_here]['dl'][0]) * pars['C12a_x=-L'])
                    jac[ju_here['i0'][2][k],ju_here['i0'][i][self.mm]] += - (pars['C12e_x=-L'][k-1] * pars_Q[key_here] + Ttid[3][:,k-1])
                    jac[ju_here['i0'][2][k],ju_here['obcR'][i]] = -Ttic[0][:,k-1,0]
                    jac[ju_here['i0'][2][k],ju_here['obcI'][i]] = -Ttic[1][:,k-1,0]
                
                elif locs_now[i] == 'loc x=0':
                    jac[ju_here['i0'][2][k],ju_here['i0'][i][0]] = Ttid[4][k-1]
                    jac[ju_here['i0'][2][k],ju_here['i1'][i][0]] = Ttid[5][k-1] 
                    jac[ju_here['i0'][2][k],ju_here['i2'][i][0]] = Ttid[6][k-1]
                    jac[ju_here['i0'][2][k],ju_here['i0'][i][k]] = 3/(2*self.ch_pars[key_here]['dl'][-1]) * pars['C12a_x=0'] + pars['C12d_x=0'][k-1]*pars_Q[key_here]
                    jac[ju_here['i0'][2][k],ju_here['i1'][i][k]] =-4/(2*self.ch_pars[key_here]['dl'][-1]) * pars['C12a_x=0']
                    jac[ju_here['i0'][2][k],ju_here['i2'][i][k]] = 1/(2*self.ch_pars[key_here]['dl'][-1]) * pars['C12a_x=0']
                    jac[ju_here['i0'][2][k],ju_here['i0'][i][self.mm]] += pars['C12e_x=0'][k-1] * pars_Q[key_here] + Ttid[7][:,k-1]
                    jac[ju_here['i0'][2][k],ju_here['obcR'][i]] = Ttic[0][:,k-1,-1]
                    jac[ju_here['i0'][2][k],ju_here['obcI'][i]] = Ttic[1][:,k-1,-1]
                    
                else: print('ERROR')

        # =============================================================================
        # boundary layer correction 
        # =============================================================================
        '''
        for i in range(3):
            if locs_now[i] == 'loc x=-L':
                jac[self.ch_inds[keys_now[i]]['ob_x=-L_bcR'],self.ch_inds[keys_now[i]]['ob_x=-L_bcR']] = 1
                jac[self.ch_inds[keys_now[i]]['ob_x=-L_bcI'],self.ch_inds[keys_now[i]]['ob_x=-L_bcI']] = 1
            elif locs_now[i] == 'loc x=0':
                jac[self.ch_inds[keys_now[i]]['ob_x=0_bcR'],self.ch_inds[keys_now[i]]['ob_x=0_bcR']] = 1
                jac[self.ch_inds[keys_now[i]]['ob_x=0_bcI'],self.ch_inds[keys_now[i]]['ob_x=0_bcI']] = 1
            else: print('Error!')
        
        '''
        jac_bc = func_jac_blc(self, ju_here)


        #speed up the life, hope it is correct.      

        #self.m0 = np.arange(self.M)
        zi_here = self.z_inds[self.m0]
        
        mr = np.repeat(self.m0,self.M)
        mt = np.tile(self.m0,self.M)
        cos_here = np.cos(mt*np.pi*self.z_nd[self.z_inds[mt]])

        # =============================================================================
        #  first condition: 1-2=0
        # =============================================================================
        # #real part
        # #afgeleides naar Bs van hetzelfde kanaal            
        #jac[ju_here['obcR'][0][0]+mr,ju_here['obcR'][0][0]+mt] = cos_here
        # #afgeleiders naar Bs van het andere kanaal
        #jac[ju_here['obcR'][0][0]+mr,ju_here['obcR'][1][0]+mt] = - cos_here

        #print(ju_here['obcR'][0][0]+self.m0)
        #print( ju_here['i0'][0][0] )
        
        #print(jac_bc[0][0].shape)
        #print(jac_bc[0][0][0][zi_here])

        # #afgeleides naar st van hetzelfde kanaal
        jac[ju_here['obcR'][0][0]+self.m0,ju_here['i0'][0][0]] = jac_bc[0][0][0][zi_here]
        jac[ju_here['obcR'][0][0]+self.m0,ju_here['i1'][0][0]] = jac_bc[0][0][1][zi_here]
        jac[ju_here['obcR'][0][0]+self.m0,ju_here['i2'][0][0]] = jac_bc[0][0][2][zi_here]
        # jac[ju_here['obcR'][0][0]+m,ju_here['i0'][0][0]+self.mm] = jac_bc[0][0][3][:,zi_here]
        # #afgeleides naar st van het andere kanaal
        jac[ju_here['obcR'][0][0]+self.m0,ju_here['i0'][1][0]] = -jac_bc[1][0][0][zi_here]
        jac[ju_here['obcR'][0][0]+self.m0,ju_here['i1'][1][0]] = -jac_bc[1][0][1][zi_here]
        jac[ju_here['obcR'][0][0]+self.m0,ju_here['i2'][1][0]] = -jac_bc[1][0][2][zi_here]
        # jac[ju_here['obcR'][0][0]+m,ju_here['i0'][1][0]+self.mm] = -jac_bc[1][0][3][:,zi_here]
        
        
        # #imaginairy part 
        # #afgeleides naar Bs van hetzelfde kanaal
        # jac[ju_here['obcI'][0][0]+m,ju_here['obcI'][0][0]+self.m0] = cos_here 
        # #afgeleiders naar Bs van het andere kanaal
        # jac[ju_here['obcI'][0][0]+m,ju_here['obcI'][1][0]+self.m0] = - cos_here
        # #afgeleides naar st van hetzelfde kanaal
        jac[ju_here['obcI'][0][0]+self.m0,ju_here['i0'][0][0]] = jac_bc[0][1][0][zi_here]
        jac[ju_here['obcI'][0][0]+self.m0,ju_here['i1'][0][0]] = jac_bc[0][1][1][zi_here]
        jac[ju_here['obcI'][0][0]+self.m0,ju_here['i2'][0][0]] = jac_bc[0][1][2][zi_here]
        # jac[ju_here['obcI'][0][0]+m,ju_here['i0'][0][0]+self.mm] = jac_bc[0][1][3][:,zi_here]
        # #afgeleides naar st van het andere kanaal
        jac[ju_here['obcI'][0][0]+self.m0,ju_here['i0'][1][0]] = -jac_bc[1][1][0][zi_here]
        jac[ju_here['obcI'][0][0]+self.m0,ju_here['i1'][1][0]] = -jac_bc[1][1][1][zi_here]
        jac[ju_here['obcI'][0][0]+self.m0,ju_here['i2'][1][0]] = -jac_bc[1][1][2][zi_here]
        # jac[ju_here['obcI'][0][0]+m,ju_here['i0'][1][0]+self.mm] = -jac_bc[1][1][3][:,zi_here]
        
            
        # # =============================================================================
        # #  second condition: 2-3=0
        # # =============================================================================
        
        # #real part
        # #afgeleides naar Bs van hetzelfde kanaal
        # jac[ju_here['obcR'][1][0]+m,ju_here['obcR'][1][0]+self.m0] = cos_here
        # #afgeleiders naar Bs van het andere kanaal
        # jac[ju_here['obcR'][1][0]+m,ju_here['obcR'][2][0]+self.m0] = - cos_here
        # #afgeleides naar st van hetzelfde kanaal
        jac[ju_here['obcR'][1][0]+self.m0,ju_here['i0'][1][0]] = jac_bc[1][0][0][zi_here]
        jac[ju_here['obcR'][1][0]+self.m0,ju_here['i1'][1][0]] = jac_bc[1][0][1][zi_here]
        jac[ju_here['obcR'][1][0]+self.m0,ju_here['i2'][1][0]] = jac_bc[1][0][2][zi_here]
        # jac[ju_here['obcR'][1][0]+m,ju_here['i0'][1][0]+self.mm] = jac_bc[1][0][3][:,zi_here]

        # #afgeleides naar st van het andere kanaal
        jac[ju_here['obcR'][1][0]+self.m0,ju_here['i0'][2][0]] = -jac_bc[2][0][0][zi_here]
        jac[ju_here['obcR'][1][0]+self.m0,ju_here['i1'][2][0]] = -jac_bc[2][0][1][zi_here]
        jac[ju_here['obcR'][1][0]+self.m0,ju_here['i2'][2][0]] = -jac_bc[2][0][2][zi_here]
        # jac[ju_here['obcR'][1][0]+m,ju_here['i0'][2][0]+self.mm] = -jac_bc[2][0][3][:,zi_here]
       
        # #imaginairy part
        # #afgeleides naar Bs van hetzelfde kanaal
        # jac[ju_here['obcI'][1][0]+m,ju_here['obcI'][1][0]+self.m0] = cos_here
        # #afgeleiders naar Bs van het andere kanaal
        # jac[ju_here['obcI'][1][0]+m,ju_here['obcI'][2][0]+self.m0] = - cos_here
        # #afgeleides naar st van hetzelfde kanaal
        jac[ju_here['obcI'][1][0]+self.m0, ju_here['i0'][1][0]] = jac_bc[1][1][0][zi_here]
        jac[ju_here['obcI'][1][0]+self.m0,ju_here['i1'][1][0]] = jac_bc[1][1][1][zi_here]
        jac[ju_here['obcI'][1][0]+self.m0,ju_here['i2'][1][0]] = jac_bc[1][1][2][zi_here]
        # jac[ju_here['obcI'][1][0]+m,ju_here['i0'][1][0]+self.mm] = jac_bc[1][1][3][:,zi_here]
        # #afgeleides naar st van het andere kanaal
        jac[ju_here['obcI'][1][0]+self.m0,ju_here['i0'][2][0]] = -jac_bc[2][1][0][zi_here]
        jac[ju_here['obcI'][1][0]+self.m0,ju_here['i1'][2][0]] = -jac_bc[2][1][1][zi_here]
        jac[ju_here['obcI'][1][0]+self.m0,ju_here['i2'][2][0]] = -jac_bc[2][1][2][zi_here]
        # jac[ju_here['obcI'][1][0]+m,ju_here['i0'][2][0]+self.mm] = -jac_bc[2][1][3][:,zi_here]
          
        # # =============================================================================
        # # third condition: diffusive flux in tidal cycle conserved
        # # =============================================================================
        # #real part
        # #channel 1
        if ju_here['loc'][0] == 'loc x=-L': sign,eps_ind = -1 , 0
        elif ju_here['loc'][0] == 'loc x=0': sign,eps_ind = 1 , -1
                
        # #afgeleides naar Bs van C1
        # jac[ju_here['obcR'][2][0]+m,ju_here['obcR'][0][0]+self.m0] = - cos_here * self.ch_pars[key]['eps']**0.5 * jac_bc[0][2]
        # #afgeleides naar st van C1
        jac[ju_here['obcR'][2][0]+self.m0,ju_here['i0'][0][0]] = sign * self.ch_pars[ju_here['key'][0]]['eps'][eps_ind] * jac_bc[0][0][4][zi_here]
        jac[ju_here['obcR'][2][0]+self.m0,ju_here['i1'][0][0]] = sign * self.ch_pars[ju_here['key'][0]]['eps'][eps_ind] * jac_bc[0][0][5][zi_here]
        jac[ju_here['obcR'][2][0]+self.m0,ju_here['i2'][0][0]] = sign * self.ch_pars[ju_here['key'][0]]['eps'][eps_ind] * jac_bc[0][0][6][zi_here]
        jac[ju_here['obcR'][2][0]+self.m0,ju_here['i3'][0][0]] = sign * self.ch_pars[ju_here['key'][0]]['eps'][eps_ind] * jac_bc[0][0][7][zi_here]
        # jac[ju_here['obcR'][2][0]+m,ju_here['i0'][0][0]+self.mm] = sign * self.ch_pars[key]['eps'] * jac_bc[0][0][8 ][:,zi_here]
        # jac[ju_here['obcR'][2][0]+m,ju_here['i1'][0][0]+self.mm] = sign * self.ch_pars[key]['eps'] * jac_bc[0][0][9 ][:,zi_here]
        # jac[ju_here['obcR'][2][0]+m,ju_here['i2'][0][0]+self.mm] = sign * self.ch_pars[key]['eps'] * jac_bc[0][0][10][:,zi_here]
           
            
        # #channel 2
        if ju_here['loc'][1] == 'loc x=-L': sign,eps_ind = -1 , 0
        elif ju_here['loc'][1] == 'loc x=0': sign,eps_ind = 1 , -1
        
        # #afgeleides naar Bs van C2
        # jac[ju_here['obcR'][2][0]+m,ju_here['obcR'][1][0]+self.m0] = - cos_here * self.ch_pars[key]['eps']**0.5 * jac_bc[1][2]
        # #afgeleides naar st van C2
        jac[ju_here['obcR'][2][0]+self.m0,ju_here['i0'][1][0]] = sign *self.ch_pars[ju_here['key'][1]]['eps'][eps_ind] * jac_bc[1][0][4][zi_here]
        jac[ju_here['obcR'][2][0]+self.m0,ju_here['i1'][1][0]] = sign *self.ch_pars[ju_here['key'][1]]['eps'][eps_ind] * jac_bc[1][0][5][zi_here]
        jac[ju_here['obcR'][2][0]+self.m0,ju_here['i2'][1][0]] = sign *self.ch_pars[ju_here['key'][1]]['eps'][eps_ind] * jac_bc[1][0][6][zi_here]
        jac[ju_here['obcR'][2][0]+self.m0,ju_here['i3'][1][0]] = sign *self.ch_pars[ju_here['key'][1]]['eps'][eps_ind] * jac_bc[1][0][7][zi_here]
        # jac[ju_here['obcR'][2][0]+m,ju_here['i0'][1][0]+self.mm] = sign *self.ch_pars[key]['eps'] * jac_bc[1][0][8 ][:,zi_here]
        # jac[ju_here['obcR'][2][0]+m,ju_here['i1'][1][0]+self.mm] = sign *self.ch_pars[key]['eps'] * jac_bc[1][0][9 ][:,zi_here]
        # jac[ju_here['obcR'][2][0]+m,ju_here['i2'][1][0]+self.mm] = sign *self.ch_pars[key]['eps'] * jac_bc[1][0][10][:,zi_here]
        
        
        # #channel 3
        if ju_here['loc'][2] == 'loc x=-L': sign,eps_ind = -1 , 0
        elif ju_here['loc'][2] == 'loc x=0':sign,eps_ind = 1 , -1
        
        # #afgeleides naar Bs van C3
        # jac[ju_here['obcR'][2][0]+m,ju_here['obcR'][2][0]+self.m0] = - cos_here * self.ch_pars[key]['eps']**0.5 * jac_bc[2][2]
        # #afgeleides naar st van C3
        jac[ju_here['obcR'][2][0]+self.m0,ju_here['i0'][2][0]] = sign *self.ch_pars[ju_here['key'][2]]['eps'][eps_ind] * jac_bc[2][0][4][zi_here]
        jac[ju_here['obcR'][2][0]+self.m0,ju_here['i1'][2][0]] = sign *self.ch_pars[ju_here['key'][2]]['eps'][eps_ind] * jac_bc[2][0][5][zi_here]
        jac[ju_here['obcR'][2][0]+self.m0,ju_here['i2'][2][0]] = sign *self.ch_pars[ju_here['key'][2]]['eps'][eps_ind] * jac_bc[2][0][6][zi_here]
        jac[ju_here['obcR'][2][0]+self.m0,ju_here['i3'][2][0]] = sign *self.ch_pars[ju_here['key'][2]]['eps'][eps_ind] * jac_bc[2][0][7][zi_here]
        # jac[ju_here['obcR'][2][0]+m,ju_here['i0'][2][0]+self.mm] = sign *self.ch_pars[key]['eps'] * jac_bc[2][0][8 ][:,zi_here]
        # jac[ju_here['obcR'][2][0]+m,ju_here['i1'][2][0]+self.mm] = sign *self.ch_pars[key]['eps'] * jac_bc[2][0][9 ][:,zi_here]
        # jac[ju_here['obcR'][2][0]+m,ju_here['i2'][2][0]+self.mm] = sign *self.ch_pars[key]['eps'] * jac_bc[2][0][10][:,zi_here]
        
        # #imaginairy part follows here
        # #channel 1
        if ju_here['loc'][0] == 'loc x=-L': sign,eps_ind = -1 , 0
        elif ju_here['loc'][0] == 'loc x=0': sign,eps_ind = 1 , -1
        
        # #afgeleides naar Bs van C1
        # jac[ju_here['obcI'][2][0]+m,ju_here['obcI'][0][0]+self.m0] = - cos_here * self.ch_pars[key]['eps']**0.5 * jac_bc[0][2]
        # #afgeleides naar st van C1
        jac[ju_here['obcI'][2][0]+self.m0,ju_here['i0'][0][0]] = sign * self.ch_pars[ju_here['key'][0]]['eps'][eps_ind] * jac_bc[0][1][4][zi_here]
        jac[ju_here['obcI'][2][0]+self.m0,ju_here['i1'][0][0]] = sign * self.ch_pars[ju_here['key'][0]]['eps'][eps_ind] * jac_bc[0][1][5][zi_here]
        jac[ju_here['obcI'][2][0]+self.m0,ju_here['i2'][0][0]] = sign * self.ch_pars[ju_here['key'][0]]['eps'][eps_ind] * jac_bc[0][1][6][zi_here]
        jac[ju_here['obcI'][2][0]+self.m0,ju_here['i3'][0][0]] = sign * self.ch_pars[ju_here['key'][0]]['eps'][eps_ind] * jac_bc[0][1][7][zi_here]
        # jac[ju_here['obcI'][2][0]+m,ju_here['i0'][0][0]+self.mm] = sign * self.ch_pars[key]['eps'] * jac_bc[0][1][8 ][:,zi_here]
        # jac[ju_here['obcI'][2][0]+m,ju_here['i1'][0][0]+self.mm] = sign * self.ch_pars[key]['eps'] * jac_bc[0][1][9 ][:,zi_here]
        # jac[ju_here['obcI'][2][0]+m,ju_here['i2'][0][0]+self.mm] = sign * self.ch_pars[key]['eps'] * jac_bc[0][1][10][:,zi_here]
           
        # #channel 2
        if ju_here['loc'][1] == 'loc x=-L': sign,eps_ind = -1 , 0
        elif ju_here['loc'][1] == 'loc x=0': sign,eps_ind = 1 , -1
        
        # #afgeleides naar Bs van C2
        # jac[ju_here['obcI'][2][0]+m,ju_here['obcI'][1][0]+self.m0] = - cos_here * self.ch_pars[key]['eps']**0.5 * jac_bc[1][2]
        # #afgeleides naar st van C2
        jac[ju_here['obcI'][2][0]+self.m0,ju_here['i0'][1][0]] = sign *self.ch_pars[ju_here['key'][1]]['eps'][eps_ind] * jac_bc[1][1][4][zi_here]
        jac[ju_here['obcI'][2][0]+self.m0,ju_here['i1'][1][0]] = sign *self.ch_pars[ju_here['key'][1]]['eps'][eps_ind] * jac_bc[1][1][5][zi_here]
        jac[ju_here['obcI'][2][0]+self.m0,ju_here['i2'][1][0]] = sign *self.ch_pars[ju_here['key'][1]]['eps'][eps_ind] * jac_bc[1][1][6][zi_here]
        jac[ju_here['obcI'][2][0]+self.m0,ju_here['i3'][1][0]] = sign *self.ch_pars[ju_here['key'][1]]['eps'][eps_ind] * jac_bc[1][1][7][zi_here]
        # jac[ju_here['obcI'][2][0]+m,ju_here['i0'][1][0]+self.mm] = sign *self.ch_pars[key]['eps'] * jac_bc[1][1][8 ][:,zi_here]
        # jac[ju_here['obcI'][2][0]+m,ju_here['i1'][1][0]+self.mm] = sign *self.ch_pars[key]['eps'] * jac_bc[1][1][9 ][:,zi_here]
        # jac[ju_here['obcI'][2][0]+m,ju_here['i2'][1][0]+self.mm] = sign *self.ch_pars[key]['eps'] * jac_bc[1][1][10][:,zi_here]
        
        # #channel 3
        if ju_here['loc'][2] == 'loc x=-L': sign,eps_ind = -1 , 0
        elif ju_here['loc'][2] == 'loc x=0': sign,eps_ind = 1 , -1
        
        # #afgeleides naar Bs van C3
        # jac[ju_here['obcI'][2][0]+m,ju_here['obcI'][2][0]+self.m0] = - cos_here * self.ch_pars[key]['eps']**0.5 * jac_bc[2][2]
        # #afgeleides naar st van C3
        jac[ju_here['obcI'][2][0]+self.m0,ju_here['i0'][2][0]] = sign * self.ch_pars[ju_here['key'][2]]['eps'][eps_ind] * jac_bc[2][1][4][zi_here]
        jac[ju_here['obcI'][2][0]+self.m0,ju_here['i1'][2][0]] = sign * self.ch_pars[ju_here['key'][2]]['eps'][eps_ind] * jac_bc[2][1][5][zi_here]
        jac[ju_here['obcI'][2][0]+self.m0,ju_here['i2'][2][0]] = sign * self.ch_pars[ju_here['key'][2]]['eps'][eps_ind] * jac_bc[2][1][6][zi_here]
        jac[ju_here['obcI'][2][0]+self.m0,ju_here['i3'][2][0]] = sign * self.ch_pars[ju_here['key'][2]]['eps'][eps_ind] * jac_bc[2][1][7][zi_here]
        # jac[ju_here['obcI'][2][0]+m,ju_here['i0'][2][0]+self.mm] = sign *self.ch_pars[key]['eps'] * jac_bc[2][1][8 ][:,zi_here]
        # jac[ju_here['obcI'][2][0]+m,ju_here['i1'][2][0]+self.mm] = sign *self.ch_pars[key]['eps'] * jac_bc[2][1][9 ][:,zi_here]
        # jac[ju_here['obcI'][2][0]+m,ju_here['i2'][2][0]+self.mm] = sign *self.ch_pars[key]['eps'] * jac_bc[2][1][10][:,zi_here]    



        for m in range(self.M): #TODO: for now a loop, later replace with indices allocation. but that is less easy to check. 
            
            #shortcuts
            zi_here = self.z_inds[m]
            cos_here = np.cos(self.m0*np.pi*self.z_nd[zi_here])
            
            # =============================================================================
            #  first condition: 1-2=0
            # =============================================================================
            #real part
            #afgeleides naar Bs van hetzelfde kanaal            
            jac[ju_here['obcR'][0][0]+m,ju_here['obcR'][0][0]+self.m0] = cos_here
            #afgeleiders naar Bs van het andere kanaal
            jac[ju_here['obcR'][0][0]+m,ju_here['obcR'][1][0]+self.m0] = - cos_here
            
            
            
            #afgeleides naar st van hetzelfde kanaal
            # jac[ju_here['obcR'][0][0]+m,ju_here['i0'][0][0]] = jac_bc[0][0][0][zi_here]
            # jac[ju_here['obcR'][0][0]+m,ju_here['i1'][0][0]] = jac_bc[0][0][1][zi_here]
            # jac[ju_here['obcR'][0][0]+m,ju_here['i2'][0][0]] = jac_bc[0][0][2][zi_here]
            jac[ju_here['obcR'][0][0]+m,ju_here['i0'][0][0]+self.mm] = jac_bc[0][0][3][:,zi_here]
            #afgeleides naar st van het andere kanaal
            # jac[ju_here['obcR'][0][0]+m,ju_here['i0'][1][0]] = -jac_bc[1][0][0][zi_here]
            # jac[ju_here['obcR'][0][0]+m,ju_here['i1'][1][0]] = -jac_bc[1][0][1][zi_here]
            # jac[ju_here['obcR'][0][0]+m,ju_here['i2'][1][0]] = -jac_bc[1][0][2][zi_here]
            jac[ju_here['obcR'][0][0]+m,ju_here['i0'][1][0]+self.mm] = -jac_bc[1][0][3][:,zi_here]
            
            
            #imaginairy part 
            #afgeleides naar Bs van hetzelfde kanaal
            jac[ju_here['obcI'][0][0]+m,ju_here['obcI'][0][0]+self.m0] = cos_here 
            #afgeleiders naar Bs van het andere kanaal
            jac[ju_here['obcI'][0][0]+m,ju_here['obcI'][1][0]+self.m0] = - cos_here
            #afgeleides naar st van hetzelfde kanaal
            # jac[ju_here['obcI'][0][0]+m,ju_here['i0'][0][0]] = jac_bc[0][1][0][zi_here]
            # jac[ju_here['obcI'][0][0]+m,ju_here['i1'][0][0]] = jac_bc[0][1][1][zi_here]
            # jac[ju_here['obcI'][0][0]+m,ju_here['i2'][0][0]] = jac_bc[0][1][2][zi_here]
            jac[ju_here['obcI'][0][0]+m,ju_here['i0'][0][0]+self.mm] = jac_bc[0][1][3][:,zi_here]
            #afgeleides naar st van het andere kanaal
            # jac[ju_here['obcI'][0][0]+m,ju_here['i0'][1][0]] = -jac_bc[1][1][0][zi_here]
            # jac[ju_here['obcI'][0][0]+m,ju_here['i1'][1][0]] = -jac_bc[1][1][1][zi_here]
            # jac[ju_here['obcI'][0][0]+m,ju_here['i2'][1][0]] = -jac_bc[1][1][2][zi_here]
            jac[ju_here['obcI'][0][0]+m,ju_here['i0'][1][0]+self.mm] = -jac_bc[1][1][3][:,zi_here]
            
                
            # =============================================================================
            #  second condition: 2-3=0
            # =============================================================================
            
            #real part
            #afgeleides naar Bs van hetzelfde kanaal
            jac[ju_here['obcR'][1][0]+m,ju_here['obcR'][1][0]+self.m0] = cos_here
            #afgeleiders naar Bs van het andere kanaal
            jac[ju_here['obcR'][1][0]+m,ju_here['obcR'][2][0]+self.m0] = - cos_here
            #afgeleides naar st van hetzelfde kanaal
            # jac[ju_here['obcR'][1][0]+m,ju_here['i0'][1][0]] = jac_bc[1][0][0][zi_here]
            # jac[ju_here['obcR'][1][0]+m,ju_here['i1'][1][0]] = jac_bc[1][0][1][zi_here]
            # jac[ju_here['obcR'][1][0]+m,ju_here['i2'][1][0]] = jac_bc[1][0][2][zi_here]
            jac[ju_here['obcR'][1][0]+m,ju_here['i0'][1][0]+self.mm] = jac_bc[1][0][3][:,zi_here]

            #afgeleides naar st van het andere kanaal
            # jac[ju_here['obcR'][1][0]+m,ju_here['i0'][2][0]] = -jac_bc[2][0][0][zi_here]
            # jac[ju_here['obcR'][1][0]+m,ju_here['i1'][2][0]] = -jac_bc[2][0][1][zi_here]
            # jac[ju_here['obcR'][1][0]+m,ju_here['i2'][2][0]] = -jac_bc[2][0][2][zi_here]
            jac[ju_here['obcR'][1][0]+m,ju_here['i0'][2][0]+self.mm] = -jac_bc[2][0][3][:,zi_here]
           
            #imaginairy part
            #afgeleides naar Bs van hetzelfde kanaal
            jac[ju_here['obcI'][1][0]+m,ju_here['obcI'][1][0]+self.m0] = cos_here
            #afgeleiders naar Bs van het andere kanaal
            jac[ju_here['obcI'][1][0]+m,ju_here['obcI'][2][0]+self.m0] = - cos_here
            #afgeleides naar st van hetzelfde kanaal
            # jac[ju_here['obcI'][1][0]+m,ju_here['i0'][1][0]] = jac_bc[1][1][0][zi_here]
            # jac[ju_here['obcI'][1][0]+m,ju_here['i1'][1][0]] = jac_bc[1][1][1][zi_here]
            # jac[ju_here['obcI'][1][0]+m,ju_here['i2'][1][0]] = jac_bc[1][1][2][zi_here]
            jac[ju_here['obcI'][1][0]+m,ju_here['i0'][1][0]+self.mm] = jac_bc[1][1][3][:,zi_here]
            #afgeleides naar st van het andere kanaal
            # jac[ju_here['obcI'][1][0]+m,ju_here['i0'][2][0]] = -jac_bc[2][1][0][zi_here]
            # jac[ju_here['obcI'][1][0]+m,ju_here['i1'][2][0]] = -jac_bc[2][1][1][zi_here]
            # jac[ju_here['obcI'][1][0]+m,ju_here['i2'][2][0]] = -jac_bc[2][1][2][zi_here]
            jac[ju_here['obcI'][1][0]+m,ju_here['i0'][2][0]+self.mm] = -jac_bc[2][1][3][:,zi_here]
              
            # =============================================================================
            # third condition: diffusive flux in tidal cycle conserved
            # =============================================================================
            #real part
            #channel 1
            if ju_here['loc'][0] == 'loc x=-L': sign,eps_ind = -1 , 0
            elif ju_here['loc'][0] == 'loc x=0': sign,eps_ind = 1 , -1
            
            #afgeleides naar Bs van C1
            jac[ju_here['obcR'][2][0]+m,ju_here['obcR'][0][0]+self.m0] = - cos_here * self.ch_pars[ju_here['key'][0]]['eps'][eps_ind]**0.5 * jac_bc[0][2]
            #afgeleides naar st van C1
            # jac[ju_here['obcR'][2][0]+m,ju_here['i0'][0][0]] = sign * self.ch_pars[key]['eps'] * jac_bc[0][0][4][zi_here]
            # jac[ju_here['obcR'][2][0]+m,ju_here['i1'][0][0]] = sign * self.ch_pars[key]['eps'] * jac_bc[0][0][5][zi_here]
            # jac[ju_here['obcR'][2][0]+m,ju_here['i2'][0][0]] = sign * self.ch_pars[key]['eps'] * jac_bc[0][0][6][zi_here]
            # jac[ju_here['obcR'][2][0]+m,ju_here['i3'][0][0]] = sign * self.ch_pars[key]['eps'] * jac_bc[0][0][7][zi_here]
            jac[ju_here['obcR'][2][0]+m,ju_here['i0'][0][0]+self.mm] = sign * self.ch_pars[ju_here['key'][0]]['eps'][eps_ind] * jac_bc[0][0][8 ][:,zi_here]
            jac[ju_here['obcR'][2][0]+m,ju_here['i1'][0][0]+self.mm] = sign * self.ch_pars[ju_here['key'][0]]['eps'][eps_ind] * jac_bc[0][0][9 ][:,zi_here]
            jac[ju_here['obcR'][2][0]+m,ju_here['i2'][0][0]+self.mm] = sign * self.ch_pars[ju_here['key'][0]]['eps'][eps_ind] * jac_bc[0][0][10][:,zi_here]
               
                
            #channel 2
            if ju_here['loc'][1] == 'loc x=-L': sign,eps_ind = -1 , 0
            elif ju_here['loc'][1] == 'loc x=0': sign,eps_ind = 1 , -1
            
            #afgeleides naar Bs van C2
            jac[ju_here['obcR'][2][0]+m,ju_here['obcR'][1][0]+self.m0] = - cos_here * self.ch_pars[ju_here['key'][1]]['eps'][eps_ind]**0.5 * jac_bc[1][2]
            #afgeleides naar st van C2
            # jac[ju_here['obcR'][2][0]+m,ju_here['i0'][1][0]] = sign *self.ch_pars[key]['eps'] * jac_bc[1][0][4][zi_here]
            # jac[ju_here['obcR'][2][0]+m,ju_here['i1'][1][0]] = sign *self.ch_pars[key]['eps'] * jac_bc[1][0][5][zi_here]
            # jac[ju_here['obcR'][2][0]+m,ju_here['i2'][1][0]] = sign *self.ch_pars[key]['eps'] * jac_bc[1][0][6][zi_here]
            # jac[ju_here['obcR'][2][0]+m,ju_here['i3'][1][0]] = sign *self.ch_pars[key]['eps'] * jac_bc[1][0][7][zi_here]
            jac[ju_here['obcR'][2][0]+m,ju_here['i0'][1][0]+self.mm] = sign * self.ch_pars[ju_here['key'][1]]['eps'][eps_ind] * jac_bc[1][0][8 ][:,zi_here]
            jac[ju_here['obcR'][2][0]+m,ju_here['i1'][1][0]+self.mm] = sign * self.ch_pars[ju_here['key'][1]]['eps'][eps_ind] * jac_bc[1][0][9 ][:,zi_here]
            jac[ju_here['obcR'][2][0]+m,ju_here['i2'][1][0]+self.mm] = sign * self.ch_pars[ju_here['key'][1]]['eps'][eps_ind] * jac_bc[1][0][10][:,zi_here]
            
            
            #channel 3
            if ju_here['loc'][2] == 'loc x=-L': sign,eps_ind = -1 , 0
            elif ju_here['loc'][2] == 'loc x=0': sign,eps_ind = 1 , -1
            
            #afgeleides naar Bs van C3
            jac[ju_here['obcR'][2][0]+m,ju_here['obcR'][2][0]+self.m0] = - cos_here * self.ch_pars[ju_here['key'][2]]['eps'][eps_ind]**0.5 * jac_bc[2][2]
            #afgeleides naar st van C3
            # jac[ju_here['obcR'][2][0]+m,ju_here['i0'][2][0]] = sign *self.ch_pars[key]['eps'] * jac_bc[2][0][4][zi_here]
            # jac[ju_here['obcR'][2][0]+m,ju_here['i1'][2][0]] = sign *self.ch_pars[key]['eps'] * jac_bc[2][0][5][zi_here]
            # jac[ju_here['obcR'][2][0]+m,ju_here['i2'][2][0]] = sign *self.ch_pars[key]['eps'] * jac_bc[2][0][6][zi_here]
            # jac[ju_here['obcR'][2][0]+m,ju_here['i3'][2][0]] = sign *self.ch_pars[key]['eps'] * jac_bc[2][0][7][zi_here]
            jac[ju_here['obcR'][2][0]+m,ju_here['i0'][2][0]+self.mm] = sign * self.ch_pars[ju_here['key'][2]]['eps'][eps_ind] * jac_bc[2][0][8 ][:,zi_here]
            jac[ju_here['obcR'][2][0]+m,ju_here['i1'][2][0]+self.mm] = sign * self.ch_pars[ju_here['key'][2]]['eps'][eps_ind] * jac_bc[2][0][9 ][:,zi_here]
            jac[ju_here['obcR'][2][0]+m,ju_here['i2'][2][0]+self.mm] = sign * self.ch_pars[ju_here['key'][2]]['eps'][eps_ind] * jac_bc[2][0][10][:,zi_here]
            
            #imaginairy part follows here
            #channel 1
            if ju_here['loc'][0] == 'loc x=-L': sign,eps_ind = -1 , 0
            elif ju_here['loc'][0] == 'loc x=0': sign,eps_ind = 1 , -1
            
            #afgeleides naar Bs van C1
            jac[ju_here['obcI'][2][0]+m,ju_here['obcI'][0][0]+self.m0] = - cos_here * self.ch_pars[ju_here['key'][0]]['eps'][eps_ind]**0.5 * jac_bc[0][2]
            #afgeleides naar st van C1
            # jac[ju_here['obcI'][2][0]+m,ju_here['i0'][0][0]] = sign * self.ch_pars[key]['eps'] * jac_bc[0][1][4][zi_here]
            # jac[ju_here['obcI'][2][0]+m,ju_here['i1'][0][0]] = sign * self.ch_pars[key]['eps'] * jac_bc[0][1][5][zi_here]
            # jac[ju_here['obcI'][2][0]+m,ju_here['i2'][0][0]] = sign * self.ch_pars[key]['eps'] * jac_bc[0][1][6][zi_here]
            # jac[ju_here['obcI'][2][0]+m,ju_here['i3'][0][0]] = sign * self.ch_pars[key]['eps'] * jac_bc[0][1][7][zi_here]
            jac[ju_here['obcI'][2][0]+m,ju_here['i0'][0][0]+self.mm] = sign * self.ch_pars[ju_here['key'][0]]['eps'][eps_ind] * jac_bc[0][1][8 ][:,zi_here]
            jac[ju_here['obcI'][2][0]+m,ju_here['i1'][0][0]+self.mm] = sign * self.ch_pars[ju_here['key'][0]]['eps'][eps_ind] * jac_bc[0][1][9 ][:,zi_here]
            jac[ju_here['obcI'][2][0]+m,ju_here['i2'][0][0]+self.mm] = sign * self.ch_pars[ju_here['key'][0]]['eps'][eps_ind] * jac_bc[0][1][10][:,zi_here]
               
            #channel 2
            if ju_here['loc'][1] == 'loc x=-L': sign,eps_ind = -1 , 0
            elif ju_here['loc'][1] == 'loc x=0': sign,eps_ind = 1 , -1
            
            #afgeleides naar Bs van C2
            jac[ju_here['obcI'][2][0]+m,ju_here['obcI'][1][0]+self.m0] = - cos_here * self.ch_pars[ju_here['key'][1]]['eps'][eps_ind]**0.5 * jac_bc[1][2]
            #afgeleides naar st van C2
            # jac[ju_here['obcI'][2][0]+m,ju_here['i0'][1][0]] = sign *self.ch_pars[key]['eps'] * jac_bc[1][1][4][zi_here]
            # jac[ju_here['obcI'][2][0]+m,ju_here['i1'][1][0]] = sign *self.ch_pars[key]['eps'] * jac_bc[1][1][5][zi_here]
            # jac[ju_here['obcI'][2][0]+m,ju_here['i2'][1][0]] = sign *self.ch_pars[key]['eps'] * jac_bc[1][1][6][zi_here]
            # jac[ju_here['obcI'][2][0]+m,ju_here['i3'][1][0]] = sign *self.ch_pars[key]['eps'] * jac_bc[1][1][7][zi_here]
            jac[ju_here['obcI'][2][0]+m,ju_here['i0'][1][0]+self.mm] = sign * self.ch_pars[ju_here['key'][1]]['eps'][eps_ind] * jac_bc[1][1][8 ][:,zi_here]
            jac[ju_here['obcI'][2][0]+m,ju_here['i1'][1][0]+self.mm] = sign * self.ch_pars[ju_here['key'][1]]['eps'][eps_ind] * jac_bc[1][1][9 ][:,zi_here]
            jac[ju_here['obcI'][2][0]+m,ju_here['i2'][1][0]+self.mm] = sign * self.ch_pars[ju_here['key'][1]]['eps'][eps_ind] * jac_bc[1][1][10][:,zi_here]
            
            #channel 3
            if ju_here['loc'][2] == 'loc x=-L': sign,eps_ind = -1 , 0
            elif ju_here['loc'][2] == 'loc x=0': sign,eps_ind = 1 , -1
            
            #afgeleides naar Bs van C3
            jac[ju_here['obcI'][2][0]+m,ju_here['obcI'][2][0]+self.m0] = - cos_here * self.ch_pars[ju_here['key'][2]]['eps'][eps_ind]**0.5 * jac_bc[2][2]
            #afgeleides naar st van C3
            # jac[ju_here['obcI'][2][0]+m,ju_here['i0'][2][0]] = sign *self.ch_pars[key]['eps'] * jac_bc[2][1][4][zi_here]
            # jac[ju_here['obcI'][2][0]+m,ju_here['i1'][2][0]] = sign *self.ch_pars[key]['eps'] * jac_bc[2][1][5][zi_here]
            # jac[ju_here['obcI'][2][0]+m,ju_here['i2'][2][0]] = sign *self.ch_pars[key]['eps'] * jac_bc[2][1][6][zi_here]
            # jac[ju_here['obcI'][2][0]+m,ju_here['i3'][2][0]] = sign *self.ch_pars[key]['eps'] * jac_bc[2][1][7][zi_here]
            jac[ju_here['obcI'][2][0]+m,ju_here['i0'][2][0]+self.mm] = sign * self.ch_pars[ju_here['key'][2]]['eps'][eps_ind] * jac_bc[2][1][8 ][:,zi_here]
            jac[ju_here['obcI'][2][0]+m,ju_here['i1'][2][0]+self.mm] = sign * self.ch_pars[ju_here['key'][2]]['eps'][eps_ind] * jac_bc[2][1][9 ][:,zi_here]
            jac[ju_here['obcI'][2][0]+m,ju_here['i2'][2][0]+self.mm] = sign * self.ch_pars[ju_here['key'][2]]['eps'][eps_ind] * jac_bc[2][1][10][:,zi_here]       
            
    #'''
    return jac

def jac_junc_tot_vary(self, ans, junc_gegs, tid_inp, pars_Q):
    # =============================================================================
    # The jacobian matrix for the jucntions
    # =============================================================================
    
    #create empty vector
    jac = np.zeros((self.ch_inds[self.ch_keys[-1]]['totx'][-1] * self.M,self.ch_inds[self.ch_keys[-1]]['totx'][-1] * self.M))
    
    for j in range(1,self.n_j+1):
        
        ju_here = junc_gegs['j'+str(j)]
        keys_now = ju_here['key']
        locs_now = ju_here['loc']
        
        
        #third:transport
        
        #for depth-averaged transport
        for i in range(3): #calculate contributions from channels seperately 
            #prepare    
            key_here = keys_now[i]
            pars = self.ch_parja[key_here]
            
            if locs_now[i] == 'loc x=-L':
                jac[ju_here['i0'][2][0],ju_here['i0'][i][0]] = -(- 3/(2*self.ch_pars[key_here]['dl'][0]) * np.sum(ans[ju_here['i0'][i][1:]]* pars['C13c_x=-L']) ) 
                jac[ju_here['i0'][2][0],ju_here['i1'][i][0]] = -(  4/(2*self.ch_pars[key_here]['dl'][0]) * np.sum(ans[ju_here['i0'][i][1:]]* pars['C13c_x=-L']) ) 
                jac[ju_here['i0'][2][0],ju_here['i2'][i][0]] = -(- 1/(2*self.ch_pars[key_here]['dl'][0]) * np.sum(ans[ju_here['i0'][i][1:]]* pars['C13c_x=-L']) ) 
                jac[ju_here['i0'][2][0],ju_here['i0'][i][self.mm]] = -( pars['C13c_x=-L'] * (-3*ans[ju_here['i0'][i][0]]+4*ans[ju_here['i1'][i][0]]-ans[ju_here['i2'][i][0]])/(2*self.ch_pars[key_here]['dl'][0]) ) 

                
            elif locs_now[i] == 'loc x=0':
                jac[ju_here['i0'][2][0],ju_here['i0'][i][0]] =  3/(2*self.ch_pars[key_here]['dl'][-1]) * np.sum(ans[ju_here['i0'][i][1:]] * pars['C13c_x=0'])
                jac[ju_here['i0'][2][0],ju_here['i1'][i][0]] = -4/(2*self.ch_pars[key_here]['dl'][-1]) * np.sum(ans[ju_here['i0'][i][1:]] * pars['C13c_x=0'])
                jac[ju_here['i0'][2][0],ju_here['i2'][i][0]] =  1/(2*self.ch_pars[key_here]['dl'][-1]) * np.sum(ans[ju_here['i0'][i][1:]] * pars['C13c_x=0']) 
                jac[ju_here['i0'][2][0],ju_here['i0'][i][self.mm]] = pars['C13c_x=0'] * (ans[ju_here['i2'][i][0]] -4*ans[ju_here['i1'][i][0]] +3*ans[ju_here['i0'][i][0]] )/(2*self.ch_pars[key_here]['dl'][-1])  

                
            else: print('ERROR')
        
        #transport at every vertical level , with tidal contriubtion
        for k in range(1,self.M):
            for i in range(3): #calculate contributions from channels seperately 
                #prepare
                key_here = keys_now[i]
                pars = self.ch_parja[key_here]
                
                if locs_now[i] == 'loc x=-L':
                    jac[ju_here['i0'][2][k],ju_here['i0'][i][0]] = - (-3/(2*self.ch_pars[key_here]['dl'][0]) * pars['C12b_x=-L'][k-1] * ans[ju_here['i0'][i][k]] - 3/(2*self.ch_pars[key_here]['dl'][0]) * np.sum(pars['C12c_x=-L'][k-1]*ans[ju_here['i0'][i][1:]]))
                    jac[ju_here['i0'][2][k],ju_here['i1'][i][0]] = - (4/(2*self.ch_pars[key_here]['dl'][0]) * pars['C12b_x=-L'][k-1] * ans[ju_here['i0'][i][k]] + 4/(2*self.ch_pars[key_here]['dl'][0])  * np.sum(pars['C12c_x=-L'][k-1]*ans[ju_here['i0'][i][1:]]))
                    jac[ju_here['i0'][2][k],ju_here['i2'][i][0]] = - (-1/(2*self.ch_pars[key_here]['dl'][0]) * pars['C12b_x=-L'][k-1] * ans[ju_here['i0'][i][k]] - 1/(2*self.ch_pars[key_here]['dl'][0])  * np.sum(pars['C12c_x=-L'][k-1]*ans[ju_here['i0'][i][1:]]))
                    jac[ju_here['i0'][2][k],ju_here['i0'][i][k]] = - ( pars['C12b_x=-L'][k-1] * (-3*ans[ju_here['i0'][i][0]]+4*ans[ju_here['i1'][i][0]]-ans[ju_here['i2'][i][0]])/(2*self.ch_pars[key_here]['dl'][0]) )
                    jac[ju_here['i0'][2][k],ju_here['i0'][i][self.mm]] += - (pars['C12c_x=-L'][k-1] * (-3*ans[ju_here['i0'][i][0]]+4*ans[ju_here['i1'][i][0]]-ans[ju_here['i2'][i][0]])/(2*self.ch_pars[key_here]['dl'][0]) )

                
                elif locs_now[i] == 'loc x=0':
                    jac[ju_here['i0'][2][k],ju_here['i0'][i][0]] = 3/(2*self.ch_pars[key_here]['dl'][-1]) * pars['C12b_x=0'][k-1] * ans[ju_here['i0'][i][k]] + 3/(2*self.ch_pars[key_here]['dl'][-1]) * np.sum([pars['C12c_x=0'][k-1]*ans[ju_here['i0'][i][1:]]]) 
                    jac[ju_here['i0'][2][k],ju_here['i1'][i][0]] =-4/(2*self.ch_pars[key_here]['dl'][-1]) * pars['C12b_x=0'][k-1] * ans[ju_here['i0'][i][k]] - 4/(2*self.ch_pars[key_here]['dl'][-1]) * np.sum([pars['C12c_x=0'][k-1]*ans[ju_here['i0'][i][1:]]])
                    jac[ju_here['i0'][2][k],ju_here['i2'][i][0]] = 1/(2*self.ch_pars[key_here]['dl'][-1]) * pars['C12b_x=0'][k-1] * ans[ju_here['i0'][i][k]] + 1/(2*self.ch_pars[key_here]['dl'][-1]) * np.sum([pars['C12c_x=0'][k-1]*ans[ju_here['i0'][i][1:]]])
                    jac[ju_here['i0'][2][k],ju_here['i0'][i][k]] = pars['C12b_x=0'][k-1] * (ans[ju_here['i2'][i][0]]-4*ans[ju_here['i1'][i][0]]+3*ans[ju_here['i0'][i][0]])/(2*self.ch_pars[key_here]['dl'][-1]) 
                    jac[ju_here['i0'][2][k],ju_here['i0'][i][self.mm]] +=  pars['C12c_x=0'][k-1] * (ans[ju_here['i2'][i][0]]-4*ans[ju_here['i1'][i][0]]+3*ans[ju_here['i0'][i][0]])/(2*self.ch_pars[key_here]['dl'][-1]) 

                else: print('ERROR')


    return jac

