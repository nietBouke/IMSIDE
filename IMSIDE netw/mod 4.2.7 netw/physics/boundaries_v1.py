# =============================================================================
# file where functions are defined which calculate the conditions at the boundaries 
# of the channels in a channel network
# junctions are not treated here but in a seperate file 
# =============================================================================

import numpy as np

def func_sol_Tbnd(self, tid_inp, key):   
    # =============================================================================
    # Tidal transport at boundaries, to add to solution vector 
    # =============================================================================
    #load from inp
    st = tid_inp['st'][[0,-1]]
    
    #depth-averaged transport
    flux = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][[0,-1]] * np.mean(1/4 * np.real(self.ch_pars[key]['ut'][:,[0,-1]]*np.conj(st) + np.conj(self.ch_pars[key]['ut'][:,[0,-1]])*st) , axis=2)[0] / self.soc_sca 

    return flux 


def func_jac_Tbnd(self,  key):  
    # =============================================================================
    # Tidal transport at boundaries, to add to jacobian 
    # =============================================================================
    
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
    nph         = self.nn*np.pi/self.ch_gegs[key]['H']
    dsdc2_here  = self.ch_pars[key]['c2c'][:,[0,-1]]
    dsdc3_here  = self.ch_pars[key]['c3c'][:,[0,-1]]
    dsdc4_here  = self.ch_pars[key]['c4c'][:,[0,-1]]
    
    #depth-averaged
    #derivatives for x=-L
    dT0_dsb0 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0] * np.real( ut_here[:,0] * np.conj( dsdc2_here * self.go2 * detadx_here[:,0] * -3/(2*dx_here[0]) ) + np.conj(ut_here[:,0]) * dsdc2_here * self.go2 * detadx_here[:,0] * -3/(2*dx_here[0]) ).mean(2)
    dT0_dsb1 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0] * np.real( ut_here[:,0] * np.conj( dsdc2_here * self.go2 * detadx_here[:,0] *  4/(2*dx_here[0]) ) + np.conj(ut_here[:,0]) * dsdc2_here * self.go2 * detadx_here[:,0] *  4/(2*dx_here[0]) ).mean(2)
    dT0_dsb2 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0] * np.real( ut_here[:,0] * np.conj( dsdc2_here * self.go2 * detadx_here[:,0] * -1/(2*dx_here[0]) ) + np.conj(ut_here[:,0]) * dsdc2_here * self.go2 * detadx_here[:,0] * -1/(2*dx_here[0]) ).mean(2)
    dT0_dsn0 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0] * np.real( ut_here[:,0] * np.conj( dsdc3_here * - nph*eta_here[:,0] + dsdc4_here * - nph * self.go2 * (detadx2_here[:,0] + detadx_here[:,0]/bn_here[0]) ) 
                                                           + np.conj(ut_here[:,0]) * ( dsdc3_here * - nph*eta_here[:,0] + dsdc4_here * - nph * self.go2 * (detadx2_here[:,0] + detadx_here[:,0]/bn_here[0]) ) ).mean(2) 
    
    #derivatives for x=0
    dT_1_dsb_1 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1] * np.real( ut_here[:,1] * np.conj( dsdc2_here * self.go2 * detadx_here[:,1] *  3/(2*dx_here[1]) ) + np.conj(ut_here[:,1]) * dsdc2_here * self.go2 * detadx_here[:,1] *  3/(2*dx_here[1]) ).mean(2)
    dT_1_dsb_2 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1] * np.real( ut_here[:,1] * np.conj( dsdc2_here * self.go2 * detadx_here[:,1] * -4/(2*dx_here[1]) ) + np.conj(ut_here[:,1]) * dsdc2_here * self.go2 * detadx_here[:,1] * -4/(2*dx_here[1]) ).mean(2)
    dT_1_dsb_3 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1] * np.real( ut_here[:,1] * np.conj( dsdc2_here * self.go2 * detadx_here[:,1] *  1/(2*dx_here[1]) ) + np.conj(ut_here[:,1]) * dsdc2_here * self.go2 * detadx_here[:,1] *  1/(2*dx_here[1]) ).mean(2)
    dT_1_dsn_1 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1] * np.real( ut_here[:,1] * np.conj( dsdc3_here * - nph*eta_here[:,1] + dsdc4_here * - nph * self.go2 * (detadx2_here[:,1] + detadx_here[:,1]/bn_here[1]) ) 
                                                           + np.conj(ut_here[:,1]) * ( dsdc3_here * - nph*eta_here[:,1] + dsdc4_here * - nph * self.go2 * (detadx2_here[:,1] + detadx_here[:,1]/bn_here[1]) ) ).mean(2) 
    
    return (dT0_dsb0[0,0],dT0_dsb1[0,0],dT0_dsb2[0,0],dT0_dsn0[:,0] , dT_1_dsb_1[0,0],dT_1_dsb_2[0,0],dT_1_dsb_3[0,0],dT_1_dsn_1[:,0]) 


def func_sol_cor(self, ans, tid_inp, key):
    # =============================================================================
    # solution vector for the matching conditions for the boundary layer correction
    # we could get around inputting ans here. 
    # =============================================================================
    
    #prepare
    indi = self.ch_inds[key]
    
    # =============================================================================
    # calculate tidal salinity , not corrected
    # =============================================================================
    st        = tid_inp['st']
    dstdx     = tid_inp['dstidx']
    st_lft    = st[indi['di2_x=-L']]
    st_rgt    = st[indi['di2_x=0']]
    dstdx_lft = dstdx[indi['di2_x=-L']]
    dstdx_rgt = dstdx[indi['di2_x=0']]
    
    #the normalized tidal salinity, complex number
    P_lft = st_lft/self.soc_sca
    P_rgt = st_rgt/self.soc_sca
    #the normalized tidal salinity gradient, complex number
    dPdx_lft = dstdx_lft / self.soc_sca*self.Lsc
    dPdx_rgt = dstdx_rgt / self.soc_sca*self.Lsc
    # =============================================================================
    # calculate salinity correction
    # =============================================================================
    #real and imaginary parts equal - I first project and then set equal. Probably this projection is not nessecary - but lets keep it consequent        
    Bm_Re_lft = np.sum(ans[indi['ib_x=-L_bcR']][:,:,np.newaxis] * np.cos(self.npzh) , axis=0)#real part of salinity correction
    Bm_Im_lft = np.sum(ans[indi['ib_x=-L_bcI']][:,:,np.newaxis] * np.cos(self.npzh) , axis=0)#imaginary part of salinity correction
    Bm_Re_rgt = np.sum(ans[indi['ib_x=0_bcR']][:,:,np.newaxis] * np.cos(self.npzh) , axis=0)#real part of salinity correction
    Bm_Im_rgt = np.sum(ans[indi['ib_x=0_bcI']][:,:,np.newaxis] * np.cos(self.npzh) , axis=0)#imaginary part of salinity correction

    # =============================================================================
    # calculate the matching conditions for the solution vector
    # =============================================================================
    #real parts C1 and C2 equal
    sol_pt1 = ((np.real(P_rgt)+Bm_Re_rgt) - (np.real(P_lft)+Bm_Re_lft))[:,self.z_inds]
    #imaginairy parts C1 and C2 equal
    sol_pt2 = ((np.imag(P_rgt)+Bm_Im_rgt) - (np.imag(P_lft)+Bm_Im_lft))[:,self.z_inds]
    
    #(diffusive) transport in tidal cycle conserved
    sol_pt3 = (- self.eps * (np.real(dPdx_lft) + Bm_Re_lft/np.sqrt(self.eps)) + self.eps * (np.real(dPdx_rgt) - Bm_Re_rgt/np.sqrt(self.eps)))[:,self.z_inds]
    sol_pt4 = (- self.eps * (np.imag(dPdx_lft) + Bm_Im_lft/np.sqrt(self.eps)) + self.eps * (np.imag(dPdx_rgt) - Bm_Im_rgt/np.sqrt(self.eps)))[:,self.z_inds]

    #return the results 
    return sol_pt1,sol_pt2,sol_pt3,sol_pt4


def func_jac_cor(self, key):
    # =============================================================================
    # jacobian associated with the matching conditions for the boundary layer correction 
    # =============================================================================
    #prepare
    indi  = self.ch_inds[key]
    nph   = self.nn*np.pi/self.ch_gegs[key]['H']

    #local parameters
    dx_lft      = self.ch_gegs[key]['dx'][np.newaxis,1:,np.newaxis]
    bn_lft      = self.ch_pars[key]['bn'][np.newaxis,1:,np.newaxis]
    eta_lft     = self.ch_pars[key]['eta'][:,self.ch_pars[key]['di'][1:-1]]
    detadx_lft  = self.ch_pars[key]['detadx'][:,self.ch_pars[key]['di'][1:-1]]
    detadx2_lft = self.ch_pars[key]['detadx2'][:,self.ch_pars[key]['di'][1:-1]]
    detadx3_lft = self.ch_pars[key]['detadx3'][:,self.ch_pars[key]['di'][1:-1]]
    dsdc2_lft   = self.ch_pars[key]['c2c']#[:,self.ch_pars[key]['di'][1:-1]]
    dsdc3_lft   = self.ch_pars[key]['c3c']#[:,self.ch_pars[key]['di'][1:-1]]
    dsdc4_lft   = self.ch_pars[key]['c4c']#[:,self.ch_pars[key]['di'][1:-1]]
    
    dx_rgt      = self.ch_gegs[key]['dx'][np.newaxis,:-1,np.newaxis]
    bn_rgt      = self.ch_pars[key]['bn'][np.newaxis,:-1,np.newaxis]
    eta_rgt     = self.ch_pars[key]['eta'][:,self.ch_pars[key]['di'][1:-1]-1]
    detadx_rgt  = self.ch_pars[key]['detadx'][:,self.ch_pars[key]['di'][1:-1]-1]
    detadx2_rgt = self.ch_pars[key]['detadx2'][:,self.ch_pars[key]['di'][1:-1]-1]
    detadx3_rgt = self.ch_pars[key]['detadx3'][:,self.ch_pars[key]['di'][1:-1]-1]
    dsdc2_rgt   = self.ch_pars[key]['c2c']#[:,self.ch_pars[key]['di'][1:-1]-1]
    dsdc3_rgt   = self.ch_pars[key]['c3c']#[:,self.ch_pars[key]['di'][1:-1]-1]
    dsdc4_rgt   = self.ch_pars[key]['c4c']#[:,self.ch_pars[key]['di'][1:-1]-1]
    
    # =============================================================================
    # derivatives for st and dstdx
    # =============================================================================
    #left, or PvdA
    dst_lft_dsb0 = dsdc2_lft * self.go2 * detadx_lft * -3/(2*dx_lft)
    dst_lft_dsb1 = dsdc2_lft * self.go2 * detadx_lft *  4/(2*dx_lft) 
    dst_lft_dsb2 = dsdc2_lft * self.go2 * detadx_lft * -1/(2*dx_lft) 
    dst_lft_dsn0 = (dsdc3_lft * - nph*eta_lft + dsdc4_lft * - nph * self.go2 * (detadx2_lft + detadx_lft/bn_lft))

    dst_lft_x_dsb0 = self.Lsc*dsdc2_lft * self.go2 * (detadx2_lft * -3/(2*dx_lft) + detadx_lft *  2/(dx_lft**2) )
    dst_lft_x_dsb1 = self.Lsc*dsdc2_lft * self.go2 * (detadx2_lft *  4/(2*dx_lft) + detadx_lft * -5/(dx_lft**2) )
    dst_lft_x_dsb2 = self.Lsc*dsdc2_lft * self.go2 * (detadx2_lft * -1/(2*dx_lft) + detadx_lft *  4/(dx_lft**2) )
    dst_lft_x_dsb3 = self.Lsc*dsdc2_lft * self.go2 *  detadx_lft * -1/(dx_lft**2)
    dst_lft_x_dsn0 = self.Lsc*(dsdc3_lft * - nph * detadx_lft + dsdc4_lft * -nph * self.go2 * (detadx3_lft + detadx2_lft/bn_lft) \
                + (dsdc3_lft * -nph *eta_lft + dsdc4_lft * -nph *  self.go2 * (detadx2_lft + detadx_lft/bn_lft) ) * -3/(2*dx_lft))
    dst_lft_x_dsn1 = self.Lsc*(dsdc3_lft * -nph *eta_lft + dsdc4_lft * -nph *  self.go2 * (detadx2_lft + detadx_lft/bn_lft) ) *  4/(2*dx_lft)
    dst_lft_x_dsn2 = self.Lsc*(dsdc3_lft * -nph *eta_lft + dsdc4_lft * -nph *  self.go2 * (detadx2_lft + detadx_lft/bn_lft) ) * -1/(2*dx_lft)

    #right, or VVD
    dst_rgt_dsb_1 = dsdc2_rgt * self.go2 * detadx_rgt *  3/(2*dx_rgt) 
    dst_rgt_dsb_2 = dsdc2_rgt * self.go2 * detadx_rgt * -4/(2*dx_rgt) 
    dst_rgt_dsb_3 = dsdc2_rgt * self.go2 * detadx_rgt *  1/(2*dx_rgt) 
    dst_rgt_dsn_1 = (dsdc3_rgt * - nph*eta_rgt + dsdc4_rgt * - nph * self.go2 * (detadx2_rgt + detadx_rgt/bn_rgt)) 
    
    dst_rgt_x_dsb_1 = self.Lsc*dsdc2_rgt * self.go2 * (detadx2_rgt *  3/(2*dx_rgt) + detadx_rgt *  2/(dx_rgt**2) )
    dst_rgt_x_dsb_2 = self.Lsc*dsdc2_rgt * self.go2 * (detadx2_rgt * -4/(2*dx_rgt) + detadx_rgt * -5/(dx_rgt**2) )
    dst_rgt_x_dsb_3 = self.Lsc*dsdc2_rgt * self.go2 * (detadx2_rgt *  1/(2*dx_rgt) + detadx_rgt *  4/(dx_rgt**2) )
    dst_rgt_x_dsb_4 = self.Lsc*dsdc2_rgt * self.go2 *  detadx_rgt * -1/(dx_rgt**2)
    dst_rgt_x_dsn_1 = self.Lsc*(dsdc3_rgt * - nph * detadx_rgt + dsdc4_rgt * -nph * self.go2 * (detadx3_rgt + detadx2_rgt/bn_rgt) \
                  + (dsdc3_rgt * -nph *eta_rgt + dsdc4_rgt * -nph *  self.go2 * (detadx2_rgt + detadx_rgt/bn_rgt) ) *  3/(2*dx_rgt))
    dst_rgt_x_dsn_2 = self.Lsc*(dsdc3_rgt * -nph *eta_rgt + dsdc4_rgt * -nph *  self.go2 * (detadx2_rgt + detadx_rgt/bn_rgt) ) * -4/(2*dx_rgt)
    dst_rgt_x_dsn_3 = self.Lsc*(dsdc3_rgt * -nph *eta_rgt + dsdc4_rgt * -nph *  self.go2 * (detadx2_rgt + detadx_rgt/bn_rgt) ) *  1/(2*dx_rgt)
            
    #real and imaginairy           
    return (np.real(dst_lft_dsb0), np.real(dst_lft_dsb1), np.real(dst_lft_dsb2),
            np.real(dst_lft_dsn0),
            np.real(dst_lft_x_dsb0),np.real(dst_lft_x_dsb1),np.real(dst_lft_x_dsb2),np.real(dst_lft_x_dsb3),
            np.real(dst_lft_x_dsn0),np.real(dst_lft_x_dsn1),np.real(dst_lft_x_dsn2)) , \
           (np.imag(dst_lft_dsb0), np.imag(dst_lft_dsb1), np.imag(dst_lft_dsb2),
            np.imag(dst_lft_dsn0),
            np.imag(dst_lft_x_dsb0),np.imag(dst_lft_x_dsb1),np.imag(dst_lft_x_dsb2),np.imag(dst_lft_x_dsb3),
            np.imag(dst_lft_x_dsn0),np.imag(dst_lft_x_dsn1),np.imag(dst_lft_x_dsn2)) ,\
           (np.real(dst_rgt_dsb_1), np.real(dst_rgt_dsb_2), np.real(dst_rgt_dsb_3),
            np.real(dst_rgt_dsn_1),
            np.real(dst_rgt_x_dsb_1), np.real(dst_rgt_x_dsb_2), np.real(dst_rgt_x_dsb_3), np.real(dst_rgt_x_dsb_4),
            np.real(dst_rgt_x_dsn_1), np.real(dst_rgt_x_dsn_2), np.real(dst_rgt_x_dsn_3)) ,\
           (np.imag(dst_rgt_dsb_1), np.imag(dst_rgt_dsb_2), np.imag(dst_rgt_dsb_3),
            np.imag(dst_rgt_dsn_1),
            np.imag(dst_rgt_x_dsb_1), np.imag(dst_rgt_x_dsb_2), np.imag(dst_rgt_x_dsb_3), np.imag(dst_rgt_x_dsb_4),
            np.imag(dst_rgt_x_dsn_1), np.imag(dst_rgt_x_dsn_2), np.imag(dst_rgt_x_dsn_3)) 
  




def sol_bound(self, key, ans, tid_inp, pars_Q, pars_s):
    #load salinities
    sri, swe, soc = pars_s
    # =============================================================================
    # the solution vector for the boundary conditions     
    # =============================================================================
    
    #create empty vector
    so = np.zeros(self.ch_inds[key]['di3'][-1]*self.M)
    #local variables, for shorter notation
    inds = self.ch_inds[key].copy()
    dl = self.ch_pars[key]['dl'].copy()
    pars = self.ch_parja[key].copy()
    
    #subtract variables
    sb_bnd0 = ans[2*self.M]
    sb_bnd1 = ans[3*self.M]
    sb_bnd2 = ans[4*self.M]
    #sb_bnd3 = ans[5*self.M]
    
    sn_bnd0 = ans[2*self.M + self.mm]
    sn_bnd1 = ans[3*self.M + self.mm]
    sn_bnd2 = ans[4*self.M + self.mm]
    #sn_bnd3 = ans[5*self.M + self.mm]
    
    sb_bnd_1 = ans[-3*self.M]
    sb_bnd_2 = ans[-4*self.M]
    sb_bnd_3 = ans[-5*self.M]
    #sb_bnd_4 = ans[-6*self.M]

    sn_bnd_1 = ans[-3*self.M + self.mm]
    sn_bnd_2 = ans[-4*self.M + self.mm]
    sn_bnd_3 = ans[-5*self.M + self.mm]
    #sn_bnd_4 = ans[-6*self.M + self.mm]
    
     
    # =============================================================================
    # river boundaries - sriv prescribed
    # =============================================================================
    if self.ch_gegs[key]['loc x=-L'][0] == 'r':
        so[2*self.M] = sb_bnd0 - sri[int(self.ch_gegs[key]['loc x=-L'][1])-1]/self.soc_sca
        so[2*self.M+1:3*self.M] = sn_bnd0
    elif self.ch_gegs[key]['loc x=0'][0] == 'r':
        so[-3*self.M] = sb_bnd_1 - sri[int(self.ch_gegs[key]['loc x=0'][1])-1]/self.soc_sca
        so[-3*self.M+1:-2*self.M] = sn_bnd_1
        
    
    
    # =============================================================================
    # weir boundaries: flux prescribed
    # =============================================================================
    if self.ch_gegs[key]['loc x=-L'][0] == 'w':
        #prepare tides
        Ttid = func_sol_Tbnd(self, tid_inp, key)        
        if pars_Q[key] > 0: #flux is equal to the advective flux through weir
            so[2*self.M] = pars['C13a_x=-L']*pars_Q[key]*sb_bnd0 - pars['C13a_x=-L']*pars_Q[key] * swe[int(self.ch_gegs[key]['loc x=-L'][1])-1]/self.soc_sca \
                + pars['C13d_x=-L'] * (-3*sb_bnd0 + 4*sb_bnd1 - sb_bnd2) / (2*dl[0]) \
                + pars_Q[key] * np.sum(pars['C13b_x=-L'] * sn_bnd0) \
                + (-3*sb_bnd0 + 4*sb_bnd1 - sb_bnd2)/(2*dl[0]) * np.sum(pars['C13c_x=-L'] * sn_bnd0) \
                + Ttid[0]
                    
        elif pars_Q[key]<=0: #flux is equal to the advective flux through weir, set by river salinity
            so[2*self.M] = pars['C13d_x=-L'] * (-3*sb_bnd0 + 4*sb_bnd1 - sb_bnd2) / (2*dl[0]) \
            + pars_Q[key] * np.sum(pars['C13b_x=-L'] * sn_bnd0) \
            + (-3*sb_bnd0 + 4*sb_bnd1 - sb_bnd2)/(2*dl[0]) * np.sum(pars['C13c_x=-L'] * sn_bnd0) \
            + Ttid[0] 
        #no flux at depth - 
        so[2*self.M+1:3*self.M] = (-3*sn_bnd0+4*sn_bnd1-sn_bnd2)/(2*dl[0]) 
        
    elif self.ch_gegs[key]['loc x=0'][0] == 'w':
        #prepare tides
        Ttid = func_sol_Tbnd(self, tid_inp, key)   
        if pars_Q[key]>=0: #flux is equal to the advective flux through weir
            so[-3*self.M] = pars['C13d_x=0'] * (3*sb_bnd_1 - 4*sb_bnd_2 + sb_bnd_3)/(2*dl[-1]) \
                + pars_Q[key] * np.sum(pars['C13b_x=0'] * sn_bnd_1) \
                + (3*sb_bnd_1 - 4*sb_bnd_2 + sb_bnd_3)/(2*dl[-1]) * np.sum(pars['C13c_x=0'] * sn_bnd_1) \
                + Ttid[-1]
        elif pars_Q[key]<0: #flux is equal to the advective flux through weir, set by weir salinity
            so[-3*self.M] =  pars['C13a_x=0'] * pars_Q[key]*sb_bnd_1 - pars['C13a_x=0'] * pars_Q[key] * swe[int(self.ch_gegs[key]['loc x=0'][1])-1]/self.soc_sca \
                    + pars['C13d_x=0'] * (3*sb_bnd_1 - 4*sb_bnd_2 + sb_bnd_3)/(2*dl[-1]) \
                    + pars_Q[key] * np.sum(pars['C13b_x=0'] * sn_bnd_1) \
                    + (3*sb_bnd_1 - 4*sb_bnd_2 + sb_bnd_3)/(2*dl[-1]) * np.sum(pars['C13c_x=0'] * sn_bnd_1) \
                    + Ttid[-1]
         #no flux at depth 
        so[-2*self.M-self.N:-2*self.M] =  (3*sn_bnd_1-4*sn_bnd_2+sn_bnd_3)/(2*dl[-1])            
       
    # =============================================================================
    # har boundaries: only seaward flux 
    # =============================================================================
    if self.ch_gegs[key]['loc x=-L'][0] == 'h':
        #prepare tides
        Ttid = func_sol_Tbnd(self, tid_inp, key)   
        #only seaward flux
        so[2*self.M] = pars['C13d_x=-L'] * (-3*sb_bnd0 + 4*sb_bnd1 - sb_bnd2) / (2*dl[0]) \
        + pars_Q[key] * np.sum(pars['C13b_x=-L'] * sn_bnd0) \
        + (-3*sb_bnd0 + 4*sb_bnd1 - sb_bnd2)/(2*dl[0]) * np.sum(pars['C13c_x=-L'] * sn_bnd0) \
        + Ttid[0]

        #no flux at depth - 
        so[2*self.M+1:3*self.M] = (-3*sn_bnd0+4*sn_bnd1-sn_bnd2)/(2*dl[0]) 

    elif self.ch_gegs[key]['loc x=0'][0] == 'h':
        #prepare tides
        Ttid = func_sol_Tbnd(self, tid_inp, key)   
        #only seaward flux
        so[-3*self.M] = pars['C13d_x=0'] * (3*sb_bnd_1 - 4*sb_bnd_2 + sb_bnd_3)/(2*dl[-1]) \
            + pars_Q[key] * np.sum(pars['C13b_x=0'] * sn_bnd_1) \
            + (3*sb_bnd_1 - 4*sb_bnd_2 + sb_bnd_3)/(2*dl[-1]) * np.sum(pars['C13c_x=0'] * sn_bnd_1) \
            + Ttid[-1]
        #no diffusive flux at depth
        so[-2*self.M-self.N:-2*self.M] = (3*sn_bnd_1-4*sn_bnd_2+sn_bnd_3)/(2*dl[-1])  
    
    # =============================================================================
    # sea boundaries - soc prescribed
    # =============================================================================
    if self.ch_gegs[key]['loc x=-L'][0] == 's':
        so[2*self.M] = sb_bnd0 - soc[int(self.ch_gegs[key]['loc x=-L'][1])-1]/self.soc_sca
        so[2*self.M+1:3*self.M] = sn_bnd0
    elif self.ch_gegs[key]['loc x=0'][0] == 's':
        so[-3*self.M] = sb_bnd_1 - soc[int(self.ch_gegs[key]['loc x=0'][1])-1]/self.soc_sca
        so[-2*self.M-self.N:-2*self.M] = sn_bnd_1
    
    
    # =============================================================================
    # boundary layer correction for outer boundaries is absent
    # =============================================================================
    if self.ch_gegs[key]['loc x=-L'][0] in ['s','r','w','h']:
        so[:2*self.M] = ans[:2*self.M]
    if self.ch_gegs[key]['loc x=0'][0] in ['s','r','w','h']:
        so[-2*self.M:] = ans[-2*self.M:]

    
    # =============================================================================
    # inner boundaries 
    # keep simple formulation for now. - when including boundary layer correction, this should be adjusted. maybe. 
    # =============================================================================
    #salinities at boundaries   
    s_1 = ans[inds['i_sa_1']]
    s_2 = ans[inds['i_sa_2']]
    s_3 = ans[inds['i_sa_3']]

    s0 = ans[inds['i_sa0']]
    s1 = ans[inds['i_sa1']]
    s2 = ans[inds['i_sa2']]
    
    #salintiy equal (subtidal)
    #print( so[inds['bnd_rgt']] , s_1 - s0)
    so[inds['ib_rgt']] = s_1 - s0
    #salt derivative equal (subtidal)
    so[inds['ib_lft']] = (3*s_1 - 4*s_2 + s_3)/(2*dl[inds['i_p_1']]) - (-3*s0 + 4*s1 - s2)/(2*dl[inds['i_p0']]) 

    #transport formulation - not required for boundary layer correction, but for if we want to change the depth of the sections.  
    
    # =============================================================================
    # boundary layer correction around the inner boundaries.     
    # =============================================================================
    #'''
    correction = func_sol_cor(self, ans, tid_inp, key)
    #(simple) equations for the boundary layer corrections
    so[inds['ibc_lft'][:,:self.M]] = correction[0]
    so[inds['ibc_lft'][:,self.M:]] = correction[1]    
    so[inds['ibc_rgt'][:,:self.M]] = correction[2]
    so[inds['ibc_rgt'][:,self.M:]] = correction[3]
    '''
    
    so[inds['ibc_lft'][:,:self.M]] = ans[inds['ibc_lft'][:,:self.M]]
    so[inds['ibc_lft'][:,self.M:]] = ans[inds['ibc_lft'][:,self.M:]]
    so[inds['ibc_rgt'][:,:self.M]] = ans[inds['ibc_rgt'][:,:self.M]]
    so[inds['ibc_rgt'][:,self.M:]] = ans[inds['ibc_rgt'][:,self.M:]]
    #'''

    return so




    
        
def jac_bound_fix(self, key, pars_Q):    
    # =============================================================================
    # the jacobian matrix for the boundary conditions     
    # =============================================================================
    
    #create empty vector
    jac = np.zeros((self.ch_inds[key]['di3'][-1]*self.M,self.ch_inds[key]['di3'][-1]*self.M))
    #local variables, for shorter notation
    inds = self.ch_inds[key].copy()
    dl = self.ch_pars[key]['dl'].copy()
    pars = self.ch_parja[key].copy()
       
    
    # =============================================================================
    # river boundaries and sea boundaries
    # =============================================================================
    if self.ch_gegs[key]['loc x=-L'][0]  == 'r' or self.ch_gegs[key]['loc x=-L'][0] == 's': jac[ 2*self.M + self.m0, 2*self.M + self.m0] = 1
    elif self.ch_gegs[key]['loc x=0'][0] == 'r' or self.ch_gegs[key]['loc x=0'][0]  == 's': jac[-3*self.M + self.m0,-3*self.M + self.m0] = 1
    
    # =============================================================================
    # weir boundaries: flux prescribed
    # =============================================================================    
    if self.ch_gegs[key]['loc x=-L'][0] == 'w':
        Ttid = func_jac_Tbnd(self, key)
        if pars_Q[key]>0: #flux is equal to the advective flux through weir
            jac[2*self.M,2*self.M] = pars['C13a_x=-L']*pars_Q[key] -3/(2*dl[0]) * pars['C13d_x=-L']  + Ttid[0]
            jac[2*self.M,3*self.M] =  4/(2*dl[0]) * pars['C13d_x=-L'] + Ttid[1] 
            jac[2*self.M,4*self.M] = -1/(2*dl[0]) * pars['C13d_x=-L'] + Ttid[2]       
            for j in range(1,self.M): jac[2*self.M,2*self.M + j] = pars['C13b_x=-L'][j-1]*pars_Q[key] + Ttid[3][j-1]

        elif pars_Q[key]<=0: #flux is equal to the advective flux through weir, set by river salinity
            jac[2*self.M,2*self.M] = -3/(2*dl[0]) * pars['C13d_x=-L'] + Ttid[0] 
            jac[2*self.M,3*self.M] =  4/(2*dl[0]) * pars['C13d_x=-L'] + Ttid[1]
            jac[2*self.M,4*self.M] = -1/(2*dl[0]) * pars['C13d_x=-L'] + Ttid[2]             
            for j in range(1,self.M): jac[2*self.M,2*self.M + j] = pars['C13b_x=-L'][j-1]*pars_Q[key]  + Ttid[3][j-1]

        #no diff flux at depth
        jac[2*self.M+self.mm, 2*self.M+self.mm] = -3/(2*dl[0])  
        jac[2*self.M+self.mm, 3*self.M+self.mm] =  4/(2*dl[0]) 
        jac[2*self.M+self.mm, 4*self.M+self.mm] = -1/(2*dl[0]) 

    elif self.ch_gegs[key]['loc x=0'][0] == 'w':
        Ttid = func_jac_Tbnd(self, tid_inp, key) 
        if pars_Q[key]>0: #flux is equal to the advective flux through weir
            jac[-3*self.M,-3*self.M] = 3/(2*dl[-1]) * pars['C13d_x=0'] + Ttid[4]
            jac[-3*self.M,-4*self.M] =-4/(2*dl[-1]) * pars['C13d_x=0'] + Ttid[5] 
            jac[-3*self.M,-5*self.M] = 1/(2*dl[-1]) * pars['C13d_x=0'] + Ttid[6]        
            for j in range(1,self.M): jac[-3*self.M,-3*self.M + j] = pars['C13b_x=0'][j-1]*pars_Q[key] +  Ttid[7][j-1]

        elif pars_Q[key]<=0: #flux is equal to the advective flux through weir, set by river salinity
            jac[-3*self.M,-3*self.M] = pars['C13a_x=0']*pars_Q[key] + 3/(2*dl[-1]) * pars['C13d_x=0']  + Ttid[4]
            jac[-3*self.M,-4*self.M] =-4/(2*dl[-1]) * pars['C13d_x=0'] + Ttid[5]
            jac[-3*self.M,-5*self.M] = 1/(2*dl[-1]) * pars['C13d_x=0'] + Ttid[6]
            for j in range(1,self.M): jac[-3*self.M,-3*self.M + j] = pars['C13b_x=0'][j-1]*pars_Q[key] + Ttid[7][j-1]

        #no diffusive flux at depth
        jac[-3*self.M+self.mm, -3*self.M+self.mm] = 3/(2*dl[-1])  
        jac[-3*self.M+self.mm, -4*self.M+self.mm] = -4/(2*dl[-1]) 
        jac[-3*self.M+self.mm, -5*self.M+self.mm] = 1/(2*dl[-1]) 
     
    # =============================================================================
    # har boundaries: only seaward flux 
    # =============================================================================    
    if self.ch_gegs[key]['loc x=-L'][0] == 'h':
        Ttid = func_jac_Tbnd(self, key)  
        #no total flux
        jac[2*self.M,2*self.M] = -3/(2*dl[0]) * pars['C13d_x=-L'] + Ttid[0]
        jac[2*self.M,3*self.M] =  4/(2*dl[0]) * pars['C13d_x=-L'] + Ttid[1]
        jac[2*self.M,4*self.M] = -1/(2*dl[0]) * pars['C13d_x=-L'] + Ttid[2]
        for j in range(1,self.M): jac[2*self.M,2*self.M + j] = pars['C13b_x=-L'][j-1]*pars_Q[key] + Ttid[3][j-1]
        #no flux at depth
        jac[2*self.M+self.mm, 2*self.M+self.mm] = -3/(2*dl[0])  
        jac[2*self.M+self.mm, 3*self.M+self.mm] =  4/(2*dl[0]) 
        jac[2*self.M+self.mm, 4*self.M+self.mm] = -1/(2*dl[0]) 

    elif self.ch_gegs[key]['loc x=0'][0] == 'h':
        Ttid = func_jac_Tbnd(self, key)  
        #no total flux
        jac[-3*self.M,-3*self.M] = 3/(2*dl[-1]) * pars['C13d_x=0'] + Ttid[4]
        jac[-3*self.M,-4*self.M] =-4/(2*dl[-1]) * pars['C13d_x=0'] + Ttid[5]
        jac[-3*self.M,-5*self.M] = 1/(2*dl[-1]) * pars['C13d_x=0'] + Ttid[6]
        for j in range(1,self.M): jac[-3*self.M,-3*self.M + j] = pars['C13b_x=0'][j-1]*pars_Q[key] + Ttid[7][j-1]

        #no diffusive flux at depth
        jac[-3*self.M+self.mm, -3*self.M+self.mm] = 3/(2*dl[-1])  
        jac[-3*self.M+self.mm, -4*self.M+self.mm] = -4/(2*dl[-1]) 
        jac[-3*self.M+self.mm, -5*self.M+self.mm] = 1/(2*dl[-1]) 
    
    
    # =============================================================================
    # boundary layer correction for outer boundaries is absent
    # =============================================================================
    if self.ch_gegs[key]['loc x=-L'][0] in ['s','r','w','h']:
        jac[np.arange(2*self.M),np.arange(2*self.M)] = 1
    if self.ch_gegs[key]['loc x=0'][0] in ['s','r','w','h']:
        jac[np.arange(-2*self.M,0),np.arange(-2*self.M,0)] = 1

    
    # =============================================================================
    # inner boundaries 
    # keep simple formulation for now. - when including boundary layer correction, this should be adjusted. maybe. 
    # =============================================================================
    #salinities at boundaries        
    jac[inds['ib_rgt'] , inds['i_sa_1']] = 1
    jac[inds['ib_rgt'] , inds['i_sa0']] = -1
   
    jac[inds['ib_lft'] , inds['i_sa_3']] += 1/(2*dl[inds['i_p_1']])
    jac[inds['ib_lft'] , inds['i_sa_2']] += -4/(2*dl[inds['i_p_1']])
    jac[inds['ib_lft'] , inds['i_sa_1']] += 3/(2*dl[inds['i_p_1']])
    jac[inds['ib_lft'] , inds['i_sa0']] += 3/(2*dl[inds['i_p0']])
    jac[inds['ib_lft'] , inds['i_sa1']] += -4/(2*dl[inds['i_p0']])
    jac[inds['ib_lft'] , inds['i_sa2']] += 1/(2*dl[inds['i_p0']])
    
    
    
    '''
    jac[inds['ibc_lft'][:,:self.M] , inds['ibc_lft'][:,:self.M]] = 1
    jac[inds['ibc_lft'][:,self.M:] , inds['ibc_lft'][:,self.M:]] = 1
    jac[inds['ibc_rgt'][:,:self.M] , inds['ibc_rgt'][:,:self.M]] = 1
    jac[inds['ibc_rgt'][:,self.M:] , inds['ibc_rgt'][:,self.M:]] = 1
    
    
    
    '''
    # =============================================================================
    # boundary layer correction around the inner boundaries.     
    # TODO: idnices from somewhere else
    # =============================================================================
    jac_bc = func_jac_cor(self, key)
    # first condition: lft = rgt
    #real
    #correction
    jac[inds['ibc_lft2'] , np.tile(inds['ibc_rgt'][:,:self.M],self.M).flatten()] = np.tile(-np.cos(self.m0[:,np.newaxis]*np.pi*self.z_nd[self.z_inds]).T.flatten(),self.ch_pars[key]['n_seg']-1)
    jac[inds['ibc_lft2'] , np.tile(inds['ibc_lft'][:,:self.M],self.M).flatten()] = np.tile(np.cos(self.m0[:,np.newaxis]*np.pi*self.z_nd[self.z_inds]).T.flatten(),self.ch_pars[key]['n_seg']-1)
    
    #then other parts - lft
    jac[inds['ibc_lft2'] ,np.repeat(inds['i_sb0'],self.M**2)] += np.repeat(-jac_bc[0][0][0,:,self.z_inds].T,self.M)
    jac[inds['ibc_lft2'] ,np.repeat(inds['i_sb1'],self.M**2)] += np.repeat(-jac_bc[0][1][0,:,self.z_inds].T,self.M)
    jac[inds['ibc_lft2'] ,np.repeat(inds['i_sb2'],self.M**2)] += np.repeat(-jac_bc[0][2][0,:,self.z_inds].T,self.M)
    #jac[np.repeat(inds['ibc_lft'][:,:self.M],N) ,(np.repeat(inds['i_s0'],self.M)[:,np.newaxis]+self.mm).flatten()] += -jac_bc[0][3][:,:,self.z_inds].flatten()
    
    #then other parts - rgt
    jac[inds['ibc_lft2'] ,np.repeat(inds['i_sb_1'],self.M**2)] += np.repeat(jac_bc[2][0][0,:,self.z_inds].T,self.M)
    jac[inds['ibc_lft2'] ,np.repeat(inds['i_sb_2'],self.M**2)] += np.repeat(jac_bc[2][1][0,:,self.z_inds].T,self.M)
    jac[inds['ibc_lft2'] ,np.repeat(inds['i_sb_3'],self.M**2)] += np.repeat(jac_bc[2][2][0,:,self.z_inds].T,self.M)

    #imaginary
    #correction
    jac[self.M+inds['ibc_lft2'] , np.tile(inds['ibc_rgt'][:,self.M:],self.M).flatten()] = np.tile(-np.cos(self.m0[:,np.newaxis]*np.pi*self.z_nd[self.z_inds]).T.flatten(),self.ch_pars[key]['n_seg']-1)
    jac[self.M+inds['ibc_lft2'] , np.tile(inds['ibc_lft'][:,self.M:],self.M).flatten()] = np.tile(np.cos(self.m0[:,np.newaxis]*np.pi*self.z_nd[self.z_inds]).T.flatten(),self.ch_pars[key]['n_seg']-1)
    
    #then other parts - lft
    jac[self.M+inds['ibc_lft2'] ,np.repeat(inds['i_sb0'],self.M**2)] += np.repeat(-jac_bc[1][0][0,:,self.z_inds].T,self.M)
    jac[self.M+inds['ibc_lft2'] ,np.repeat(inds['i_sb1'],self.M**2)] += np.repeat(-jac_bc[1][1][0,:,self.z_inds].T,self.M)
    jac[self.M+inds['ibc_lft2'] ,np.repeat(inds['i_sb2'],self.M**2)] += np.repeat(-jac_bc[1][2][0,:,self.z_inds].T,self.M)


    #then other parts - rgt
    jac[self.M+inds['ibc_lft2'] ,np.repeat(inds['i_sb_1'],self.M**2)] += np.repeat(jac_bc[3][0][0,:,self.z_inds].T,self.M)
    jac[self.M+inds['ibc_lft2'] ,np.repeat(inds['i_sb_2'],self.M**2)] += np.repeat(jac_bc[3][1][0,:,self.z_inds].T,self.M)
    jac[self.M+inds['ibc_lft2'] ,np.repeat(inds['i_sb_3'],self.M**2)] += np.repeat(jac_bc[3][2][0,:,self.z_inds].T,self.M)

    # second condition: flux is constant
    
    #real
    #correction
    jac[inds['ibc_rgt2'] , np.tile(inds['ibc_lft'][:,:self.M],self.M).flatten()] = np.tile(-np.cos(self.m0[:,np.newaxis]*np.pi*self.z_nd[self.z_inds]).T.flatten(),self.ch_pars[key]['n_seg']-1) * self.eps**0.5
    jac[inds['ibc_rgt2'] , np.tile(inds['ibc_rgt'][:,:self.M],self.M).flatten()] = np.tile(-np.cos(self.m0[:,np.newaxis]*np.pi*self.z_nd[self.z_inds]).T.flatten(),self.ch_pars[key]['n_seg']-1) * self.eps**0.5

    #then other parts - lft
    jac[inds['ibc_rgt2'] ,np.repeat(inds['i_sb0'],self.M**2)] += np.repeat(-jac_bc[0][4][0,:,self.z_inds].T,self.M) * self.eps
    jac[inds['ibc_rgt2'] ,np.repeat(inds['i_sb1'],self.M**2)] += np.repeat(-jac_bc[0][5][0,:,self.z_inds].T,self.M) * self.eps
    jac[inds['ibc_rgt2'] ,np.repeat(inds['i_sb2'],self.M**2)] += np.repeat(-jac_bc[0][6][0,:,self.z_inds].T,self.M) * self.eps
    jac[inds['ibc_rgt2'] ,np.repeat(inds['i_sb3'],self.M**2)] += np.repeat(-jac_bc[0][7][0,:,self.z_inds].T,self.M) * self.eps

    #then other parts - rgt
    jac[inds['ibc_rgt2'] ,np.repeat(inds['i_sb_1'],self.M**2)] += np.repeat(jac_bc[2][4][0,:,self.z_inds].T,self.M) * self.eps
    jac[inds['ibc_rgt2'] ,np.repeat(inds['i_sb_2'],self.M**2)] += np.repeat(jac_bc[2][5][0,:,self.z_inds].T,self.M) * self.eps
    jac[inds['ibc_rgt2'] ,np.repeat(inds['i_sb_3'],self.M**2)] += np.repeat(jac_bc[2][6][0,:,self.z_inds].T,self.M) * self.eps
    jac[inds['ibc_rgt2'] ,np.repeat(inds['i_sb_4'],self.M**2)] += np.repeat(jac_bc[2][7][0,:,self.z_inds].T,self.M) * self.eps
    
    #imaginary
    #correction
    jac[self.M+inds['ibc_rgt2'] , np.tile(inds['ibc_lft'][:,self.M:],self.M).flatten()] += np.tile(-np.cos(self.m0[:,np.newaxis]*np.pi*self.z_nd[self.z_inds]).T.flatten(),self.ch_pars[key]['n_seg']-1) * self.eps**0.5
    jac[self.M+inds['ibc_rgt2'] , np.tile(inds['ibc_rgt'][:,self.M:],self.M).flatten()] += np.tile(-np.cos(self.m0[:,np.newaxis]*np.pi*self.z_nd[self.z_inds]).T.flatten(),self.ch_pars[key]['n_seg']-1) * self.eps**0.5

    #then other parts - lft
    jac[self.M+inds['ibc_rgt2'] ,np.repeat(inds['i_sb0'],self.M**2)] += np.repeat(-jac_bc[1][4][0,:,self.z_inds].T,self.M) * self.eps
    jac[self.M+inds['ibc_rgt2'] ,np.repeat(inds['i_sb1'],self.M**2)] += np.repeat(-jac_bc[1][5][0,:,self.z_inds].T,self.M) * self.eps
    jac[self.M+inds['ibc_rgt2'] ,np.repeat(inds['i_sb2'],self.M**2)] += np.repeat(-jac_bc[1][6][0,:,self.z_inds].T,self.M) * self.eps
    jac[self.M+inds['ibc_rgt2'] ,np.repeat(inds['i_sb3'],self.M**2)] += np.repeat(-jac_bc[1][7][0,:,self.z_inds].T,self.M) * self.eps

    #then other parts - rgt
    jac[self.M+inds['ibc_rgt2'] ,np.repeat(inds['i_sb_1'],self.M**2)] += np.repeat(jac_bc[3][4][0,:,self.z_inds].T,self.M) * self.eps
    jac[self.M+inds['ibc_rgt2'] ,np.repeat(inds['i_sb_2'],self.M**2)] += np.repeat(jac_bc[3][5][0,:,self.z_inds].T,self.M) * self.eps
    jac[self.M+inds['ibc_rgt2'] ,np.repeat(inds['i_sb_3'],self.M**2)] += np.repeat(jac_bc[3][6][0,:,self.z_inds].T,self.M) * self.eps
    jac[self.M+inds['ibc_rgt2'] ,np.repeat(inds['i_sb_4'],self.M**2)] += np.repeat(jac_bc[3][7][0,:,self.z_inds].T,self.M) * self.eps


    #TODO: loops should be replaced with indsces 
    for x9 in range(self.ch_pars[key]['n_seg']-1):
        for m in range(self.M):
            # =============================================================================
            # first condition: lft = rgt
            # =============================================================================

            #real        
            #then other parts - lft
            jac[inds['ibc_lft'][x9,m], inds['i_sb0'][x9]+self.mm] += -jac_bc[0][3][:,x9,self.z_inds[m]]
   
            #then other parts - rgt
            jac[inds['ibc_lft'][x9,m], inds['i_sb_1'][x9] + self.mm] += jac_bc[2][3][:,x9,self.z_inds[m]]

            #imaginary
            #then other parts - lft
            jac[inds['ibc_lft'][x9,self.M+m], inds['i_sb0'][x9]+self.mm] += -jac_bc[1][3][:,x9,self.z_inds[m]]

            #then other parts - rgt
            jac[inds['ibc_lft'][x9,self.M+m], inds['i_sb_1'][x9] + self.mm] += jac_bc[3][3][:,x9,self.z_inds[m]]
            # =============================================================================
            # second condition: flux is constant
            # =============================================================================
            
            #real
            
            #then other parts - lft
            jac[inds['ibc_rgt'][x9,m], inds['i_sb0'][x9]+self.mm] += -jac_bc[0][8][:,x9,self.z_inds[m]] * self.eps
            jac[inds['ibc_rgt'][x9,m], inds['i_sb1'][x9]+self.mm] += -jac_bc[0][9][:,x9,self.z_inds[m]] * self.eps
            jac[inds['ibc_rgt'][x9,m], inds['i_sb2'][x9]+self.mm] += -jac_bc[0][10][:,x9,self.z_inds[m]] * self.eps
            
            #then other parts - rgt
            jac[inds['ibc_rgt'][x9,m], inds['i_sb_1'][x9]+self.mm] += jac_bc[2][8][:,x9,self.z_inds[m]] * self.eps
            jac[inds['ibc_rgt'][x9,m], inds['i_sb_2'][x9]+self.mm] += jac_bc[2][9][:,x9,self.z_inds[m]] * self.eps
            jac[inds['ibc_rgt'][x9,m], inds['i_sb_3'][x9]+self.mm] += jac_bc[2][10][:,x9,self.z_inds[m]] * self.eps
            
            #imaginary
            
            #then other parts - lft
            jac[inds['ibc_rgt'][x9,self.M+m], inds['i_sb0'][x9]+self.mm] += -jac_bc[1][8][:,x9,self.z_inds[m]] * self.eps
            jac[inds['ibc_rgt'][x9,self.M+m], inds['i_sb1'][x9]+self.mm] += -jac_bc[1][9][:,x9,self.z_inds[m]] * self.eps
            jac[inds['ibc_rgt'][x9,self.M+m], inds['i_sb2'][x9]+self.mm] += -jac_bc[1][10][:,x9,self.z_inds[m]] * self.eps
            
            #then other parts - rgt
            jac[inds['ibc_rgt'][x9,self.M+m], inds['i_sb_1'][x9]+self.mm] += jac_bc[3][8][:,x9,self.z_inds[m]] * self.eps
            jac[inds['ibc_rgt'][x9,self.M+m], inds['i_sb_2'][x9]+self.mm] += jac_bc[3][9][:,x9,self.z_inds[m]] * self.eps
            jac[inds['ibc_rgt'][x9,self.M+m], inds['i_sb_3'][x9]+self.mm] += jac_bc[3][10][:,x9,self.z_inds[m]] * self.eps

    #'''
    return jac



def jac_bound_vary(self, key, ans, tid_inp, pars_Q):    
    # =============================================================================
    # the jacobian matrix for the boundary conditions     
    # =============================================================================
    
    #create empty vector
    jac = np.zeros((self.ch_inds[key]['di3'][-1]*self.M,self.ch_inds[key]['di3'][-1]*self.M))
    #local variables, for shorter notation
    inds = self.ch_inds[key].copy()
    dl = self.ch_pars[key]['dl'].copy()
    pars = self.ch_parja[key].copy()
    
    #subtract variables
    sb_bnd0 = ans[2*self.M]
    sb_bnd1 = ans[3*self.M]
    sb_bnd2 = ans[4*self.M]
    #sb_bnd3 = ans[5*self.M]
    
    sn_bnd0 = ans[2*self.M + self.mm]
    sn_bnd1 = ans[3*self.M + self.mm]
    sn_bnd2 = ans[4*self.M + self.mm]
    #sn_bnd3 = ans[5*self.M + self.mm]
    
    sb_bnd_1 = ans[-3*self.M]
    sb_bnd_2 = ans[-4*self.M]
    sb_bnd_3 = ans[-5*self.M]
    #sb_bnd_4 = ans[-6*self.M]

    sn_bnd_1 = ans[-3*self.M + self.mm]
    sn_bnd_2 = ans[-4*self.M + self.mm]
    sn_bnd_3 = ans[-5*self.M + self.mm]
    #sn_bnd_4 = ans[-6*self.M + self.mm]
    
    
    # =============================================================================
    # weir boundaries: flux prescribed
    # =============================================================================    
    if self.ch_gegs[key]['loc x=-L'][0] == 'w':
        C13_sum = np.sum(pars['C13c_x=-L'] * sn_bnd0)
        if pars_Q[key]>0: #flux is equal to the advective flux through weir
            jac[2*self.M,2*self.M] = -3/(2*dl[0]) * C13_sum
            jac[2*self.M,3*self.M] =  4/(2*dl[0]) * C13_sum
            jac[2*self.M,4*self.M] = -1/(2*dl[0]) * C13_sum      
            for j in range(1,self.M): jac[2*self.M,2*self.M + j] = (-3*sb_bnd0 + 4*sb_bnd1 - sb_bnd2)/(2*dl[0]) * pars['C13c_x=-L'][j-1] 

        elif pars_Q[key]<=0: #flux is equal to the advective flux through weir, set by river salinity
            jac[2*self.M,2*self.M] = -3/(2*dl[0]) * C13_sum
            jac[2*self.M,3*self.M] =  4/(2*dl[0]) * C13_sum
            jac[2*self.M,4*self.M] = -1/(2*dl[0]) * C13_sum
            for j in range(1,self.M): jac[2*self.M,2*self.M + j] = (-3*sb_bnd0 + 4*sb_bnd1 - sb_bnd2)/(2*dl[0]) * pars['C13c_x=-L'][j-1] 


    elif self.ch_gegs[key]['loc x=0'][0] == 'w':
        C13_sum = np.sum(pars['C13c_x=0'] * sn_bnd0)
        if pars_Q[key]>0: #flux is equal to the advective flux through weir
            jac[-3*self.M,-3*self.M] = 3/(2*dl[-1]) * C13_sum
            jac[-3*self.M,-4*self.M] =-4/(2*dl[-1]) * C13_sum
            jac[-3*self.M,-5*self.M] = 1/(2*dl[-1]) * C13_sum 
            for j in range(1,self.M): jac[-3*self.M,-3*self.M + j] = (3*sb_bnd_1 - 4*sb_bnd_2 + sb_bnd_3)/(2*dl[-1]) * pars['C13c_x=0'][j-1] 

        elif pars_Q[key]<=0: #flux is equal to the advective flux through weir, set by river salinity
            jac[-3*self.M,-3*self.M] = 3/(2*dl[-1]) * C13_sum
            jac[-3*self.M,-4*self.M] =-4/(2*dl[-1]) * C13_sum
            jac[-3*self.M,-5*self.M] = 1/(2*dl[-1]) * C13_sum
            for j in range(1,self.M): jac[-3*self.M,-3*self.M + j] = (3*sb_bnd_1 - 4*sb_bnd_2 + sb_bnd_3)/(2*dl[-1]) * pars['C13c_x=0'][j-1] 

     
    # =============================================================================
    # har boundaries: only seaward flux 
    # =============================================================================    
    if self.ch_gegs[key]['loc x=-L'][0] == 'h':
        C13_sum = np.sum(pars['C13c_x=-L'] * sn_bnd0)
        #no total flux
        jac[2*self.M,2*self.M] = -3/(2*dl[0]) * C13_sum
        jac[2*self.M,3*self.M] =  4/(2*dl[0]) * C13_sum
        jac[2*self.M,4*self.M] = -1/(2*dl[0]) * C13_sum
        for j in range(1,self.M): jac[2*self.M,2*self.M + j] = (-3*sb_bnd0 + 4*sb_bnd1 - sb_bnd2)/(2*dl[0]) * pars['C13c_x=-L'][j-1] 


    elif self.ch_gegs[key]['loc x=0'][0] == 'h':
        C13_sum = np.sum(pars['C13c_x=0'] * sn_bnd0)
        #no total flux
        jac[-3*self.M,-3*self.M] = 3/(2*dl[-1]) * C13_sum
        jac[-3*self.M,-4*self.M] =-4/(2*dl[-1]) * C13_sum
        jac[-3*self.M,-5*self.M] = 1/(2*dl[-1]) * C13_sum
        for j in range(1,self.M): jac[-3*self.M,-3*self.M + j] =  (3*sb_bnd_1 - 4*sb_bnd_2 + sb_bnd_3)/(2*dl[-1]) * pars['C13c_x=0'][j-1] 
    

    return jac