# =============================================================================
# here we define the contirbution to the jacobian and solution vector by the boundaries 
# also the equations for the boundary layer correction are here
# =============================================================================
import numpy as np

def corsol(self, zout, inp, tid_geg, indi):    
    # =============================================================================
    # solution vector for the matching conditions for the boundary layer correction 
    # =============================================================================
    #load salinity, not corrected
    st = inp['st']
    dstdx = inp['dstidx']
    
    st_lft = st[indi['ih_lft']]
    st_rgt = st[indi['ih_rgt']]
    dstdx_lft = dstdx[indi['ih_lft']]
    dstdx_rgt = dstdx[indi['ih_rgt']]
        
    #the normalized tidal salinity, complex number
    P_lft = st_lft/self.soc
    P_rgt = st_rgt/self.soc
    #the normalized tidal salinity gradient, complex number
    dPdx_lft = dstdx_lft / self.soc*self.Lsc
    dPdx_rgt = dstdx_rgt / self.soc*self.Lsc
    # =============================================================================
    # calculate salinity correction
    # =============================================================================
    #eigenlijk zijn dit indices en moet dit dus ergens anders. 
    bnl_Re_lft = ( self.di3[1:-1]   *self.M+np.arange(self.M)[:,np.newaxis])
    bnl_Im_lft = ( self.di3[1:-1]   *self.M+np.arange(self.M,2*self.M)[:,np.newaxis])
    bnl_Re_rgt = ((self.di3[1:-1]-2)*self.M+np.arange(self.M)[:,np.newaxis])
    bnl_Im_rgt = ((self.di3[1:-1]-2)*self.M+np.arange(self.M,2*self.M)[:,np.newaxis])
    
    #real and imaginary parts equal - I first project and then set equal. Probably this projection is not nessecary - but lets keep it consequent        
    Bm_Re_lft = np.sum(zout[bnl_Re_lft][:,:,np.newaxis] * np.cos(np.pi*np.arange(self.M)[:,np.newaxis,np.newaxis]*self.z_nd) , axis=0)#real part of salinity correction
    Bm_Im_lft = np.sum(zout[bnl_Im_lft][:,:,np.newaxis] * np.cos(np.pi*np.arange(self.M)[:,np.newaxis,np.newaxis]*self.z_nd) , axis=0)#imaginary part of salinity correction
    Bm_Re_rgt = np.sum(zout[bnl_Re_rgt][:,:,np.newaxis] * np.cos(np.pi*np.arange(self.M)[:,np.newaxis,np.newaxis]*self.z_nd) , axis=0)#real part of salinity correction
    Bm_Im_rgt = np.sum(zout[bnl_Im_rgt][:,:,np.newaxis] * np.cos(np.pi*np.arange(self.M)[:,np.newaxis,np.newaxis]*self.z_nd) , axis=0)#imaginary part of salinity correction

    # =============================================================================
    # calculate the matching conditions for the solution vector
    # =============================================================================
    #real parts C1 and C2 equal
    sol_pt1 = ((np.real(P_rgt)+Bm_Re_rgt) - (np.real(P_lft)+Bm_Re_lft))[:,self.z_inds]
    #imaginairy parts C1 and C2 equal
    sol_pt2 = ((np.imag(P_rgt)+Bm_Im_rgt) - (np.imag(P_lft)+Bm_Im_lft))[:,self.z_inds]
    
    #(diffusive) transport in tidal cycle conserved
    sol_pt3 = (- tid_geg['eps'] * self.A_lft[:,np.newaxis] * (np.real(dPdx_lft) + Bm_Re_lft/np.sqrt(tid_geg['eps'])) + tid_geg['eps'] * self.A_rgt[:,np.newaxis] * (np.real(dPdx_rgt) - Bm_Re_rgt/np.sqrt(tid_geg['eps'])))[:,self.z_inds]
    sol_pt4 = (- tid_geg['eps'] * self.A_lft[:,np.newaxis] * (np.imag(dPdx_lft) + Bm_Im_lft/np.sqrt(tid_geg['eps'])) + tid_geg['eps'] * self.A_rgt[:,np.newaxis] * (np.imag(dPdx_rgt) - Bm_Im_rgt/np.sqrt(tid_geg['eps'])))[:,self.z_inds]

    #return the results 
    return sol_pt1,sol_pt2,sol_pt3,sol_pt4
    
def corjac(self, zout, tid_geg, indi):
    # =============================================================================
    # jacobian associated with the matching conditions for the boundary layer correction 
    # =============================================================================
    #prepare
    dsdc2_lft = tid_geg['c2c'][:,indi['ih_lft']]
    dsdc3_lft = tid_geg['c3c'][:,indi['ih_lft']]
    dsdc4_lft = tid_geg['c4c'][:,indi['ih_lft']]
    
    dsdc2_rgt = tid_geg['c2c'][:,indi['ih_rgt']]
    dsdc3_rgt = tid_geg['c3c'][:,indi['ih_rgt']]
    dsdc4_rgt = tid_geg['c4c'][:,indi['ih_rgt']]
    
    #local parameters
    dx_lft      = self.dxn[np.newaxis,indi['ih2_lft'],np.newaxis]
    bn_lft      = self.bn[np.newaxis,indi['ih2_lft'],np.newaxis]
    eta_lft     = tid_geg['eta'][:,indi['ih_lft']]
    detadx_lft  = tid_geg['detadx'][:,indi['ih_lft']]
    detadx2_lft = tid_geg['detadx2'][:,indi['ih_lft']]
    detadx3_lft = tid_geg['detadx3'][:,indi['ih_lft']]
    
    dx_rgt      = self.dxn[np.newaxis,indi['ih2_rgt'],np.newaxis]
    bn_rgt      = self.bn[np.newaxis,indi['ih2_rgt'],np.newaxis]
    eta_rgt     = tid_geg['eta'][:,indi['ih_rgt']]
    detadx_rgt  = tid_geg['detadx'][:,indi['ih_rgt']]
    detadx2_rgt = tid_geg['detadx2'][:,indi['ih_rgt']]
    detadx3_rgt = tid_geg['detadx3'][:,indi['ih_rgt']]
    
    nph_lft = self.nph[:,indi['ih_lft']]
    nph_rgt = self.nph[:,indi['ih_rgt']]
    
    # =============================================================================
    # derivatives for st and dstdx
    # =============================================================================
    #left, or PvdA
    dst_lft_dsb0 = dsdc2_lft * self.g/tid_geg['omega']**2 * detadx_lft * -3/(2*dx_lft)
    dst_lft_dsb1 = dsdc2_lft * self.g/tid_geg['omega']**2 * detadx_lft *  4/(2*dx_lft) 
    dst_lft_dsb2 = dsdc2_lft * self.g/tid_geg['omega']**2 * detadx_lft * -1/(2*dx_lft) 
    dst_lft_dsn0 = (dsdc3_lft * - nph_lft*eta_lft + dsdc4_lft * - nph_lft * self.g/tid_geg['omega']**2 * (detadx2_lft + detadx_lft/bn_lft))

    dst_lft_x_dsb0 = self.A_lft[np.newaxis,:,np.newaxis] * self.Lsc*dsdc2_lft * self.g/tid_geg['omega']**2 * (detadx2_lft * -3/(2*dx_lft) + detadx_lft *  2/(dx_lft**2) )
    dst_lft_x_dsb1 = self.A_lft[np.newaxis,:,np.newaxis] * self.Lsc*dsdc2_lft * self.g/tid_geg['omega']**2 * (detadx2_lft *  4/(2*dx_lft) + detadx_lft * -5/(dx_lft**2) )
    dst_lft_x_dsb2 = self.A_lft[np.newaxis,:,np.newaxis] * self.Lsc*dsdc2_lft * self.g/tid_geg['omega']**2 * (detadx2_lft * -1/(2*dx_lft) + detadx_lft *  4/(dx_lft**2) )
    dst_lft_x_dsb3 = self.A_lft[np.newaxis,:,np.newaxis] * self.Lsc*dsdc2_lft * self.g/tid_geg['omega']**2 *  detadx_lft * -1/(dx_lft**2)
    dst_lft_x_dsn0 = self.A_lft[np.newaxis,:,np.newaxis] * self.Lsc*(dsdc3_lft * - nph_lft * detadx_lft + dsdc4_lft * -nph_lft * self.g/tid_geg['omega']**2 * (detadx3_lft + detadx2_lft/bn_lft) \
                + (dsdc3_lft * -nph_lft *eta_lft + dsdc4_lft * -nph_lft *  self.g/tid_geg['omega']**2 * (detadx2_lft + detadx_lft/bn_lft) ) * -3/(2*dx_lft))
    dst_lft_x_dsn1 = self.A_lft[np.newaxis,:,np.newaxis] * self.Lsc*(dsdc3_lft * -nph_lft *eta_lft + dsdc4_lft * -nph_lft *  self.g/tid_geg['omega']**2 * (detadx2_lft + detadx_lft/bn_lft) ) *  4/(2*dx_lft)
    dst_lft_x_dsn2 = self.A_lft[np.newaxis,:,np.newaxis] * self.Lsc*(dsdc3_lft * -nph_lft *eta_lft + dsdc4_lft* -nph_lft *  self.g/tid_geg['omega']**2 * (detadx2_lft + detadx_lft/bn_lft) ) * -1/(2*dx_lft)

    #right, or VVD
    dst_rgt_dsb_1 = dsdc2_rgt * self.g/tid_geg['omega']**2 * detadx_rgt *  3/(2*dx_rgt) 
    dst_rgt_dsb_2 = dsdc2_rgt * self.g/tid_geg['omega']**2 * detadx_rgt * -4/(2*dx_rgt) 
    dst_rgt_dsb_3 = dsdc2_rgt * self.g/tid_geg['omega']**2 * detadx_rgt *  1/(2*dx_rgt) 
    dst_rgt_dsn_1 = (dsdc3_rgt * - nph_rgt*eta_rgt + dsdc4_rgt * - nph_rgt * self.g/tid_geg['omega']**2 * (detadx2_rgt + detadx_rgt/bn_rgt)) 
    
    dst_rgt_x_dsb_1 = self.A_rgt[np.newaxis,:,np.newaxis] * self.Lsc*dsdc2_rgt * self.g/tid_geg['omega']**2 * (detadx2_rgt *  3/(2*dx_rgt) + detadx_rgt *  2/(dx_rgt**2) )
    dst_rgt_x_dsb_2 = self.A_rgt[np.newaxis,:,np.newaxis] * self.Lsc*dsdc2_rgt * self.g/tid_geg['omega']**2 * (detadx2_rgt * -4/(2*dx_rgt) + detadx_rgt * -5/(dx_rgt**2) )
    dst_rgt_x_dsb_3 = self.A_rgt[np.newaxis,:,np.newaxis] * self.Lsc*dsdc2_rgt * self.g/tid_geg['omega']**2 * (detadx2_rgt *  1/(2*dx_rgt) + detadx_rgt *  4/(dx_rgt**2) )
    dst_rgt_x_dsb_4 = self.A_rgt[np.newaxis,:,np.newaxis] * self.Lsc*dsdc2_rgt * self.g/tid_geg['omega']**2 *  detadx_rgt * -1/(dx_rgt**2)
    dst_rgt_x_dsn_1 = self.A_rgt[np.newaxis,:,np.newaxis] * self.Lsc*(dsdc3_rgt * - nph_rgt * detadx_rgt + dsdc4_rgt * -nph_rgt * self.g/tid_geg['omega']**2 * (detadx3_rgt + detadx2_rgt/bn_rgt) \
                  + (dsdc3_rgt * -nph_rgt *eta_rgt + dsdc4_rgt * -nph_rgt *  self.g/tid_geg['omega']**2 * (detadx2_rgt + detadx_rgt/bn_rgt) ) *  3/(2*dx_rgt))
    dst_rgt_x_dsn_2 = self.A_rgt[np.newaxis,:,np.newaxis] * self.Lsc*(dsdc3_rgt * -nph_rgt *eta_rgt + dsdc4_rgt * -nph_rgt *  self.g/tid_geg['omega']**2 * (detadx2_rgt + detadx_rgt/bn_rgt) ) * -4/(2*dx_rgt)
    dst_rgt_x_dsn_3 = self.A_rgt[np.newaxis,:,np.newaxis] * self.Lsc*(dsdc3_rgt * -nph_rgt *eta_rgt + dsdc4_rgt * -nph_rgt *  self.g/tid_geg['omega']**2 * (detadx2_rgt + detadx_rgt/bn_rgt) ) *  1/(2*dx_rgt)
            
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
  

def corsol_bnd(self, zout, tid_geg, indi):
    # =============================================================================
    # Solution vector for contribution to transport at boundaries due to boundary correction
    # =============================================================================

    #calculate salinity correction
    Bm_rgt = zout[((self.di3[1:-1]-2)*self.M+np.arange(self.M)[:,np.newaxis])] + 1j * zout[((self.di3[1:-1]-2)*self.M+np.arange(self.M,2*self.M)[:,np.newaxis])]
    Bm_lft = zout[( self.di3[1:-1]   *self.M+np.arange(self.M)[:,np.newaxis])] + 1j * zout[( self.di3[1:-1]   *self.M+np.arange(self.M,2*self.M)[:,np.newaxis])]
    
    st_cor_lft = np.sum(Bm_lft[:,:,np.newaxis] * np.cos(np.arange(self.M)[:,np.newaxis,np.newaxis]*np.pi*self.z_nd) ,0)*self.soc # I think this is the correct scaling
    st_cor_rgt = np.sum(Bm_rgt[:,:,np.newaxis] * np.cos(np.arange(self.M)[:,np.newaxis,np.newaxis]*np.pi*self.z_nd) ,0)*self.soc # I think this is the correct scaling
    
    #velocities
    ut_lft = tid_geg['ut'][indi['ih_lft']]
    ut_rgt = tid_geg['ut'][indi['ih_rgt']]
    
    
    #fluxes
    flux_lft_da = self.A_lft * np.mean(1/4 * np.real(ut_lft*np.conj(st_cor_lft) + np.conj(ut_lft)*st_cor_lft) , axis=-1) / self.soc #waarschijnlijk kan dit analytisch
    flux_rgt_da = self.A_rgt * np.mean(1/4 * np.real(ut_rgt*np.conj(st_cor_rgt) + np.conj(ut_rgt)*st_cor_rgt) , axis=-1) / self.soc #waarschijnlijk kan dit analytisch

    flux_lft_dv = self.A_lft * np.mean(1/4 * np.real(ut_lft*np.conj(st_cor_lft) + np.conj(ut_lft)*st_cor_lft)[np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd) , axis=-1) / self.soc #waarschijnlijk kan dit analytisch
    flux_rgt_dv = self.A_rgt * np.mean(1/4 * np.real(ut_rgt*np.conj(st_cor_rgt) + np.conj(ut_rgt)*st_cor_rgt)[np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd) , axis=-1) / self.soc #waarschijnlijk kan dit analytisch
       
    return flux_lft_da , flux_rgt_da, flux_lft_dv, flux_rgt_dv

def corjac_bnd(self, zout, tid_geg, indi):
    # =============================================================================
    # Jacobian for contribution to transport at boundaries due to boundary correction
    # =============================================================================
    #velocities
    ut_lft = tid_geg['ut'][indi['ih_lft']]
    ut_rgt = tid_geg['ut'][indi['ih_rgt']]
        
    #depth-averaged
    dT_lft_dRe = self.A_lft * np.mean(1/4 * 2*np.real(ut_lft)*np.cos(np.arange(self.M)[:,np.newaxis,np.newaxis]*np.pi*self.z_nd) , axis=-1) #waarschijnlijk kan dit analytisch
    dT_lft_dIm = self.A_lft * np.mean(1/4 * 2*np.imag(ut_lft)*np.cos(np.arange(self.M)[:,np.newaxis,np.newaxis]*np.pi*self.z_nd) , axis=-1) #waarschijnlijk kan dit analytisch
    
    dT_rgt_dRe = self.A_rgt * np.mean(1/4 * 2*np.real(ut_rgt)*np.cos(np.arange(self.M)[:,np.newaxis,np.newaxis]*np.pi*self.z_nd) , axis=-1) #waarschijnlijk kan dit analytisch
    dT_rgt_dIm = self.A_rgt * np.mean(1/4 * 2*np.imag(ut_rgt)*np.cos(np.arange(self.M)[:,np.newaxis,np.newaxis]*np.pi*self.z_nd) , axis=-1) #waarschijnlijk kan dit analytisch
    
    #depth-perturbed
    dTz_lft_dRe = self.A_lft * np.mean(1/4 * 2*np.real(ut_lft) * np.cos(np.arange(self.M)[:,np.newaxis,np.newaxis,np.newaxis]*np.pi*self.z_nd) * np.cos(np.arange(1,self.M)[:,np.newaxis,np.newaxis]*np.pi*self.z_nd), axis=-1) #waarschijnlijk kan dit analytisch
    dTz_lft_dIm = self.A_lft * np.mean(1/4 * 2*np.imag(ut_lft) * np.cos(np.arange(self.M)[:,np.newaxis,np.newaxis,np.newaxis]*np.pi*self.z_nd) * np.cos(np.arange(1,self.M)[:,np.newaxis,np.newaxis]*np.pi*self.z_nd), axis=-1) #waarschijnlijk kan dit analytisch
    
    dTz_rgt_dRe = self.A_rgt * np.mean(1/4 * 2*np.real(ut_rgt) * np.cos(np.arange(self.M)[:,np.newaxis,np.newaxis,np.newaxis]*np.pi*self.z_nd) * np.cos(np.arange(1,self.M)[:,np.newaxis,np.newaxis]*np.pi*self.z_nd), axis=-1) #waarschijnlijk kan dit analytisch
    dTz_rgt_dIm = self.A_rgt * np.mean(1/4 * 2*np.imag(ut_rgt) * np.cos(np.arange(self.M)[:,np.newaxis,np.newaxis,np.newaxis]*np.pi*self.z_nd) * np.cos(np.arange(1,self.M)[:,np.newaxis,np.newaxis]*np.pi*self.z_nd), axis=-1) #waarschijnlijk kan dit analytisch

    return np.concatenate([dT_lft_dRe , dT_lft_dIm ]), np.concatenate([dT_rgt_dRe , dT_rgt_dIm]) , np.concatenate([dTz_lft_dRe , dTz_lft_dIm ]), np.concatenate([dTz_rgt_dRe , dTz_rgt_dIm])




def tidsol_bnd(self, inp, tid_geg):
    # =============================================================================
    # Transport at boundaries, to add to solution vector 
    # =============================================================================
    #load inp
    st  = inp['st']  
    
    # =============================================================================
    # left sides of the domain
    # =============================================================================
    ut_lft = tid_geg['ut'][self.ii_all['ih_lft']] 
    st_lft = st[self.ii_all['ih_lft']] 

    #transport
    flux_lft = self.A_lft * np.mean(1/4 * np.real(ut_lft*np.conj(st_lft) + np.conj(ut_lft)*st_lft) , axis=1) / self.soc  
    #transport at vertical levels
    flux_lft_z = self.A_lft * np.mean(1/4 * np.real(ut_lft*np.conj(st_lft) + np.conj(ut_lft)*st_lft) * np.cos(self.nph[:,self.ii_all['ih_lft']]  * self.zlist[:,self.ii_all['ih_lft']]) , axis=2) / self.soc  
    
    # =============================================================================
    # right sides of the domain
    # =============================================================================
    ut_rgt = tid_geg['ut'][self.ii_all['ih_rgt']] 
    st_rgt = st[self.ii_all['ih_rgt']] 

    #transport
    flux_rgt = self.A_rgt * np.mean(1/4 * np.real(ut_rgt*np.conj(st_rgt) + np.conj(ut_rgt)*st_rgt) , axis=1) / self.soc  
    #transport at vertical levels
    flux_rgt_z = self.A_rgt * np.mean(1/4 * np.real(ut_rgt*np.conj(st_rgt) + np.conj(ut_rgt)*st_rgt) * np.cos(self.nph[:,self.ii_all['ih_rgt']]  * self.zlist[:,self.ii_all['ih_rgt']]) , axis=2) / self.soc  

    return flux_lft, flux_rgt , flux_lft_z, flux_rgt_z 

def tidjac_bnd(self, tid_geg):
    # =============================================================================
    # Transport at boundaries, to add to jacobian 
    # =============================================================================
    #coefficients
    #ih = np.array([di[1:-1]-1 ,di[1:-1]]).T.flatten()
    #ih2 = np.concatenate([[0],np.repeat(np.arange(1,len(nxn)-1),2),[len(nxn)-1]])

    dsdc2_lft = tid_geg['c2c'][:,self.ii_all['ih_lft']]
    dsdc3_lft = tid_geg['c3c'][:,self.ii_all['ih_lft']]
    dsdc4_lft = tid_geg['c4c'][:,self.ii_all['ih_lft']]
    
    dsdc2_rgt = tid_geg['c2c'][:,self.ii_all['ih_rgt']]
    dsdc3_rgt = tid_geg['c3c'][:,self.ii_all['ih_rgt']]
    dsdc4_rgt = tid_geg['c4c'][:,self.ii_all['ih_rgt']]

    ut_lft = tid_geg['ut'][self.ii_all['ih_lft']]
    ut_rgt = tid_geg['ut'][self.ii_all['ih_rgt']]

    # =============================================================================
    # jacobian, derivatives, 8 terms
    # =============================================================================        
    
    #depth-averaged
    #derivatives for x=-L
    dT0_dsb0 = 1/4 * self.A_lft * np.real( ut_lft * np.conj( dsdc2_lft * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_lft']] * -3/(2*self.dxn[np.newaxis,self.ii_all['ih2_lft'],np.newaxis]) ) + np.conj(ut_lft) * dsdc2_lft * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_lft']] \
                             * -3/(2*self.dxn[np.newaxis,self.ii_all['ih2_lft'],np.newaxis]) ).mean(2)
    dT0_dsb1 = 1/4 * self.A_lft * np.real( ut_lft * np.conj( dsdc2_lft * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_lft']] *  4/(2*self.dxn[np.newaxis,self.ii_all['ih2_lft'],np.newaxis]) ) + np.conj(ut_lft) * dsdc2_lft * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_lft']] \
                             *  4/(2*self.dxn[np.newaxis,self.ii_all['ih2_lft'],np.newaxis]) ).mean(2)
    dT0_dsb2 = 1/4 * self.A_lft * np.real( ut_lft * np.conj( dsdc2_lft * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_lft']] * -1/(2*self.dxn[np.newaxis,self.ii_all['ih2_lft'],np.newaxis]) ) + np.conj(ut_lft) * dsdc2_lft * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_lft']] \
                             * -1/(2*self.dxn[np.newaxis,self.ii_all['ih2_lft'],np.newaxis]) ).mean(2)
    dT0_dsn0 = 1/4 * self.A_lft * np.real( ut_lft * np.conj( dsdc3_lft * - self.nph[:,self.ii_all['ih_lft']]*tid_geg['eta'][:,self.ii_all['ih_lft']] + dsdc4_lft * - self.nph[:,self.ii_all['ih_lft']] * self.g/tid_geg['omega']**2 * (tid_geg['detadx2'][:,self.ii_all['ih_lft']] \
                             + tid_geg['detadx'][:,self.ii_all['ih_lft']]/self.bn[np.newaxis,self.ii_all['ih2_lft'],np.newaxis]) ) \
                            + np.conj(ut_lft) * ( dsdc3_lft * - self.nph[:,self.ii_all['ih_lft']] * tid_geg['eta'][:,self.ii_all['ih_lft']] \
                             + dsdc4_lft * - self.nph[:,self.ii_all['ih_lft']] * self.g/tid_geg['omega']**2 * (tid_geg['detadx2'][:,self.ii_all['ih_lft']] + tid_geg['detadx'][:,self.ii_all['ih_lft']]/self.bn[np.newaxis,self.ii_all['ih2_lft'],np.newaxis]) ) ).mean(2) 
      
    #derivatives for x=0
    dT_1_dsb_1 = 1/4 * self.A_rgt * np.real( ut_rgt * np.conj( dsdc2_rgt * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_rgt']] *  3/(2*self.dxn[np.newaxis,self.ii_all['ih2_rgt'],np.newaxis]) ) + np.conj(ut_rgt) * dsdc2_rgt * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_rgt']] \
                               *  3/(2*self.dxn[np.newaxis,self.ii_all['ih2_rgt'],np.newaxis]) ).mean(2)
    dT_1_dsb_2 = 1/4 * self.A_rgt * np.real( ut_rgt * np.conj( dsdc2_rgt * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_rgt']] * -4/(2*self.dxn[np.newaxis,self.ii_all['ih2_rgt'],np.newaxis]) ) + np.conj(ut_rgt) * dsdc2_rgt * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_rgt']] \
                               * -4/(2*self.dxn[np.newaxis,self.ii_all['ih2_rgt'],np.newaxis]) ).mean(2)
    dT_1_dsb_3 = 1/4 * self.A_rgt * np.real( ut_rgt * np.conj( dsdc2_rgt * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_rgt']] *  1/(2*self.dxn[np.newaxis,self.ii_all['ih2_rgt'],np.newaxis]) ) + np.conj(ut_rgt) * dsdc2_rgt * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_rgt']] \
                               *  1/(2*self.dxn[np.newaxis,self.ii_all['ih2_rgt'],np.newaxis]) ).mean(2)
    dT_1_dsn_1 = 1/4 * self.A_rgt * np.real( ut_rgt * np.conj( dsdc3_rgt * - self.nph[:,self.ii_all['ih_rgt']]*tid_geg['eta'][:,self.ii_all['ih_rgt']] + dsdc4_rgt * - self.nph[:,self.ii_all['ih_rgt']] * self.g/tid_geg['omega']**2 * (tid_geg['detadx2'][:,self.ii_all['ih_rgt']] \
                               + tid_geg['detadx'][:,self.ii_all['ih_rgt']]/self.bn[np.newaxis,self.ii_all['ih2_rgt'],np.newaxis]) ) + np.conj(ut_rgt) * ( dsdc3_rgt * - self.nph[:,self.ii_all['ih_rgt']] * tid_geg['eta'][:,self.ii_all['ih_rgt']] \
                               + dsdc4_rgt * - self.nph[:,self.ii_all['ih_rgt']] * self.g/tid_geg['omega']**2 * (tid_geg['detadx2'][:,self.ii_all['ih_rgt']] + tid_geg['detadx'][:,self.ii_all['ih_rgt']]/self.bn[np.newaxis,self.ii_all['ih2_rgt'],np.newaxis]) ) ).mean(2) 
                                                                                                                                                                                                                                                                                                        
    #z levels
    #derivatives for x=-L
    dTz0_dsb0 = 1/4 * self.A_lft * np.mean(np.real( ut_lft * np.conj( dsdc2_lft * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_lft']]* -3/(2*self.dxn[np.newaxis,self.ii_all['ih2_lft'],np.newaxis]) ) + np.conj(ut_lft) * dsdc2_lft * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_lft']] \
                                      * -3/(2*self.dxn[np.newaxis,self.ii_all['ih2_lft'],np.newaxis]) ) * np.cos(self.nph[:,self.ii_all['ih_lft']]*self.zlist[:,self.ii_all['ih_lft']]), axis=-1)
    dTz0_dsb1 = 1/4 * self.A_lft * np.mean(np.real( ut_lft * np.conj( dsdc2_lft * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_lft']]*  4/(2*self.dxn[np.newaxis,self.ii_all['ih2_lft'],np.newaxis]) ) + np.conj(ut_lft) * dsdc2_lft * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_lft']] \
                                      *  4/(2*self.dxn[np.newaxis,self.ii_all['ih2_lft'],np.newaxis]) ) * np.cos(self.nph[:,self.ii_all['ih_lft']]*self.zlist[:,self.ii_all['ih_lft']]), axis=-1)
    dTz0_dsb2 = 1/4 * self.A_lft * np.mean(np.real( ut_lft * np.conj( dsdc2_lft * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_lft']]* -1/(2*self.dxn[np.newaxis,self.ii_all['ih2_lft'],np.newaxis]) ) + np.conj(ut_lft) * dsdc2_lft * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_lft']]\
                                      * -1/(2*self.dxn[np.newaxis,self.ii_all['ih2_lft'],np.newaxis]) ) * np.cos(self.nph[:,self.ii_all['ih_lft']]*self.zlist[:,self.ii_all['ih_lft']]), axis=-1)
    dTz0_dsn0 = 1/4 * self.A_lft * np.mean(np.real( ut_lft * np.conj( dsdc3_lft * - self.nph[:,self.ii_all['ih_lft']] * tid_geg['eta'][:,self.ii_all['ih_lft']] + dsdc4_lft * - self.nph[:,self.ii_all['ih_lft']] * self.g/tid_geg['omega']**2 * (tid_geg['detadx2'][:,self.ii_all['ih_lft']] \
                                      + tid_geg['detadx'][:,self.ii_all['ih_lft']]/self.bn[np.newaxis,self.ii_all['ih2_lft'],np.newaxis]) ) + np.conj(ut_lft) * ( dsdc3_lft * - self.nph[:,self.ii_all['ih_lft']] * tid_geg['eta'][:,self.ii_all['ih_lft']] \
                                      + dsdc4_lft * - self.nph[:,self.ii_all['ih_lft']] * self.g/tid_geg['omega']**2 * (tid_geg['detadx2'][:,self.ii_all['ih_lft']] + tid_geg['detadx'][:,self.ii_all['ih_lft']]/self.bn[np.newaxis,self.ii_all['ih2_lft'],np.newaxis]) ) )[:,np.newaxis,:,:] * np.cos(self.nph[:,self.ii_all['ih_lft']]*self.zlist[:,self.ii_all['ih_lft']]) , axis = -1)
    
    #derivatives for x=0
    dTz_1_dsb_1 = 1/4 * self.A_rgt * np.mean(np.real( ut_rgt * np.conj( dsdc2_rgt * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_rgt']] *  3/(2*self.dxn[np.newaxis,self.ii_all['ih2_rgt'],np.newaxis]) ) \
                                                + np.conj(ut_rgt) * dsdc2_rgt * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_rgt']] *  3/(2*self.dxn[np.newaxis,self.ii_all['ih2_rgt'],np.newaxis]) ) * np.cos(self.nph[:,self.ii_all['ih_rgt']]*self.zlist[:,self.ii_all['ih_rgt']]) , axis=-1)
    dTz_1_dsb_2 = 1/4 * self.A_rgt * np.mean(np.real( ut_rgt * np.conj( dsdc2_rgt * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_rgt']] * -4/(2*self.dxn[np.newaxis,self.ii_all['ih2_rgt'],np.newaxis]) ) \
                                                + np.conj(ut_rgt) * dsdc2_rgt * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_rgt']] * -4/(2*self.dxn[np.newaxis,self.ii_all['ih2_rgt'],np.newaxis]) ) * np.cos(self.nph[:,self.ii_all['ih_rgt']]*self.zlist[:,self.ii_all['ih_rgt']]) , axis=-1)
    dTz_1_dsb_3 = 1/4 * self.A_rgt * np.mean(np.real( ut_rgt * np.conj( dsdc2_rgt * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_rgt']] *  1/(2*self.dxn[np.newaxis,self.ii_all['ih2_rgt'],np.newaxis]) ) \
                                                + np.conj(ut_rgt) * dsdc2_rgt * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,self.ii_all['ih_rgt']] *  1/(2*self.dxn[np.newaxis,self.ii_all['ih2_rgt'],np.newaxis]) ) * np.cos(self.nph[:,self.ii_all['ih_rgt']]*self.zlist[:,self.ii_all['ih_rgt']]) , axis=-1)
    dTz_1_dsn_1 = 1/4 * self.A_rgt * np.mean(np.real( ut_rgt * np.conj( dsdc3_rgt * - self.nph[:,self.ii_all['ih_rgt']]*tid_geg['eta'][:,self.ii_all['ih_rgt']] + dsdc4_rgt * - self.nph[:,self.ii_all['ih_rgt']] * self.g/tid_geg['omega']**2 * (tid_geg['detadx2'][:,self.ii_all['ih_rgt']] \
                                                + tid_geg['detadx'][:,self.ii_all['ih_rgt']]/self.bn[np.newaxis,self.ii_all['ih2_rgt'],np.newaxis]) ) + np.conj(ut_rgt) * ( dsdc3_rgt * - self.nph[:,self.ii_all['ih_rgt']]*tid_geg['eta'][:,self.ii_all['ih_rgt']] \
                                                + dsdc4_rgt * - self.nph[:,self.ii_all['ih_rgt']] * self.g/tid_geg['omega']**2 * (tid_geg['detadx2'][:,self.ii_all['ih_rgt']] \
                                                + tid_geg['detadx'][:,self.ii_all['ih_rgt']]/self.bn[np.newaxis,self.ii_all['ih2_rgt'],np.newaxis]) ) )[:,np.newaxis,:,:] * np.cos(self.nph[:,self.ii_all['ih_rgt']]*self.zlist[:,self.ii_all['ih_rgt']]) , axis=-1)
    
    return (dT0_dsb0[0],dT0_dsb1[0],dT0_dsb2[0],dT0_dsn0 ),( dT_1_dsb_1[0],dT_1_dsb_2[0],dT_1_dsb_3[0],dT_1_dsn_1) , (dTz0_dsb0,dTz0_dsb1,dTz0_dsb2,dTz0_dsn0 ),( dTz_1_dsb_1,dTz_1_dsb_2,dTz_1_dsb_3,dTz_1_dsn_1) 

def tidsol_riv(self, inp, tid_geg):
    # =============================================================================
    # Transport at river boundary, to add to solution vector 
    # =============================================================================
    #load inp
    st  = inp['st']  
    ut_riv = tid_geg['ut'][0] 
    st_riv = st[0] 
        
    #transport
    flux_riv = np.mean(1/4 * np.real(ut_riv*np.conj(st_riv) + np.conj(ut_riv)*st_riv)) / self.soc  

    return flux_riv

def tidjac_riv(self, tid_geg):
    # =============================================================================
    # Transport at river boundaries, to add to jacobian 
    # =============================================================================
    #prepare
    dsdc2 = tid_geg['c2c'][:,0,np.newaxis]
    dsdc3 = tid_geg['c3c'][:,0,np.newaxis]
    dsdc4 = tid_geg['c4c'][:,0,np.newaxis]
    ut_riv = tid_geg['ut'][0]

    # jacobian, derivatives,      
    #depth-averaged
    dT_dsb0 = 1/4 * np.real( ut_riv * np.conj( dsdc2 * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,0] * -3/(2*self.dxn[np.newaxis,0,np.newaxis]) ) + np.conj(ut_riv) * dsdc2 * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,0] * -3/(2*self.dxn[np.newaxis,0,np.newaxis]) ).mean(2)
    dT_dsb1 = 1/4 * np.real( ut_riv * np.conj( dsdc2 * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,0] *  4/(2*self.dxn[np.newaxis,0,np.newaxis]) ) + np.conj(ut_riv) * dsdc2 * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,0] *  4/(2*self.dxn[np.newaxis,0,np.newaxis]) ).mean(2)
    dT_dsb2 = 1/4 * np.real( ut_riv * np.conj( dsdc2 * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,0] * -1/(2*self.dxn[np.newaxis,0,np.newaxis]) ) + np.conj(ut_riv) * dsdc2 * self.g/tid_geg['omega']**2 * tid_geg['detadx'][:,0] * -1/(2*self.dxn[np.newaxis,0,np.newaxis]) ).mean(2)
    dT_dsn0 = 1/4 * np.real( ut_riv * np.conj( dsdc3 * - self.nph*tid_geg['eta'][:,0] + dsdc4 * - self.nph * self.g/tid_geg['omega']**2 * (tid_geg['detadx2'][:,0] + tid_geg['detadx'][:,0]/self.bn[np.newaxis,0,np.newaxis]) ) 
                                                           + np.conj(ut_riv) * ( dsdc3 * - self.nph*tid_geg['eta'][:,0] + dsdc4 * - self.nph * self.g/tid_geg['omega']**2 * (tid_geg['detadx2'][:,0] + tid_geg['detadx'][:,0]/self.bn[np.newaxis,0,np.newaxis]) ) ).mean(2) 
        
    return dT_dsb0 , dT_dsb1 , dT_dsb2 , dT_dsn0[:,0]


def solu_bnd_tidal(self, ans, inp, tid_geg, indi, version):

    so = np.zeros(self.di3[-1]*self.M)
    
    #no equations for the river and sea boundaries correction
    so[indi['bnl_bnd']] = ans[indi['bnl_bnd']]
    
    if version in ['A','B','C']:
    #if version in ['A','B','C','D']:
        #(simple) equations for the boundary layer corrections
        so[indi['bnl_lft']] = ans[indi['bnl_lft']]
        so[indi['bnl_rgt']] = ans[indi['bnl_rgt']]
    
    
    # =============================================================================
    # inner boundary - flux - tidal contribution
    # =============================================================================
    if version in ['B','C','D']:
        
        #contribution to river boundary 
        so[2*self.M] += tidsol_riv(self, inp, tid_geg)
        #boundaries 
        ftl_da, ftr_da, ftl_dv, ftr_dv = tidsol_bnd(self, inp, tid_geg)
        
        #for depth-averaged flux
        so[indi['i_s_1']] +=  ftr_da - ftl_da  
    
    

    if version in ['D']:
        
        #ftl_da, ftr_da, ftl_dv, ftr_dv = tidsol_bnd(self, inp, tid_geg)

        #for depth-perturbed flux
        so[indi['bnd2_rgt']] += (ftr_dv.T - ftl_dv.T).flatten()
        # =============================================================================
        # inner boundary - flux - boundary layer correction
        # =============================================================================
        
        #boundary layer correction itself
        sol_bc = corsol(self, ans, inp, tid_geg, indi)
    
        so[indi['bnl_rgt'][:,:self.M]] = sol_bc[0]
        so[indi['bnl_rgt'][:,self.M:]] = sol_bc[1] 
        so[indi['bnl_lft'][:,:self.M]] = sol_bc[2]
        so[indi['bnl_lft'][:,self.M:]] = sol_bc[3]  
        
        #effect on fluxes
        sol_bcT = corsol_bnd(self, ans, tid_geg, indi)

        #for depth-averaged flux
        so[indi['i_s_1']] += -sol_bcT[0] + sol_bcT[1] 
        
        #new part here, depth-perturbed flux
        so[indi['i_s_1'][:,np.newaxis] + np.arange(1, self.M)] += - sol_bcT[2].T  +  sol_bcT[3].T
        
    return so


def jaco_bnd_tidal(self, ans, tid_geg, indi, version):
    
    jac = np.zeros((self.di3[-1]*self.M,self.di3[-1]*self.M))
    # =============================================================================
    # boundary layer correctoion
    # =============================================================================
    
    #no equations for points at boundaries
    jac[indi['bnl_bnd'],indi['bnl_bnd']] += 1
    
    #if version in ['A','B','C','D']:
    if version in ['A','B','C']:
        #(simple) equations for the boundary layer corrections
        jac[indi['bnl_lft'],indi['bnl_lft']] += 1
        jac[indi['bnl_rgt'],indi['bnl_rgt']] += 1
    
    if version in ['B','C','D']:
        
        #river boundary
        tj_riv = tidjac_riv(self, tid_geg)
        jac[self.M*2, self.M*2] += tj_riv[0]
        jac[self.M*2, self.M*3] += tj_riv[1]
        jac[self.M*2, self.M*4] += tj_riv[2]
        jac[self.M*2, self.M*2 + np.arange(1,self.M)] += tj_riv[3]
        # =============================================================================
        # inner boundary - flux - tidal contribution
        # =============================================================================      
        jtl_da, jtr_da, jtl_dv, jtr_dv = tidjac_bnd(self, tid_geg)
        
        #print(jtr_da[3] , jtl_da[3])
        

        #depth-averaged 
        jac[indi['i_s_1'] , indi['i_s_1']] += jtr_da[0]
        jac[indi['i_s_1'] , indi['i_s_2']] += jtr_da[1]
        jac[indi['i_s_1'] , indi['i_s_3']] += jtr_da[2]
        
        jac[indi['i_s_1'] , indi['i_s0']] += - jtl_da[0]
        jac[indi['i_s_1'] , indi['i_s1']] += - jtl_da[1]
        jac[indi['i_s_1'] , indi['i_s2']] += - jtl_da[2]
        
        for k in range(1,self.M):
            jac[indi['i_s_1'] , indi['i_s_1'] + k] +=  jtr_da[3][k-1]  #TODO: fix error here 
            jac[indi['i_s_1'] , indi['i_s0']  + k] +=- jtl_da[3][k-1] #TODO: fix error here
    
    
    
    
    if version in ['D']:
        
        #jtl_da, jtr_da, jtl_dv, jtr_dv = tidjac_bnd(self, tid_geg)

        #depth-perturbed
        jac[indi['bnd2_rgt'],np.repeat(indi['i_s_3'],self.N)] +=  jtr_dv[2].T.flatten() #sb_3
        jac[indi['bnd2_rgt'],np.repeat(indi['i_s_2'],self.N)] +=  jtr_dv[1].T.flatten() #sb_2
        jac[indi['bnd2_rgt'],np.repeat(indi['i_s_1'],self.N)] +=  jtr_dv[0].T.flatten()  #sb_1
        
        jac[indi['bnd2_rgt'],np.repeat(indi['i_s0'],self.N)] += - jtl_dv[0].T.flatten()#sb0
        jac[indi['bnd2_rgt'],np.repeat(indi['i_s1'],self.N)] += - jtl_dv[1].T.flatten()#sb1
        jac[indi['bnd2_rgt'],np.repeat(indi['i_s2'],self.N)] += - jtl_dv[2].T.flatten()#sb2
            
        for k in range(1,self.M):
            jac[indi['bnd2_rgt'] , np.repeat(indi['i_s_1'],self.N) + k] += jtr_dv[3][k-1].T.flatten() #sn_1 in sum
            jac[indi['bnd2_rgt'] , np.repeat(indi['i_s0'] ,self.N) + k] +=-jtl_dv[3][k-1].T.flatten() #sn0 in sum
        
        # =============================================================================
        # boundary layer correction itself
        # =============================================================================
        jac_bc = corjac(self, ans, tid_geg, indi)   
        # first condition: lft = rgt
        #real
        #correction
        jac[np.repeat(indi['bnl_rgt'][:,:self.M],self.M) , np.tile(indi['bnl_rgt'][:,:self.M],self.M).flatten()] = np.tile(np.cos(np.arange(self.M)[:,np.newaxis]*np.pi*self.z_nd[self.z_inds]).T.flatten(),self.ndom-1)
        jac[np.repeat(indi['bnl_rgt'][:,:self.M],self.M) , np.tile(indi['bnl_lft'][:,:self.M],self.M).flatten()] = np.tile(-np.cos(np.arange(self.M)[:,np.newaxis]*np.pi*self.z_nd[self.z_inds]).T.flatten(),self.ndom-1)
        #then other parts - lft
        jac[np.repeat(indi['bnl_rgt'][:,:self.M],self.M) ,np.repeat(indi['i_s0'],self.M**2)] += np.repeat(-jac_bc[0][0][0,:,self.z_inds].T,self.M)
        jac[np.repeat(indi['bnl_rgt'][:,:self.M],self.M) ,np.repeat(indi['i_s1'],self.M**2)] += np.repeat(-jac_bc[0][1][0,:,self.z_inds].T,self.M)
        jac[np.repeat(indi['bnl_rgt'][:,:self.M],self.M) ,np.repeat(indi['i_s2'],self.M**2)] += np.repeat(-jac_bc[0][2][0,:,self.z_inds].T,self.M)
        #jac[np.repeat(indi['bnl_rgt'][:,:self.M],N) ,(np.repeat(indi['i_s0'],self.M)[:,np.newaxis]+np.arange(1,self.M)).flatten()] += -jac_bc[0][3][:,:,self.z_inds].flatten()
        
        #then other parts - rgt
        jac[np.repeat(indi['bnl_rgt'][:,:self.M],self.M) ,np.repeat(indi['i_s_1'],self.M**2)] += np.repeat(jac_bc[2][0][0,:,self.z_inds].T,self.M)
        jac[np.repeat(indi['bnl_rgt'][:,:self.M],self.M) ,np.repeat(indi['i_s_2'],self.M**2)] += np.repeat(jac_bc[2][1][0,:,self.z_inds].T,self.M)
        jac[np.repeat(indi['bnl_rgt'][:,:self.M],self.M) ,np.repeat(indi['i_s_3'],self.M**2)] += np.repeat(jac_bc[2][2][0,:,self.z_inds].T,self.M)
        
        #imaginary
        #correction
        jac[self.M+np.repeat(indi['bnl_rgt'][:,:self.M],self.M) , np.tile(indi['bnl_rgt'][:,self.M:],self.M).flatten()] = np.tile(np.cos(np.arange(self.M)[:,np.newaxis]*np.pi*self.z_nd[self.z_inds]).T.flatten(),self.ndom-1)
        jac[self.M+np.repeat(indi['bnl_rgt'][:,:self.M],self.M) , np.tile(indi['bnl_lft'][:,self.M:],self.M).flatten()] = np.tile(-np.cos(np.arange(self.M)[:,np.newaxis]*np.pi*self.z_nd[self.z_inds]).T.flatten(),self.ndom-1)
        
        #then other parts - lft
        jac[self.M+np.repeat(indi['bnl_rgt'][:,:self.M],self.M) ,np.repeat(indi['i_s0'],self.M**2)] += np.repeat(-jac_bc[1][0][0,:,self.z_inds].T,self.M)
        jac[self.M+np.repeat(indi['bnl_rgt'][:,:self.M],self.M) ,np.repeat(indi['i_s1'],self.M**2)] += np.repeat(-jac_bc[1][1][0,:,self.z_inds].T,self.M)
        jac[self.M+np.repeat(indi['bnl_rgt'][:,:self.M],self.M) ,np.repeat(indi['i_s2'],self.M**2)] += np.repeat(-jac_bc[1][2][0,:,self.z_inds].T,self.M)
    
    
        #then other parts - rgt
        jac[self.M+np.repeat(indi['bnl_rgt'][:,:self.M],self.M) ,np.repeat(indi['i_s_1'],self.M**2)] += np.repeat(jac_bc[3][0][0,:,self.z_inds].T,self.M)
        jac[self.M+np.repeat(indi['bnl_rgt'][:,:self.M],self.M) ,np.repeat(indi['i_s_2'],self.M**2)] += np.repeat(jac_bc[3][1][0,:,self.z_inds].T,self.M)
        jac[self.M+np.repeat(indi['bnl_rgt'][:,:self.M],self.M) ,np.repeat(indi['i_s_3'],self.M**2)] += np.repeat(jac_bc[3][2][0,:,self.z_inds].T,self.M)
        
        # second condition: flux is constant
        #real
        #correction
        jac[np.repeat(indi['bnl_lft'][:,:self.M],self.M) , np.tile(indi['bnl_lft'][:,:self.M],self.M).flatten()] = (self.A_lft[:,np.newaxis,np.newaxis] *-np.cos(np.arange(self.M)[:,np.newaxis]*np.pi*self.z_nd[self.z_inds]).T).flatten() * tid_geg['eps']**0.5
        jac[np.repeat(indi['bnl_lft'][:,:self.M],self.M) , np.tile(indi['bnl_rgt'][:,:self.M],self.M).flatten()] = (self.A_rgt[:,np.newaxis,np.newaxis] *-np.cos(np.arange(self.M)[:,np.newaxis]*np.pi*self.z_nd[self.z_inds]).T).flatten() * tid_geg['eps']**0.5

        #then other parts - lft
        jac[np.repeat(indi['bnl_lft'][:,:self.M],self.M) ,np.repeat(indi['i_s0'],self.M**2)] += np.repeat(-jac_bc[0][4][0,:,self.z_inds].T,self.M) * tid_geg['eps']
        jac[np.repeat(indi['bnl_lft'][:,:self.M],self.M) ,np.repeat(indi['i_s1'],self.M**2)] += np.repeat(-jac_bc[0][5][0,:,self.z_inds].T,self.M) * tid_geg['eps']
        jac[np.repeat(indi['bnl_lft'][:,:self.M],self.M) ,np.repeat(indi['i_s2'],self.M**2)] += np.repeat(-jac_bc[0][6][0,:,self.z_inds].T,self.M) * tid_geg['eps']
        jac[np.repeat(indi['bnl_lft'][:,:self.M],self.M) ,np.repeat(indi['i_s3'],self.M**2)] += np.repeat(-jac_bc[0][7][0,:,self.z_inds].T,self.M) * tid_geg['eps']
    
        #then other parts - rgt
        jac[np.repeat(indi['bnl_lft'][:,:self.M],self.M) ,np.repeat(indi['i_s_1'],self.M**2)] += np.repeat(jac_bc[2][4][0,:,self.z_inds].T,self.M) * tid_geg['eps']
        jac[np.repeat(indi['bnl_lft'][:,:self.M],self.M) ,np.repeat(indi['i_s_2'],self.M**2)] += np.repeat(jac_bc[2][5][0,:,self.z_inds].T,self.M) * tid_geg['eps']
        jac[np.repeat(indi['bnl_lft'][:,:self.M],self.M) ,np.repeat(indi['i_s_3'],self.M**2)] += np.repeat(jac_bc[2][6][0,:,self.z_inds].T,self.M) * tid_geg['eps']
        jac[np.repeat(indi['bnl_lft'][:,:self.M],self.M) ,np.repeat(indi['i_s_4'],self.M**2)] += np.repeat(jac_bc[2][7][0,:,self.z_inds].T,self.M) * tid_geg['eps']
        
        
        #imaginary
        #correction
        jac[self.M+np.repeat(indi['bnl_lft'][:,:self.M],self.M) , np.tile(indi['bnl_lft'][:,self.M:],self.M).flatten()] += (self.A_lft[:,np.newaxis,np.newaxis] *-np.cos(np.arange(self.M)[:,np.newaxis]*np.pi*self.z_nd[self.z_inds]).T).flatten() * tid_geg['eps']**0.5
        jac[self.M+np.repeat(indi['bnl_lft'][:,:self.M],self.M) , np.tile(indi['bnl_rgt'][:,self.M:],self.M).flatten()] += (self.A_rgt[:,np.newaxis,np.newaxis] *-np.cos(np.arange(self.M)[:,np.newaxis]*np.pi*self.z_nd[self.z_inds]).T).flatten() * tid_geg['eps']**0.5

        #then other parts - lft
        jac[self.M+np.repeat(indi['bnl_lft'][:,:self.M],self.M) ,np.repeat(indi['i_s0'],self.M**2)] += np.repeat(-jac_bc[1][4][0,:,self.z_inds].T,self.M) * tid_geg['eps']
        jac[self.M+np.repeat(indi['bnl_lft'][:,:self.M],self.M) ,np.repeat(indi['i_s1'],self.M**2)] += np.repeat(-jac_bc[1][5][0,:,self.z_inds].T,self.M) * tid_geg['eps']
        jac[self.M+np.repeat(indi['bnl_lft'][:,:self.M],self.M) ,np.repeat(indi['i_s2'],self.M**2)] += np.repeat(-jac_bc[1][6][0,:,self.z_inds].T,self.M) * tid_geg['eps']
        jac[self.M+np.repeat(indi['bnl_lft'][:,:self.M],self.M) ,np.repeat(indi['i_s3'],self.M**2)] += np.repeat(-jac_bc[1][7][0,:,self.z_inds].T,self.M) * tid_geg['eps']
    
        #then other parts - rgt
        jac[self.M+np.repeat(indi['bnl_lft'][:,:self.M],self.M) ,np.repeat(indi['i_s_1'],self.M**2)] += np.repeat(jac_bc[3][4][0,:,self.z_inds].T,self.M) * tid_geg['eps']
        jac[self.M+np.repeat(indi['bnl_lft'][:,:self.M],self.M) ,np.repeat(indi['i_s_2'],self.M**2)] += np.repeat(jac_bc[3][5][0,:,self.z_inds].T,self.M) * tid_geg['eps']
        jac[self.M+np.repeat(indi['bnl_lft'][:,:self.M],self.M) ,np.repeat(indi['i_s_3'],self.M**2)] += np.repeat(jac_bc[3][6][0,:,self.z_inds].T,self.M) * tid_geg['eps']
        jac[self.M+np.repeat(indi['bnl_lft'][:,:self.M],self.M) ,np.repeat(indi['i_s_4'],self.M**2)] += np.repeat(jac_bc[3][7][0,:,self.z_inds].T,self.M) * tid_geg['eps']
        
    
        #TODO: loops should be replaced with indices 
        for x9 in range(self.ndom-1):
            for m in range(self.M):
                # =============================================================================
                # first condition: lft = rgt
                # =============================================================================
    
                #real        
                #then other parts - lft
                jac[indi['bnl_rgt'][x9,m], indi['i_s0'][x9]+np.arange(1,self.M)] += -jac_bc[0][3][:,x9,self.z_inds[m]]
       
                #then other parts - rgt
                jac[indi['bnl_rgt'][x9,m], indi['i_s_1'][x9] + np.arange(1,self.M)] += jac_bc[2][3][:,x9,self.z_inds[m]]
                
                #imaginary
                #then other parts - lft
                jac[indi['bnl_rgt'][x9,self.M+m], indi['i_s0'][x9]+np.arange(1,self.M)] += -jac_bc[1][3][:,x9,self.z_inds[m]]
    
                #then other parts - rgt
                jac[indi['bnl_rgt'][x9,self.M+m], indi['i_s_1'][x9] + np.arange(1,self.M)] += jac_bc[3][3][:,x9,self.z_inds[m]]
                # =============================================================================
                # second condition: flux is constant
                # =============================================================================
                #real
                
                #then other parts - lft
                jac[indi['bnl_lft'][x9,m], indi['i_s0'][x9]+np.arange(1,self.M)] += -jac_bc[0][8][:,x9,self.z_inds[m]] * tid_geg['eps']
                jac[indi['bnl_lft'][x9,m], indi['i_s1'][x9]+np.arange(1,self.M)] += -jac_bc[0][9][:,x9,self.z_inds[m]] * tid_geg['eps']
                jac[indi['bnl_lft'][x9,m], indi['i_s2'][x9]+np.arange(1,self.M)] += -jac_bc[0][10][:,x9,self.z_inds[m]] * tid_geg['eps']
                
                #then other parts - rgt
                jac[indi['bnl_lft'][x9,m], indi['i_s_1'][x9]+np.arange(1,self.M)] += jac_bc[2][8][:,x9,self.z_inds[m]] * tid_geg['eps']
                jac[indi['bnl_lft'][x9,m], indi['i_s_2'][x9]+np.arange(1,self.M)] += jac_bc[2][9][:,x9,self.z_inds[m]] * tid_geg['eps']
                jac[indi['bnl_lft'][x9,m], indi['i_s_3'][x9]+np.arange(1,self.M)] += jac_bc[2][10][:,x9,self.z_inds[m]] * tid_geg['eps']


                #imaginary
                
                #then other parts - lft
                jac[indi['bnl_lft'][x9,self.M+m], indi['i_s0'][x9]+np.arange(1,self.M)] += -jac_bc[1][8][:,x9,self.z_inds[m]] * tid_geg['eps']
                jac[indi['bnl_lft'][x9,self.M+m], indi['i_s1'][x9]+np.arange(1,self.M)] += -jac_bc[1][9][:,x9,self.z_inds[m]] * tid_geg['eps']
                jac[indi['bnl_lft'][x9,self.M+m], indi['i_s2'][x9]+np.arange(1,self.M)] += -jac_bc[1][10][:,x9,self.z_inds[m]] * tid_geg['eps']
                
                #then other parts - rgt
                jac[indi['bnl_lft'][x9,self.M+m], indi['i_s_1'][x9]+np.arange(1,self.M)] += jac_bc[3][8][:,x9,self.z_inds[m]] * tid_geg['eps']
                jac[indi['bnl_lft'][x9,self.M+m], indi['i_s_2'][x9]+np.arange(1,self.M)] += jac_bc[3][9][:,x9,self.z_inds[m]] * tid_geg['eps']
                jac[indi['bnl_lft'][x9,self.M+m], indi['i_s_3'][x9]+np.arange(1,self.M)] += jac_bc[3][10][:,x9,self.z_inds[m]] * tid_geg['eps']
       
        
       
        #effect on fluxes
        #for depth-averaged flux
        Tjc = corjac_bnd(self, ans, tid_geg, indi) 
        #boundary layer correction
        jac[np.repeat(indi['i_s_1'],2*self.M).reshape((self.ndom-1,2*self.M)).T , indi['bnl_rgt'].T] += Tjc[1]
        jac[np.repeat(indi['i_s_1'],2*self.M).reshape((self.ndom-1,2*self.M)).T , indi['bnl_lft'].T] += -Tjc[0]  
        
        #new part here - vertical variation 
        for x9 in range(self.ndom-1):
            for n in range(1,self.M):
                for c in range(2*self.M):
                    jac[indi['i_s_1'][x9] + n , indi['i_s_1'][x9] + c + 3*self.M] += - Tjc[2][c,n-1,x9]
                    jac[indi['i_s_1'][x9] + n , indi['i_s0'][x9] + c - 4*self.M] +=  Tjc[3][c,n-1,x9]
        
    return jac


def calc_cor(self, zout, indi, index): #for plotting purposes afterwards 
    
    #calculate salinity correction
    Bm_rgt = zout[((self.di3[1:-1]-2)*self.M+np.arange(self.M)[:,np.newaxis])] + 1j * zout[((self.di3[1:-1]-2)*self.M+np.arange(self.M,2*self.M)[:,np.newaxis])]
    Bm_lft = zout[( self.di3[1:-1]   *self.M+np.arange(self.M)[:,np.newaxis])] + 1j * zout[( self.di3[1:-1]   *self.M+np.arange(self.M,2*self.M)[:,np.newaxis])]
           
    tid_set = self.tid_sets[self.tid_comp[index]]
    tid_geg = self.tid_gegs[self.tid_comp[index]]

    #calculate x 
    x_temp = [np.arange(self.nxn[i])*self.dxn[i] for i in range(self.ndom)]
    x_lft = [x_temp[i][np.where(x_temp[i]<-np.sqrt(tid_geg['epsL'])*np.log(self.tol))[0]] for i in range(self.ndom)][1:]
    x_rgt = [x_temp[i][np.where(x_temp[i]>x_temp[i][-1]+np.sqrt(tid_geg['epsL'])*np.log(self.tol))[0]]-self.Ln[i] for i in range(self.ndom)][:-1]
           
    stc_lft   = [np.exp(-x_lft[i][:,np.newaxis]/np.sqrt(tid_geg['epsL'])) * np.sum(Bm_lft[:,i] * np.cos(np.arange(self.M)*np.pi*self.z_nd[:,np.newaxis]) ,1) * self.soc for i in range(self.ndom-1)]
    stc_rgt   = [np.exp(x_rgt[i][:,np.newaxis]/np.sqrt(tid_geg['epsL'])) * np.sum(Bm_rgt[:,i] * np.cos(np.arange(self.M)*np.pi*self.z_nd[:,np.newaxis]) ,1) * self.soc for i in range(self.ndom-1)]
            
    #salinity for plotting
    #t= np.linspace(0,tid_set['tid_per'],self.nt)
    stc = np.zeros((self.di[-1],self.nz) , dtype = complex) 
    for i in range(self.ndom-1):
        stc[self.di[1+i]+np.arange(len(x_lft[i]))]    = stc_lft[i]
        stc[self.di[1+i]+np.arange(-len(x_rgt[i]),0)] = stc_rgt[i]
    '''
    #terms for fluxes 
    flux = np.zeros(self.di[-1]) 
    for i in range(self.ndom-1):
        flux[self.di[1+i]+np.arange(len(x_lft[i]))]    = -1/4 * np.real(self.ut[self.di[1+i]+np.arange(len(x_lft[i]))] *np.conj(stc_lft[i]) + np.conj(self.ut[self.di[1+i]+np.arange(len(x_lft[i]))]) *stc_lft[i]).mean(1) 
        flux[self.di[1+i]+np.arange(-len(x_rgt[i]),0)] = -1/4 * np.real(self.ut[self.di[1+i]+np.arange(-len(x_rgt[i]),0)]*np.conj(stc_rgt[i]) + np.conj(self.ut[self.di[1+i]+np.arange(-len(x_rgt[i]),0)])*stc_rgt[i]).mean(1) 
    '''
    
    #print(np.max(np.abs(stc)))
    
    return stc
