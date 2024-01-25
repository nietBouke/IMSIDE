# =============================================================================
# tidal module
# the aim here is also to make it cleaner and clearer than the previous functions. 
# =============================================================================
import numpy as np
import time as time
from physics.tide_funcs5 import sti_coefs, sti, utisti

def tidal_module(self, tid_set):
    # =============================================================================
    # this function calculates the quantities which are required for futher calculations
    # moslty solves for the tidal water level and associated currents
    # =============================================================================

    # =============================================================================
    # solve for water level in different segments (except for the sea domain)
    # matrix equation, as in fork
    # use Jinyang notation here - bit weird but ok
    # =============================================================================
    #parameters for equations
    omega = 2*np.pi/tid_set['tid_per']
    deA = (1+1j)*self.H/np.sqrt(2*tid_set['Av_ti']/omega)
    if self.choice_bottomslip_ti  == 'sf' : B = (np.cosh(deA) + tid_set['Av_ti']/(tid_set['sf_ti']*self.H) * deA * np.sinh(deA))**-1
    elif self.choice_bottomslip_ti== 'rr' : B = (np.cosh(deA) + (tid_set['rr_ti']+np.zeros(self.di[-1])) * deA * np.sinh(deA))**-1
    else: print('ERROR: invalid choice of tidal bottom slip')
    ka = np.sqrt(0.25*self.bex**-2 + omega**2/(self.g*self.H) * (B/deA*np.sinh(deA)-1)**-1)
    

    sol = np.zeros(2*len(self.nxn)+1,dtype=complex)
    matr = np.zeros((2*len(self.nxn)+1,2*len(self.nxn)+1),dtype=complex)
    
    #river boundary
    matr[0,0] = -1/(2*self.bn[0]) + ka[0]
    matr[0,1] = -1/(2*self.bn[0]) - ka[0]
    sol[0] = 0
        
    fQ = self.b * self.H * (B/deA * np.sinh(deA) - 1)
    
    for dom in range(len(self.nxn)-1): 
        #water level equal
        matr[dom*2+1, dom*2+0] = np.exp(-self.Ln[dom]/(2*self.bn[dom])) * np.exp(self.Ln[dom]*ka[self.di[dom+1]-1])
        matr[dom*2+1, dom*2+1] = np.exp(-self.Ln[dom]/(2*self.bn[dom])) * np.exp(-self.Ln[dom]*ka[self.di[dom+1]-1])
        matr[dom*2+1, dom*2+2] = -1
        matr[dom*2+1, dom*2+3] = -1
        #discharge equal, i.e. water level gradient
        matr[dom*2+2, dom*2+0] = (np.exp(-self.Ln[dom]/(2*self.bn[dom])) * np.exp( self.Ln[dom]*ka[self.di[dom+1]-1]) * (-1/(2*self.bn[dom]) + ka[self.di[dom+1]-1])) * fQ[self.di[dom+1]-1]
        matr[dom*2+2, dom*2+1] = (np.exp(-self.Ln[dom]/(2*self.bn[dom])) * np.exp(-self.Ln[dom]*ka[self.di[dom+1]-1]) * (-1/(2*self.bn[dom]) - ka[self.di[dom+1]-1])) * fQ[self.di[dom+1]-1]
        matr[dom*2+2, dom*2+2] = -(-1/(2*self.bn[dom+1]) + ka[self.di[dom+1]]) * fQ[self.di[dom+1]]
        matr[dom*2+2, dom*2+3] = -(-1/(2*self.bn[dom+1]) - ka[self.di[dom+1]]) * fQ[self.di[dom+1]]
    
        #solution vector is zero everywhere, no need to specify that
        #print(self.di[dom+1]-1, self.di[dom+1])
    
    #sea boundary
    
    #first condition: at the sea boundary, the level is equal to a to be determined level
    matr[-2,-3] = np.exp(-self.Ln[-1]/(2*self.bn[-1])) * np.exp( ka[-1]*self.Ln[-1])
    matr[-2,-2] = np.exp(-self.Ln[-1]/(2*self.bn[-1])) * np.exp(-ka[-1]*self.Ln[-1])
    matr[-2,-1] = -1
    
    #second condition: the water level difference in the sea domain equals the difference between the prescribed sea level and the to be determined level
    matr[-1,-3] = np.exp(-self.Ln[-1]/(2*self.bn[-1])) * np.exp( ka[-1]*self.Ln[-1]) - 1
    matr[-1,-2] = np.exp(-self.Ln[-1]/(2*self.bn[-1])) * np.exp(-ka[-1]*self.Ln[-1]) - 1
    matr[-1,-1] = -1 
    sol[-1] = tid_set['a_tide'] * np.exp(-1j*tid_set['p_tide']/180*np.pi)
    
    #solve this set of equations
    oplossing = np.linalg.solve(matr,sol)
    coef_eta = oplossing[:-1].reshape((int(len(self.nxn)),2))
       
    #print(oplossing)

    # =============================================================================
    # calculate water level and derivatives
    # =============================================================================    
    eta     = np.zeros(self.di[-1],dtype=complex)
    etar    = np.zeros(self.di[-1],dtype=complex)
    detadx  = np.zeros(self.di[-1],dtype=complex)
    detadx2 = np.zeros(self.di[-1],dtype=complex)
    detadx3 = np.zeros(self.di[-1],dtype=complex)
    for dom in range(self.ndom):
        x_here = np.linspace(-self.Ln[dom],0,self.nxn[dom])
        eta[self.di[dom]:self.di[dom+1]]  = np.exp(-(x_here+self.Ln[dom])/(2*self.bn[dom])) * (coef_eta[dom,0]*np.exp((x_here+self.Ln[dom])*ka[self.di[dom]:self.di[dom+1]]) + coef_eta[dom,1]*np.exp(-(x_here+self.Ln[dom])*ka[self.di[dom]:self.di[dom+1]]) )
        etar[self.di[dom]:self.di[dom+1]] = np.exp(-(x_here+self.Ln[dom])/(2*self.bn[dom])) * (coef_eta[dom,0]*np.exp((x_here+self.Ln[dom])*ka[self.di[dom]:self.di[dom+1]]) - coef_eta[dom,1]*np.exp(-(x_here+self.Ln[dom])*ka[self.di[dom]:self.di[dom+1]]) )
        
        detadx[self.di[dom]:self.di[dom+1]]  = ka[self.di[dom]:self.di[dom+1]] * etar[self.di[dom]:self.di[dom+1]] - eta[self.di[dom]:self.di[dom+1]]/(2*self.bn[dom])
        detadx2[self.di[dom]:self.di[dom+1]] = ka[self.di[dom]:self.di[dom+1]]**2*eta[self.di[dom]:self.di[dom+1]] + eta[self.di[dom]:self.di[dom+1]]/(4*self.bn[dom]**2) - ka[self.di[dom]:self.di[dom+1]]/self.bn[dom] * etar[self.di[dom]:self.di[dom+1]]
        detadx3[self.di[dom]:self.di[dom+1]] = ka[self.di[dom]:self.di[dom+1]]**3 * etar[self.di[dom]:self.di[dom+1]] - eta[self.di[dom]:self.di[dom+1]]/(8*self.bn[dom]**3) - 3*eta[self.di[dom]:self.di[dom+1]]*ka[self.di[dom]:self.di[dom+1]]**2/(2*self.bn[dom]) + 3*ka[self.di[dom]:self.di[dom+1]]/(4*self.bn[dom]**2) * etar[self.di[dom]:self.di[dom+1]]
    

    #correct shape 
    eta     = eta[np.newaxis,:,np.newaxis]
    detadx  = detadx[np.newaxis,:,np.newaxis]
    detadx2 = detadx2[np.newaxis,:,np.newaxis]
    detadx3 = detadx3[np.newaxis,:,np.newaxis]
    
    #velocities
    ut    = (self.g/(1j*omega) * detadx  * (B[:,np.newaxis]*np.cosh(deA[:,np.newaxis]*self.z_nd) - 1))[0]
    dutdx = (self.g/(1j*omega) * detadx2 * (B[:,np.newaxis]*np.cosh(deA[:,np.newaxis]*self.z_nd) - 1))[0]
    wt    = 1j*omega*eta - self.g/(1j*omega) * (detadx2 + detadx/self.bex[np.newaxis,:self.di[-1],np.newaxis]) * (B[:,np.newaxis]*self.H[:,np.newaxis]/deA[:,np.newaxis]*np.sinh(deA[:,np.newaxis]*self.z_nd) - self.zlist)

    utb = (self.g/(1j*omega) * detadx[0,:,0] * (B/deA*np.sinh(deA)-1))
    utp = (self.g/(1j*omega) * detadx[0] * B[:,np.newaxis] * (np.cosh(deA[:,np.newaxis]*self.z_nd) - np.sinh(deA[:,np.newaxis])/deA[:,np.newaxis]))
    
    #coefficients for tidal salinty - new developments on a sunny afternoon, Jan 19 2024
    c1 = - tid_set['Kv_ti']/(1j*omega)
    c2c, c3c, c4c , c2pc, c3pc, c4pc , c2c_z, c3c_z, c4c_z, = sti_coefs(self.zlist,c1,(deA[np.newaxis,:,np.newaxis],self.H[np.newaxis,:,np.newaxis],B[np.newaxis,:,np.newaxis],self.nn))

    # =============================================================================
    #     save in dictionary
    # =============================================================================
    save = {}
    save['omega'] = omega
    save['deA'] = deA
    save['B']   = B
    
    save['eta']     = eta
    save['detadx']  = detadx
    save['detadx2'] = detadx2
    save['detadx3'] = detadx3
    
    save['ut']    = ut
    save['dutdx'] = dutdx
    save['wt']    = wt
    
    save['utb'] = utb
    save['utp'] = utp
 
    #other 
    save['eps']  = tid_set['Kh_ti']/(omega*self.Lsc**2)       #epsilon, the normalised horizontal diffusion
    save['epsL'] = tid_set['Kh_ti']/(omega)        #epsilon, the normalised horizontal diffusion
    
    #coefficients for tidal salinty
    save['c2c'] = c2c
    save['c3c'] = c3c
    save['c4c'] = c4c   
    save['c2pc'] = c2pc
    save['c3pc'] = c3pc
    save['c4pc'] = c4pc   
    save['c2c_z'] = c2c_z
    save['c3c_z'] = c3c_z
    save['c4c_z'] = c4c_z
    
    return save
    

def tidal_salinity(self, zout, tid_set, tid_geg):
    # =============================================================================
    # this function calculates tidal salinity, as a function of subtidal salinity. 
    # these variables need to be calculated once every iteration step
    # =============================================================================
    
    # =============================================================================
    # subtidal salinity gradients 
    # =============================================================================

    s9 = np.zeros(self.di[-1]*self.M)+np.nan
    for i in range(self.ndom): s9[self.di[i]*self.M:self.di[i+1]*self.M] = zout[(self.di3[i]+2)*self.M:(self.di3[i+1]-2)*self.M]
    ss = s9.reshape(self.di[-1],self.M)
    
    sb = self.soc*ss[:,0]
    sn = self.soc*ss[:,1:].T
    dsbdx, dsbdx2, dsndx = np.zeros(self.di[-1]), np.zeros(self.di[-1]), np.zeros((self.N,self.di[-1]))
    for dom in range(self.ndom):
        dsbdx[self.di[dom]] = (-3*sb[self.di[dom]] + 4*sb[self.di[dom]+1] - sb[self.di[dom]+2] )/(2*self.dxn[dom])
        dsbdx[self.di[dom]+1:self.di[dom+1]-1] = (sb[self.di[dom]+2:self.di[dom+1]] - sb[self.di[dom]:self.di[dom+1]-2])/(2*self.dxn[dom])
        dsbdx[self.di[dom+1]-1] = (sb[self.di[dom+1]-3] - 4*sb[self.di[dom+1]-2] + 3*sb[self.di[dom+1]-1] )/(2*self.dxn[dom])
   
        dsbdx2[self.di[dom]] = (2*sb[self.di[dom]] - 5*sb[self.di[dom]+1] +4*sb[self.di[dom]+2] -1*sb[self.di[dom]+3] )/(self.dxn[dom]**2)
        dsbdx2[self.di[dom]+1:self.di[dom+1]-1] = (sb[self.di[dom]+2:self.di[dom+1]] - 2*sb[self.di[dom]+1:self.di[dom+1]-1] + sb[self.di[dom]:self.di[dom+1]-2])/(self.dxn[dom]**2)
        dsbdx2[self.di[dom+1]-1] = (-sb[self.di[dom+1]-4] + 4*sb[self.di[dom+1]-3] - 5*sb[self.di[dom+1]-2] + 2*sb[self.di[dom+1]-1] )/(self.dxn[dom]**2)
   
        dsndx[:,self.di[dom]] = (-3*sn[:,self.di[dom]] + 4*sn[:,self.di[dom]+1] - sn[:,self.di[dom]+2] )/(2*self.dxn[dom])
        dsndx[:,self.di[dom]+1:self.di[dom+1]-1] = (sn[:,self.di[dom]+2:self.di[dom+1]] - sn[:,self.di[dom]:self.di[dom+1]-2])/(2*self.dxn[dom])
        dsndx[:,self.di[dom+1]-1] = (sn[:,self.di[dom+1]-3] - 4*sn[:,self.di[dom+1]-2] + 3*sn[:,self.di[dom+1]-1] )/(2*self.dxn[dom])
   
    sn = sn[:,:,np.newaxis]
    dsbdx = dsbdx[np.newaxis,:,np.newaxis]
    dsbdx2 = dsbdx2[np.newaxis,:,np.newaxis]
    dsndx = dsndx[:,:,np.newaxis]
    
    # =============================================================================
    # Now the tidal salinty and the derivatives   
    # =============================================================================
        
    c1 = - tid_set['Kv_ti']/(1j*tid_geg['omega'])
    c2, c3, c4 = np.zeros((self.N,self.di[-1],self.nz),dtype=complex) , np.zeros((self.N,self.di[-1],self.nz),dtype=complex) , np.zeros((self.N,self.di[-1],self.nz),dtype=complex)
    for dom in range(self.ndom): 
        c2[:,self.di[dom]:self.di[dom+1]] = self.g/(tid_geg['omega']**2) * tid_geg['detadx'][:,self.di[dom]:self.di[dom+1]] * dsbdx[:,self.di[dom]:self.di[dom+1]]
        c3[:,self.di[dom]:self.di[dom+1]] = - self.nn*np.pi/self.H[np.newaxis,self.di[dom]:self.di[dom+1],np.newaxis] * sn[:,self.di[dom]:self.di[dom+1]] * tid_geg['eta'][:,self.di[dom]:self.di[dom+1]]
        c4[:,self.di[dom]:self.di[dom+1]] = - self.nn*np.pi/self.H[np.newaxis,self.di[dom]:self.di[dom+1],np.newaxis] * sn[:,self.di[dom]:self.di[dom+1]] * self.g/(tid_geg['omega']**2) * (tid_geg['detadx2'][:,self.di[dom]:self.di[dom+1]] + tid_geg['detadx'][:,self.di[dom]:self.di[dom+1]]/self.bn[dom])
    
    st = sti((c2,c3,c4),(tid_geg['c2c'],tid_geg['c3c'],tid_geg['c4c']))    
    stb = np.mean(st,1)[:,np.newaxis] #kan in theorie analytisch - ik heb wat geprobeerd maar dat lijkt niet sneller, ik stel voor het lekker zo te laten. 
    stp = st-stb #kan in theorie analytisch

    # ===========================================================================
    # calculate salt flux derivative
    # ===========================================================================
    dc2dx, dc3dx, dc4dx = np.zeros((self.N,self.di[-1],self.nz),dtype=complex) , np.zeros((self.N,self.di[-1],self.nz),dtype=complex) , np.zeros((self.N,self.di[-1],self.nz),dtype=complex)
    for dom in range(self.ndom): 
        dc2dx[:,self.di[dom]:self.di[dom+1]] = self.g/(tid_geg['omega']**2) * (tid_geg['detadx2'][:,self.di[dom]:self.di[dom+1]]*dsbdx[:,self.di[dom]:self.di[dom+1]] + tid_geg['detadx'][:,self.di[dom]:self.di[dom+1]]*dsbdx2[:,self.di[dom]:self.di[dom+1]])
        dc3dx[:,self.di[dom]:self.di[dom+1]] = - self.nn*np.pi/self.H[np.newaxis,self.di[dom]:self.di[dom+1],np.newaxis] * (tid_geg['eta'][:,self.di[dom]:self.di[dom+1]]*dsndx[:,self.di[dom]:self.di[dom+1]] + sn[:,self.di[dom]:self.di[dom+1]]*tid_geg['detadx'][:,self.di[dom]:self.di[dom+1]])
        dc4dx[:,self.di[dom]:self.di[dom+1]] = - self.nn*np.pi/self.H[np.newaxis,self.di[dom]:self.di[dom+1],np.newaxis] * self.g/(tid_geg['omega']**2) * (dsndx[:,self.di[dom]:self.di[dom+1]] * (tid_geg['detadx2'][:,self.di[dom]:self.di[dom+1]] \
                                     + tid_geg['detadx'][:,self.di[dom]:self.di[dom+1]]/self.bn[dom]) + sn[:,self.di[dom]:self.di[dom+1]] * (tid_geg['detadx3'][:,self.di[dom]:self.di[dom+1]] + tid_geg['detadx2'][:,self.di[dom]:self.di[dom+1]]/self.bn[dom]))
    
    dstidx = (tid_geg['c2c'] * dc2dx + (tid_geg['c3c'] * dc3dx).sum(0) + (tid_geg['c4c'] * dc4dx).sum(0))[0]
    dstibdx = np.mean(dstidx,1)[:,np.newaxis] #kan in theorie analytisch
    dstipdx = dstidx-dstibdx #kan in theorie analytisch
           
    #derivatives to z 
    dstdz = sti((c2,c3,c4),(tid_geg['c2c_z'],tid_geg['c3c_z'],tid_geg['c4c_z']))  

    '''  
    # =============================================================================
    # calculate tranport directly - works but is slower than numerically apparently (maybe not anymore but needs a check)
    # =============================================================================
    c5 = self.g/(1j*tid_geg['omega']) * tid_geg['detadx']

    
    #hier nog vrij lang, kan ook allemaal in tidefuncs bestand. - dit werkt, maar is factor ~50 trager dan mn andere methode...
    tijd = time.time()
    utst_b = 1/4 * np.real( utisti((self.H[np.newaxis],self.nn[:,0]) , 
                                (c1,c2[:,:,0],c3[:,:,0],c4[:,:,0],tid_geg['deA'][np.newaxis],tid_geg['B'][np.newaxis]) ,
                                (np.conj(c5[:,:,0]),np.conj(tid_geg['deA'])[np.newaxis],np.conj(tid_geg['B'])[np.newaxis])) +
                            utisti((self.H[np.newaxis],self.nn[:,0]) , 
                                (np.conj(c1),np.conj(c2[:,:,0]),np.conj(c3[:,:,0]),np.conj(c4[:,:,0]),np.conj(tid_geg['deA'])[np.newaxis],np.conj(tid_geg['B'])[np.newaxis]) ,
                                (c5[:,:,0],tid_geg['deA'][np.newaxis],tid_geg['B'][np.newaxis]))
                            )
    
    print('analytical takes', time.time()-tijd , ' seconds')
    '''
    
    #save variables which I need - not in object, since they change during the calculation
    save = { 'st' : st , 
            'stb': stb ,
            'stp': stp ,
            
            'dstidx': dstidx ,
            'dstibdx': dstibdx ,
            'dstipdx': dstipdx ,      

            'dstdz': dstdz , 
            
            #'utst_b': utst_b
            }
        
    return save


def tidsol_inda(self, inp, tid_geg):
    # =============================================================================
    # solution vector elements for contirubtion to tide to the depth-averaged inner domain
    # =============================================================================
    #load inp
    st     = inp['st']
    dstidx = inp['dstidx']
    
    #calculate contribution - numerically, in theory this can be done analytically. 
    dfdx = np.mean(-1/4 * np.real(tid_geg['ut']*np.conj(dstidx) + np.conj(tid_geg['ut'])*dstidx + tid_geg['dutdx']*np.conj(st) + np.conj(tid_geg['dutdx'])*st) , axis=1)
    flux = np.mean(-1/4 * np.real(tid_geg['ut']*np.conj(st) + np.conj(tid_geg['ut'])*st) , axis=1) #part for dbdx
    
    tides = (dfdx + flux / self.bex) * self.Lsc/self.soc 
    
    
    return tides




def tidjac_inda(self, tid_geg):
    # =============================================================================
    # jacobian matrix elements for contirubtion to tide to the depth-averaged inner domain
    # =============================================================================

    #load inp
    dsdc2 = tid_geg['c2c']
    dsdc3 = tid_geg['c3c']
    dsdc4 = tid_geg['c4c']

    dT_dsb1,dT_dsb0,dT_dsb_1 = np.zeros(self.di[-1]) , np.zeros(self.di[-1]) , np.zeros(self.di[-1])
    dT_dsn1,dT_dsn0,dT_dsn_1 = np.zeros((self.N,self.di[-1])) , np.zeros((self.N,self.di[-1])) , np.zeros((self.N,self.di[-1]))
            
    for dom in range(self.ndom):     
        ut_here      = tid_geg['ut'][np.newaxis,self.di[dom]:self.di[dom+1]]
        dutdx_here   = tid_geg['dutdx'][np.newaxis,self.di[dom]:self.di[dom+1]]
        eta_here     = tid_geg['eta'][:,self.di[dom]:self.di[dom+1]]
        detadx_here  = tid_geg['detadx'][:,self.di[dom]:self.di[dom+1]]
        detadx2_here = tid_geg['detadx2'][:,self.di[dom]:self.di[dom+1]]
        detadx3_here = tid_geg['detadx3'][:,self.di[dom]:self.di[dom+1]]
        
        dsdc2_here = dsdc2[:,self.di[dom]:self.di[dom+1]]
        dsdc3_here = dsdc3[:,self.di[dom]:self.di[dom+1]]
        dsdc4_here = dsdc4[:,self.di[dom]:self.di[dom+1]]
        nph_here = self.nph[:,self.di[dom]:self.di[dom+1]]
        
        dT_dsb1[self.di[dom]:self.di[dom+1]] = -1/4*self.Lsc * np.real(dutdx_here * np.conj( dsdc2_here * self.g/tid_geg['omega']**2 * detadx_here * 1/(2*self.dxn[dom]) ) 
                               + np.conj(dutdx_here) * dsdc2_here * self.g/tid_geg['omega']**2 * detadx_here * 1/(2*self.dxn[dom])
                               + ut_here * np.conj(dsdc2_here * self.g/tid_geg['omega']**2 * (detadx2_here/(2*self.dxn[dom]) + detadx_here/(self.dxn[dom]**2)))
                               + np.conj(ut_here)*(dsdc2_here * self.g/tid_geg['omega']**2 * (detadx2_here/(2*self.dxn[dom]) + detadx_here/(self.dxn[dom]**2))) 
                               + ut_here/self.bn[dom] * np.conj( dsdc2_here * self.g/tid_geg['omega']**2 * detadx_here * 1/(2*self.dxn[dom]) ) 
                               + np.conj(ut_here/self.bn[dom]) * dsdc2_here * self.g/tid_geg['omega']**2 * detadx_here * 1/(2*self.dxn[dom])
                               ).mean(2)
        
        dT_dsb0[self.di[dom]:self.di[dom+1]] = -1/4*self.Lsc * np.real( ut_here * np.conj(dsdc2_here * -2 * self.g/tid_geg['omega']**2 * detadx_here/(self.dxn[dom]**2)) 
                               +np.conj(ut_here)*(dsdc2_here * -2 * self.g/tid_geg['omega']**2 * detadx_here/(self.dxn[dom]**2)) ).mean(2) 
        
        dT_dsb_1[self.di[dom]:self.di[dom+1]] = -1/4*self.Lsc * np.real(dutdx_here * np.conj( dsdc2_here *-self.g/tid_geg['omega']**2 * detadx_here * 1/(2*self.dxn[dom]) ) 
                                                            + np.conj(dutdx_here)* dsdc2_here *-self.g/tid_geg['omega']**2 * detadx_here * 1/(2*self.dxn[dom])
                                                            + ut_here * np.conj(dsdc2_here * self.g/tid_geg['omega']**2 * (-detadx2_here/(2*self.dxn[dom]) + detadx_here/(self.dxn[dom]**2)))
                                                            + np.conj(ut_here)*(dsdc2_here * self.g/tid_geg['omega']**2 * (-detadx2_here/(2*self.dxn[dom]) + detadx_here/(self.dxn[dom]**2))) 
                                                            + ut_here/self.bn[dom] * np.conj( dsdc2_here *-self.g/tid_geg['omega']**2 * detadx_here * 1/(2*self.dxn[dom]) ) 
                                                            + np.conj(ut_here/self.bn[dom]) * dsdc2_here *-self.g/tid_geg['omega']**2 * detadx_here * 1/(2*self.dxn[dom])
                                                            ).mean(2)

        dT_dsn1[:,self.di[dom]:self.di[dom+1]] = -1/4*self.Lsc * np.real( ut_here * np.conj(dsdc3_here * -nph_here * eta_here/(2*self.dxn[dom]) + dsdc4_here * -nph_here * self.g/tid_geg['omega']**2 /(2*self.dxn[dom]) * (detadx2_here + detadx_here/self.bn[dom])) 
                                                      + np.conj(ut_here)*(dsdc3_here * -nph_here * eta_here/(2*self.dxn[dom]) + dsdc4_here * -nph_here * self.g/tid_geg['omega']**2 /(2*self.dxn[dom]) * (detadx2_here + detadx_here/self.bn[dom])) ).mean(2)
        
        dT_dsn0[:,self.di[dom]:self.di[dom+1]] = -1/4*self.Lsc * np.real(  dutdx_here * np.conj( dsdc3_here * - nph_here*eta_here + dsdc4_here * - nph_here * self.g/tid_geg['omega']**2 * (detadx2_here + detadx_here/self.bn[dom]) ) 
                                                    + np.conj(dutdx_here)*( dsdc3_here * - nph_here*eta_here + dsdc4_here * - nph_here * self.g/tid_geg['omega']**2 * (detadx2_here + detadx_here/self.bn[dom]) ) 
                                                    + ut_here * np.conj(dsdc3_here * (- nph_here * detadx_here) + dsdc4_here * (-nph_here * self.g/tid_geg['omega']**2 * (detadx3_here + detadx2_here/self.bn[dom])))
                                                    + np.conj(ut_here)*(dsdc3_here * (- nph_here * detadx_here) + dsdc4_here * (-nph_here * self.g/tid_geg['omega']**2 * (detadx3_here + detadx2_here/self.bn[dom])))
                                                    + ut_here/self.bn[dom] * np.conj( dsdc3_here * - nph_here*eta_here + dsdc4_here * - nph_here * self.g/tid_geg['omega']**2 * (detadx2_here + detadx_here/self.bn[dom]) ) 
                                                    + np.conj(ut_here)/self.bn[dom]*( dsdc3_here * - nph_here*eta_here + dsdc4_here * - nph_here * self.g/tid_geg['omega']**2 * (detadx2_here + detadx_here/self.bn[dom]) ) 
                                                    ).mean(2)
        
        dT_dsn_1[:,self.di[dom]:self.di[dom+1]] = -1/4*self.Lsc * np.real( ut_here * np.conj(dsdc3_here * nph_here * eta_here/(2*self.dxn[dom]) + dsdc4_here * nph_here * self.g/tid_geg['omega']**2 /(2*self.dxn[dom]) * (detadx2_here + detadx_here/self.bn[dom])) 
                                 + np.conj(ut_here)*(dsdc3_here * nph_here * eta_here/(2*self.dxn[dom]) + dsdc4_here * nph_here * self.g/tid_geg['omega']**2 /(2*self.dxn[dom]) * (detadx2_here + detadx_here/self.bn[dom])) ).mean(2)
    
    
    return dT_dsb1,dT_dsb0,dT_dsb_1 , dT_dsn1.T,dT_dsn0.T,dT_dsn_1.T

    

def tidsol_indv(self, inp, tid_geg):
    # =============================================================================
    # solution vector elements for contirubtion to tide to the depth-perturbed inner domain
    # =============================================================================
    
    #load inp
    st  = inp['st']          
    stb = inp['stb']
    stp = inp['stp']

    dstidx  = inp['dstidx']
    dstibdx = inp['dstibdx']
    dstipdx = inp['dstipdx']
    dstdz   = inp['dstdz']
        
    term1 = self.Lsc/self.soc * (-1/4*np.real(tid_geg['utp']*np.conj(dstidx) +  np.conj(tid_geg['utp'])*dstidx)[np.newaxis,:,:] * np.cos(self.nph*self.zlist)).mean(2)
    term2 = self.Lsc/self.soc * (-1/4*np.real(tid_geg['utb'][:,np.newaxis]*np.conj(dstipdx) + np.conj(tid_geg['utb'][:,np.newaxis])*dstipdx)[np.newaxis,:,:] * np.cos(self.nph*self.zlist)).mean(2)
    term3 = self.Lsc/self.soc * (-1/4*np.real(tid_geg['wt']  * np.conj(dstdz) + np.conj(tid_geg['wt'] ) * dstdz) * np.cos(self.nph*self.zlist)).mean(2)

    out = term1 + term2 + term3

    return out


def tidjac_indv(self, tid_geg):
    # =============================================================================
    # jacobian matrix elements for contirubtion to tide to the depth-perturbed inner domain
    # =============================================================================
    #load inp
    dsdc2 = tid_geg['c2c']  
    dsdc3 = tid_geg['c3c']  
    dsdc4 = tid_geg['c4c']    
    dspdc2= tid_geg['c2pc']
    dspdc3= tid_geg['c3pc']
    dspdc4= tid_geg['c4pc']
    dstdz_dc2 = tid_geg['c2c_z']
    dstdz_dc3 = tid_geg['c3c_z']
    dstdz_dc4 = tid_geg['c4c_z']
    
    dt1_dsb1,dt1_dsb0,dt1_dsb_1 = np.zeros((self.N,self.di[-1])) , np.zeros((self.N,self.di[-1])) , np.zeros((self.N,self.di[-1])) 
    dt1_dsn1,dt1_dsn0,dt1_dsn_1 = np.zeros((self.N,self.N,self.di[-1])) , np.zeros((self.N,self.N,self.di[-1])) , np.zeros((self.N,self.N,self.di[-1]))
    dt2_dsb1,dt2_dsb0,dt2_dsb_1 = np.zeros((self.N,self.di[-1])) , np.zeros((self.N,self.di[-1])) , np.zeros((self.N,self.di[-1])) 
    dt2_dsn1,dt2_dsn0,dt2_dsn_1 = np.zeros((self.N,self.N,self.di[-1])) , np.zeros((self.N,self.N,self.di[-1])) , np.zeros((self.N,self.N,self.di[-1]))
    dt3_dsb1,dt3_dsb_1,dt3_dsn0 = np.zeros((self.N,self.di[-1])) , np.zeros((self.N,self.di[-1])) , np.zeros((self.N,self.N,self.di[-1])) 
            
    
    for dom in range(self.ndom):     
        utp_here     = tid_geg['utp'][np.newaxis,self.di[dom]:self.di[dom+1]]
        utb_here     = tid_geg['utb'][np.newaxis,self.di[dom]:self.di[dom+1],np.newaxis]
        wt_here      = tid_geg['wt'][:,self.di[dom]:self.di[dom+1]]
        eta_here     = tid_geg['eta'][:,self.di[dom]:self.di[dom+1]]
        detadx_here  = tid_geg['detadx'][:,self.di[dom]:self.di[dom+1]]
        detadx2_here = tid_geg['detadx2'][:,self.di[dom]:self.di[dom+1]]
        detadx3_here = tid_geg['detadx3'][:,self.di[dom]:self.di[dom+1]]
        
        dsdc2_here = dsdc2[:,self.di[dom]:self.di[dom+1]]
        dsdc3_here = dsdc3[:,self.di[dom]:self.di[dom+1]]
        dsdc4_here = dsdc4[:,self.di[dom]:self.di[dom+1]]
        dspdc2_here = dspdc2[:,self.di[dom]:self.di[dom+1]]
        dspdc3_here = dspdc3[:,self.di[dom]:self.di[dom+1]]
        dspdc4_here = dspdc4[:,self.di[dom]:self.di[dom+1]]
        dstdz_dc2_here = dstdz_dc2[:,self.di[dom]:self.di[dom+1]]
        dstdz_dc3_here = dstdz_dc3[:,self.di[dom]:self.di[dom+1]]
        dstdz_dc4_here = dstdz_dc4[:,self.di[dom]:self.di[dom+1]]
        nph_here   = self.nph[:,self.di[dom]:self.di[dom+1]]
        z_here     = self.zlist[:,self.di[dom]:self.di[dom+1]]



        #term 1
        dt1_dsb1[:,self.di[dom]:self.di[dom+1]] = -1/4*self.Lsc/self.soc * (np.real(utp_here * np.conj( dsdc2_here * self.g/tid_geg['omega']**2 * (detadx2_here/(2*self.dxn[dom]) + detadx_here/(self.dxn[dom]**2)) ) 
                               + np.conj(utp_here) * dsdc2_here * self.g/tid_geg['omega']**2 * (detadx2_here/(2*self.dxn[dom]) + detadx_here/(self.dxn[dom]**2))) * np.cos(nph_here*z_here)).mean(2)
        
        dt1_dsb0[:,self.di[dom]:self.di[dom+1]] = -1/4*self.Lsc/self.soc * (np.real( utp_here * np.conj(dsdc2_here * -2 * self.g/tid_geg['omega']**2 * detadx_here/(self.dxn[dom]**2)) 
                               + np.conj(utp_here) * (dsdc2_here * -2 * self.g/tid_geg['omega']**2 * detadx_here/(self.dxn[dom]**2))) * np.cos(nph_here*z_here)).mean(2)

        dt1_dsb_1[:,self.di[dom]:self.di[dom+1]] = -1/4*self.Lsc/self.soc * (np.real(utp_here * np.conj(dsdc2_here * self.g/tid_geg['omega']**2 * (-detadx2_here/(2*self.dxn[dom]) + detadx_here/(self.dxn[dom]**2)))
                                + np.conj(utp_here) * dsdc2_here * self.g/tid_geg['omega']**2 * (-detadx2_here/(2*self.dxn[dom]) + detadx_here/(self.dxn[dom]**2))) * np.cos(nph_here*z_here)).mean(2)
                                                           
        dt1_dsn1[:,:,self.di[dom]:self.di[dom+1]] = -1/4*self.Lsc/self.soc * (np.real( utp_here * np.conj(dsdc3_here * -nph_here * eta_here/(2*self.dxn[dom]) + dsdc4_here * -nph_here * self.g/tid_geg['omega']**2 /(2*self.dxn[dom]) * (detadx2_here + detadx_here/self.bn[dom])) 
                                + np.conj(utp_here)*(dsdc3_here * -nph_here * eta_here/(2*self.dxn[dom]) + dsdc4_here * -nph_here * self.g/tid_geg['omega']**2 /(2*self.dxn[dom]) * (detadx2_here + detadx_here/self.bn[dom])))[:,np.newaxis,:,:] * np.cos(nph_here*z_here) ).mean(-1)

        dt1_dsn0[:,:,self.di[dom]:self.di[dom+1]] = -1/4*self.Lsc/self.soc * (np.real(    utp_here * np.conj(dsdc3_here * (- nph_here * detadx_here) + dsdc4_here * (-nph_here * self.g/tid_geg['omega']**2 * (detadx3_here + detadx2_here/self.bn[dom])))
                                                                  + np.conj(utp_here) * (dsdc3_here * (- nph_here * detadx_here) + dsdc4_here * (-nph_here * self.g/tid_geg['omega']**2 * (detadx3_here + detadx2_here/self.bn[dom]))))[:,np.newaxis,:,:] * np.cos(nph_here*z_here)).mean(-1)

        dt1_dsn_1[:,:,self.di[dom]:self.di[dom+1]] = -1/4*self.Lsc/self.soc * (np.real( utp_here * np.conj(dsdc3_here * -nph_here * eta_here/(2*self.dxn[dom]) + dsdc4_here * -nph_here * self.g/tid_geg['omega']**2 /(2*self.dxn[dom]) * (detadx2_here + detadx_here/self.bn[dom])) 
                                + np.conj(utp_here)*(dsdc3_here * -nph_here * eta_here/(2*self.dxn[dom]) + dsdc4_here * -nph_here * self.g/tid_geg['omega']**2 /(2*self.dxn[dom]) * (detadx2_here + detadx_here/self.bn[dom])) )[:,np.newaxis,:,:] * np.cos(nph_here*z_here) ).mean(-1) *-1 
        #term 2
        dt2_dsb1[:,self.di[dom]:self.di[dom+1]] = -1/4*self.Lsc/self.soc * (np.real(utb_here * np.conj( dspdc2_here * self.g/tid_geg['omega']**2 * (detadx2_here/(2*self.dxn[dom]) + detadx_here/(self.dxn[dom]**2)) ) 
                               + np.conj(utb_here) * dspdc2_here * self.g/tid_geg['omega']**2 * (detadx2_here/(2*self.dxn[dom]) + detadx_here/(self.dxn[dom]**2))) * np.cos(nph_here*z_here)).mean(2)
        
        dt2_dsb0[:,self.di[dom]:self.di[dom+1]] = -1/4*self.Lsc/self.soc * (np.real( utb_here * np.conj(dspdc2_here * -2 * self.g/tid_geg['omega']**2 * detadx_here/(self.dxn[dom]**2)) 
                               + np.conj(utb_here) * (dspdc2_here * -2 * self.g/tid_geg['omega']**2 * detadx_here/(self.dxn[dom]**2))) * np.cos(nph_here*z_here)).mean(2)

        dt2_dsb_1[:,self.di[dom]:self.di[dom+1]] = -1/4*self.Lsc/self.soc * (np.real(utb_here * np.conj(dspdc2_here * self.g/tid_geg['omega']**2 * (-detadx2_here/(2*self.dxn[dom]) + detadx_here/(self.dxn[dom]**2)))
                                + np.conj(utb_here) * dspdc2_here * self.g/tid_geg['omega']**2 * (-detadx2_here/(2*self.dxn[dom]) + detadx_here/(self.dxn[dom]**2))) * np.cos(nph_here*z_here)).mean(2)
                                                           
        dt2_dsn1[:,:,self.di[dom]:self.di[dom+1]] = -1/4*self.Lsc/self.soc * (np.real( utb_here * np.conj(dspdc3_here * -nph_here * eta_here/(2*self.dxn[dom]) + dspdc4_here * -nph_here * self.g/tid_geg['omega']**2 /(2*self.dxn[dom]) * (detadx2_here + detadx_here/self.bn[dom])) 
                                + np.conj(utb_here)*(dspdc3_here * -nph_here * eta_here/(2*self.dxn[dom]) + dspdc4_here * -nph_here * self.g/tid_geg['omega']**2 /(2*self.dxn[dom]) * (detadx2_here + detadx_here/self.bn[dom])))[:,np.newaxis,:,:] * np.cos(nph_here*z_here) ).mean(-1)

        dt2_dsn0[:,:,self.di[dom]:self.di[dom+1]] = -1/4*self.Lsc/self.soc * (np.real(    utb_here * np.conj(dspdc3_here * (- nph_here * detadx_here) + dspdc4_here * (-nph_here * self.g/tid_geg['omega']**2 * (detadx3_here + detadx2_here/self.bn[dom])))
                                                                  + np.conj(utb_here) * (dspdc3_here * (- nph_here * detadx_here) + dspdc4_here * (-nph_here * self.g/tid_geg['omega']**2 * (detadx3_here + detadx2_here/self.bn[dom]))))[:,np.newaxis,:,:] * np.cos(nph_here*z_here)).mean(-1)

        dt2_dsn_1[:,:,self.di[dom]:self.di[dom+1]] = -1/4*self.Lsc/self.soc * (np.real( utb_here * np.conj(dspdc3_here * -nph_here * eta_here/(2*self.dxn[dom]) + dspdc4_here * -nph_here * self.g/tid_geg['omega']**2 /(2*self.dxn[dom]) * (detadx2_here + detadx_here/self.bn[dom])) 
                                + np.conj(utb_here)*(dspdc3_here * -nph_here * eta_here/(2*self.dxn[dom]) + dspdc4_here * -nph_here * self.g/tid_geg['omega']**2 /(2*self.dxn[dom]) * (detadx2_here + detadx_here/self.bn[dom])) )[:,np.newaxis,:,:] * np.cos(nph_here*z_here) ).mean(-1) *-1 
        
        #term 3
        dt3_dsb1[:,self.di[dom]:self.di[dom+1]]  = -1/4*self.Lsc/self.soc * (np.real(wt_here * np.conj(dstdz_dc2_here * self.g/tid_geg['omega']**2 * detadx_here * 1/(2*self.dxn[dom])) + np.conj(wt_here) * (dstdz_dc2_here * self.g/tid_geg['omega']**2 * detadx_here * 1/(2*self.dxn[dom])))* np.cos(nph_here*z_here) ).mean(-1)
        dt3_dsb_1[:,self.di[dom]:self.di[dom+1]] = -1/4*self.Lsc/self.soc * (np.real(wt_here * np.conj(dstdz_dc2_here * self.g/tid_geg['omega']**2 * detadx_here *-1/(2*self.dxn[dom])) + np.conj(wt_here) * (dstdz_dc2_here * self.g/tid_geg['omega']**2 * detadx_here *-1/(2*self.dxn[dom])))* np.cos(nph_here*z_here) ).mean(-1)
        dt3_dsn0[:,:,self.di[dom]:self.di[dom+1]]= -1/4*self.Lsc/self.soc * (np.real(wt_here * np.conj(- nph_here * eta_here * dstdz_dc3_here - nph_here * self.g/tid_geg['omega']**2 * (detadx2_here + detadx_here/self.bn[dom]) * dstdz_dc4_here ) \
                                        + np.conj(wt_here) * (- nph_here * eta_here * dstdz_dc3_here - nph_here * self.g/tid_geg['omega']**2 * (detadx2_here + detadx_here/self.bn[dom]) * dstdz_dc4_here   ))[:,np.newaxis,:,:] * np.cos(nph_here*z_here) ).mean(-1)
        
    return (dt1_dsb_1,dt1_dsb0,dt1_dsb1 , dt1_dsn_1,dt1_dsn0,dt1_dsn1) , (dt2_dsb_1,dt2_dsb0,dt2_dsb1 , dt2_dsn_1,dt2_dsn0,dt2_dsn1) , (dt3_dsb_1,dt3_dsb1,dt3_dsn0)
       

def corsol_ins(self, zout, tid_geg, indi):
    #calculate salinity correction
    Bm_rgt = zout[((self.di3[1:-1]-2)*self.M+np.arange(self.M)[:,np.newaxis])] + 1j * zout[((self.di3[1:-1]-2)*self.M+np.arange(self.M,2*self.M)[:,np.newaxis])]
    Bm_lft = zout[( self.di3[1:-1]   *self.M+np.arange(self.M)[:,np.newaxis])] + 1j * zout[( self.di3[1:-1]   *self.M+np.arange(self.M,2*self.M)[:,np.newaxis])]
    
    #calculate x 
    x_temp = [np.arange(self.nxn[i])*self.dxn[i] for i in range(self.ndom)]
    x_lft = [x_temp[i][np.where(x_temp[i]<-np.sqrt(tid_geg['epsL'])*np.log(self.tol))[0]][1:] for i in range(self.ndom)][1:]
    x_rgt = [x_temp[i][np.where(x_temp[i]>x_temp[i][-1]+np.sqrt(tid_geg['epsL'])*np.log(self.tol))[0]][:-1]-self.Ln[i] for i in range(self.ndom)][:-1]
    
    ih3_lft = [indi['ih_lft'][i] + np.arange(1,len(x_lft[i])+1) for i in range(self.ndom-1)]
    ih3_rgt = [indi['ih_rgt'][i] + np.arange(-len(x_rgt[i]),0) for i in range(self.ndom-1)]
     
    #test if the boundary layer is not larger than the domain of the segment
    for i in range(self.ndom-1):
        if len(x_rgt[i])>0:
            if -x_rgt[i][0] >= self.Ln[i]:
                print('ERROR: boundary layer larger than segment, this gives problems. choose different segment size (or Kh_ti or tol)')
        if len(x_lft[i])>0:
            if x_lft[i][-1] >= self.Ln[i+1]:
                print('ERROR: boundary layer larger than segment, this gives problems. choose different segment size (or Kh_ti or tol)')
           
    stc_lft   = [np.exp(-x_lft[i][:,np.newaxis]/np.sqrt(tid_geg['epsL'])) * np.sum(Bm_lft[:,i] * np.cos(np.arange(self.M)*np.pi*self.z_nd[:,np.newaxis]) ,1) * self.soc for i in range(self.ndom-1)]
    stc_lft_x = [stc_lft[i] * -1/np.sqrt(tid_geg['epsL']) for i in range(self.ndom-1)]
    stpc_lft_x= [-1/np.sqrt(tid_geg['epsL']) * np.exp(-x_lft[i][:,np.newaxis]/np.sqrt(tid_geg['epsL'])) * np.sum(Bm_lft[1:,i] * np.cos(np.arange(1,self.M)*np.pi*self.z_nd[:,np.newaxis]) ,1) * self.soc for i in range(self.ndom-1)]
    stc_lft_z = [np.exp(-x_lft[i]/np.sqrt(tid_geg['epsL']))[:,np.newaxis] * np.sum(Bm_lft[:,i,np.newaxis,np.newaxis] * -np.arange(self.M)[:,np.newaxis,np.newaxis]*np.pi/self.H[ih3_lft[i],np.newaxis] * np.sin(np.arange(self.M)[:,np.newaxis,np.newaxis]*np.pi*self.z_nd) , 0) * self.soc for i in range(self.ndom-1)]
    
    
    stc_rgt   = [np.exp(x_rgt[i][:,np.newaxis]/np.sqrt(tid_geg['epsL'])) * np.sum(Bm_rgt[:,i] * np.cos(np.arange(self.M)*np.pi*self.z_nd[:,np.newaxis]) ,1) * self.soc for i in range(self.ndom-1)]
    stc_rgt_x = [stc_rgt[i] * 1/np.sqrt(tid_geg['epsL']) for i in range(self.ndom-1)]
    stpc_rgt_x= [1/np.sqrt(tid_geg['epsL']) * np.exp(x_rgt[i][:,np.newaxis]/np.sqrt(tid_geg['epsL'])) * np.sum(Bm_rgt[1:,i] * np.cos(np.arange(1,self.M)*np.pi*self.z_nd[:,np.newaxis]) ,1) * self.soc for i in range(self.ndom-1)]
    stc_rgt_z = [np.exp(x_rgt[i]/np.sqrt(tid_geg['epsL']))[:,np.newaxis] * np.sum(Bm_rgt[:,i,np.newaxis,np.newaxis] * -np.arange(self.M)[:,np.newaxis,np.newaxis]*np.pi/self.H[ih3_rgt[i],np.newaxis] * np.sin(np.arange(self.M)[:,np.newaxis,np.newaxis]*np.pi*self.z_nd) , 0) *self.soc for i in range(self.ndom-1)]
               
    cda_lft = [-np.mean(-1/4 * np.real(tid_geg['ut'][ih3_lft[i]]*np.conj(stc_lft_x[i]) + np.conj(tid_geg['ut'][ih3_lft[i]]) * stc_lft_x[i] \
                                 + tid_geg['dutdx'][ih3_lft[i]]*np.conj(stc_lft[i]) + np.conj(tid_geg['dutdx'][ih3_lft[i]])*stc_lft[i] \
                                 + (tid_geg['ut'][ih3_lft[i]]*np.conj(stc_lft[i]) + np.conj(tid_geg['ut'][ih3_lft[i]]) * stc_lft[i])/self.bn[i+1]) , -1)* self.Lsc/self.soc for i in range(self.ndom-1)]
             
    t1_lft = [np.mean(-1/4 * np.real(tid_geg['utp'][ih3_lft[i]]*np.conj(stc_lft_x[i])  + np.conj(tid_geg['utp'][ih3_lft[i]])* stc_lft_x[i])[np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc for i in range(self.ndom-1)]
    t2_lft = [np.mean(-1/4 * np.real(tid_geg['utb'][ih3_lft[i],np.newaxis]*np.conj(stpc_lft_x[i]) + np.conj(tid_geg['utb'][ih3_lft[i],np.newaxis])*stpc_lft_x[i])[np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc for i in range(self.ndom-1)]
    t3_lft = [np.mean(-1/4 * np.real(tid_geg['wt'][0,ih3_lft[i]]*np.conj(stc_lft_z[i]) + np.conj(tid_geg['wt'][0,ih3_lft[i]])*stc_lft_z[i]  )[np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc for i in range(self.ndom-1)]
    
    cdv_lft = [t1_lft[i] + t2_lft[i] + t3_lft[i] for i in range(self.ndom-1)]
    
    cda_rgt = [-np.mean(-1/4 * np.real(tid_geg['ut'][ih3_rgt[i]]*np.conj(stc_rgt_x[i]) + np.conj(tid_geg['ut'][ih3_rgt[i]]) * stc_rgt_x[i] \
                                 + tid_geg['dutdx'][ih3_rgt[i]]*np.conj(stc_rgt[i]) + np.conj(tid_geg['dutdx'][ih3_rgt[i]]) * stc_rgt[i] \
                                 + (tid_geg['ut'][ih3_rgt[i]]*np.conj(stc_rgt[i]) + np.conj(tid_geg['ut'][ih3_rgt[i]]) * stc_rgt[i])/self.bn[i])  , -1) * self.Lsc/self.soc for i in range(self.ndom-1)]
    
    t1_rgt = [np.mean(-1/4 * np.real(tid_geg['utp'][ih3_rgt[i]]*np.conj(stc_rgt_x[i] ) + np.conj(tid_geg['utp'][ih3_rgt[i]])* stc_rgt_x[i])[np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc for i in range(self.ndom-1)]
    t2_rgt = [np.mean(-1/4 * np.real(tid_geg['utb'][ih3_rgt[i],np.newaxis]*np.conj(stpc_rgt_x[i]) + np.conj(tid_geg['utb'][ih3_rgt[i],np.newaxis])*stpc_rgt_x[i])[np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc for i in range(self.ndom-1)]
    t3_rgt = [np.mean(-1/4 * np.real(tid_geg['wt'][0,ih3_rgt[i]] *np.conj(stc_rgt_z[i] ) + np.conj(tid_geg['wt'][0,ih3_rgt[i]]) * stc_rgt_z[i])[np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc/self.soc for i in range(self.ndom-1)]
            
    cdv_rgt = [t1_rgt[i] + t2_rgt[i] + t3_rgt[i] for i in range(self.ndom-1)]
    
    return cda_lft, cda_rgt, cdv_lft , cdv_rgt


def corjac_ins(self, zout, tid_geg, indi):
    #calculate x 
    x_temp = [np.arange(self.nxn[i])*self.dxn[i] for i in range(self.ndom)]
    x_lft = [x_temp[i][np.where(x_temp[i]<-np.sqrt(tid_geg['epsL'])*np.log(self.tol))[0]][1:] for i in range(self.ndom)][1:]
    x_rgt = [x_temp[i][np.where(x_temp[i]>x_temp[i][-1]+np.sqrt(tid_geg['epsL'])*np.log(self.tol))[0]][:-1]-self.Ln[i] for i in range(self.ndom)][:-1]
    
    ih3_lft = [indi['ih_lft'][i] + np.arange(1,len(x_lft[i])+1) for i in range(self.ndom-1)]
    ih3_rgt = [indi['ih_rgt'][i] + np.arange(-len(x_rgt[i]),0) for i in range(self.ndom-1)]
                  

    #derivatives for jacobian
    dstc_lft     = [np.exp(-x_lft[i][np.newaxis,:,np.newaxis]/np.sqrt(tid_geg['epsL'])) * np.cos(np.arange(self.M)[:,np.newaxis,np.newaxis]*np.pi*self.z_nd[np.newaxis,np.newaxis,:]) for i in range(self.ndom-1)]
    dstcp_lft    = [np.zeros(dstc_lft[i].shape) for i in range(self.ndom-1)]
    for i in range(self.ndom-1): dstcp_lft[i][1:]= dstc_lft[i][1:]       
    dstc_lft_z   = [np.exp(-x_lft[i][:,np.newaxis]/np.sqrt(tid_geg['epsL'])) * -(np.arange(self.M)[:,np.newaxis,np.newaxis]*np.pi/self.H[ih3_lft[i],np.newaxis]) * np.sin(np.arange(self.M)[:,np.newaxis,np.newaxis]*np.pi*self.z_nd) for i in range(self.ndom-1)]
    
    dstc_rgt   = [np.exp(x_rgt[i][np.newaxis,:,np.newaxis]/np.sqrt(tid_geg['epsL'])) * np.cos(np.arange(self.M)[:,np.newaxis,np.newaxis]*np.pi*self.z_nd[np.newaxis,np.newaxis,:])  for i in range(self.ndom-1)]
    dstcp_rgt  = [np.zeros(dstc_rgt[i].shape) for i in range(self.ndom-1)]
    for i in range(self.ndom-1): dstcp_rgt[i][1:]= dstc_rgt[i][1:]       
    dstc_rgt_z = [np.exp(x_rgt[i][:,np.newaxis]/np.sqrt(tid_geg['epsL'])) * -(np.arange(self.M)[:,np.newaxis,np.newaxis]*np.pi/self.H[ih3_rgt[i],np.newaxis]) * np.sin(np.arange(self.M)[:,np.newaxis,np.newaxis]*np.pi*self.z_nd) for i in range(self.ndom-1)]
    
    #left, terms for jacobian
    #da
    dT_lft_dBR = [np.mean(1/4 * (2*np.real(tid_geg['ut'][ih3_lft[i]]) * dstc_lft[i] * -1/np.sqrt(tid_geg['epsL']) + 2*np.real(tid_geg['dutdx'][ih3_lft[i]]) * dstc_lft[i] \
                             + (2*np.real(tid_geg['ut'][ih3_lft[i]]) * dstc_lft[i])/self.bn[i+1])  , -1)* self.Lsc for i in range(self.ndom-1)]
    dT_lft_dBI = [np.mean(1/4 * (2*np.imag(tid_geg['ut'][ih3_lft[i]]) * dstc_lft[i] * -1/np.sqrt(tid_geg['epsL']) + 2*np.imag(tid_geg['dutdx'][ih3_lft[i]]) * dstc_lft[i] \
                              + (2*np.imag(tid_geg['ut'][ih3_lft[i]]) * dstc_lft[i])/self.bn[i+1])  , -1)* self.Lsc for i in range(self.ndom-1)]
    #dv
    dt1_lft_dBR = [np.mean(1/4 * (2*np.real(tid_geg['utp'][ih3_lft[i]]) * -1/np.sqrt(tid_geg['epsL']) *  dstc_lft[i])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc for i in range(self.ndom-1)]
    dt2_lft_dBR = [np.mean(1/4 * (2*np.real(tid_geg['utb'][ih3_lft[i],np.newaxis]) * -1/np.sqrt(tid_geg['epsL']) * dstcp_lft[i])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc for i in range(self.ndom-1)]
    dt3_lft_dBR = [np.mean(1/4 * (2*np.real(tid_geg['wt'][0,ih3_lft[i]])  * dstc_lft_z[i])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc for i in range(self.ndom-1)]

    dt1_lft_dBI = [np.mean(1/4 * (2*np.imag(tid_geg['utp'][ih3_lft[i]]) * -1/np.sqrt(tid_geg['epsL']) *  dstc_lft[i])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc for i in range(self.ndom-1)]
    dt2_lft_dBI = [np.mean(1/4 * (2*np.imag(tid_geg['utb'][ih3_lft[i],np.newaxis]) * -1/np.sqrt(tid_geg['epsL']) * dstcp_lft[i])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc for i in range(self.ndom-1)]
    dt3_lft_dBI = [np.mean(1/4 * (2*np.imag(tid_geg['wt'][0,ih3_lft[i]])  * dstc_lft_z[i])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc for i in range(self.ndom-1)]

    dTz_lft_dBR = [-dt1_lft_dBR[i]-dt2_lft_dBR[i]-dt3_lft_dBR[i] for i in range(self.ndom-1)]
    dTz_lft_dBI = [-dt1_lft_dBI[i]-dt2_lft_dBI[i]-dt3_lft_dBI[i] for i in range(self.ndom-1)]
    
    #right, terms for Jacobian
    #da
    dT_rgt_dBR = [np.mean(1/4 * (2*np.real(tid_geg['ut'][ih3_rgt[i]]) * dstc_rgt[i] * 1/np.sqrt(tid_geg['epsL']) + 2*np.real(tid_geg['dutdx'][ih3_rgt[i]]) * dstc_rgt[i] \
                              + (2*np.real(tid_geg['ut'][ih3_rgt[i]]) * dstc_rgt[i])/self.bn[i])  , -1)* self.Lsc for i in range(self.ndom-1)]
    dT_rgt_dBI = [np.mean(1/4 * (2*np.imag(tid_geg['ut'][ih3_rgt[i]]) * dstc_rgt[i] * 1/np.sqrt(tid_geg['epsL']) + 2*np.imag(tid_geg['dutdx'][ih3_rgt[i]]) * dstc_rgt[i] \
                              + (2*np.imag(tid_geg['ut'][ih3_rgt[i]]) * dstc_rgt[i])/self.bn[i])  , -1)* self.Lsc for i in range(self.ndom-1)]
    #dv
    dt1_rgt_dBR = [np.mean(1/4 * (2*np.real(tid_geg['utp'][ih3_rgt[i]]) * 1/np.sqrt(tid_geg['epsL']) *  dstc_rgt[i])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc for i in range(self.ndom-1)]
    dt2_rgt_dBR = [np.mean(1/4 * (2*np.real(tid_geg['utb'][ih3_rgt[i],np.newaxis]) * 1/np.sqrt(tid_geg['epsL']) * dstcp_rgt[i])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc for i in range(self.ndom-1)]
    dt3_rgt_dBR = [np.mean(1/4 * (2*np.real(tid_geg['wt'][0,ih3_rgt[i]])  * dstc_rgt_z[i])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc for i in range(self.ndom-1)]

    dt1_rgt_dBI = [np.mean(1/4 * (2*np.imag(tid_geg['utp'][ih3_rgt[i]]) * 1/np.sqrt(tid_geg['epsL']) *  dstc_rgt[i])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc for i in range(self.ndom-1)]
    dt2_rgt_dBI = [np.mean(1/4 * (2*np.imag(tid_geg['utb'][ih3_rgt[i],np.newaxis]) * 1/np.sqrt(tid_geg['epsL']) * dstcp_rgt[i])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc for i in range(self.ndom-1)]
    dt3_rgt_dBI = [np.mean(1/4 * (2*np.imag(tid_geg['wt'][0,ih3_rgt[i]])  * dstc_rgt_z[i])[:,np.newaxis,:,:] * np.cos(self.nn*np.pi*self.z_nd)  , -1) * self.Lsc for i in range(self.ndom-1)]

    dTz_rgt_dBR = [-dt1_rgt_dBR[i]-dt2_rgt_dBR[i]-dt3_rgt_dBR[i] for i in range(self.ndom-1)]
    dTz_rgt_dBI = [-dt1_rgt_dBI[i]-dt2_rgt_dBI[i]-dt3_rgt_dBI[i] for i in range(self.ndom-1)]
    
    return dT_lft_dBR, dT_lft_dBI, dTz_lft_dBR, dTz_lft_dBI, dT_rgt_dBR, dT_rgt_dBI, dTz_rgt_dBR, dTz_rgt_dBI


def solu_tidal(self, ans, inp, tid_geg, indi, version):

    # =============================================================================
    # contribution to the solution vector by the tidal dynamics
    # =============================================================================
    so = np.zeros(self.di3[-1]*self.M)
    
    if version in ['B','C','D']:
        #depth=averaged
        so[indi['xn_m']] += - tidsol_inda(self,inp,tid_geg)[indi['x']] 
    
    if version in ['D']:
        #add tides to the vertical balance
        so[indi['xnr_mj']] += tidsol_indv(self,inp,tid_geg)[indi['j1'],indi['xr']]

        #boundary layer correction
        cda_lft, cda_rgt, cdv_lft , cdv_rgt = corsol_ins(self, ans, tid_geg, indi)
        for i in range(self.ndom-1): #for all segment boundaries. 
            #lft 
            ih = (self.di3[i+1]+3+np.arange(len(cda_lft[i]))) * self.M #this can be done somewhere else
            so[ih] += cda_lft[i] #da
            so[ih+np.arange(1,self.M)[:,np.newaxis]] += cdv_lft[i] #dv
    
            #rgt 
            ih = (self.di3[i+1]-3+np.arange(-len(cda_rgt[i]),0)) * self.M #this should be done somewhere else
            so[ih] += cda_rgt[i] #da
            so[ih+np.arange(1,self.M)[:,np.newaxis]] += cdv_rgt[i] #dv
        
    return so 



def jaco_tidal(self, ans, tid_geg, indi, version):

    # =============================================================================
    # contribution to the solution vector by the tidal dynamics
    # =============================================================================
       
    jac = np.zeros((self.di3[-1]*self.M,self.di3[-1]*self.M))
    
    if version in ['B','C','D']:
        # =============================================================================
        #  depth-averaged
        # =============================================================================
        #contribution by tides, lets hope it works this way.
        tid_raw = tidjac_inda(self,tid_geg)
        
        #left
        jac[indi['xn_m'], indi['xnm_m']] +=  - tid_raw[2][indi['x']]
        jac[indi['xnr_m'], indi['xnrm_mj']] +=  - tid_raw[5][indi['xr'],indi['j1']]
        #center
        jac[indi['xn_m'],indi['xn_m']] +=  - tid_raw[1][indi['x']]
        jac[indi['xnr_m'], indi['xnr_mj']] += - tid_raw[4][indi['xr'],indi['j1']] 
        #right
        jac[indi['xn_m'], indi['xnp_m']] += - tid_raw[0][indi['x']]
        jac[indi['xnr_m'], indi['xnrp_mj']] += - tid_raw[3][indi['xr'],indi['j1']]

    if version in ['D']: 
        # =============================================================================
        # add tides to the vertical balance
        # =============================================================================
        NRv = tidjac_indv(self,tid_geg)
        t1a,t1b,t1c,t1d,t1e,t1f = NRv[0]
        t2a,t2b,t2c,t2d,t2e,t2f = NRv[1]
        t3a,t3c,t3e = NRv[2]
        
        jac[indi['xnr_mj'],indi['xnrm_m']] = jac[indi['xnr_mj'],indi['xnrm_m']] + (t1a[indi['j1'],indi['xr']]+t2a[indi['j1'],indi['xr']]+t3a[indi['j1'],indi['xr']])*self.soc
        jac[indi['xnr_mj'],indi['xnr_m']]  = jac[indi['xnr_mj'],indi['xnr_m']] +  (t1b[indi['j1'],indi['xr']]+t2b[indi['j1'],indi['xr']])*self.soc 
        jac[indi['xnr_mj'],indi['xnrp_m']] = jac[indi['xnr_mj'],indi['xnrp_m']] + (t1c[indi['j1'],indi['xr']]+t2c[indi['j1'],indi['xr']]+t3c[indi['j1'],indi['xr']])*self.soc 
        
        jac[indi['xnr_mj2'],indi['xnrm_mk']] = jac[indi['xnr_mj2'],indi['xnrm_mk']] + (t1d[indi['k1'],indi['j12'],indi['xr2']]+t2d[indi['k1'],indi['j12'],indi['xr2']])*self.soc
        jac[indi['xnr_mj2'], indi['xnr_mk']] = jac[indi['xnr_mj2'], indi['xnr_mk']] + (t1e[indi['k1'],indi['j12'],indi['xr2']]+t2e[indi['k1'],indi['j12'],indi['xr2']]+t3e[indi['k1'],indi['j12'],indi['xr2']])*self.soc
        jac[indi['xnr_mj2'],indi['xnrp_mk']] = jac[indi['xnr_mj2'],indi['xnrp_mk']] + (t1f[indi['k1'],indi['j12'],indi['xr2']]+t2f[indi['k1'],indi['j12'],indi['xr2']])*self.soc
        
        
        # =============================================================================
        # boundary layer effect on salinity in inner domain 
        # =============================================================================
        dT_lft_dBR, dT_lft_dBI, dTz_lft_dBR, dTz_lft_dBI, dT_rgt_dBR, dT_rgt_dBI, dTz_rgt_dBR, dTz_rgt_dBI = corjac_ins(self, ans, tid_geg, indi) 
        
        for i in range(self.ndom-1): #for all segment boundaries. 
            #lft 
            lx   = len(dT_lft_dBR[i][0])
            ih_l = np.tile((self.di3[i+1]+3+np.arange(lx)) * self.M , self.M).reshape((self.M,lx))
            ih_R = (np.repeat(indi['bnl_lft'][i][:self.M],lx).reshape((self.M,lx)))+2*self.M
            ih_I = (np.repeat(indi['bnl_lft'][i][self.M:],lx).reshape((self.M,lx)))+2*self.M
            
            ih_lz = ih_l[:,np.newaxis,:]+np.arange(1,self.M)[np.newaxis,:,np.newaxis]
            ih_Rz = ih_R[:,np.newaxis,:]+np.zeros(self.N,dtype=int)[np.newaxis,:,np.newaxis]
            ih_Iz = ih_I[:,np.newaxis,:]+np.zeros(self.N,dtype=int)[np.newaxis,:,np.newaxis]
            
            jac[ih_l,ih_R] += dT_lft_dBR[i]
            jac[ih_l,ih_I] += dT_lft_dBI[i]
            jac[ih_lz,ih_Rz] += dTz_lft_dBR[i]
            jac[ih_lz,ih_Iz] += dTz_lft_dBI[i]      
    
            #rgt
            lx   = len(dT_rgt_dBR[i][0])
            ih_r = np.tile((self.di3[i+1]-3+np.arange(-lx,0)) * self.M , self.M).reshape((self.M,lx))
            ih_R = (np.repeat(indi['bnl_rgt'][i][:self.M],lx).reshape((self.M,lx)))-2*self.M
            ih_I = (np.repeat(indi['bnl_rgt'][i][self.M:],lx).reshape((self.M,lx)))-2*self.M
            
            ih_rz = ih_r[:,np.newaxis,:]+np.arange(1,self.M)[np.newaxis,:,np.newaxis]
            ih_Rz = ih_R[:,np.newaxis,:]+np.zeros(self.N,dtype=int)[np.newaxis,:,np.newaxis]
            ih_Iz = ih_I[:,np.newaxis,:]+np.zeros(self.N,dtype=int)[np.newaxis,:,np.newaxis]
            
            jac[ih_r,ih_R] += dT_rgt_dBR[i]
            jac[ih_r,ih_I] += dT_rgt_dBI[i]
            jac[ih_rz,ih_Rz] += dTz_rgt_dBR[i]
            jac[ih_rz,ih_Iz] += dTz_rgt_dBI[i]
        
    
    return jac  
    

'''
def calc_tid7(self, out, indi, xhere):
    ss2 = np.delete(out , np.concatenate([indi['bnl_rgt'].flatten(),indi['bnl_lft'].flatten(),indi['bnl_bnd']]))

    #calculate salinity
    sn = np.reshape(ss2,(self.di[-1],self.M))[:,1:] * self.soc
    s_p = np.sum([sn[xhere,n-1]*np.cos(np.pi*n *self.z_nd) for n in range(1,self.M)],0) 
    dspdz  = np.sum([sn[xhere,n-1]*np.pi*n/self.H*-np.sin(np.pi*n*self.z_nd) for n in range(1,self.M)],0)
    
    iop = self.eta[0,xhere,0] * (s_p[-1]-s_p[0])/self.H \
        + self.g/self.omega**2 * (self.detadx2[0,xhere,0] + self.detadx[0,xhere,0]/self.bex[xhere]) *\
        np.mean((self.B*self.H/self.deA*np.sinh(self.deA*self.z_nd) - self.zlist) * dspdz)
    
    T_Tb = -1/4 * np.real(tid_geg['utb'][xhere,0] * np.conj(iop) + np.conj(tid_geg['utb'][xhere,0]) * iop) * self.b[xhere] * self.H
    
    
    return T_Tb
    
ii = run.di[-2]-3
print(calc_tid7(run,out,run.ii_all,ii))
res = run.plot_transport(out , run.ii_all, vers)[4][ii]
print(res)
print(res/calc_tid7(run,out,run.ii_all,ii))
'''
