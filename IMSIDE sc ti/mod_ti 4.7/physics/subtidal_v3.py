# =============================================================================
# create here the parameters for the subtidal module
# maybe also add how to make the solution vector and jacobian to this file. 
# for now: only time-independent. make also a time-dependent version of this. 
# =============================================================================

import numpy as np

def subtidal_module(self):
    # =============================================================================
    # build physical parameters
    # =============================================================================

    #different parametrizations possible
    
    #vertical viscosity
    if self.choice_viscosityv_st == 'constant': Av_st = self.Av_st + np.zeros(self.di[-1]) #do nothing, value is specified
    elif self.choice_viscosityv_st == 'cuh': Av_st = self.cv_st * self.Ut * self.H
    else: print('ERROR: no valid op option for choice vertical viscosity subtidal')

    #vertical diffusivity
    if self.choice_diffusivityv_st == 'constant': Kv_st = self.Kv_st + np.zeros(self.di[-1]) #do nothing, value is specified
    elif self.choice_diffusivityv_st == 'as': Kv_st = Av_st / self.Sc_st
    else: print('ERROR: no valid option for choice vertical diffusivity subtidal')

    #horizontal diffusivity
    if self.choice_diffusivityh_st == 'constant':             
        Kh = self.Kh_st + np.zeros(self.di[-1])
        Kh[self.di[-2]:]= self.Kh_st * self.b[self.di[-2]:]/self.b[self.di[-2]] #sea domain
    if self.choice_diffusivityh_st == 'cub':             
        Kh = self.ch_st * self.Ut * self.b
        
    #derivative - numerical, can also analytical, should not matter too much. 
    Kh_x = np.zeros(Kh.shape) + np.nan
    Kh_x[1:-1] = (Kh[2:] - Kh[:-2])/ (2*self.dl[1:-1]*self.Lsc)
    Kh_x[[0,-1]] ,  Kh_x[self.di[1:-1]] ,  Kh_x[self.di[1:-1]-1] = np.nan,np.nan,np.nan
    
    #coefficients
    if self.choice_bottomslip_st == 'sf': rr = Av_st/(self.sf_st*self.H)
    elif self.choice_bottomslip_st == 'rr': rr = self.rr_st + np.zeros(self.di[-1])
    else: print('ERROR: no valid op option for choice bottomslip subtidal')
    g1 = -1 + (1.5+3*rr) / (1+ 3 *rr)
    g2 =  -3 / (2+6*rr)
    g3 = (1+4*rr) / (1+3*rr) * (9+18*rr) - 8 - 24*rr
    g4 = -9 * (1+4*rr) / (1+3*rr)
    g5 = - 8
    #g6 = 4+4*rr -12*(0.5+rr)**2/(1+3*rr)
    #g7 = 4
    #g8 = (3+6*rr) / (1+3*rr)
    #g1,g2,g3,g4,g5 = 1/2 , -3/2 , 1, -9 ,-8
    
    u_bar = self.Q/(self.H*self.b)
    alf   = self.g*self.Be*self.H**3/(48*Av_st)
    
    #lists of ks and ns and pis
    kkp = np.linspace(1,self.N,self.N)*np.pi #k*pi
    nnp = np.linspace(1,self.N,self.N)*np.pi #n*pi
    pkn = np.array([kkp]*self.N) + np.transpose([nnp]*self.N) #pi*(n+k)
    pnk = np.array([nnp]*self.N) - np.transpose([kkp]*self.N) #pi*(n-k)
    np.fill_diagonal(pkn,None),np.fill_diagonal(pnk,None)
    
    # =============================================================================
    #  Subtidal terms of the equations for salinity
    # =============================================================================
    pars_st = {}
    
    #dimensions: x,k,n
    #k=n is not possible here. 
    pars_st['C1a'] = u_bar/2 
    
    pars_st['C2a'] = u_bar[:,np.newaxis] * (g1[:,np.newaxis]/2 + g2[:,np.newaxis]/6 + g2[:,np.newaxis]/(4*kkp**2))
    pars_st['C2b'] = alf[:,np.newaxis]*self.soc/self.Lsc * (g3[:,np.newaxis]/2 + g4[:,np.newaxis]*(1/6 + 1/(4*kkp**2)) -g5*(1/8 + 3/(8*kkp**2)) )
    pars_st['C2c'] = u_bar[:,np.newaxis,np.newaxis]*g2[:,np.newaxis,np.newaxis]* ( np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2 ) 
    pars_st['C2d'] = self.soc/self.Lsc*alf[:,np.newaxis,np.newaxis] * (g4[:,np.newaxis,np.newaxis]*(np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2) + g5* ((3*np.cos(pkn)-3)/pkn**4 + (3*np.cos(pnk)-3)/pnk**4 - 3/2*np.cos(pkn)/pkn**2 - 3/2*np.cos(pnk)/pnk**2) )
    
    pars_st['C3a'] = 2*u_bar[:,np.newaxis]*g2[:,np.newaxis]/(kkp**2)*np.cos(kkp) 
    pars_st['C3b'] = alf[:,np.newaxis]*self.soc/self.Lsc * ( 2*g4[:,np.newaxis]/kkp**2 * np.cos(kkp) - g5/kkp**4 *(6-6*np.cos(kkp) +3*kkp**2*np.cos(kkp)) )
    
    pars_st['C5a'] = alf[:,np.newaxis]*self.soc/self.Lsc *kkp* ( -9*g5+6*g4[:,np.newaxis]+kkp**2*(-12*g3[:,np.newaxis]-4*g4[:,np.newaxis]+3*g5) ) / (48*kkp**3)
    pars_st['C5b'] = alf[:,np.newaxis]*self.soc*self.bex[:,np.newaxis]**(-1)*kkp * ( -9*g5+6*g4[:,np.newaxis]+kkp**2*(-12*g3[:,np.newaxis]-4*g4[:,np.newaxis]+3*g5) ) / (48*kkp**3)    
    pars_st['C5c'] = alf[:,np.newaxis,np.newaxis]*self.soc/self.Lsc*nnp* ( (3*g5*np.cos(pkn)-3*g5)/pkn**5 + np.cos(pkn)/pkn**3 * (g4[:,np.newaxis,np.newaxis]-1.5*g5) + np.cos(pkn)/pkn * (g5/8 - g4[:,np.newaxis,np.newaxis]/6 - g3[:,np.newaxis,np.newaxis]/2) 
                            +(3*g5*np.cos(pnk)-3*g5)/pnk**5 + np.cos(pnk)/pnk**3 * (g4[:,np.newaxis,np.newaxis]-1.5*g5) + np.cos(pnk)/pnk * (g5/8 - g4[:,np.newaxis,np.newaxis]/6 - g3[:,np.newaxis,np.newaxis]/2))
    pars_st['C5d'] = alf[:,np.newaxis,np.newaxis]*self.soc*nnp*self.bex[:,np.newaxis,np.newaxis]**(-1)*( (3*g5*np.cos(pkn)-3*g5)/pkn**5 + np.cos(pkn)/pkn**3 * (g4[:,np.newaxis,np.newaxis]-1.5*g5) 
                                + np.cos(pkn)/pkn * (g5/8 - g4[:,np.newaxis,np.newaxis]/6 - g3[:,np.newaxis,np.newaxis]/2)
                                +(3*g5*np.cos(pnk)-3*g5)/pnk**5 + np.cos(pnk)/pnk**3 * (g4[:,np.newaxis,np.newaxis]-1.5*g5) + np.cos(pnk)/pnk * (g5/8 - g4[:,np.newaxis,np.newaxis]/6 - g3[:,np.newaxis,np.newaxis]/2))
    pars_st['C5e'] = 0
    pars_st['C5f'] = 0
    
    pars_st['C6a'] = self.Lsc*Kv_st[:,np.newaxis]*kkp**2/(2*self.H[:,np.newaxis]**2)
    
    pars_st['C7a'] = -self.bex**-1*Kh/2 - Kh_x/2
    pars_st['C7b'] = -Kh/(2*self.Lsc)
    
    #in the following terms k=n is possible -  huh why did i say this, no k dependence of course
    pars_st['C10a'] = (u_bar-self.bex**(-1)*Kh-Kh_x)
    pars_st['C10b'] = - Kh / self.Lsc
    pars_st['C10c'] = self.bex[:,np.newaxis]**(-1)*alf[:,np.newaxis]*self.soc * ( 2*g4[:,np.newaxis]/nnp**2*np.cos(nnp) - g5/nnp**4 * (6-6*np.cos(nnp)+3*nnp**2*np.cos(nnp)) )
    pars_st['C10d'] = alf[:,np.newaxis]*self.soc/self.Lsc * ( 2*g4[:,np.newaxis]/nnp**2*np.cos(nnp) - g5/nnp**4 * (6-6*np.cos(nnp)+3*nnp**2*np.cos(nnp)) )
    pars_st['C10e'] = (2*u_bar[:,np.newaxis]*g2[:,np.newaxis]) / nnp**2 * np.cos(nnp) 
    pars_st['C10f'] = alf[:,np.newaxis]*self.soc/self.Lsc * ( 2*g4[:,np.newaxis]/nnp**2*np.cos(nnp) - g5/nnp**4 * (6-6*np.cos(nnp)+3*nnp**2*np.cos(nnp)) )
    pars_st['C10g'] = 0
    
    
    #boundaries of segments - inspired from network code.
    A_1 = self.b[self.di[1:-1]-1] * self.H[self.di[1:-1]-1]
    A0  = self.b[self.di[1:-1]] * self.H[self.di[1:-1]]
    '''
    #vertical - sounds good, does not work 
    pars_st['C12a_1'] = 1*A_1 * 0.5*-Kh[self.di[1:-1]-1]
    pars_st['C12b_1'] = 1*A_1[:,np.newaxis] * alf[self.di[1:-1]-1,np.newaxis] * self.soc * (1/2 * g3[self.di[1:-1]-1,np.newaxis] + (1/6 + 1/(4*kkp**2)) * g4[self.di[1:-1]-1,np.newaxis] + (-1/8 - 3/(8*kkp**2)) * g5 )
    pars_st['C12c_1'] = 1*A_1[:,np.newaxis,np.newaxis] * alf[self.di[1:-1]-1,np.newaxis,np.newaxis] * self.soc * ((np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2) * g4[self.di[1:-1]-1,np.newaxis,np.newaxis] \
                                                            + ((3*np.cos(pkn)-3)/pkn**4 - 3/2*np.cos(pkn)/pkn**2 + (3*np.cos(pnk)-3)/pnk**4 - 3/2*np.cos(pnk)/pnk**2 ) * g5)
    pars_st['C12d_1'] = 1*A_1[:,np.newaxis] * self.Lsc * u_bar[self.di[1:-1]-1,np.newaxis] * (1/2 + 1/2 * g1[self.di[1:-1]-1,np.newaxis] + (1/6 + 1/(4*kkp**2)) * g2[self.di[1:-1]-1,np.newaxis])
    pars_st['C12e_1'] = 1*A_1[:,np.newaxis,np.newaxis] * self.Lsc*u_bar[self.di[1:-1]-1,np.newaxis,np.newaxis] * ((np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2) * g2[self.di[1:-1]-1,np.newaxis,np.newaxis])
    pars_st['C12f_1'] = 1*A_1[:,np.newaxis] * self.Lsc*u_bar[self.di[1:-1]-1,np.newaxis]* 2 * np.cos(kkp)/kkp**2 * g2[self.di[1:-1]-1,np.newaxis]
    pars_st['C12g_1'] = 1*A_1[:,np.newaxis] * alf[self.di[1:-1]-1,np.newaxis]*self.soc * (g4[self.di[1:-1]-1,np.newaxis]*2 * np.cos(kkp)/kkp**2 + g5*((6*np.cos(kkp)-6)/kkp**4 - 3*np.cos(kkp)/kkp**2))

    pars_st['C12a0'] = 1*A0 * 0.5*-Kh[self.di[1:-1]]
    pars_st['C12b0'] = 1*A0[:,np.newaxis] * alf[self.di[1:-1],np.newaxis]*self.soc * (1/2 * g3[self.di[1:-1],np.newaxis] + (1/6 + 1/(4*kkp**2)) * g4[self.di[1:-1],np.newaxis] + (-1/8 - 3/(8*kkp**2)) * g5 )
    pars_st['C12c0'] = 1*A0[:,np.newaxis,np.newaxis] * alf[self.di[1:-1],np.newaxis,np.newaxis] * self.soc * ((np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2) * g4[self.di[1:-1],np.newaxis,np.newaxis] \
                                                            + ((3*np.cos(pkn)-3)/pkn**4 - 3/2*np.cos(pkn)/pkn**2 + (3*np.cos(pnk)-3)/pnk**4 - 3/2*np.cos(pnk)/pnk**2 ) * g5)
    pars_st['C12d0'] = 1*A0[:,np.newaxis] * self.Lsc*u_bar[self.di[1:-1],np.newaxis] * (1/2 + 1/2 * g1[self.di[1:-1],np.newaxis] + (1/6 + 1/(4*kkp**2)) * g2[self.di[1:-1],np.newaxis])
    pars_st['C12e0'] = 1*A0[:,np.newaxis,np.newaxis] * self.Lsc*u_bar[self.di[1:-1],np.newaxis,np.newaxis] * ((np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2) * g2[self.di[1:-1],np.newaxis,np.newaxis])
    pars_st['C12f0'] = 1*A0[:,np.newaxis] * self.Lsc*u_bar[self.di[1:-1],np.newaxis] * 2 * np.cos(kkp)/kkp**2 * g2[self.di[1:-1],np.newaxis]
    pars_st['C12g0'] = 1*A0[:,np.newaxis] * alf[self.di[1:-1],np.newaxis]*self.soc * (g4[self.di[1:-1],np.newaxis]*2 * np.cos(kkp)/kkp**2 + g5*((6*np.cos(kkp)-6)/kkp**4 - 3*np.cos(kkp)/kkp**2))

    pars_st['C12c_1'][np.where(np.isnan(pars_st['C12c_1']))] = 0 
    pars_st['C12c0'][np.where(np.isnan(pars_st['C12c0']))]   = 0 
    pars_st['C12e_1'][np.where(np.isnan(pars_st['C12e_1']))] = 0 
    pars_st['C12e0'][np.where(np.isnan(pars_st['C12e0']))]   = 0    
    '''
    
    # depth-averaged 
    pars_st['C13a_1'] = A_1 * u_bar[self.di[1:-1]-1]
    pars_st['C13b_1'] = A_1[:,np.newaxis] * u_bar[self.di[1:-1]-1,np.newaxis]*2*g2[self.di[1:-1]-1,np.newaxis]*np.cos(nnp)/nnp**2
    pars_st['C13c_1'] = A_1[:,np.newaxis] * alf[self.di[1:-1]-1,np.newaxis]*self.soc/self.Lsc * (2*g4[self.di[1:-1]-1,np.newaxis]*np.cos(nnp)/nnp**2 + g5*(6*np.cos(nnp)-6)/nnp**4 -g5*3*np.cos(nnp)/nnp**2 )
    pars_st['C13d_1'] = A_1 * -Kh[self.di[1:-1]-1]/self.Lsc

    pars_st['C13a0'] = A0 * u_bar[self.di[1:-1]]
    pars_st['C13b0'] = A0[:,np.newaxis] * u_bar[self.di[1:-1],np.newaxis]*2*g2[self.di[1:-1],np.newaxis]*np.cos(nnp)/nnp**2
    pars_st['C13c0'] = A0[:,np.newaxis] * alf[self.di[1:-1],np.newaxis]*self.soc/self.Lsc * (2*g4[self.di[1:-1],np.newaxis]*np.cos(nnp)/nnp**2 + g5*(6*np.cos(nnp)-6)/nnp**4 -g5*3*np.cos(nnp)/nnp**2 )
    pars_st['C13d0'] = A0 * -Kh[self.di[1:-1]]/self.Lsc
    
    #set the contribution to the sum to zero of k=n (the diagonal)
    pars_st['C2c'][np.where(np.isnan(pars_st['C2c']))] = 0 
    pars_st['C2d'][np.where(np.isnan(pars_st['C2d']))] = 0
    pars_st['C5c'][np.where(np.isnan(pars_st['C5c']))] = 0
    pars_st['C5d'][np.where(np.isnan(pars_st['C5d']))] = 0
    #C5f'][np.where(np.isnan(C5f))] = 0
    
    #C13, but for the river side 
    pars_st['C14a'] = u_bar[0]
    pars_st['C14b'] = u_bar[0,np.newaxis]*2*g2[0]*np.cos(nnp)/nnp**2
    pars_st['C14c'] = alf[0]*self.soc/self.Lsc * (2*g4[0]*np.cos(nnp)/nnp**2 + g5*(6*np.cos(nnp)-6)/nnp**4 -g5*3*np.cos(nnp)/nnp**2 )
    pars_st['C14d'] = -Kh[0]/self.Lsc

    
    # =============================================================================
    # new options for matching conditions in the vertical 
    # to solve issues when there is a jump in depth. 
    # =============================================================================

    Hbnd = np.array([(self.Hn[i]+self.Hn[i+1])/2 for i in range(self.ndom-1)])
    
    #vertical viscosity
    if self.choice_viscosityv_st == 'constant': Av_stb = self.Av_st + np.zeros(self.ndom-1) #do nothing, value is specified
    elif self.choice_viscosityv_st == 'cuh': Av_stb = self.cv_st * self.Ut * Hbnd
    else: print('ERROR: no valid op option for choice vertical viscosity subtidal')
    
    #coefficients
    if self.choice_bottomslip_st == 'sf': rrb = Av_st/(self.sf_st*Hbnd)
    elif self.choice_bottomslip_st == 'rr': rrb = self.rr_st + np.zeros(self.ndom-1)
    else: print('ERROR: no valid op option for choice bottomslip subtidal')
    g1b = -1 + (1.5+3*rrb) / (1+ 3 *rrb)
    g2b =  -3 / (2+6*rrb)
    g3b = (1+4*rrb) / (1+3*rrb) * (9+18*rrb) - 8 - 24*rrb
    g4b = -9 * (1+4*rrb) / (1+3*rrb)
    g5b = - 8
    
    u_bar_1b = self.Q/(Hbnd*self.b[self.di[1:-1]-1])
    u_bar0b  = self.Q/(Hbnd*self.b[self.di[1:-1]])
    alfb     = self.g*self.Be*Hbnd**3/(48*Av_stb)
    
    #vertical
    pars_st['C15a_1'] = A_1 * 0.5*-Kh[self.di[1:-1]-1]
    pars_st['C15b_1'] = A_1[:,np.newaxis] * alfb[:,np.newaxis] * self.soc * (1/2 * g3b[:,np.newaxis] + (1/6 + 1/(4*kkp**2)) * g4b[:,np.newaxis] + (-1/8 - 3/(8*kkp**2)) * g5b )
    pars_st['C15c_1'] = A_1[:,np.newaxis,np.newaxis] * alfb[:,np.newaxis,np.newaxis] * self.soc * ((np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2) * g4b[:,np.newaxis,np.newaxis] \
                                                            + ((3*np.cos(pkn)-3)/pkn**4 - 3/2*np.cos(pkn)/pkn**2 + (3*np.cos(pnk)-3)/pnk**4 - 3/2*np.cos(pnk)/pnk**2 ) * g5b)
    pars_st['C15d_1'] = A_1[:,np.newaxis] * self.Lsc * u_bar_1b[:,np.newaxis] * (1/2 + 1/2 * g1b[:,np.newaxis] + (1/6 + 1/(4*kkp**2)) * g2b[:,np.newaxis])
    pars_st['C15e_1'] = A_1[:,np.newaxis,np.newaxis] * self.Lsc*u_bar_1b[:,np.newaxis,np.newaxis] * ((np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2) * g2b[:,np.newaxis,np.newaxis])

    
    pars_st['C15a0'] = A0 * 0.5*-Kh[self.di[1:-1]]
    pars_st['C15b0'] = A0[:,np.newaxis] * alfb[:,np.newaxis]*self.soc * (1/2 * g3b[:,np.newaxis] + (1/6 + 1/(4*kkp**2)) * g4b[:,np.newaxis] + (-1/8 - 3/(8*kkp**2)) * g5b )
    pars_st['C15c0'] = A0[:,np.newaxis,np.newaxis] * alfb[:,np.newaxis,np.newaxis] * self.soc * ((np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2) * g4b[:,np.newaxis,np.newaxis] \
                                                            + ((3*np.cos(pkn)-3)/pkn**4 - 3/2*np.cos(pkn)/pkn**2 + (3*np.cos(pnk)-3)/pnk**4 - 3/2*np.cos(pnk)/pnk**2 ) * g5b)
    pars_st['C15d0'] = A0[:,np.newaxis] * self.Lsc*u_bar0b[:,np.newaxis] * (1/2 + 1/2 * g1b[:,np.newaxis] + (1/6 + 1/(4*kkp**2)) * g2b[:,np.newaxis])
    pars_st['C15e0'] = A0[:,np.newaxis,np.newaxis] * self.Lsc*u_bar0b[:,np.newaxis,np.newaxis] * ((np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2) * g2b[:,np.newaxis,np.newaxis])

    pars_st['C15c_1'][np.where(np.isnan(pars_st['C15c_1']))] = 0 
    pars_st['C15c0'][np.where(np.isnan(pars_st['C15c0']))]   = 0 
    pars_st['C15e_1'][np.where(np.isnan(pars_st['C15e_1']))] = 0 
    pars_st['C15e0'][np.where(np.isnan(pars_st['C15e0']))]   = 0    

    
    return pars_st
    


def solu_subtidal(self, ans, pars_st, indi, version):
    # =============================================================================
    # contribution to the solution vector by the subtidal dynamics
    # =============================================================================   
    so = np.zeros(self.di3[-1]*self.M)
    


    # =============================================================================
    # contribution to solution vector due to average salt balance
    # =============================================================================
    so[indi['xn_m']] += pars_st['C10a'][indi['x']]*(ans[indi['xnp_m']]-ans[indi['xnm_m']])/(2*self.dl[indi['x']]) + pars_st['C10b'][indi['x']]*(ans[indi['xnp_m']] - 2*ans[indi['xn_m']] + ans[indi['xnm_m']])/(self.dl[indi['x']]**2) 

    if version in ['A','C','D']:
        so[indi['xn_m']] += ((ans[indi['xnp_m']]-ans[indi['xnm_m']])/(2*self.dl[indi['x']]) * np.sum([pars_st['C10c'][indi['x'],n-1]*ans[indi['xn_m']+n] for n in range(1,self.M)],0) 
                            + (ans[indi['xnp_m']] - 2*ans[indi['xn_m']] + ans[indi['xnm_m']])/(self.dl[indi['x']]**2) * np.sum([pars_st['C10d'][indi['x'],n-1]*ans[indi['xn_m']+n] for n in range(1,self.M)],0) 
                            + np.sum([pars_st['C10e'][indi['x'],n-1]*(ans[indi['xnp_m']+n]-ans[indi['xnm_m']+n])/(2*self.dl[indi['x']]) for n in range(1,self.M)],0) 
                            + (ans[indi['xnp_m']]-ans[indi['xnm_m']])/(2*self.dl[indi['x']]) * np.sum([pars_st['C10f'][indi['x'],n-1]*(ans[indi['xnp_m']+n]-ans[indi['xnm_m']+n])/(2*self.dl[indi['x']]) for n in range(1,self.M)],0)
                            + np.sum([pars_st['C10g']*ans[indi['xn_m']+n] for n in range(1,self.M)],0) )
    
    # =============================================================================
    # contribution to solution vector due to vertical salt balance
    # =============================================================================
    if version in ['D']: #contribution to solution vector due to term T1
        so[indi['xnr_mj']] = so[indi['xnr_mj']] + pars_st['C1a'][indi['xr']] * ((ans[indi['xnrp_mj']]-ans[indi['xnrm_mj']])/(2*self.dl[indi['xr']]))

    if version in ['D']: #contribution to solution vector due to term T2
        so[indi['xnr_mj']] = so[indi['xnr_mj']] + (pars_st['C2a'][indi['xr'],indi['j1']]*(ans[indi['xnrp_mj']]-ans[indi['xnrm_mj']])/(2*self.dl[indi['xr']])
                                + pars_st['C2b'][indi['xr'],indi['j1']]*((ans[indi['xnrp_mj']]-ans[indi['xnrm_mj']])/(2*self.dl[indi['xr']]))*((ans[indi['xnrp_m']]-ans[indi['xnrm_m']])/(2*self.dl[indi['xr']]))
                                +np.sum([pars_st['C2c'][indi['xr'],indi['j1'],n-1] * (ans[indi['xnrp_m']+n]-ans[indi['xnrm_m']+n]) for n in range(1,self.M)],0)/(2*self.dl[indi['xr']])
                                +np.sum([pars_st['C2d'][indi['xr'],indi['j1'],n-1] * (ans[indi['xnrp_m']+n]-ans[indi['xnrm_m']+n]) for n in range(1,self.M)],0)/(2*self.dl[indi['xr']])
                                *(ans[indi['xnrp_m']]-ans[indi['xnrm_m']])/(2*self.dl[indi['xr']]) )

    if version in ['A','C','D']: #contribution to solution vector due to term T3
        so[indi['xnr_mj']] = so[indi['xnr_mj']] + pars_st['C3a'][indi['xr'],indi['j1']] * (ans[indi['xnrp_m']]-ans[indi['xnrm_m']])/(2*self.dl[indi['xr']]) + pars_st['C3b'][indi['xr'],indi['j1']] * ((ans[indi['xnrp_m']]-ans[indi['xnrm_m']])/(2*self.dl[indi['xr']]))**2

    if version in ['D']: #contribution to jacobian due to term T5
        so[indi['xnr_mj']] = so[indi['xnr_mj']] + (pars_st['C5a'][indi['xr'],indi['j1']]*(ans[indi['xnrp_m']] - 2*ans[indi['xnr_m']] + ans[indi['xnrm_m']])/(self.dl[indi['xr']]**2)*ans[indi['xnr_mj']] 
                                 + pars_st['C5b'][indi['xr'],indi['j1']]*(ans[indi['xnrp_m']]-ans[indi['xnrm_m']])/(2*self.dl[indi['xr']]) * ans[indi['xnr_mj']]  
                                   +np.sum([pars_st['C5c'][indi['xr'],indi['j1'],n-1]*ans[indi['xnr_m']+n] for n in range(1,self.M)],0) * (ans[indi['xnrp_m']] - 2*ans[indi['xnr_m']] + ans[indi['xnrm_m']])/(self.dl[indi['xr']]**2) 
                                   +np.sum([pars_st['C5d'][indi['xr'],indi['j1'],n-1]*ans[indi['xnr_m']+n] for n in range(1,self.M)],0) * (ans[indi['xnrp_m']]-ans[indi['xnrm_m']])/(2*self.dl[indi['xr']])
                                   + pars_st['C5e']*ans[indi['xnr_mj']] + np.sum([pars_st['C5f']*ans[indi['xnr_m']+n] for n in range(1,self.M)],0))

    if version in ['A','C','D']: #contribution to solution vector due to term T6
        so[indi['xnr_mj']] = so[indi['xnr_mj']] + pars_st['C6a'][indi['xr'],indi['j1']] * ans[indi['xnr_mj']]

    if version in ['D']: 
        so[indi['xnr_mj']] = so[indi['xnr_mj']] + pars_st['C7a'][indi['xr']]*(ans[indi['xnrp_mj']]-ans[indi['xnrm_mj']])/(2*self.dl[indi['xr']]) + pars_st['C7b'][indi['xr']]*(ans[indi['xnrp_mj']] - 2*ans[indi['xnr_mj']] + ans[indi['xnrm_mj']])/(self.dl[indi['xr']]**2) 
 
    
    if version in ['B']:    #in this case, there is no stratification, so this all has to be zero
        so[indi['xnr_mj']] = ans[indi['xnr_mj']]
    
    return so


def jaco_subtidal(self, ans, pars_st, indi, version):
    
    
    jac = np.zeros((self.di3[-1]*self.M,self.di3[-1]*self.M))
    

    # =============================================================================
    # contribution to solution vector due to depth-averaged balance
    # ============================================================================= 
    #left
    jac[indi['xn_m'], indi['xnm_m']] += - pars_st['C10a'][indi['x']]/(2*self.dl[indi['x']]) + pars_st['C10b'][indi['x']]/(self.dl[indi['x']]**2) 
    #center
    jac[indi['xn_m'],indi['xn_m']] += - 2/(self.dl[indi['x']]**2)*pars_st['C10b'][indi['x']] 
    #right
    jac[indi['xn_m'], indi['xnp_m']] +=  pars_st['C10a'][indi['x']]/(2*self.dl[indi['x']]) + pars_st['C10b'][indi['x']]/(self.dl[indi['x']]**2) 

    if version in ['A','C','D']: #contribution to jacobian due to term T6
        jac[indi['xn_m'], indi['xnm_m']] += - 1/(2*self.dl[indi['x']])*np.sum([pars_st['C10c'][indi['x'],n-1]*ans[indi['xn_m']+n] for n in range(1,self.M)],0) \
                                       + 1/(self.dl[indi['x']]**2)*np.sum([pars_st['C10d'][indi['x'],n-1]*ans[indi['xn_m']+n] for n in range(1,self.M)],0) \
                                       - 1/(2*self.dl[indi['x']])*np.sum([pars_st['C10f'][indi['x'],n-1]*(ans[indi['xnp_m']+n]-ans[indi['xnm_m']+n])/(2*self.dl[indi['x']]) for n in range(1,self.M)],0) 
        jac[indi['xnr_m'], indi['xnrm_mj']] +=  - pars_st['C10e'][indi['xr'],indi['j1']]/(2*self.dl[indi['xr']]) - pars_st['C10f'][indi['xr'],indi['j1']]/(2*self.dl[indi['xr']])*(ans[indi['xnrp_m']]-ans[indi['xnrm_m']])/(2*self.dl[indi['xr']])
        #center
        jac[indi['xn_m'],indi['xn_m']] += - 2/(self.dl[indi['x']]**2)*np.sum([pars_st['C10d'][indi['x'],n-1]*ans[indi['xn_m']+n] for n in range(1,self.M)],0)
        jac[indi['xnr_m'], indi['xnr_mj']] += (ans[indi['xnrp_m']]-ans[indi['xnrm_m']])/(2*self.dl[indi['xr']]) * pars_st['C10c'][indi['xr'],indi['j1']] \
                                        + (ans[indi['xnrp_m']]-2*ans[indi['xnr_m']]+ans[indi['xnrm_m']])/(self.dl[indi['xr']]**2) * pars_st['C10d'][indi['xr'],indi['j1']]
        #right
        jac[indi['xn_m'], indi['xnp_m']] +=  1/(2*self.dl[indi['x']])*np.sum([pars_st['C10c'][indi['x'],n-1]*ans[indi['xn_m']+n] for n in range(1,self.M)],0) \
                                       + 1/(self.dl[indi['x']]**2)*np.sum([pars_st['C10d'][indi['x'],n-1]*ans[indi['xn_m']+n] for n in range(1,self.M)],0) \
                                       + 1/(2*self.dl[indi['x']])*np.sum([pars_st['C10f'][indi['x'],n-1]*(ans[indi['xnp_m']+n]-ans[indi['xnm_m']+n])/(2*self.dl[indi['x']]) for n in range(1,self.M)],0) 
        jac[indi['xnr_m'], indi['xnrp_mj']] +=  pars_st['C10e'][indi['xr'],indi['j1']]/(2*self.dl[indi['xr']]) + pars_st['C10f'][indi['xr'],indi['j1']]/(2*self.dl[indi['xr']])*(ans[indi['xnrp_m']]-ans[indi['xnrm_m']])/(2*self.dl[indi['xr']])
    
    
    # =============================================================================
    # contribution to solution vector due to vertical salt balance
    # =============================================================================
    if version in ['D'] : #contribution to jacobian due to term T1
        jac[indi['xnr_mj'],indi['xnrm_mj']] = jac[indi['xnr_mj'],indi['xnrm_mj']] - pars_st['C1a'][indi['xr']]/(2*self.dl[indi['xr']]) 
        jac[indi['xnr_mj'],indi['xnrp_mj']] = jac[indi['xnr_mj'],indi['xnrp_mj']] + pars_st['C1a'][indi['xr']]/(2*self.dl[indi['xr']]) 
          

    if version in ['D']: #contribution to jacobian due to term T2
        jac[indi['xnr_mj'],indi['xnrm_mj']] = jac[indi['xnr_mj'],indi['xnrm_mj']] - pars_st['C2a'][indi['xr'],indi['j1']]/(2*self.dl[indi['xr']]) - pars_st['C2b'][indi['xr'],indi['j1']]/(2*self.dl[indi['xr']]) * (ans[indi['xnrp_m']]-ans[indi['xnrm_m']])/(2*self.dl[indi['xr']])
        jac[indi['xnr_mj'],indi['xnrp_mj']] = jac[indi['xnr_mj'],indi['xnrp_mj']] + pars_st['C2a'][indi['xr'],indi['j1']]/(2*self.dl[indi['xr']]) + pars_st['C2b'][indi['xr'],indi['j1']]/(2*self.dl[indi['xr']]) * (ans[indi['xnrp_m']]-ans[indi['xnrm_m']])/(2*self.dl[indi['xr']])
        jac[indi['xnr_mj'],indi['xnrm_m']] = (jac[indi['xnr_mj'],indi['xnrm_m']] - pars_st['C2b'][indi['xr'],indi['j1']]/(2*self.dl[indi['xr']]) * (ans[indi['xnrp_mj']]-ans[indi['xnrm_mj']])/(2*self.dl[indi['xr']]) 
                       - np.sum([pars_st['C2d'][indi['xr'],indi['j1'],n-1] * (ans[indi['xnrp_m']+n]-ans[indi['xnrm_m']+n]) for n in range(1,self.M)],0)/(4*self.dl[indi['xr']]**2) )
        jac[indi['xnr_mj'],indi['xnrp_m']] = (jac[indi['xnr_mj'],indi['xnrp_m']]  + pars_st['C2b'][indi['xr'],indi['j1']]/(2*self.dl[indi['xr']]) * (ans[indi['xnrp_mj']]-ans[indi['xnrm_mj']])/(2*self.dl[indi['xr']]) 
                                           + np.sum([pars_st['C2d'][indi['xr'],indi['j1'],n-1] * (ans[indi['xnrp_m']+n]-ans[indi['xnrm_m']+n]) for n in range(1,self.M)],0)/(4*self.dl[indi['xr']]**2) )
        jac[indi['xnr_mj2'],indi['xnrm_mk']] = jac[indi['xnr_mj2'],indi['xnrm_mk']] - pars_st['C2c'][indi['xr2'],indi['j12'],indi['k1']]/(2*self.dl[indi['xr2']]) - pars_st['C2d'][indi['xr2'],indi['j12'],indi['k1']]*(ans[indi['xnrp_m2']]-ans[indi['xnrm_m2']])/(4*self.dl[indi['xr2']]**2) 
        jac[indi['xnr_mj2'],indi['xnrp_mk']] = jac[indi['xnr_mj2'],indi['xnrp_mk']] + pars_st['C2c'][indi['xr2'],indi['j12'],indi['k1']]/(2*self.dl[indi['xr2']]) + pars_st['C2d'][indi['xr2'],indi['j12'],indi['k1']]*(ans[indi['xnrp_m2']]-ans[indi['xnrm_m2']])/(4*self.dl[indi['xr2']]**2) 
    
    if version in ['A','C','D']: #contribution to jacobian due to term T3
        jac[indi['xnr_mj'],indi['xnrm_m']] = jac[indi['xnr_mj'],indi['xnrm_m']] - pars_st['C3a'][indi['xr'],indi['j1']]/(2*self.dl[indi['xr']]) - pars_st['C3b'][indi['xr'],indi['j1']]/self.dl[indi['xr']] * (ans[indi['xnrp_m']]-ans[indi['xnrm_m']])/(2*self.dl[indi['xr']])
        jac[indi['xnr_mj'],indi['xnrp_m']] = jac[indi['xnr_mj'],indi['xnrp_m']] + pars_st['C3a'][indi['xr'],indi['j1']]/(2*self.dl[indi['xr']]) + pars_st['C3b'][indi['xr'],indi['j1']]/self.dl[indi['xr']] * (ans[indi['xnrp_m']]-ans[indi['xnrm_m']])/(2*self.dl[indi['xr']])


    if version in ['D']: #contribution to jacobian due to term T5
        jac[indi['xnr_mj'], indi['xnrm_m']] = (jac[indi['xnr_mj'], indi['xnrm_m']] + pars_st['C5a'][indi['xr'],indi['j1']]*ans[indi['xnr_mj']]/(self.dl[indi['xr']]**2) - pars_st['C5b'][indi['xr'],indi['j1']]/(2*self.dl[indi['xr']])*ans[indi['xnr_mj']]
                               + np.sum([ans[indi['xnr_m']+n]*pars_st['C5c'][indi['xr'],indi['j1'],n-1]/(self.dl[indi['xr']]**2) for n in range(1,self.M)],0) 
                               - np.sum([ans[indi['xnr_m']+n]*pars_st['C5d'][indi['xr'],indi['j1'],n-1]/(2*self.dl[indi['xr']]) for n in range(1,self.M)],0) )
        jac[indi['xnr_mj'], indi['xnr_m']] = jac[indi['xnr_mj'], indi['xnr_m']] - 2*pars_st['C5a'][indi['xr'],indi['j1']]*ans[indi['xnr_mj']]/(self.dl[indi['xr']]**2) -2* np.sum([ans[indi['xnr_m']+n]*pars_st['C5c'][indi['xr'],indi['j1'],n-1]/(self.dl[indi['xr']]**2) for n in range(1,self.M)],0)
        jac[indi['xnr_mj'], indi['xnrp_m']] = (jac[indi['xnr_mj'], indi['xnrp_m']] + pars_st['C5a'][indi['xr'],indi['j1']]*ans[indi['xnr_mj']]/(self.dl[indi['xr']]**2) + pars_st['C5b'][indi['xr'],indi['j1']]/(2*self.dl[indi['xr']])*ans[indi['xnr_mj']]
                                       + np.sum([ans[indi['xnr_m']+n]*pars_st['C5c'][indi['xr'],indi['j1'],n-1]/(self.dl[indi['xr']]**2) for n in range(1,self.M)],0)
                                       + np.sum([ans[indi['xnr_m']+n]*pars_st['C5d'][indi['xr'],indi['j1'],n-1]/(2*self.dl[indi['xr']]) for n in range(1,self.M)],0) )
        jac[indi['xnr_mj'],indi['xnr_mj']] = (jac[indi['xnr_mj'], indi['xnr_mj']] + pars_st['C5a'][indi['xr'],indi['j1']]*(ans[indi['xnrp_m']]-2*ans[indi['xnr_m']]+ans[indi['xnrm_m']])/(self.dl[indi['xr']]**2) 
                                         + pars_st['C5b'][indi['xr'],indi['j1']]*(ans[indi['xnrp_m']]-ans[indi['xnrm_m']])/(2*self.dl[indi['xr']]) + pars_st['C5e'] )
        jac[indi['xnr_mj2'], indi['xnr_mk']] = (jac[indi['xnr_mj2'], indi['xnr_mk']] + pars_st['C5c'][indi['xr2'],indi['j12'],indi['k1']]*(ans[indi['xnrp_m2']]-2*ans[indi['xnr_m2']]+ans[indi['xnrm_m2']])/(self.dl[indi['xr2']]**2)
                                             + pars_st['C5d'][indi['xr2'],indi['j12'],indi['k1']]*(ans[indi['xnrp_m2']]-ans[indi['xnrm_m2']])/(2*self.dl[indi['xr2']]) +pars_st['C5f'] )
    

    if version in ['A','C','D']: #contribution to jacobian due to term T6
        jac[indi['xnr_mj'],indi['xnr_mj']] = jac[indi['xnr_mj'],indi['xnr_mj']] + pars_st['C6a'][indi['xr'],indi['j1']]

    if version in ['D']: 
        jac[indi['xnr_mj'],indi['xnrm_mj']] = jac[indi['xnr_mj'],indi['xnrm_mj']] - pars_st['C7a'][indi['xr']]/(2*self.dl[indi['xr']]) + pars_st['C7b'][indi['xr']] / (self.dl[indi['xr']]**2)
        jac[indi['xnr_mj'],indi['xnr_mj']] = jac[indi['xnr_mj'],indi['xnr_mj']] - 2*pars_st['C7b'][indi['xr']]/(self.dl[indi['xr']]**2)
        jac[indi['xnr_mj'],indi['xnrp_mj']] = jac[indi['xnr_mj'],indi['xnrp_mj']] + pars_st['C7a'][indi['xr']]/(2*self.dl[indi['xr']]) + pars_st['C7b'][indi['xr']] / (self.dl[indi['xr']]**2)


    if version in ['B']:    #in this case, there is no stratification, so this all has to be zero
        jac[indi['xnr_mj'],indi['xnr_mj']] = 1

    
    return jac
    
    
    
    
    
    
    
    
    







