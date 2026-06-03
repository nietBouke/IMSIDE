# =============================================================================
# The equations to be solved in the inside of the canal
# For IMSIDE_CANAL
# =============================================================================

import numpy as np

#values constants
g =9.81 #gravity
Be=7.6e-4 #isohaline contractie

def build_indices(self):
    # =============================================================================
    # function to build the indices, for the definition of the solution vector and Jacobian. 
    # too hard to explain every variable. Just to put the equations in the right place in the matrices
    # =============================================================================

    inds = {}

    #remove the boundary points for new variable 
    inds['di2'] = np.zeros((len(self.di)-2)*2)
    for i in range(1,len(self.di)-1):
        inds['di2'][i*2-2] = self.di[i]-1
        inds['di2'][i*2-1] = self.di[i]
    inds['di2'] = np.array(inds['di2'], dtype=int)
    
    #build the lists of indices for the different places in the Jacobian
    inds['x'] = np.delete(np.arange(self.di[-1]),inds['di2'])[1:-1] # x coordinates for the points which are not on a aboundary
    inds['xr'] = inds['x'].repeat(self.N) #x for N values, mostly i in old code
    inds['xr_m'] = inds['xr']*self.M #M*i in old code, diagonal 
    inds['xrm_m'] = (inds['xr']-1)*self.M #M*(i-1) in old code, left of the boundary
    inds['xrp_m'] = (inds['xr']+1)*self.M #M*(i+1) in old code, right of the boundary
    inds['xr_mj'] = inds['xr_m']+np.tile(np.arange(1,self.M),self.di[-1]-2-(len(self.di[1:-1])*2)) #M*i+j in old code, diagonal fourier coefficients
    inds['xrp_mj'] = inds['xrp_m']+np.tile(np.arange(1,self.M),self.di[-1]-2-(len(self.di[1:-1])*2)) #M*(i-1)+j in old code, left fourier coefficients
    inds['xrm_mj'] = inds['xrm_m']+np.tile(np.arange(1,self.M),self.di[-1]-2-(len(self.di[1:-1])*2)) #M*(i+1)+j in old code, right fourier coefficients
    inds['j1'] = np.tile(np.arange(self.N),self.di[-1]-2-(len(self.di[1:-1])*2)) #j in old code
    
    #for the sums in the jacobian we need to repeat some arrays
    inds['xr_mj2'] = np.repeat(inds['xr_m'],self.N)+np.tile(np.arange(1,self.M),(self.di[-1]-2-(len(self.di[1:-1])*2))).repeat(self.N)
    inds['xr2'] = inds['xr'].repeat(self.N)
    inds['j12'] = inds['j1'].repeat(self.N)
    inds['xr_m2'] = inds['xr_m'].repeat(self.N)
    inds['xrp_m2'] = inds['xrp_m'].repeat(self.N)
    inds['xrm_m2'] = inds['xrm_m'].repeat(self.N)
    
    #and repeat of the repetition
    inds['k1'] = np.tile(inds['j1'],self.N)
    inds['xr_mk']=np.repeat(inds['xr_m'],self.N)+np.tile(np.tile(np.arange(1,self.M),(self.di[-1]-2-(len(self.di[1:-1])*2))),self.N)
    inds['xrp_mk']=np.repeat(inds['xrp_m'],self.N)+np.tile(np.tile(np.arange(1,self.M),(self.di[-1]-2-(len(self.di[1:-1])*2))),self.N)
    inds['xrm_mk']=np.repeat(inds['xrm_m'],self.N)+np.tile(np.tile(np.arange(1,self.M),(self.di[-1]-2-(len(self.di[1:-1])*2))),self.N)
    #the same story, but for the averag salt equation, which requires less repetition
    inds['x_m'] = inds['x']*self.M
    inds['xp_m'] = (inds['x']+1)*self.M
    inds['xm_m'] = (inds['x']-1)*self.M
    
    #for the boundaries 
    inds['bnd'] = (self.di[np.arange(1,len(self.nxn))]*self.M+np.arange(self.M)[:,np.newaxis]).flatten()
    inds['bnd_dl'] = np.repeat(np.arange(1,len(self.nxn)),self.M) #not used yet
    
    
    self.inds = inds
    
    return

def build_coeffs(self):
    # =============================================================================
    # this function builds the coefficients of the different terms in the salt transport equations
    # =============================================================================

    # =============================================================================
    # first convert some physical quantities 
    # =============================================================================
     
    #build parameters which will be used later
    u_bar = self.Q/(self.H*self.b) #depth-averaged velocity
    #Kh = ch*Ut[:,np.newaxis]*b #horizontal diffusion coefficient
    #Av = cv*Ut[:,np.newaxis]*H #vertical viscosity coefficient
    #Kv = Av/Sc #vertical diffusion coefficient
    alf = g*Be*self.H**3/(48*self.Av) #parameter determining the strength of the density driven current
    
    #Partial slip coefficients - from momentum equation solution
    rnow = self.Av/(self.sf*self.H)
    
    g1 = -1 + (1.5+3*rnow) / (1+ 3 *rnow)
    g2 =  -3 / (2+6*rnow)
    g3 = (1+4*rnow) / (1+3*rnow) * (9+18*rnow) - 8 - 24*rnow
    g4 = -9 * (1+4*rnow) / (1+3*rnow)
    g5 = - 8
    #below are required when wind is taken into account 
    # g6 = 4+4*Av/(sf*H) -12*(0.5+Av/(sf*H))**2/(1+3*Av/(sf*H))
    # g7 = 4
    # g8 = (3+6*Av/(sf*H)) / (1+3*Av/(sf*H))
    #g1,g2,g3,g4,g5 = 1/2 , -3/2 , 1, -9 ,-8 #for no slip
    
    #derivative of Kh
    Kh_x = np.zeros((self.T,self.di[-1]))
    for t3 in range(self.T):
        temp = np.concatenate([[0],(self.Kh[t3,2:]-self.Kh[t3,:-2])/(2*self.DX[1:-1]),[0]])
        temp[[0,-1]] , temp[self.di[1:-1]] , temp[self.di[1:-1]-1] = None , None , None
        Kh_x[t3] = temp

    #multiplications of pi, in vector and matrix form
    kkp = np.linspace(1,self.N,self.N)*np.pi #k*pi
    nnp = np.linspace(1,self.N,self.N)*np.pi #n*pi
    pkn = np.array([kkp]*self.N) + np.transpose([nnp]*self.N) #pi*(n+k)
    pnk = np.array([nnp]*self.N) - np.transpose([kkp]*self.N) #pi*(n-k)
    np.fill_diagonal(pkn,None),np.fill_diagonal(pnk,None)
    

    # =============================================================================
    # then build the coefficients 
    # for the definitions of the coefficients: I have somewhere a pdf file. 
    # not super fast but it is supposed to be short with respect to the total simulation time
    # =============================================================================
    pphys = {}

    #term 1
    pphys['C1a'] = np.zeros((self.T,self.di[-1]))
    for t3 in range(self.T): 
        pphys['C1a'][t3] = u_bar[t3]/2
        
    #term 2
    pphys['C2a'], pphys['C2b'], pphys['C2c'], pphys['C2d'] = np.zeros((self.T, self.di[-1], self.N)) , np.zeros((self.T, self.di[-1], self.N)) , \
        np.zeros((self.T, self.di[-1], self.N, self.N)) , np.zeros((self.T, self.di[-1], self.N, self.N))
    for t3 in range(self.T):
        pphys['C2a'][t3] = u_bar[t3,:,np.newaxis] * (g1[t3,:,np.newaxis]/2 + g2[t3,:,np.newaxis]/6 + g2[t3,:,np.newaxis]/(4*kkp**2)) 
        pphys['C2b'][t3] = alf[t3,:, np.newaxis]*self.soc_sca/self.Lsc * (g3[t3,:,np.newaxis]/2 + g4[t3,:,np.newaxis]*(1/6 + 1/(4*kkp**2)) -g5*(1/8 + 3/(8*kkp**2)) )
        pphys['C2c'][t3] = u_bar[t3,:,np.newaxis, np.newaxis] * g2[t3,:,np.newaxis,np.newaxis]* ( np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2 )  
        pphys['C2d'][t3] = alf[t3,:, np.newaxis, np.newaxis] * self.soc_sca/self.Lsc * (g4[t3,:,np.newaxis,np.newaxis]*(np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2) + g5* ((3*np.cos(pkn)-3)/pkn**4 + (3*np.cos(pnk)-3)/pnk**4 - 3/2*np.cos(pkn)/pkn**2 - 3/2*np.cos(pnk)/pnk**2) )     
    pphys['C2c'][np.where(np.isnan(pphys['C2c']))] = 0 
    pphys['C2d'][np.where(np.isnan(pphys['C2d']))] = 0

    #term 3
    pphys['C3a'], pphys['C3b'] = np.zeros((self.T, self.di[-1], self.N)) , np.zeros((self.T, self.di[-1], self.N))
    for t3 in range(self.T):
        pphys['C3a'][t3] = u_bar[t3,:,np.newaxis] *2*g2[t3,:,np.newaxis]/(kkp**2)*np.cos(kkp)
        pphys['C3b'][t3] = alf[t3,:, np.newaxis] *self.soc_sca/self.Lsc * ( 2*g4[t3,:,np.newaxis]/kkp**2 * np.cos(kkp) - g5/kkp**4 *(6-6*np.cos(kkp) +3*kkp**2*np.cos(kkp)) )
    
    #term 4 does not contribute to salt transport
    
    #term 5
    pphys['C5a'], pphys['C5b'], pphys['C5c'], pphys['C5d'], pphys['C5e'], pphys['C5f'] = np.zeros((self.T, self.di[-1], self.N)) , np.zeros((self.T, self.di[-1], self.N)) , \
        np.zeros((self.T, self.di[-1], self.N, self.N)) , np.zeros((self.T, self.di[-1], self.N, self.N)) , np.zeros((self.T,self.di[-1],self.N)), np.zeros((self.T,self.di[-1],self.N,self.N))
    for t3 in range(self.T):
        pphys['C5a'][t3] = alf[t3,:,np.newaxis] *self.soc_sca/self.Lsc * nnp* ( -9*g5+6*g4[t3,:,np.newaxis]+nnp**2*(-12*g3[t3,:,np.newaxis]-4*g4[t3,:,np.newaxis]+3*g5) ) / (48*nnp**3)
        pphys['C5b'][t3] = alf[t3,:,np.newaxis] *self.soc_sca*(self.dbdx/self.b)[:,np.newaxis] * nnp* ( -9*g5+6*g4[t3,:,np.newaxis]+nnp**2*(-12*g3[t3,:,np.newaxis]-4*g4[t3,:,np.newaxis]+3*g5) ) / (48*nnp**3)
        pphys['C5c'][t3] = alf[t3,:,np.newaxis,np.newaxis] *self.soc_sca/self.Lsc*nnp* ( (3*g5*np.cos(pkn)-3*g5)/pkn**5 + np.cos(pkn)/pkn**3 * (g4[t3,:,np.newaxis,np.newaxis]-1.5*g5) 
                        + np.cos(pkn)/pkn * (g5/8 - g4[t3,:,np.newaxis,np.newaxis]/6 - g3[t3,:,np.newaxis,np.newaxis]/2) 
                        +(3*g5*np.cos(pnk)-3*g5)/pnk**5 + np.cos(pnk)/pnk**3 * (g4[t3,:,np.newaxis,np.newaxis]-1.5*g5) + np.cos(pnk)/pnk * (g5/8 - g4[t3,:,np.newaxis,np.newaxis]/6 - g3[t3,:,np.newaxis,np.newaxis]/2))
        pphys['C5d'][t3] = alf[t3,:,np.newaxis,np.newaxis] *self.soc_sca*(self.dbdx/self.b)[:,np.newaxis,np.newaxis]*nnp*( (3*g5*np.cos(pkn)-3*g5)/pkn**5 + np.cos(pkn)/pkn**3 * (g4[t3,:,np.newaxis,np.newaxis]-1.5*g5) 
                        + np.cos(pkn)/pkn * (g5/8 - g4[t3,:,np.newaxis,np.newaxis]/6 - g3[t3,:,np.newaxis,np.newaxis]/2) 
                        +(3*g5*np.cos(pnk)-3*g5)/pnk**5 + np.cos(pnk)/pnk**3 * (g4[t3,:,np.newaxis,np.newaxis]-1.5*g5) + np.cos(pnk)/pnk * (g5/8 - g4[t3,:,np.newaxis,np.newaxis]/6 - g3[t3,:,np.newaxis,np.newaxis]/2))
        pphys['C5e'][t3] = 0 #wind effects are turned off
        pphys['C5f'][t3] = 0 #wind effects are turned off
    pphys['C5c'][np.where(np.isnan(pphys['C5c']))] = 0 
    pphys['C5d'][np.where(np.isnan(pphys['C5d']))] = 0
    pphys['C5f'][np.where(np.isnan(pphys['C5f']))] = 0
    
    #term 6
    pphys['C6a'] = np.zeros((self.T, self.di[-1], self.N))
    for t3 in range(self.T):
        pphys['C6a'][t3] = self.Kv[t3, :, np.newaxis]*self.Lsc*kkp**2/(2*self.H[:,np.newaxis]**2)

    #term 7
    pphys['C7a'], pphys['C7b'] = np.zeros((self.T,self.di[-1])) , np.zeros((self.T,self.di[-1]))
    for t3 in range(self.T):
        pphys['C7a'][t3] = -(self.dbdx/self.b) * self.Kh[t3] /2 - Kh_x[t3] /2
        pphys['C7b'][t3] = -self.Kh[t3]/(2*self.Lsc)
        
        
    #equation for depth=averaged salt 
    pphys['C10a'], pphys['C10b'], pphys['C10c'], pphys['C10d'], pphys['C10e'], pphys['C10f'], pphys['C10g'] = np.zeros((self.T,self.di[-1])) , np.zeros((self.T,self.di[-1])) , np.zeros((self.T,self.di[-1],self.N)), np.zeros((self.T,self.di[-1],self.N)), np.zeros((self.T,self.di[-1],self.N)), np.zeros((self.T,self.di[-1],self.N)) , np.zeros((self.T,self.di[-1],self.N))
    for t3 in range(self.T):
        pphys['C10a'][t3] = (u_bar[t3]-(self.dbdx/self.b)*self.Kh[t3]-Kh_x[t3])
        pphys['C10b'][t3] = -self.Kh[t3]/self.Lsc
        pphys['C10c'][t3] = (self.dbdx/self.b)[:,np.newaxis]*alf[t3,:, np.newaxis] *self.soc_sca * ( 2*g4[t3,:,np.newaxis]/nnp**2*np.cos(nnp) - g5/nnp**4 * (6-6*np.cos(nnp)+3*nnp**2*np.cos(nnp)) )
        pphys['C10d'][t3] = alf[t3, :, np.newaxis] *self.soc_sca/self.Lsc * ( 2*g4[t3,:,np.newaxis]/nnp**2*np.cos(nnp) - g5/nnp**4 * (6-6*np.cos(nnp)+3*nnp**2*np.cos(nnp)) )
        pphys['C10e'][t3] = u_bar[t3,:, np.newaxis] *2*g2[t3,:,np.newaxis] / nnp**2 * np.cos(nnp) 
        pphys['C10f'][t3] = alf[t3,:, np.newaxis]*self.soc_sca/self.Lsc * ( 2*g4[t3,:,np.newaxis]/nnp**2*np.cos(nnp) - g5/nnp**4 * (6-6*np.cos(nnp)+3*nnp**2*np.cos(nnp)) )
        pphys['C10g'][t3] = 0 #wind effects are turned off
    
    #coefficients for salt transport at upstream boundary
    pphys['C11a'] = u_bar[:,0]*self.Lsc
    pphys['C11b'] = - self.Kh[:,0]
    pphys['C11c'] = 2 * g2[:,0,np.newaxis]*u_bar[:,0,np.newaxis]*self.Lsc*np.cos(nnp)/nnp**2 
    pphys['C11d'] = alf[:,0,np.newaxis]*self.soc_sca*(2*g4[:,0,np.newaxis]*np.cos(nnp)/nnp**2 + g5*((6*np.cos(nnp)-6)/nnp**4 - 3*np.cos(nnp)/nnp**2))
    
    #coefficients for salt transport at boundaries     
    pphys['C12a'], pphys['C12b'], pphys['C12c'], pphys['C12d'], pphys['C12e'] = np.zeros(self.T) , np.zeros((self.T,self.N)) ,\
        np.zeros((self.T,self.N,self.N)), np.zeros((self.T,self.N)) , np.zeros((self.T,self.N,self.N))
    for t3 in range(self.T):
        pphys['C12a'][t3] = 1#0.5*Kh[t3,0]
        pphys['C12b'][t3] = 0*alf[t3,0]*self.soc_sca*kkp* ( g5*(kkp**2-3)/(16*kkp**3) + g4[t3,0]*(3-2*kkp**2)/(24*kkp**3) - g3[t3,0]/(4*kkp) )
        pphys['C12c'][t3] = 0*alf[t3,0]*self.soc_sca*nnp* ( g5/8*(np.cos(pkn)/pkn**5*(24-12*pkn**2+pkn**4) - 24/pkn**5 + np.cos(pnk)/pnk**5*(24-12*pnk**2+pnk**4) - 24/pnk**5 )
                                          +g4[t3,0]/6*((6-pkn**2)*np.cos(pkn)/pkn**3  + (6-pnk**2)*np.cos(pnk)/pnk**3 ) + g3[t3,0]/2*(np.cos(pkn)/pkn + np.cos(pnk)/pnk) )
        pphys['C12d'][t3] = 0*self.Lsc*u_bar[t3,0] * (g1[t3,0]/2 + g2[t3,0]/6 + g3[t3,0]/(4*kkp**2))
        pphys['C12e'][t3] = 0*self.Lsc*u_bar[t3,0] * g2[t3,0]*(np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2)
    pphys['C12c'][np.where(np.isnan(pphys['C12c']))] = 0  
    pphys['C12e'][np.where(np.isnan(pphys['C12e']))] = 0  
    
    self.pphys = pphys
    
    return 

def build_sol(self, ans_n, ans_o,  t):
    # =============================================================================
    # fucntion to build the solution vector 
    # =============================================================================
    so = np.zeros(self.di[-1]*self.M)
    #short notation for the variables from the object
    inds = self.inds
    pphys= self.pphys
    
    #timedependence parts
    so[inds['x_m'] ] = so[inds['x_m'] ] + ans_n[inds['x_m']] / self.dt_sca[t] - ans_o[inds['x_m']] / self.dt_sca[t]
    so[inds['xr_mj']] = so[inds['xr_mj']] + 1/(2*self.dt_sca[t])*ans_n[inds['xr_mj']]  - 1/(2*self.dt_sca[t])*ans_o[inds['xr_mj']] 
    
    #river boundary - salt prescribed
    #so[0] = ans_n[0]-sri[t]/soc_sca
    #so[1:self.M] = ans_n[1:self.M]
    

    #river boundary - flux is zero
    so[0] = pphys['C11a'][t]*ans_n[0] + pphys['C11b'][t] * (-3*ans_n[0] + 4*ans_n[self.M] - ans_n[2*self.M])/(2*self.dl[0]) \
        + np.sum([pphys['C11c'][t,n]*ans_n[n+1] for n in range(self.N)]) \
        + (-3*ans_n[0] + 4*ans_n[self.M] - ans_n[2*self.M])/(2*self.dl[0]) * np.sum([ans_n[n+1]*pphys['C11d'][t,n] for n in range(self.N)])
    
    for j in range(1,self.M): so[j] =  pphys['C12a'][t]*(-3*ans_n[j]+4*ans_n[self.M+j]-ans_n[2*self.M+j])/(2*self.dl[0]) \
        + pphys['C12b'][t,j-1]*ans_n[j]*(-3*ans_n[0]+4*ans_n[self.M]-ans_n[2*self.M])/(2*self.dl[0]) \
        + np.sum([pphys['C12c'][t,j-1,n]*ans_n[n+1] for n in range(self.N)])*(-3*ans_n[0]+4*ans_n[self.M]-ans_n[2*self.M])/(2*self.dl[0]) \
        + pphys['C12d'][t,j-1]*ans_n[j] + np.sum([pphys['C12e'][t,j-1,n]*ans_n[n+1] for n in range(self.N)])
    

    #average salt
    so[inds['x_m']] = so[inds['x_m']] + (1-self.theta)*(pphys['C10a'][t,inds['x']]*(ans_o[inds['xp_m']]-ans_o[inds['xm_m']])/(2*self.dl[inds['x']]) \
                + pphys['C10b'][t,inds['x']]*(ans_o[inds['xp_m']] - 2*ans_o[inds['x_m']] + ans_o[inds['xm_m']])/(self.dl[inds['x']]**2) 
                + (ans_o[inds['xp_m']]-ans_o[inds['xm_m']])/(2*self.dl[inds['x']]) * np.sum([pphys['C10c'][t,inds['x'],n-1]*ans_o[inds['x_m']+n] for n in range(1,self.M)],0) 
                + (ans_o[inds['xp_m']] - 2*ans_o[inds['x_m']] + ans_o[inds['xm_m']])/(self.dl[inds['x']]**2) * np.sum([pphys['C10d'][t,inds['x'],n-1]*ans_o[inds['x_m']+n] for n in range(1,self.M)],0) 
                + np.sum([pphys['C10e'][t,inds['x'],n-1]*(ans_o[inds['xp_m']+n]-ans_o[inds['xm_m']+n])/(2*self.dl[inds['x']]) for n in range(1,self.M)],0) 
                + (ans_o[inds['xp_m']]-ans_o[inds['xm_m']])/(2*self.dl[inds['x']]) * np.sum([pphys['C10f'][t,inds['x'],n-1]*(ans_o[inds['xp_m']+n]-ans_o[inds['xm_m']+n])/(2*self.dl[inds['x']]) for n in range(1,self.M)],0)
                + np.sum([pphys['C10g'][t,inds['x'],n-1]*ans_o[inds['x_m']+n] for n in range(1,self.M)],0) )    
    so[inds['x_m']] = so[inds['x_m']] + self.theta*(pphys['C10a'][t,inds['x']]*(ans_n[inds['xp_m']]-ans_n[inds['xm_m']])/(2*self.dl[inds['x']]) \
                + pphys['C10b'][t,inds['x']]*(ans_n[inds['xp_m']] - 2*ans_n[inds['x_m']] + ans_n[inds['xm_m']])/(self.dl[inds['x']]**2) 
                + (ans_n[inds['xp_m']]-ans_n[inds['xm_m']])/(2*self.dl[inds['x']]) * np.sum([pphys['C10c'][t,inds['x'],n-1]*ans_n[inds['x_m']+n] for n in range(1,self.M)],0) 
                + (ans_n[inds['xp_m']] - 2*ans_n[inds['x_m']] + ans_n[inds['xm_m']])/(self.dl[inds['x']]**2) * np.sum([pphys['C10d'][t,inds['x'],n-1]*ans_n[inds['x_m']+n] for n in range(1,self.M)],0) 
                + np.sum([pphys['C10e'][t,inds['x'],n-1]*(ans_n[inds['xp_m']+n]-ans_n[inds['xm_m']+n])/(2*self.dl[inds['x']]) for n in range(1,self.M)],0) 
                + (ans_n[inds['xp_m']]-ans_n[inds['xm_m']])/(2*self.dl[inds['x']]) * np.sum([pphys['C10f'][t,inds['x'],n-1]*(ans_n[inds['xp_m']+n]-ans_n[inds['xm_m']+n])/(2*self.dl[inds['x']]) for n in range(1,self.M)],0)
                + np.sum([pphys['C10g'][t,inds['x'],n-1]*ans_n[inds['x_m']+n] for n in range(1,self.M)],0) )   
    
    #perturbed salt
    #contribution to solution vector due to term T1
    so[inds['xr_mj']] = so[inds['xr_mj']] + (1-self.theta)* (pphys['C1a'][t,inds['xr']] * ((ans_o[inds['xrp_mj']]-ans_o[inds['xrm_mj']])/(2*self.dl[inds['xr']])))
    so[inds['xr_mj']] = so[inds['xr_mj']] + self.theta*    (pphys['C1a'][t,inds['xr']] * ((ans_n[inds['xrp_mj']]-ans_n[inds['xrm_mj']])/(2*self.dl[inds['xr']])))

    #contribution to solution vector due to term T2
    so[inds['xr_mj']] = so[inds['xr_mj']] + (1-self.theta)*(pphys['C2a'][t,inds['xr'],inds['j1']]*(ans_o[inds['xrp_mj']]-ans_o[inds['xrm_mj']])/(2*self.dl[inds['xr']])
                            + pphys['C2b'][t,inds['xr'],inds['j1']]*((ans_o[inds['xrp_mj']]-ans_o[inds['xrm_mj']])/(2*self.dl[inds['xr']]))*((ans_o[inds['xrp_m']]-ans_o[inds['xrm_m']])/(2*self.dl[inds['xr']]))
                            +np.sum([pphys['C2c'][t,inds['xr'],inds['j1'],n-1] * (ans_o[inds['xrp_m']+n]-ans_o[inds['xrm_m']+n]) for n in range(1,self.M)],0)/(2*self.dl[inds['xr']])
                            +np.sum([pphys['C2d'][t,inds['xr'],inds['j1'],n-1] * (ans_o[inds['xrp_m']+n]-ans_o[inds['xrm_m']+n]) for n in range(1,self.M)],0)/(2*self.dl[inds['xr']])
                            *(ans_o[inds['xrp_m']]-ans_o[inds['xrm_m']])/(2*self.dl[inds['xr']]) )
    so[inds['xr_mj']] = so[inds['xr_mj']] + self.theta*   (pphys['C2a'][t,inds['xr'],inds['j1']]*(ans_n[inds['xrp_mj']]-ans_n[inds['xrm_mj']])/(2*self.dl[inds['xr']])
                            + pphys['C2b'][t,inds['xr'],inds['j1']]*((ans_n[inds['xrp_mj']]-ans_n[inds['xrm_mj']])/(2*self.dl[inds['xr']]))*((ans_n[inds['xrp_m']]-ans_n[inds['xrm_m']])/(2*self.dl[inds['xr']]))
                            +np.sum([pphys['C2c'][t,inds['xr'],inds['j1'],n-1] * (ans_n[inds['xrp_m']+n]-ans_n[inds['xrm_m']+n]) for n in range(1,self.M)],0)/(2*self.dl[inds['xr']])
                            +np.sum([pphys['C2d'][t,inds['xr'],inds['j1'],n-1] * (ans_n[inds['xrp_m']+n]-ans_n[inds['xrm_m']+n]) for n in range(1,self.M)],0)/(2*self.dl[inds['xr']])
                            *(ans_n[inds['xrp_m']]-ans_n[inds['xrm_m']])/(2*self.dl[inds['xr']]) )

    #contribution to solution vector due to term T3
    so[inds['xr_mj']] = so[inds['xr_mj']] + (1-self.theta)*(pphys['C3a'][t,inds['xr'],inds['j1']] * (ans_o[inds['xrp_m']]-ans_o[inds['xrm_m']])/(2*self.dl[inds['xr']]) \
                                                            + pphys['C3b'][t,inds['xr'],inds['j1']] * ((ans_o[inds['xrp_m']]-ans_o[inds['xrm_m']])/(2*self.dl[inds['xr']]))**2)
    so[inds['xr_mj']] = so[inds['xr_mj']] + self.theta*    (pphys['C3a'][t,inds['xr'],inds['j1']] * (ans_n[inds['xrp_m']]-ans_n[inds['xrm_m']])/(2*self.dl[inds['xr']]) \
                                                            + pphys['C3b'][t,inds['xr'],inds['j1']] * ((ans_n[inds['xrp_m']]-ans_n[inds['xrm_m']])/(2*self.dl[inds['xr']]))**2)

    #contribution to jacobian due to term T5
    so[inds['xr_mj']] = so[inds['xr_mj']] + (1-self.theta)*(pphys['C5a'][t,inds['xr'],inds['j1']]*(ans_o[inds['xrp_m']] - 2*ans_o[inds['xr_m']] + ans_o[inds['xrm_m']])/(self.dl[inds['xr']]**2)*ans_o[inds['xr_mj']] 
                             + pphys['C5b'][t,inds['xr'],inds['j1']]*(ans_o[inds['xrp_m']]-ans_o[inds['xrm_m']])/(2*self.dl[inds['xr']]) * ans_o[inds['xr_mj']]  
                             + np.sum([pphys['C5c'][t,inds['xr'],inds['j1'],n-1]*ans_o[inds['xr_m']+n] for n in range(1,self.M)],0) * (ans_o[inds['xrp_m']] - 2*ans_o[inds['xr_m']] + ans_o[inds['xrm_m']])/(self.dl[inds['xr']]**2) 
                             + np.sum([pphys['C5d'][t,inds['xr'],inds['j1'],n-1]*ans_o[inds['xr_m']+n] for n in range(1,self.M)],0) * (ans_o[inds['xrp_m']]-ans_o[inds['xrm_m']])/(2*self.dl[inds['xr']])
                             + pphys['C5e'][t,inds['xr'],inds['j1']]*ans_o[inds['xr_mj']] + np.sum([pphys['C5f'][t,inds['xr'],inds['j1'],n-1]*ans_o[inds['xr_m']+n] for n in range(1,self.M)],0))
    so[inds['xr_mj']] = so[inds['xr_mj']] + self.theta*    (pphys['C5a'][t,inds['xr'],inds['j1']]*(ans_n[inds['xrp_m']] - 2*ans_n[inds['xr_m']] + ans_n[inds['xrm_m']])/(self.dl[inds['xr']]**2)*ans_n[inds['xr_mj']] 
                             + pphys['C5b'][t,inds['xr'],inds['j1']]*(ans_n[inds['xrp_m']]-ans_n[inds['xrm_m']])/(2*self.dl[inds['xr']]) * ans_n[inds['xr_mj']]  
                             + np.sum([pphys['C5c'][t,inds['xr'],inds['j1'],n-1]*ans_n[inds['xr_m']+n] for n in range(1,self.M)],0) * (ans_n[inds['xrp_m']] - 2*ans_n[inds['xr_m']] + ans_n[inds['xrm_m']])/(self.dl[inds['xr']]**2) 
                             + np.sum([pphys['C5d'][t,inds['xr'],inds['j1'],n-1]*ans_n[inds['xr_m']+n] for n in range(1,self.M)],0) * (ans_n[inds['xrp_m']]-ans_n[inds['xrm_m']])/(2*self.dl[inds['xr']])
                             + pphys['C5e'][t,inds['xr'],inds['j1']]*ans_n[inds['xr_mj']] + np.sum([pphys['C5f'][t,inds['xr'],inds['j1'],n-1]*ans_n[inds['xr_m']+n] for n in range(1,self.M)],0))

    #contribution to solution vector due to term T6
    so[inds['xr_mj']] = so[inds['xr_mj']] + (1-self.theta)*(pphys['C6a'][t,inds['xr'],inds['j1']]*ans_o[inds['xr_mj']])
    so[inds['xr_mj']] = so[inds['xr_mj']] + self.theta*    (pphys['C6a'][t,inds['xr'],inds['j1']]*ans_n[inds['xr_mj']])

    #contribution to solution vector due to term T7
    so[inds['xr_mj']] = so[inds['xr_mj']] + (1-self.theta)*(pphys['C7a'][t,inds['xr']]*(ans_o[inds['xrp_mj']]-ans_o[inds['xrm_mj']])/(2*self.dl[inds['xr']]) \
                                                            + pphys['C7b'][t,inds['xr']]*(ans_o[inds['xrp_mj']] - 2*ans_o[inds['xr_mj']] + ans_o[inds['xrm_mj']])/(self.dl[inds['xr']]**2) )
    so[inds['xr_mj']] = so[inds['xr_mj']] + self.theta*    (pphys['C7a'][t,inds['xr']]*(ans_n[inds['xrp_mj']]-ans_n[inds['xrm_mj']])/(2*self.dl[inds['xr']]) \
                                                            + pphys['C7b'][t,inds['xr']]*(ans_n[inds['xrp_mj']] - 2*ans_n[inds['xr_mj']] + ans_n[inds['xrm_mj']])/(self.dl[inds['xr']]**2) )
                
    #inner boundary - this for loop may live on as a living fossil
    #equations for inner boundary
    sol_in2 =  {'se': lambda S_1, S0: S_1 - S0,
                'dse': lambda S_3,S_2,S_1,S0,S1,S2,d: (3*S_1-4*S_2+S_3)/(2*self.dl[self.di[d]-1]) - (-3*S0+4*S1-S2)/(2*self.dl[self.di[d]])
                }
    
    for d in range(1,len(self.nxn)):
        for j in range(self.M):
            s_3, s_2, s_1 = ans_n[(self.di[d]-3)*self.M:(self.di[d]-2)*self.M], ans_n[(self.di[d]-2)*self.M:(self.di[d]-1)*self.M], ans_n[(self.di[d]-1)*self.M:self.di[d]*self.M]
            s0, s1, s2 = ans_n[self.di[d]*self.M:(self.di[d]+1)*self.M], ans_n[(self.di[d]+1)*self.M:(self.di[d]+2)*self.M], ans_n[(self.di[d]+2)*self.M:(self.di[d]+3)*self.M]
            so[(self.di[d]-1)*self.M+j] =  sol_in2['dse'](s_3[j],s_2[j],s_1[j],s0[j],s1[j],s2[j],d)
            so[self.di[d]*self.M+j] =  sol_in2['se'](s_1[j],s0[j])
            
    #sea boundary
    so[self.M*(self.di[-1]-1)] = (self.co_slu[t]*self.soc[t] + (1-self.co_slu[t])*ans_n[self.di[-2]*self.M]*self.soc_sca)/self.soc_sca - ans_n[self.M*(self.di[-1]-1)]
    so[self.M*(self.di[-1]-1)+1:self.M*self.di[-1]] = ans_n[self.M*(self.di[-1]-1)+1:self.M*self.di[-1]]

    return so

def build_jac(self,ans,t):
    # =============================================================================
    # fucntion to build the jacobian associated with the solution vector 
    # =============================================================================
    
    jac = np.zeros((self.di[-1]*self.M,self.di[-1]*self.M))

    #short notation for the variables from the object
    inds = self.inds
    pphys= self.pphys
    
    #contribution to jacobian due to term T1
    jac[inds['xr_mj'],inds['xrm_mj']] = jac[inds['xr_mj'],inds['xrm_mj']] - pphys['C1a'][t,inds['xr']]/(2*self.dl[inds['xr']]) #left of diagonal
    jac[inds['xr_mj'],inds['xrp_mj']] = jac[inds['xr_mj'],inds['xrp_mj']] + pphys['C1a'][t,inds['xr']]/(2*self.dl[inds['xr']]) #right of diagonal
               
    #contribution to jacobian due to term T2            
    jac[inds['xr_mj'],inds['xrm_mj']] = jac[inds['xr_mj'],inds['xrm_mj']] - pphys['C2a'][t,inds['xr'],inds['j1']]/(2*self.dl[inds['xr']]) \
        - pphys['C2b'][t,inds['xr'],inds['j1']]/(2*self.dl[inds['xr']]) * (ans[inds['xrp_m']]-ans[inds['xrm_m']])/(2*self.dl[inds['xr']])
    jac[inds['xr_mj'],inds['xrp_mj']] = jac[inds['xr_mj'],inds['xrp_mj']] + pphys['C2a'][t,inds['xr'],inds['j1']]/(2*self.dl[inds['xr']]) \
        + pphys['C2b'][t,inds['xr'],inds['j1']]/(2*self.dl[inds['xr']]) * (ans[inds['xrp_m']]-ans[inds['xrm_m']])/(2*self.dl[inds['xr']])
    jac[inds['xr_mj'],inds['xrm_m']] = (jac[inds['xr_mj'],inds['xrm_m']] - pphys['C2b'][t,inds['xr'],inds['j1']]/(2*self.dl[inds['xr']]) * (ans[inds['xrp_mj']]-ans[inds['xrm_mj']]) /(2*self.dl[inds['xr']]) 
                   - np.sum([pphys['C2d'][t,inds['xr'],inds['j1'],n-1] * (ans[inds['xrp_m']+n]-ans[inds['xrm_m']+n]) for n in range(1,self.M)],0)/(4*self.dl[inds['xr']]**2) )
    jac[inds['xr_mj'],inds['xrp_m']] = (jac[inds['xr_mj'],inds['xrp_m']]  + pphys['C2b'][t,inds['xr'],inds['j1']]/(2*self.dl[inds['xr']]) * (ans[inds['xrp_mj']]-ans[inds['xrm_mj']])/(2*self.dl[inds['xr']]) 
                                       + np.sum([pphys['C2d'][t,inds['xr'],inds['j1'],n-1] * (ans[inds['xrp_m']+n]-ans[inds['xrm_m']+n]) for n in range(1,self.M)],0)/(4*self.dl[inds['xr']]**2) )
    jac[inds['xr_mj2'],inds['xrm_mk']] = jac[inds['xr_mj2'],inds['xrm_mk']] - pphys['C2c'][t,inds['xr2'],inds['j12'],inds['k1']]/(2*self.dl[inds['xr2']]) \
        - pphys['C2d'][t,inds['xr2'],inds['j12'],inds['k1']]*(ans[inds['xrp_m2']]-ans[inds['xrm_m2']])/(4*self.dl[inds['xr2']]**2) 
    jac[inds['xr_mj2'],inds['xrp_mk']] = jac[inds['xr_mj2'],inds['xrp_mk']] + pphys['C2c'][t,inds['xr2'],inds['j12'],inds['k1']]/(2*self.dl[inds['xr2']]) \
        + pphys['C2d'][t,inds['xr2'],inds['j12'],inds['k1']]*(ans[inds['xrp_m2']]-ans[inds['xrm_m2']])/(4*self.dl[inds['xr2']]**2) 
    
    #contribution to jacobian due to term T3
    jac[inds['xr_mj'],inds['xrm_m']] = jac[inds['xr_mj'],inds['xrm_m']] - pphys['C3a'][t,inds['xr'],inds['j1']]/(2*self.dl[inds['xr']]) \
        - pphys['C3b'][t,inds['xr'],inds['j1']]/self.dl[inds['xr']] * (ans[inds['xrp_m']]-ans[inds['xrm_m']])/(2*self.dl[inds['xr']])
    jac[inds['xr_mj'],inds['xrp_m']] = jac[inds['xr_mj'],inds['xrp_m']] + pphys['C3a'][t,inds['xr'],inds['j1']]/(2*self.dl[inds['xr']]) \
        + pphys['C3b'][t,inds['xr'],inds['j1']]/self.dl[inds['xr']] * (ans[inds['xrp_m']]-ans[inds['xrm_m']])/(2*self.dl[inds['xr']])
    
    
    #contribution to jacobian due to term T5
    jac[inds['xr_mj'], inds['xrm_m']] = (jac[inds['xr_mj'], inds['xrm_m']] + pphys['C5a'][t,inds['xr'],inds['j1']]*ans[inds['xr_mj']]/(self.dl[inds['xr']]**2) \
                                         - pphys['C5b'][t,inds['xr'],inds['j1']]/(2*self.dl[inds['xr']])*ans[inds['xr_mj']]
                           + np.sum([ans[inds['xr_m']+n]*pphys['C5c'][t,inds['xr'],inds['j1'],n-1]/(self.dl[inds['xr']]**2) for n in range(1,self.M)],0) 
                           - np.sum([ans[inds['xr_m']+n]*pphys['C5d'][t,inds['xr'],inds['j1'],n-1]/(2*self.dl[inds['xr']]) for n in range(1,self.M)],0) )
    jac[inds['xr_mj'], inds['xr_m']] = jac[inds['xr_mj'], inds['xr_m']] - 2*pphys['C5a'][t,inds['xr'],inds['j1']]*ans[inds['xr_mj']]/(self.dl[inds['xr']]**2) \
        - 2* np.sum([ans[inds['xr_m']+n]*pphys['C5c'][t,inds['xr'],inds['j1'],n-1]/(self.dl[inds['xr']]**2) for n in range(1,self.M)],0)
    jac[inds['xr_mj'], inds['xrp_m']] = (jac[inds['xr_mj'], inds['xrp_m']] + pphys['C5a'][t,inds['xr'],inds['j1']]*ans[inds['xr_mj']]/(self.dl[inds['xr']]**2) \
                                   + pphys['C5b'][t,inds['xr'],inds['j1']]/(2*self.dl[inds['xr']])*ans[inds['xr_mj']]
                                   + np.sum([ans[inds['xr_m']+n]*pphys['C5c'][t,inds['xr'],inds['j1'],n-1]/(self.dl[inds['xr']]**2) for n in range(1,self.M)],0)
                                   + np.sum([ans[inds['xr_m']+n]*pphys['C5d'][t,inds['xr'],inds['j1'],n-1]/(2*self.dl[inds['xr']]) for n in range(1,self.M)],0) )
    jac[inds['xr_mj'],inds['xr_mj']] = (jac[inds['xr_mj'], inds['xr_mj']] + pphys['C5a'][t,inds['xr'],inds['j1']]*(ans[inds['xrp_m']]-2*ans[inds['xr_m']]+ans[inds['xrm_m']])/(self.dl[inds['xr']]**2) 
                                     + pphys['C5b'][t,inds['xr'],inds['j1']]*(ans[inds['xrp_m']]-ans[inds['xrm_m']])/(2*self.dl[inds['xr']]) + pphys['C5e'][t,inds['xr'],inds['j1']] )
    jac[inds['xr_mj2'], inds['xr_mk']] = (jac[inds['xr_mj2'], inds['xr_mk']] + pphys['C5c'][t,inds['xr2'],inds['j12'],inds['k1']]*(ans[inds['xrp_m2']]-2*ans[inds['xr_m2']]+ans[inds['xrm_m2']])/(self.dl[inds['xr2']]**2)
                                         + pphys['C5d'][t,inds['xr2'],inds['j12'],inds['k1']]*(ans[inds['xrp_m2']]-ans[inds['xrm_m2']])/(2*self.dl[inds['xr2']]) +pphys['C5f'][t,inds['xr2'],inds['j12'],inds['k1']] )
    
    #contribution to jacobian due to term T6
    jac[inds['xr_mj'],inds['xr_mj']] = jac[inds['xr_mj'],inds['xr_mj']] + pphys['C6a'][t,inds['xr'],inds['j1']]
    
    #contribution to jacobian due to term T7
    jac[inds['xr_mj'],inds['xrm_mj']] = jac[inds['xr_mj'],inds['xrm_mj']] - pphys['C7a'][t,inds['xr']]/(2*self.dl[inds['xr']]) + pphys['C7b'][t,inds['xr']] / (self.dl[inds['xr']]**2)
    jac[inds['xr_mj'],inds['xr_mj']] = jac[inds['xr_mj'],inds['xr_mj']] - 2*pphys['C7b'][t,inds['xr']]/(self.dl[inds['xr']]**2)
    jac[inds['xr_mj'],inds['xrp_mj']] = jac[inds['xr_mj'],inds['xrp_mj']] + pphys['C7a'][t,inds['xr']]/(2*self.dl[inds['xr']]) + pphys['C7b'][t,inds['xr']] / (self.dl[inds['xr']]**2)
    
    #contribution to jacobian due to term the average salt equation - always present
    #left
    jac[inds['x_m'], inds['xm_m']] = (jac[inds['x_m'], inds['xm_m']] - pphys['C10a'][t,inds['x']]/(2*self.dl[inds['x']]) + pphys['C10b'][t,inds['x']]/(self.dl[inds['x']]**2) \
                                   - 1/(2*self.dl[inds['x']])*np.sum([pphys['C10c'][t,inds['x'],n-1]*ans[inds['x_m']+n] for n in range(1,self.M)],0) 
                                   + 1/(self.dl[inds['x']]**2)*np.sum([pphys['C10d'][t,inds['x'],n-1]*ans[inds['x_m']+n] for n in range(1,self.M)],0) 
                                   - 1/(2*self.dl[inds['x']])*np.sum([pphys['C10f'][t,inds['x'],n-1]*(ans[inds['xp_m']+n]-ans[inds['xm_m']+n])/(2*self.dl[inds['x']]) for n in range(1,self.M)],0) )
    jac[inds['xr_m'], inds['xrm_mj']] = jac[inds['xr_m'], inds['xrm_mj']] - pphys['C10e'][t,inds['xr'],inds['j1']]/(2*self.dl[inds['xr']]) - pphys['C10f'][t,inds['xr'],inds['j1']]/(2*self.dl[inds['xr']])*(ans[inds['xrp_m']]-ans[inds['xrm_m']])/(2*self.dl[inds['xr']])
    #center
    jac[inds['x_m'],inds['x_m']] = jac[inds['x_m'],inds['x_m']] - 2/(self.dl[inds['x']]**2)*pphys['C10b'][t,inds['x']] \
        - 2/(self.dl[inds['x']]**2)*np.sum([pphys['C10d'][t,inds['x'],n-1]*ans[inds['x_m']+n] for n in range(1,self.M)],0)
    jac[inds['xr_m'], inds['xr_mj']] = jac[inds['xr_m'],inds['xr_mj']] + (ans[inds['xrp_m']]-ans[inds['xrm_m']])/(2*self.dl[inds['xr']]) * pphys['C10c'][t,inds['xr'],inds['j1']] \
        + (ans[inds['xrp_m']]-2*ans[inds['xr_m']]+ans[inds['xrm_m']])/(self.dl[inds['xr']]**2) * pphys['C10d'][t,inds['xr'],inds['j1']] + pphys['C10g'][t,inds['xr'],inds['j1']]
    #right
    jac[inds['x_m'], inds['xp_m']] = (jac[inds['x_m'],inds['xp_m']] + pphys['C10a'][t,inds['x']]/(2*self.dl[inds['x']]) + pphys['C10b'][t,inds['x']]/(self.dl[inds['x']]**2) \
                                   + 1/(2*self.dl[inds['x']])*np.sum([pphys['C10c'][t,inds['x'],n-1]*ans[inds['x_m']+n] for n in range(1,self.M)],0) 
                                   + 1/(self.dl[inds['x']]**2)*np.sum([pphys['C10d'][t,inds['x'],n-1]*ans[inds['x_m']+n] for n in range(1,self.M)],0) 
                                   + 1/(2*self.dl[inds['x']])*np.sum([pphys['C10f'][t,inds['x'],n-1]*(ans[inds['xp_m']+n]-ans[inds['xm_m']+n])/(2*self.dl[inds['x']]) for n in range(1,self.M)],0) )
    jac[inds['xr_m'], inds['xrp_mj']] = jac[inds['xr_m'],inds['xrp_mj']] + pphys['C10e'][t,inds['xr'],inds['j1']]/(2*self.dl[inds['xr']]) + pphys['C10f'][t,inds['xr'],inds['j1']]/(2*self.dl[inds['xr']])*(ans[inds['xrp_m']]-ans[inds['xrm_m']])/(2*self.dl[inds['xr']])
    
    jac = jac*self.theta #weighting of this part - should not apply to the boundaries, works if I place it here
    
    #river boundary - no flux
    #jac[np.arange(self.M),np.arange(self.M)] = jac[np.arange(self.M),np.arange(self.M)] + 1    
    jac[0,0] = pphys['C11a'][t] -3*pphys['C11b'][t]/(2*self.dl[0]) -3/(2*self.dl[0]) * np.sum([ans[n+1]*pphys['C11d'][t,n] for n in range(self.N)])
    jac[0,self.M] =  4*pphys['C11b'][t]/(2*self.dl[0]) + 4/(2*self.dl[0]) * np.sum([ans[n+1]*pphys['C11d'][t,n]  for n in range(self.N)])
    jac[0,2*self.M] = -pphys['C11b'][t]/(2*self.dl[0]) - 1/(2*self.dl[0]) * np.sum([ans[n+1]*pphys['C11d'][t,n]  for n in range(self.N)])
    
    for j in range(1,self.M):
        jac[0,j] = pphys['C11c'][t,j-1] + (-3*ans[0] + 4*ans[self.M] - ans[2*self.M])/(2*self.dl[0]) * pphys['C11d'][t,j-1]          
        
    
    #depth-perturbed values 
    for j in range(1,self.M):
        #dom2
        jac[j, 2*self.M] = pphys['C12b'][t,j-1] * ans[j] *-1/ (2*self.dl[0]) + np.sum(pphys['C12c'][t,j-1]*ans[1:self.M]) *-1/ (2*self.dl[0])
        jac[j, self.M] = pphys['C12b'][t,j-1] * ans[j] * 4/ (2*self.dl[0]) + np.sum(pphys['C12c'][t,j-1]*ans[1:self.M]) * 4/ (2*self.dl[0])
        jac[j, 0] = pphys['C12b'][t,j-1] * ans[j] *-3/ (2*self.dl[0]) + np.sum(pphys['C12c'][t,j-1]*ans[1:self.M]) *-3/ (2*self.dl[0])
        
        jac[j, 2*self.M+j] = pphys['C12a'][t] *-1/(2*self.dl[0]) 
        jac[j, self.M+j] = pphys['C12a'][t] *4/(2*self.dl[0]) 
        jac[j, j] = pphys['C12a'][t] *-3/(2*self.dl[0])  + pphys['C12b'][t,j-1] * (-ans[self.M*2]+4*ans[self.M]-3*ans[0])/(2*self.dl[0]) + pphys['C12d'][t,j-1]
        for k in range(1,self.M):  jac[j, k]= jac[j, k] +  pphys['C12c'][t,j-1,k-1] * (-ans[self.M*2]+4*ans[self.M]-3*ans[0])/(2*self.dl[0]) + pphys['C12e'][t,j-1,k-1]
       
    #inner boundary 
    for d in range(1,len(self.nxn)):   #this for loop has to be replaced in a later stage 
        #derivatives equal
        jac[(self.di[d]-1)*self.M+np.arange(self.M), (self.di[d]-3)*self.M+np.arange(self.M)] =  1/(2*self.dl[self.di[d]-1])
        jac[(self.di[d]-1)*self.M+np.arange(self.M), (self.di[d]-2)*self.M+np.arange(self.M)] = - 4/(2*self.dl[self.di[d]-1])
        jac[(self.di[d]-1)*self.M+np.arange(self.M), (self.di[d]-1)*self.M+np.arange(self.M)] =  3/(2*self.dl[self.di[d]-1]) 
        
        jac[(self.di[d]-1)*self.M+np.arange(self.M), self.di[d]*self.M+np.arange(self.M)] =  3/(2*self.dl[self.di[d]]) 
        jac[(self.di[d]-1)*self.M+np.arange(self.M), (self.di[d]+1)*self.M+np.arange(self.M)] =  - 4/(2*self.dl[self.di[d]])
        jac[(self.di[d]-1)*self.M+np.arange(self.M), (self.di[d]+2)*self.M+np.arange(self.M)] =  1/(2*self.dl[self.di[d]])
    
    #points equal
    jac[inds['bnd'], inds['bnd']-self.M] = 1                   
    jac[inds['bnd'], inds['bnd']] = -1
    
    #sea boundary
    jac[self.M*(self.di[-1]-1),self.M*self.di[-2]] = (1-self.co_slu[t])
    jac[self.M*(self.di[-1]-1),self.M*(self.di[-1]-1)] = jac[self.M*(self.di[-1]-1),self.M*(self.di[-1]-1)] - 1
    jac[np.arange(self.M*(self.di[-1]-1)+1,self.M*self.di[-1]),np.arange(self.M*(self.di[-1]-1)+1,self.M*self.di[-1])] =  1
    
    #Time step               
    jac[inds['x_m'],inds['x_m']] = jac[inds['x_m'],inds['x_m']] + 1/self.dt_sca[t]
    jac[inds['xr_mj'],inds['xr_mj']] = jac[inds['xr_mj'],inds['xr_mj']] + 1/(2*self.dt_sca[t]) 
         
    return jac
