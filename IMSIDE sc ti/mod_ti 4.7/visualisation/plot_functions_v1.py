# =============================================================================
# plot functions  
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

def plot_sst(self,sss):
    
    indi = self.ii_all
    sss = np.delete(sss , np.concatenate([indi['bnl_rgt'].flatten(),indi['bnl_lft'].flatten(),indi['bnl_bnd']]))
    pxh = np.repeat(self.px, self.nz).reshape((self.di[-1],self.nz)) + 25

    # =============================================================================
    # Plot salt field
    # =============================================================================
   
    #calculate and plot total salinity 
    s_b = np.transpose([np.reshape(sss,(self.di[-1],self.M))[:,0]]*self.nz)
    sn = np.reshape(sss,(self.di[-1],self.M))[:,1:]
    s_p = np.array([np.sum([sn[i,n-1]*np.cos(np.pi*n *self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    s = (s_b+s_p)*self.soc
    s[np.where((s<0) & (s>-0.0001))]= 1e-10 #remove negative values due to numerical precision
    
    #make contourplot
    fig,ax = plt.subplots(figsize=(10,7))
    l1=ax.contourf(pxh.T,self.pz.T,  s.T, cmap='RdBu_r',levels=(np.linspace(0,self.soc,36)))
    #ax.quiver(qx,qz,qu.transpose(),qw.transpose(),color='white')
    cb0 = fig.colorbar(l1, ax=ax,orientation='horizontal', pad=0.16)
    cb0.set_label(label='Salinity [psu]',fontsize=16)
    ax.set_xlabel('$x$ [km]',fontsize=16) 
    ax.set_ylabel('$z$ [m]',fontsize=16)    
    ax.set_xlim(-50,5)
    ax.set_facecolor('black')
    plt.show()  

#plot_sst(run , out4)

def prt_numbers(self, sss, prt = True):
    indi = self.ii_all
    ss2 = np.delete(sss , np.concatenate([indi['bnl_rgt'].flatten(),indi['bnl_lft'].flatten(),indi['bnl_bnd']]))

    #calculate total salinity 
    s_b = np.transpose([np.reshape(ss2,(self.di[-1],self.M))[:,0]]*self.nz)
    sn = np.reshape(ss2,(self.di[-1],self.M))[:,1:]
    s_p = np.array([np.sum([sn[i,n-1]*np.cos(np.pi*n *self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    s = (s_b+s_p)*self.soc
    s[np.where((s<0) & (s>-0.0001))]= 1e-10 #remove negative values due to numerical precision

    Lint = -self.px[np.where(s[:,0]>2)[0][0]]-self.Ln[-1]/1000
    
    #salinity in tidal cycle
    if len(self.tid_comp)>1: print('Not sure about the relative timing of the tidal components')
    t= np.linspace(0,44700,self.nt)
    sti = 0
    for i in range(len(self.tid_comp)):
        tid_set = self.tid_sets[self.tid_comp[i]]
        tid_geg = self.tid_gegs[self.tid_comp[i]]
        tid_inp = self.tidal_salinity(sss, tid_set, tid_geg)  
        
        sti += np.real(tid_inp['st'][:,:,np.newaxis]* np.exp(1j*tid_geg['omega']*t))

    stot = s[:,:,np.newaxis] + sti
        
    if prt == True:
        print()
        print('The salt intrusion length is ',Lint,' km')
        print('The minimum salinity is ',np.min(s),' psu')
        print('The salinity at the sea boundary at the bottom is ', s[np.where(self.px==-self.Ln[-1]/1000)[0][0],0])
        print('The maximum salinity in the tidal cycle is', np.max(stot), ' and the minimum is ', np.min(stot),' psu' )
    
    return Lint

#prt_numbers(run, out4, run.ii_all, prt = True)
    
def plot_transport(self, sss, version):
    # =============================================================================
    # plot transports
    # =============================================================================
    indi = self.ii_all
    ss2 = np.delete(sss , np.concatenate([indi['bnl_rgt'].flatten(),indi['bnl_lft'].flatten(),indi['bnl_bnd']]))

    #calculate salinity
    sb = np.reshape(ss2,(self.di[-1],self.M))[:,0]  * self.soc
    sn = np.reshape(ss2,(self.di[-1],self.M))[:,1:] * self.soc
    
    dsbdx, dsbdx2 = np.zeros(self.di[-1]), np.zeros(self.di[-1])
    for dom in range(self.ndom):
        dsbdx[self.di[dom]] = (-3*sb[self.di[dom]] + 4*sb[self.di[dom]+1] - sb[self.di[dom]+2] )/(2*self.dxn[dom])
        dsbdx[self.di[dom]+1:self.di[dom+1]-1] = (sb[self.di[dom]+2:self.di[dom+1]] - sb[self.di[dom]:self.di[dom+1]-2])/(2*self.dxn[dom])
        dsbdx[self.di[dom+1]-1] = (sb[self.di[dom+1]-3] - 4*sb[self.di[dom+1]-2] + 3*sb[self.di[dom+1]-1] )/(2*self.dxn[dom])

        dsbdx2[self.di[dom]] = (2*sb[self.di[dom]] - 5*sb[self.di[dom]+1] +4*sb[self.di[dom]+2] -1*sb[self.di[dom]+3] )/(self.dxn[dom]**2)
        dsbdx2[self.di[dom]+1:self.di[dom+1]-1] = (sb[self.di[dom]+2:self.di[dom+1]] - 2*sb[self.di[dom]+1:self.di[dom+1]-1] + sb[self.di[dom]:self.di[dom+1]-2])/(self.dxn[dom]**2)
        dsbdx2[self.di[dom+1]-1] = (-sb[self.di[dom+1]-4] + 4*sb[self.di[dom+1]-3] - 5*sb[self.di[dom+1]-2] + 2*sb[self.di[dom+1]-1] )/(self.dxn[dom]**2)

    #dspdz  = np.array([np.sum([sn[i,n-1]*np.pi*n/self.H*-np.sin(np.pi*n*self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    #dspdz2 = np.array([np.sum([sn[i,n-1]*np.pi**2*n**2/self.H**2*-np.cos(np.pi*n*self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    
    #some variables for plottting
    #vertical viscosity
    if self.choice_viscosityv_st == 'constant': Av_st = self.Av_st + np.zeros(self.di[-1]) #do nothing, value is specified
    elif self.choice_viscosityv_st == 'cuh': Av_st = self.cv_st * self.Ut * self.H
    else: print('ERROR: no valid op option for choice vertical viscosity subtidal')

    if self.choice_bottomslip_st == 'sf': rr = Av_st/(self.sf_st*self.H)
    elif self.choice_bottomslip_st == 'rr': rr = self.rr_st + np.zeros(self.di[-1])
    else: print('ERROR: no valid op option for choice bottomslip subtidal')   
    g1 = -1 + (1.5+3*rr) / (1+ 3 *rr)
    g2 =  -3 / (2+6*rr)
    g3 = (1+4*rr) / (1+3*rr) * (9+18*rr) - 8 - 24*rr
    g4 = -9 * (1+4*rr) / (1+3*rr)
    g5 = - 8
    nnp = np.linspace(1,self.N,self.N)*np.pi #n*pi
    #horizontal diffusivity
    if self.choice_diffusivityh_st == 'constant':             
        Kh = self.Kh_st + np.zeros(self.di[-1])
        Kh[self.di[-2]:]= self.Kh_st * self.b[self.di[-2]:]/self.b[self.di[-2]] #sea domain
    if self.choice_diffusivityh_st == 'cub':             
        Kh = self.ch_st * self.Ut * self.b
      
    pxh = self.px + self.Ln[-1]/1000

    T_Q  = self.Q*sb
    T_E  = self.b*self.H*np.sum(sn*(2*(self.Q/(self.b*self.H))[:,np.newaxis]*g2[:,np.newaxis]*np.cos(nnp)/nnp**2 + (self.g*self.Be*self.H**3/(48*Av_st))[:,np.newaxis]*dsbdx[:,np.newaxis]*(2*g4[:,np.newaxis]*np.cos(nnp)/nnp**2 + g5*((6*np.cos(nnp)-6)/nnp**4 - 3*np.cos(nnp)/nnp**2))),1)
    T_D  = self.b*self.H*(-Kh*dsbdx)
    T_tot = T_Q+T_E+T_D
    #tidal transports 
    T_T, T_Tb, T_Tp, T_Tc = {}, {}, {} , {}
    
    for i in range(len(self.tid_comp)):
        tid_set = self.tid_sets[self.tid_comp[i]]
        tid_geg = self.tid_gegs[self.tid_comp[i]]
        tid_inp = self.tidal_salinity(sss, tid_set, tid_geg)   
        
        T_T[self.tid_comp[i]]  = self.b*self.H*1/4 * np.real(tid_geg['ut']*np.conj(tid_inp['st'])   + np.conj(tid_geg['ut'])*tid_inp['st']).mean(1)     
        T_Tb[self.tid_comp[i]] = self.b*self.H*1/4 * np.real(tid_geg['utb']*np.conj(tid_inp['stb'][:,0]) + np.conj(tid_geg['utb'])*tid_inp['stb'][:,0])   
        T_Tp[self.tid_comp[i]] = self.b*self.H*1/4 * np.real(tid_geg['utp']*np.conj(tid_inp['stp']) + np.conj(tid_geg['utp'])*tid_inp['stp']).mean(1)     
    
        if version in ['A']: T_T[self.tid_comp[i]] , T_Tb[self.tid_comp[i]], T_Tp[self.tid_comp[i]] = T_T[self.tid_comp[i]] * 0 , T_Tb[self.tid_comp[i]] * 0 , T_Tp[self.tid_comp[i]] * 0 
        
        T_Tc[self.tid_comp[i]] = self.b*self.H*1/4 * np.real(tid_geg['ut']*np.conj(self.calc_cor(sss, indi, i))   + np.conj(tid_geg['ut'])*self.calc_cor(sss, indi, i)).mean(1)  
        
        T_tot += T_T[self.tid_comp[i]] + T_Tc[self.tid_comp[i]]

      
    plt.figure(figsize=(6,3))
    plt.plot(pxh,T_Q,label='$T_Q$',lw = 2, c='g')
    plt.plot(pxh,T_E,label='$T_E$',lw = 2, c='orange')
    plt.plot(pxh,T_D,label='$T_D$',lw = 2, c='firebrick')
    #plt.plot(pxh,T_Tc,label='$T_{Tc}$',lw=2, c='olive')
    plt.plot(pxh,T_tot,label='$T_{tot}$',lw=2,c='black')
    #add tides
    ctides = ['skyblue','pink','silver']
    for i in range(len(self.tid_comp)): 
        #plt.plot(pxh,T_T[self.tid_comp[i]],label='$T_{'+self.tid_comp[i]+'}$',lw=2, c=ctides[i])
        #plt.plot(pxh,T_T[self.tid_comp[i]],label='$T_{T}$',lw=2, c=ctides[i])
        plt.plot(pxh,T_Tb[self.tid_comp[i]],label='$T_{Tb}$',lw=2, c='skyblue')#, ls = ':')
        plt.plot(pxh,T_Tp[self.tid_comp[i]],label='$T_{Tp}$',lw=2, c='darkblue')#, ls = "-.")
        #plt.plot(pxh,T_Tc[self.tid_comp[i]],lw=2, c='olive')#,label='$T_{Tc,'+self.tid_comp[i]+'}$')


    plt.legend(loc=2) #plt.legend(loc=2)#,
    plt.grid()
    #plt.xlim(-np.sum(Ln[:-1])/1000,0)
    plt.xlim(-50,0) #, plt.ylim(-1000,1000)
    plt.xlabel('$x$ [km]',fontsize=12) , plt.ylabel('$T$ [kg/s]',fontsize=12)
    plt.show()
    
    
    

    return T_Q, T_E, T_D, T_T,  T_tot,  T_Tb, T_Tp,

#plot_transport(run, out4,  'D')


def plot_next(self, sss, version):
    # =============================================================================
    # plot transports
    # =============================================================================
    i=0
    indi = self.ii_all
    tid_set = self.tid_sets[self.tid_comp[i]]
    tid_geg = self.tid_gegs[self.tid_comp[i]]
    tid_inp = self.tidal_salinity(sss, tid_set, tid_geg)
    ss2 = np.delete(sss , np.concatenate([indi['bnl_rgt'].flatten(),indi['bnl_lft'].flatten(),indi['bnl_bnd']]))

    sb = np.reshape(ss2,(self.di[-1],self.M))[:,0]  * self.soc
    dsbdx, dsbdx2 = np.zeros(self.di[-1]), np.zeros(self.di[-1])
    for dom in range(self.ndom):
        dsbdx[self.di[dom]] = (-3*sb[self.di[dom]] + 4*sb[self.di[dom]+1] - sb[self.di[dom]+2] )/(2*self.dxn[dom])
        dsbdx[self.di[dom]+1:self.di[dom+1]-1] = (sb[self.di[dom]+2:self.di[dom+1]] - sb[self.di[dom]:self.di[dom+1]-2])/(2*self.dxn[dom])
        dsbdx[self.di[dom+1]-1] = (sb[self.di[dom+1]-3] - 4*sb[self.di[dom+1]-2] + 3*sb[self.di[dom+1]-1] )/(2*self.dxn[dom])

        dsbdx2[self.di[dom]] = (2*sb[self.di[dom]] - 5*sb[self.di[dom]+1] +4*sb[self.di[dom]+2] -1*sb[self.di[dom]+3] )/(self.dxn[dom]**2)
        dsbdx2[self.di[dom]+1:self.di[dom+1]-1] = (sb[self.di[dom]+2:self.di[dom+1]] - 2*sb[self.di[dom]+1:self.di[dom+1]-1] + sb[self.di[dom]:self.di[dom+1]-2])/(self.dxn[dom]**2)
        dsbdx2[self.di[dom+1]-1] = (-sb[self.di[dom+1]-4] + 4*sb[self.di[dom+1]-3] - 5*sb[self.di[dom+1]-2] + 2*sb[self.di[dom+1]-1] )/(self.dxn[dom]**2)


    #calculate total salinity 
    s_b = np.transpose([np.reshape(ss2,(self.di[-1],self.M))[:,0]]*self.nz)
    sn = np.reshape(ss2,(self.di[-1],self.M))[:,1:]
    s_p = np.array([np.sum([sn[i,n-1]*np.cos(np.pi*n *self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    sp = s_p * self.soc
    s = (s_b+s_p)*self.soc
    s[np.where((s<0) & (s>-0.0001))]= 1e-10 #remove negative values due to numerical precision
    
    #some variables for plottting
    #vertical viscosity
    if self.choice_viscosityv_st == 'constant': Av_st = self.Av_st + np.zeros(self.di[-1]) #do nothing, value is specified
    elif self.choice_viscosityv_st == 'cuh': Av_st = self.cv_st * self.Ut * self.H
    else: print('ERROR: no valid op option for choice vertical viscosity subtidal')

    #vertical diffusivity
    if self.choice_diffusivityv_st == 'constant': Kv_st = self.Kv_st + np.zeros(self.di[-1]) #do nothing, value is specified
    elif self.choice_diffusivityv_st == 'as': Kv_st = Av_st / self.Sc_st
    else: print('ERROR: no valid option for choice vertical diffusivity subtidal')
    
    if self.choice_bottomslip_st == 'sf': rr = Av_st/(self.sf_st*self.H)
    elif self.choice_bottomslip_st == 'rr': rr = self.rr_st + np.zeros(self.di[-1])
    else: print('ERROR: no valid op option for choice bottomslip subtidal')   
    g1 = -1 + (1.5+3*rr) / (1+ 3 *rr)
    g2 =  -3 / (2+6*rr)
    g3 = (1+4*rr) / (1+3*rr) * (9+18*rr) - 8 - 24*rr
    g4 = -9 * (1+4*rr) / (1+3*rr)
    g5 = - 8
    
    
    
    #plot stratificatie
    strat =  s[:,0]-s[:,-1]
    pxh = self.px + self.Ln[-1]/1000

    '''
    plt.plot(pxh , strat)
    plt.xlim(-110,0)
    plt.ylabel(r'$\Delta s$ [psu]') , plt.xlabel('$x$ [km]')
    plt.show()
    #'''
    stb = tid_inp['stb']
    stp = tid_inp['stp']
    
    stb_pha = np.angle(stb[:,0])/np.pi*180
    utb_pha = np.angle(tid_geg['utb'])/np.pi*180
    dif_phab = stb_pha - utb_pha
    stb_amp = np.abs(stb[:,0])
    utb_amp = np.abs(tid_geg['utb'])
    
    # =============================================================================
    # estimated values 
    # =============================================================================
    
    stb_amp_guess = utb_amp / tid_geg['omega'] * dsbdx 
    u_bar = self.Q/(self.H*self.b)
    alf   = self.g*self.Be*self.H**3/(48*Av_st)
   
    #coefficients
    rr = 1/2
    g1 = -1 + (1.5+3*rr) / (1+ 3 *rr)
    g2 =  -3 / (2+6*rr)
    g3 = (1+4*rr) / (1+3*rr) * (9+18*rr) - 8 - 24*rr
    g4 = -9 * (1+4*rr) / (1+3*rr)
    g5 = - 8
    
    p3_z = (- g1/2 - g2/12) / self.H
    p4_z = (- g3/2 - g4/12 + g5/20) / self.H
    p34w = (-120*tid_geg['B']*tid_geg['deA']*(alf*tid_geg['deA']**2*dsbdx*(g3 + g4 - g5) + 2*alf*dsbdx*(g4 - 3*g5) + tid_geg['deA']**2*u_bar*(g1 + g2) + 2*g2*u_bar)*np.sinh(tid_geg['deA']) \
            + 10*tid_geg['B']*(alf*dsbdx*(4*tid_geg['deA']**4*(3*g3 + g4) + 24*tid_geg['deA']**2*g4 - g5*(3*tid_geg['deA']**4 + 36*tid_geg['deA']**2 + 72)) + 4*tid_geg['deA']**2*u_bar*(tid_geg['deA']**2*(3*g1 + g2) + 6*g2))*np.cosh(tid_geg['deA']) \
            + alf*dsbdx*(720*tid_geg['B']*g5 + tid_geg['deA']**6*(-40*g3 - 8*g4 + 5*g5)) - 8*tid_geg['deA']**6*u_bar*(5*g1 + g2))/(120*tid_geg['deA']**6)
    
    #parts of the expression
    pt1 = 1j * tid_geg['omega'] * tid_geg['eta'][0,:,0] * self.H**2/Kv_st * dsbdx * (u_bar * p3_z + alf * dsbdx * p4_z)
    pt2 = - self.g/(1j*tid_geg['omega']) * (tid_geg['detadx2'][0,:,0] + tid_geg['detadx'][0,:,0]/self.bex) * self.H**2/Kv_st * dsbdx * p34w
    pd_guess = pt1 + pt2
 
 
    # =============================================================================
    # plot 
    # =============================================================================
    fig , ax = plt.subplots(3,1,figsize=(3,7))
    ax[0].plot(pxh[1:] , dif_phab[1:])
    ax[0].plot(pxh[1:] , np.abs(pd_guess)[1:]*120000+90)
    ax[1].plot(pxh[1:] , stb_amp[1:])
    ax[1].plot(pxh[1:] , stb_amp_guess[1:])
    
    ax[2].plot(pxh[1:] , utb_amp[1:])
    [ax[i].set_xlim(-60,0) for i in range(3)], [ax[i].grid() for i in range(3)], ax[0].set_ylim(89,93+15) 
    ax[0].set_ylabel(r'$\Delta \Phi$ [deg]') , ax[2].set_ylabel(r'$\bar{u_{ti}}$ [m/s]') , ax[1].set_ylabel(r'$\bar{s_{ti}}$ [psu]') , 
    ax[2].set_xlabel('$x$ [km]')
    plt.show()
    
    #stp_pha = np.angle(stp)/np.pi*180
    #utp_pha = np.angle(self.utp)/np.pi*180
    #dif_phap = stp_pha - utp_pha
    
    '''
    plt.contourf(pxh[1:] , self.pz , dif_phap[1:].T)
    plt.colorbar()
    plt.xlim(-110,0)
    plt.ylabel(r'$\Delta \phi$ [deg]') , plt.xlabel('$x$ [km]')
    plt.show()
    
    plt.contourf(pxh[1:] , self.pz , utp_pha[1:].T)
    plt.colorbar()
    plt.xlim(-110,0)
    plt.ylabel(r'$\Delta \phi$ [deg]') , plt.xlabel('$x$ [km]')
    plt.show()
    
    plt.contourf(pxh[1:] , self.pz , stp_pha[1:].T)
    plt.colorbar()
    plt.xlim(-110,0)
    plt.ylabel(r'$\Delta \phi$ [deg]') , plt.xlabel('$x$ [km]')
    plt.show()
    '''
    '''
    # =============================================================================
    # do checks of calculation Huib 07-03-24
    # =============================================================================
    dsdz = tid_inp['dstdz']
    utb = tid_geg['utb']
    wt = tid_geg['wt']
    
    #vertical derivatives
    dsdz = np.zeros((self.di[-1],self.nz))
    dsdz[:,1:-1] = -(s[:,2:] - s[:,:-2])/(2*self.H[:,np.newaxis]/(self.nz-1))
    #TODO: check minus sign
    
    F = np.mean(wt[0] * dsdz, 1)
    #check 1
    utstti = self.b * self.H * 1/(2*tid_geg['omega']) * np.abs(utb) * np.abs(F) * np.cos(np.angle(utb) + np.pi/2 - np.angle(F))
    T_Tb = self.b*self.H*1/4 * np.real(tid_geg['utb']*np.conj(tid_inp['stb'][:,0]) + np.conj(tid_geg['utb'])*tid_inp['stb'][:,0]) 
    
    plt.plot(utstti)
    plt.plot(T_Tb)
    plt.show()
    
    #check 2
    F2 = - 1j * tid_geg['omega'] * sp[:,-1] * tid_geg['eta'][0,:,0] / self.H
    fig, ax = plt.subplots(2,1)
    ax[0].plot(np.real(F))
    ax[0].plot(np.real(F2))
    ax[1].plot(np.imag(F))
    ax[1].plot(np.imag(F2))
    plt.show()
    
    #check 3
    arg_dif = np.angle(tid_geg['eta'][0,:,0]) - np.angle(tid_geg['utb'])
    plt.plot(pxh,arg_dif)
    plt.plot(pxh,[np.pi/2]*len(arg_dif))
    plt.show()
    '''
        
    return pxh, strat, dif_phab, dsbdx, tid_geg['utb'], tid_geg['eta'][0,:,0]
    

#plot_next(run, out4, 'D')


def terms_tide(self, sss):
    # =============================================================================
    # function to calculate the phase difference generation in the depth-averaged salinity
    # variatio nin the tidal cylce
    # =============================================================================
    pxh = self.px + self.Ln[-1]/1000
    '''
    for i in range(len(self.tid_comp)): #for all tidal components
        tid_set = self.tid_sets[self.tid_comp[i]]
        tid_geg = self.tid_gegs[self.tid_comp[i]]
        tid_inp = self.tidal_salinity(sss, tid_set, tid_geg)   
    
        dstdz = tid_inp['dstdz']
        wt = tid_geg['wt'][0]
        T1 = np.mean(1/4 * np.real( dstdz * np.conj(wt) + np.conj(dstdz) * wt ) , 1)
        
        
        ut = tid_geg['ut']
        dstdx = tid_inp['dstidx']
        T2 = np.mean(1/4 * np.real( dstdx * np.conj(ut) + np.conj(dstdx) * ut ) , 1)
        
        plt.plot(pxh,T1)
        plt.plot(pxh,-T2)
        plt.xlim(-60,0)
        plt.yscale('log')
        plt.ylim(1e-10 , 1e-2)
        plt.grid()
        plt.show()
  
    '''
    
    t= np.linspace(0,44700,self.nt)[:,np.newaxis,np.newaxis]
    tid_geg = self.tid_gegs[self.tid_comp[0]]
    #tid_inp = self.tidal_salinity(sss, tid_set, tid_geg)   
    
    #dstdz = tid_inp['dstdz']
    ut = tid_geg['ut']
    wt = tid_geg['wt']
    utr = np.real(tid_geg['ut'][np.newaxis]* np.exp(1j*tid_geg['omega']*t))
    wtr = np.real(tid_geg['wt'] * np.exp(1j*tid_geg['omega']*t))
    
    '''
    tplot, xplot = 0,200
    plt.plot(utr[tplot,xplot] , self.zlist[0,xplot])
    tplot = 20
    plt.plot(utr[tplot,xplot] , self.zlist[0,xplot])
    tplot = 40
    plt.plot(utr[tplot,xplot] , self.zlist[0,xplot])
    tplot = 60
    plt.plot(utr[tplot,xplot] , self.zlist[0,xplot])
    tplot = 80
    plt.plot(utr[tplot,xplot] , self.zlist[0,xplot])
    tplot = 100
    plt.plot(utr[tplot,xplot] , self.zlist[0,xplot])

    plt.show()
    
    tplot, xplot = 0,200
    plt.plot(wtr[tplot,xplot] , self.zlist[0,xplot])
    tplot = 20
    plt.plot(wtr[tplot,xplot] , self.zlist[0,xplot])
    tplot = 40
    plt.plot(wtr[tplot,xplot] , self.zlist[0,xplot])
    tplot = 60
    plt.plot(wtr[tplot,xplot] , self.zlist[0,xplot])
    tplot = 80
    plt.plot(wtr[tplot,xplot] , self.zlist[0,xplot])
    tplot = 100
    plt.plot(wtr[tplot,xplot] , self.zlist[0,xplot])

    
    plt.grid()
    plt.show()
    print(pxh[xplot])
    '''  
    indi = self.ii_all

    ss2 = np.delete(sss , np.concatenate([indi['bnl_rgt'].flatten(),indi['bnl_lft'].flatten(),indi['bnl_bnd']]))
    sn = np.reshape(ss2,(self.di[-1],self.M))[:,1:]
    s_p = np.array([np.sum([sn[i,n-1]*np.cos(np.pi*n *self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    sp = s_p * self.soc
    
    eta_ti = tid_geg['eta'][0]
    utp = tid_geg['utp']
    
    print(eta_ti.shape , utp.shape)
    temp = 1/4 * np.real( eta_ti * np.conj(utp) + np.conj(eta_ti) * utp ) 
        
    T9 = self.b*self.H*np.mean(temp * sp ,1)
    plt.plot(pxh, T9)
    
    
    
#terms_tide(run,out4)

def animate_salt(self, sss, name_ani):
    '''
    # =============================================================================
    #  Make animation
    # =============================================================================
    #subtidal velocities
    zzz = np.repeat(zz,di[-1]).reshape((nz,di[-1]))
    u_b = np.array([u_bar]*nz)
    u_p =  (u_b*(1/2-3/2*zzz**2/H**2) + g*Be*H**3/(48*Av) * dsbdx *(1-9*zzz**2/H**2 - 8*zzz**3/H**3))
    u   = (u_b+u_p).transpose()
    w   = g*Be*H**3/(48*Av) * H * (dsbdx2 + dsbdx/bex) * (-g5/4 * zzz**4/H**4 - g4/3*zzz**3/H**3 - g3*zzz/H)
    nxq = 10
    qx, qz = np.linspace(-np.sum(-Ln[:-1]),0,nxq),np.linspace(-H,0,nz)
    '''
    #prepare animation for tides 
    #ut,st,eta,detadx = tide_module(sss, (H, Ln, b0, bs, dxn), inp_t, (nz,121),'salinity')
    
    indi = self.ii_all
    ss2 = np.delete(sss , np.concatenate([indi['bnl_rgt'].flatten(),indi['bnl_lft'].flatten(),indi['bnl_bnd']]))

    #calculate total salinity 
    s_b = np.transpose([np.reshape(ss2,(self.di[-1],self.M))[:,0]]*self.nz)
    sn = np.reshape(ss2,(self.di[-1],self.M))[:,1:]
    s_p = np.array([np.sum([sn[i,n-1]*np.cos(np.pi*n *self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    s = (s_b+s_p)*self.soc
    s[np.where((s<0) & (s>-0.0001))]= 1e-10 #remove negative values due to numerical precision

    Lint = -self.px[np.where(s[:,0]>2)[0][0]]-self.Ln[-1]/1000
    
    #salinity in tidal cycle
    if len(self.tid_comp)>1: print('Not sure about the relative timing of the tidal components')
    t= np.linspace(0,44700,self.nt)
    sti, eta = 0,0
    for i in range(len(self.tid_comp)):
        tid_set = self.tid_sets[self.tid_comp[i]]
        tid_geg = self.tid_gegs[self.tid_comp[i]]
        tid_inp = self.tidal_salinity(sss, tid_set, tid_geg)  
        
        sti += np.real((tid_inp['st'] + self.calc_cor(sss, self.ii_all, i))[:,:,np.newaxis]* np.exp(1j*tid_geg['omega']*t))
        eta += np.real(tid_geg['eta']* np.exp(1j*tid_geg['omega']*t))[0]
        
    stot = s[:,:,np.newaxis] + sti

    
    #print('The maximum salinty in the tidal cycle is ',np.max(s_tot),' and the minimum is',np.min(s_tot),' psu')
    #print('The maximum velocity in the tidal cycle is ',np.max(u_tot),' and the minimum is',np.min(u_tot),' m/s')
    #print('The maximum tidal velocity is ',np.max(ut),' and the minimum is',np.min(ut),' m/s')
  
    
    #take water level into account for htis
    px_tide = np.tile(self.px[:self.di[-1]]+self.Ln[-1]/1000,self.nz).reshape(self.nz,self.di[-1])
    pz_tide = np.zeros((self.nz,self.di[-1],self.nt))+np.nan
    for t in range(self.nt):
        pz_1t = np.zeros((self.nz,self.di[-1]))+np.nan
        for x in range(self.di[-1]):
            pz_1t[:,x] = np.linspace(-self.H[x],eta[x,t],self.nz)
        pz_tide[:,:,t] = pz_1t
        
    '''
    #for quiver
    qeta = np.zeros((nxq,nt))+np.nan #eta interpoatlino
    for t in range(nt): qeta[:,t] = np.interp(qx,px[:di[-2]]+Ln[-1]/1000,eta[0,:,t])
    qx_tide = np.tile(qx,nz).reshape(nz,nxq)        
    qz_tide = np.zeros((nz,nxq,nt))+np.nan
    for t in range(nt):
        pz_1t = np.zeros((nz,nxq))+np.nan
        for x in range(nxq): pz_1t[:,x] = np.linspace(-H,qeta[x,t],nz)
        qz_tide[:,:,t] = pz_1t
    #'''
  

    #ANIMATION 
    def init():
        #l1=ax.contourf(px,pz, s_tot[:,:,0], cmap='RdBu_r',levels=(np.linspace(np.min(s_tot),np.max(s_tot),15)))
        l1=ax.contourf(px_tide,pz_tide[:,:,0], stot[:,:,0].T, cmap='RdBu_r',levels=(np.linspace(0,35,15)))#,extend='both')
        #ax.quiver(qx_tide,qz_tide[:,:,0],qut[:,:,0].T,qw.T,color='black',scale=15)
        cb0 = fig.colorbar(l1, ax=ax,orientation='vertical')
        cb0.set_label(label='Salinity [psu]',fontsize=14)
        ax.set_xlabel('$x$ [km]',fontsize=14) 
        ax.set_ylabel('$z$ [m]',fontsize=14) 
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylim(-np.max(self.H),np.max(eta)*1.25)
        ax.set_xlim(-50,5), ax.set_xlim(px_tide[0,0],0)
  
        return #ax.cla()
    def animate(t):
        ax.clear()  
        ax.contourf(px_tide,pz_tide[:,:,t], stot[:,:,t].T, cmap='RdBu_r',levels=(np.linspace(0,35,15)))#,extend='both')
        #ax.quiver(qx_tide,qz_tide[:,:,t],qut[:,:,t].T,qw.T,color='black',scale=15)
        ax.set_xlabel('$x$ [km]',fontsize=14) 
        ax.set_ylabel('$z$ [m]',fontsize=14) 
        ax.set_title('Time in tidal cycle: '+str(np.round(t/self.nt*12.42,1))+' hours')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylim(-np.max(self.H),np.max(eta)*1.25)
        ax.set_xlim(-50,5) #, ax.set_xlim(px_tide[0,0],0)

        return ax
     
    import matplotlib.animation as ani      #make animations
    import time 
    frames = 11
    tijd = time.time()
    fig = plt.figure(figsize=(8,4))
    #fig.tight_layout()
    ax = fig.add_subplot(111)
    anim = ani.FuncAnimation(fig,animate,self.nt,init_func=init,blit=False)
    anim.save("/Users/biemo004/Documents/UU phd Saltisolutions/Output/Animations all/"+name_ani+".mp4", fps=frames, extra_args=['-vcodec', 'libx264'],bitrate=-1,dpi = 500)
    print('Making of the animation took '+ str(time.time()-tijd)+ ' seconds')
    plt.show()
    


#animate_salt(run , out4, 'test_060124_v6')

def plt_salt(self, sss):
    # =============================================================================
    #  Make animation
    # =============================================================================
       
    indi = self.ii_all
    ss2 = np.delete(sss , np.concatenate([indi['bnl_rgt'].flatten(),indi['bnl_lft'].flatten(),indi['bnl_bnd']]))

    #calculate total salinity 
    s_b = np.transpose([np.reshape(ss2,(self.di[-1],self.M))[:,0]]*self.nz)
    sn = np.reshape(ss2,(self.di[-1],self.M))[:,1:]
    s_p = np.array([np.sum([sn[i,n-1]*np.cos(np.pi*n *self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    s = (s_b+s_p)*self.soc
    s[np.where((s<0) & (s>-0.0001))]= 1e-10 #remove negative values due to numerical precision

    Lint = -self.px[np.where(s[:,0]>2)[0][0]]-self.Ln[-1]/1000
    
    #salinity in tidal cycle
    if len(self.tid_comp)>1: print('Not sure about the relative timing of the tidal components')
    t= np.linspace(0,44700,self.nt)
    sti, eta = 0,0
    for i in range(len(self.tid_comp)):
        tid_set = self.tid_sets[self.tid_comp[i]]
        tid_geg = self.tid_gegs[self.tid_comp[i]]
        tid_inp = self.tidal_salinity(sss, tid_set, tid_geg)  
        
        sti += np.real((tid_inp['st']*0 + self.calc_cor(sss, self.ii_all, i))[:,:,np.newaxis]* np.exp(1j*tid_geg['omega']*t))
        eta += np.real(tid_geg['eta']* np.exp(1j*tid_geg['omega']*t))[0]
        
    stot = s[:,:,np.newaxis]*0 + sti

    
    #take water level into account for htis
    px_tide = np.tile(self.px[:self.di[-1]]+self.Ln[-1]/1000,self.nz).reshape(self.nz,self.di[-1])
    pz_tide = np.zeros((self.nz,self.di[-1],self.nt))+np.nan
    for t in range(self.nt):
        pz_1t = np.zeros((self.nz,self.di[-1]))+np.nan
        for x in range(self.di[-1]):
            pz_1t[:,x] = np.linspace(-self.H[x],eta[x,t],self.nz)
        pz_tide[:,:,t] = pz_1t
        

    #ANIMATION 
    def contourplot(t):
        fig, ax = plt.subplots(1,1)
        l1=ax.contourf(px_tide,pz_tide[:,:,t], stot[:,:,t].T, cmap='RdBu_r')#,levels=(np.linspace(0,35,15)),extend='both')
        #ax.quiver(qx_tide,qz_tide[:,:,0],qut[:,:,0].T,qw.T,color='black',scale=15)
        cb0 = fig.colorbar(l1, ax=ax,orientation='vertical')
        cb0.set_label(label='Salinity [psu]',fontsize=14)
        ax.set_xlabel('$x$ [km]',fontsize=14) 
        ax.set_ylabel('$z$ [m]',fontsize=14) 
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylim(-np.max(self.H),np.max(eta)*1.25)
        ax.set_xlim(-10,5) #, ax.set_xlim(px_tide[0,0],0)
        
        plt.show()
        return
    
    
    return contourplot
#
'''
fun = plt_salt(run,out4)


fun(0)
fun(10)
fun(30)
fun(60)
fun(90)
#'''



def terms_vert(self, ans, xloc):
      
    #prepare
    sss = np.delete(ans , np.concatenate([self.ii_all['bnl_rgt'].flatten(),self.ii_all['bnl_lft'].flatten(),self.ii_all['bnl_bnd']]))
    #calculate and plot total salinity 
    sb = np.transpose([np.reshape(sss,(self.di[-1],self.M))[:,0]]*self.nz)*self.soc
    sn = np.reshape(sss,(self.di[-1],self.M))[:,1:]*self.soc
    s_p = np.array([np.sum([sn[i,n-1]*np.cos(np.pi*n *self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    s = (sb+s_p)
    s[np.where((s<0) & (s>-0.0001))]= 1e-10 #remove negative values due to numerical precision
    nnp = np.linspace(1,self.N,self.N)*np.pi #n*pi
    dx = self.dl*self.Lsc
    
    #coefficients
    #vertical viscosity
    if self.choice_viscosityv_st == 'constant': Av_st = self.Av_st + np.zeros(self.di[-1]) #do nothing, value is specified
    elif self.choice_viscosityv_st == 'cuh': Av_st = self.cv_st * self.Ut * self.H
    else: print('ERROR: no valid op option for choice vertical viscosity subtidal')

    #vertical diffusivity
    if self.choice_diffusivityv_st == 'constant': Kv_st = self.Kv_st + np.zeros(self.di[-1]) #do nothing, value is specified
    elif self.choice_diffusivityv_st == 'as': Kv_st = Av_st / self.Sc_st
    else: print('ERROR: no valid option for choice vertical diffusivity subtidal')
    
    if self.choice_bottomslip_st == 'sf': rr = Av_st/(self.sf_st*self.H)
    elif self.choice_bottomslip_st == 'rr': rr = self.rr_st + np.zeros(self.di[-1])
    else: print('ERROR: no valid op option for choice bottomslip subtidal')
    g1 = -1 + (1.5+3*rr) / (1+ 3 *rr)
    g2 =  -3 / (2+6*rr)
    g3 = (1+4*rr) / (1+3*rr) * (9+18*rr) - 8 - 24*rr
    g4 = -9 * (1+4*rr) / (1+3*rr)
    g5 = - 8    
    u_bar = self.Q/(self.H*self.b)
    alf   = self.g*self.Be*self.H**3/(48*Av_st)
    
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
    # =============================================================================
    # function to calculate the phase difference generation in the depth-averaged salinity
    # variatio nin the tidal cylce
    # =============================================================================
    pxh = self.px + self.Ln[-1]/1000
    pzh = self.z_nd#[zplot]
    
    xplot = np.argmin(np.abs(pxh - xloc))
    
    #print(alf.shape, g1.shape, sb.shape, self.z_nd.shape , dx.shape)
    T1 = u_bar[xplot] * np.sum([(sn[xplot+1,n]-sn[xplot-1,n])/(2*dx[xplot])* np.cos(nnp[n]*self.z_nd) for n in range(self.N)],0)
    T2 = (u_bar[xplot] * (g1[xplot] + g2[xplot]*self.z_nd**2) + alf[xplot]*(sb[xplot+1]-sb[xplot-1])/(2*dx[xplot])*(g3[xplot]+g4[xplot]*self.z_nd**2+g5*self.z_nd**3)) * np.sum([(sn[xplot+1,n]-sn[xplot-1,n])/(2*dx[xplot]) * np.cos(nnp[n]*self.z_nd) for n in range(self.N)],0)
    T3 = (u_bar[xplot] * (g1[xplot] + g2[xplot]*self.z_nd**2) + alf[xplot]*(sb[xplot+1]-sb[xplot-1])/(2*dx[xplot])*(g3[xplot]+g4[xplot]*self.z_nd**2+g5*self.z_nd**3) ) * (sb[xplot+1]-sb[xplot-1])/(2*dx[xplot])
    T4 = -(self.bex[xplot]**-1*np.sum([sn[xplot,n]*alf[xplot]*(sb[xplot+1]-sb[xplot-1])/(2*dx[xplot]) * (2*g4[xplot]*np.cos(nnp[n])/(nnp[n]**2)+g5*((6*np.cos(nnp[n])-6)/(nnp[n]**4) -3*np.cos(nnp[n])/(nnp[n]**2))) for n in range(self.N)],0) 
                                + np.sum([ sn[xplot,n] * alf[xplot]*(sb[xplot+1]-2*sb[xplot]+sb[xplot-1])/(dx[xplot]**2)*(2*g4[xplot]*np.cos(nnp[n])/(nnp[n]**2)+g5*((6*np.cos(nnp[n])-6)/(nnp[n]**4) -3*np.cos(nnp[n])/(nnp[n]**2))) for n in range(self.N)],0)
                                + np.sum([ (sn[xplot+1,n]-sn[xplot-1,n])/(2*dx[xplot]) * (2*g2[xplot]*u_bar[xplot] * np.cos(nnp[n])/nnp[n]**2 ) for n in range(self.N)],0)
                                + np.sum([ (sn[xplot+1,n]-sn[xplot-1,n])/(2*dx[xplot]) * alf[xplot]*(sb[xplot+1]-sb[xplot-1])/(2*dx[xplot])*(2*g4[xplot]*np.cos(nnp[n])/(nnp[n]**2)+g5*((6*np.cos(nnp[n])-6)/(nnp[n]**4) -3*np.cos(nnp[n])/(nnp[n]**2))) for n in range(self.N)],0) )  

    T5 =  -(alf[xplot]*self.H[xplot]* ( (sb[xplot+1]-2*sb[xplot]+sb[xplot-1])/(dx[xplot]**2) + self.bex[xplot]**-1*(sb[xplot+1]-sb[xplot-1])/(2*dx[xplot]) ) * (-g5/4*self.z_nd**4-g4[xplot]/3*self.z_nd**3 -g3[xplot]*self.z_nd)
                                 ) * np.sum([nnp[n]/self.H[xplot] * sn[xplot,n]*np.sin(nnp[n]*self.z_nd) for n in range(self.N)],0)    
    T6 = Kv_st[xplot] * np.sum([nnp[n]**2/self.H[xplot]**2 * sn[xplot,n]*np.cos(nnp[n]*self.z_nd) for n in range(self.N)],0)
    T7 = -self.bex[xplot]**(-1)*Kh[xplot]*np.sum([(sn[xplot+1,n]-sn[xplot-1,n])/(2*dx[xplot]) * np.cos(nnp[n]*self.z_nd) for n in range(self.N)],0) \
        - Kh_x[xplot]*np.sum([(sn[xplot+1,n]-sn[xplot-1,n])/(2*dx[xplot]) * np.cos(nnp[n]*self.z_nd) for n in range(self.N)],0) \
        - Kh[xplot] * np.sum([(sn[xplot+1,n]-2*sn[xplot,n]+sn[xplot-1,n])/(dx[xplot]**2) * np.cos(nnp[n]*self.z_nd) for n in range(self.N)],0)

    #'''
    #plot subtidal terms
    plt.plot(T1, pzh,label='T1')
    plt.plot(T2, pzh,label='T2')
    plt.plot(T3, pzh,label='T3')
    plt.plot(T4, pzh,label='T4')
    plt.plot(T5, pzh,label='T5')
    plt.plot(T6, pzh,label='T6')
    plt.plot(T7, pzh,label='T7')
    #'''
    
    #tidal terms
    tid_set = self.tid_sets[self.tid_comp[0]]
    tid_geg = self.tid_gegs[self.tid_comp[0]]
    tid_inp = self.tidal_salinity(ans, tid_set, tid_geg)   
    
    T1t = 1/4 * np.real(tid_geg['utb'][xplot] * np.conj(tid_inp['dstipdx'][xplot]) + np.conj(tid_geg['utb'][xplot]) * tid_inp['dstipdx'][xplot])
    T2t = 1/4 * np.real(tid_geg['utp'][xplot] * np.conj(tid_inp['dstipdx'][xplot]) + np.conj(tid_geg['utp'][xplot]) * tid_inp['dstipdx'][xplot])
    T3t = 1/4 * np.real(tid_geg['utp'][xplot] * np.conj(tid_inp['dstibdx'][xplot]) + np.conj(tid_geg['utp'][xplot]) * tid_inp['dstibdx'][xplot])
    T4t = -np.mean(1/4 * np.real(tid_geg['utp'][xplot] * np.conj(tid_inp['dstipdx'][xplot]) + np.conj(tid_geg['utp'][xplot]) * tid_inp['dstipdx'][xplot])) + np.zeros(self.nz)
    T5t = 1/4 * np.real( tid_inp['dstdz'][xplot] * np.conj(tid_geg['wt'][0][xplot]) + np.conj(tid_inp['dstdz'][xplot]) * tid_geg['wt'][0][xplot] )
    
    #'''
    plt.plot(T1t, pzh,label='T1t',ls=':')
    plt.plot(T2t, pzh,label='T2t',ls=':')
    plt.plot(T3t, pzh,label='T3t',ls=':')
    plt.plot(T4t, pzh,label='T4t',ls=':')
    plt.plot(T5t, pzh,label='T5t',ls=':')
    #'''
    plt.plot(T1+T2+T3+T4+T5+T6+T7+T1t+T2t+T3t+T4t+T5t, pzh, lw = 2, c='black')
    #plt.plot(T3+T6, pzh, lw = 2, c='black')

    plt.legend()
    plt.grid()
    plt.show()
    
    # =============================================================================
    # phases and amplitudes of the tidal currents and salinty
    # =============================================================================
    


    
    
          
    return pzh*self.H[xplot], T1,T2,T3,T4,T5,T6,T7,T1t,T2t,T3t,T4t,T5t , T1+T2+T3+T4+T5+T6+T7+T1t+T2t+T3t+T4t+T5t, \
        tid_geg['utb'][xplot], tid_geg['utp'][xplot], tid_geg['wt'][0][xplot], tid_inp['dstibdx'][xplot], tid_inp['dstipdx'][xplot], tid_inp['dstdz'][xplot]

 #terms_vert(run, out4, -15)
 
 
def plot_Kp(self, sss):
    # =============================================================================
    # potentiele energie anomaly
    # volgens papaer van Burcahrd and XX (XXXX)
    # =============================================================================
    
    nnp = np.linspace(1,self.N,self.N)*np.pi #n*pi
    indi = self.ii_all
    ss2 = np.delete(sss , np.concatenate([indi['bnl_rgt'].flatten(),indi['bnl_lft'].flatten(),indi['bnl_bnd']]))

    #calculate total salinity 
    s_b = np.transpose([np.reshape(ss2,(self.di[-1],self.M))[:,0]]*self.nz)* self.soc
    sn = np.reshape(ss2,(self.di[-1],self.M))[:,1:]*self.soc
    s_p = np.array([np.sum([sn[i,n-1]*np.cos(np.pi*n *self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    s = (s_b+s_p)
    s[np.where((s<0) & (s>-0.0001))]= 1e-10 #remove negative values due to numerical precision

    sb = np.reshape(ss2,(self.di[-1],self.M))[:,0]  * self.soc
    dsbdx, dsbdx2, dsndx, dsndx2  = np.zeros(self.di[-1]), np.zeros(self.di[-1]), np.zeros((self.di[-1],self.N)), np.zeros((self.di[-1],self.N))
    for dom in range(self.ndom):
        dsbdx[self.di[dom]] = (-3*sb[self.di[dom]] + 4*sb[self.di[dom]+1] - sb[self.di[dom]+2] )/(2*self.dxn[dom])
        dsbdx[self.di[dom]+1:self.di[dom+1]-1] = (sb[self.di[dom]+2:self.di[dom+1]] - sb[self.di[dom]:self.di[dom+1]-2])/(2*self.dxn[dom])
        dsbdx[self.di[dom+1]-1] = (sb[self.di[dom+1]-3] - 4*sb[self.di[dom+1]-2] + 3*sb[self.di[dom+1]-1] )/(2*self.dxn[dom])

        dsbdx2[self.di[dom]] = (2*sb[self.di[dom]] - 5*sb[self.di[dom]+1] +4*sb[self.di[dom]+2] -1*sb[self.di[dom]+3] )/(self.dxn[dom]**2)
        dsbdx2[self.di[dom]+1:self.di[dom+1]-1] = (sb[self.di[dom]+2:self.di[dom+1]] - 2*sb[self.di[dom]+1:self.di[dom+1]-1] + sb[self.di[dom]:self.di[dom+1]-2])/(self.dxn[dom]**2)
        dsbdx2[self.di[dom+1]-1] = (-sb[self.di[dom+1]-4] + 4*sb[self.di[dom+1]-3] - 5*sb[self.di[dom+1]-2] + 2*sb[self.di[dom+1]-1] )/(self.dxn[dom]**2)

        dsndx[self.di[dom]] = (-3*sn[self.di[dom]] + 4*sn[self.di[dom]+1] - sn[self.di[dom]+2] )/(2*self.dxn[dom])
        dsndx[self.di[dom]+1:self.di[dom+1]-1] = (sn[self.di[dom]+2:self.di[dom+1]] - sn[self.di[dom]:self.di[dom+1]-2])/(2*self.dxn[dom])
        dsndx[self.di[dom+1]-1] = (sn[self.di[dom+1]-3] - 4*sn[self.di[dom+1]-2] + 3*sn[self.di[dom+1]-1] )/(2*self.dxn[dom])
       
        dsndx2[self.di[dom]] = (2*sn[self.di[dom]] - 5*sn[self.di[dom]+1] +4*sn[self.di[dom]+2] -1*sn[self.di[dom]+3] )/(self.dxn[dom]**2)
        dsndx2[self.di[dom]+1:self.di[dom+1]-1] = (sn[self.di[dom]+2:self.di[dom+1]] - 2*sn[self.di[dom]+1:self.di[dom+1]-1] + sn[self.di[dom]:self.di[dom+1]-2])/(self.dxn[dom]**2)
        dsndx2[self.di[dom+1]-1] = (-sn[self.di[dom+1]-4] + 4*sn[self.di[dom+1]-3] - 5*sn[self.di[dom+1]-2] + 2*sn[self.di[dom+1]-1] )/(self.dxn[dom]**2)


    #coefficients
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
        Kh_st = self.Kh_st + np.zeros(self.di[-1])
        Kh_st[self.di[-2]:]= self.Kh_st * self.b[self.di[-2]:]/self.b[self.di[-2]] #sea domain
    if self.choice_diffusivityh_st == 'cub':             
        Kh_st = self.ch_st * self.Ut * self.b
    
    #derivative - numerical, can also analytical, should not matter too much. 
    Kh_st_x = np.zeros(Kh_st.shape) + np.nan
    Kh_st_x[1:-1] = (Kh_st[2:] - Kh_st[:-2])/ (2*self.dl[1:-1]*self.Lsc)
    Kh_st_x[[0,-1]] ,  Kh_st_x[self.di[1:-1]] ,  Kh_st_x[self.di[1:-1]-1] = np.nan,np.nan,np.nan
    
    if self.choice_bottomslip_st == 'sf': rr = Av_st/(self.sf_st*self.H)
    elif self.choice_bottomslip_st == 'rr': rr = self.rr_st + np.zeros(self.di[-1])
    else: print('ERROR: no valid op option for choice bottomslip subtidal')
    g1 = -1 + (1.5+3*rr) / (1+ 3 *rr)
    g2 =  -3 / (2+6*rr)
    g3 = (1+4*rr) / (1+3*rr) * (9+18*rr) - 8 - 24*rr
    g4 = -9 * (1+4*rr) / (1+3*rr)
    g5 = - 8    
    u_bar = self.Q/(self.H*self.b)
    alf   = self.g*self.Be*self.H**3/(48*Av_st)
    u_pri = u_bar[:,np.newaxis] * (g1[:,np.newaxis] + g2[:,np.newaxis]*self.z_nd**2) + alf[:,np.newaxis]*dsbdx[:,np.newaxis]*(g3[:,np.newaxis]+g4[:,np.newaxis]*self.z_nd**2+g5*self.z_nd**3)
    dspdx = np.sum((dsndx.T)[:,:,np.newaxis] * np.cos(nnp[:,np.newaxis,np.newaxis] * self.z_nd),0)
    
    rho0 = 1000
    fac_K = - self.g*self.Be*rho0/self.H
    
    #subtidal terms potential energy 
    T1kp = fac_K * np.sum(u_bar[:,np.newaxis]*self.H[:,np.newaxis]**2 * dsndx * (-np.sin(nnp)/nnp + nnp**-2 - np.cos(nnp)/nnp**2),1) 
    T2kp = fac_K * np.sum(u_pri * dspdx * self.zlist[0],1) / self.H
    T3kp = fac_K * (u_bar*self.H**2 * dsbdx * (g1/2 + g2/4) - alf*self.H**2 * dsbdx**2 * (g5/5-g3/2-g4/4))
    T5kp = fac_K * alf * self.H * (dsbdx2 + dsbdx/self.bex) * self.H**2 * np.sum(nnp/self.H[:,np.newaxis] * sn * (g3[:,np.newaxis] * (2+(nnp**2-2)*np.cos(nnp))/nnp**3 + g4[:,np.newaxis]/3 * ((24-12*nnp**2+nnp**4)*np.cos(nnp)-24)/nnp**5 + g5/4 * (- (120 - 20*nnp**2 + nnp**4)/nnp**5 * np.cos(nnp)) ) , axis = 1)
    T6kp = fac_K * np.sum(Kv_st[:,np.newaxis]* sn * (nnp * np.sin(nnp) -1 + np.cos(nnp)) , 1) 
    T7kp = fac_K * (Kh_st * np.sum(dsndx2 * (1-np.cos(nnp))/nnp**2 , 1) * self.H**2 + Kh_st/self.bex * np.sum(dsndx * (1-np.cos(nnp))/nnp**2 , 1) * self.H**2 + Kh_st_x * np.sum(dsndx * (1-np.cos(nnp))/nnp**2 , 1) * self.H**2)
    
    #tidal terms potential energy anomaly
    tid_set = self.tid_sets[self.tid_comp[0]]
    tid_geg = self.tid_gegs[self.tid_comp[0]]
    tid_inp = self.tidal_salinity(sss, tid_set, tid_geg)   

    # =============================================================================
    # correctie 
    # =============================================================================
    st_cor = self.calc_cor(sss, indi, 0)
    stb_cor = np.mean(st_cor, 1)
    stp_cor = st_cor - stb_cor[:,np.newaxis]
    
    dstbc_dx, dstpc_dx  = np.zeros(self.di[-1],dtype=complex), np.zeros((self.di[-1],self.nz),dtype=complex)
    for dom in range(self.ndom):
        dstbc_dx[self.di[dom]] = (-3*stb_cor[self.di[dom]] + 4*stb_cor[self.di[dom]+1] - stb_cor[self.di[dom]+2] )/(2*self.dxn[dom])
        dstbc_dx[self.di[dom]+1:self.di[dom+1]-1] = (stb_cor[self.di[dom]+2:self.di[dom+1]] - stb_cor[self.di[dom]:self.di[dom+1]-2])/(2*self.dxn[dom])
        dstbc_dx[self.di[dom+1]-1] = (stb_cor[self.di[dom+1]-3] - 4*stb_cor[self.di[dom+1]-2] + 3*stb_cor[self.di[dom+1]-1] )/(2*self.dxn[dom])
        
        dstpc_dx[self.di[dom]] = (-3*stp_cor[self.di[dom]] + 4*stp_cor[self.di[dom]+1] - stp_cor[self.di[dom]+2] )/(2*self.dxn[dom])
        dstpc_dx[self.di[dom]+1:self.di[dom+1]-1] = (stp_cor[self.di[dom]+2:self.di[dom+1]] - stp_cor[self.di[dom]:self.di[dom+1]-2])/(2*self.dxn[dom])
        dstpc_dx[self.di[dom+1]-1] = (stp_cor[self.di[dom+1]-3] - 4*stp_cor[self.di[dom+1]-2] + 3*stp_cor[self.di[dom+1]-1] )/(2*self.dxn[dom])

    #vertical derivatives
    dstc_dz = np.zeros((self.di[-1],self.nz),dtype=complex)
    dstc_dz[:,1:-1] = (st_cor[:,2:] - st_cor[:,:-2])/(2*self.H[:,np.newaxis]/(self.nz-1))

    dstbc_dx = dstbc_dx[:,np.newaxis]

    # print(dstbc_dx.shape , tid_inp['dstibdx'].shape)

    T1tkp = fac_K * np.sum(1/4 * np.real(tid_geg['utb'][:,np.newaxis] * np.conj(tid_inp['dstipdx']-dstpc_dx) + np.conj(tid_geg['utb'][:,np.newaxis]) * (tid_inp['dstipdx']-dstpc_dx)) * self.zlist[0], 1) / self.H
    T2tkp = fac_K * np.sum(1/4 * np.real(tid_geg['utp'] * np.conj(tid_inp['dstipdx']-dstpc_dx) + np.conj(tid_geg['utp']) * (tid_inp['dstipdx']-dstpc_dx)) * self.zlist[0], 1) / self.H
    T3tkp = fac_K * np.sum(1/4 * np.real(tid_geg['utp'] * np.conj(tid_inp['dstibdx']-dstbc_dx) + np.conj(tid_geg['utp']) * (tid_inp['dstibdx']-dstbc_dx)) * self.zlist[0], 1) / self.H
    T4tkp = fac_K * np.sum(-np.mean(1/4 * np.real(tid_geg['utp'] * np.conj(tid_inp['dstipdx']-dstpc_dx) + np.conj(tid_geg['utp']) * (tid_inp['dstipdx']-dstpc_dx)) , 1)[:,np.newaxis] * self.zlist[0], 1) / self.H
    T5tkp = fac_K * np.sum(1/4 * np.real( (tid_inp['dstdz']-dstc_dz) * np.conj(tid_geg['wt'][0]) + np.conj(tid_inp['dstdz']-dstc_dz) * tid_geg['wt'][0] ) * self.zlist[0], 1) / self.H 
    
    # =============================================================================
    # visualize
    # =============================================================================
    pxh = self.px + self.Ln[-1]/1000



    # plt.plot(pxh , T1kp,label='T1',ls=':',c='grey')
    # plt.plot(pxh , T2kp,label='T2',ls=':',c='grey')
    # plt.plot(pxh , T3kp,label='T3',ls=':',c='blue')
    # plt.plot(pxh , T5kp,label='T5',ls=':',c='grey')
    # plt.plot(pxh , T6kp,label='T6',ls=':',c='red')
    # plt.plot(pxh , T7kp,label='T7',ls=':',c='grey')
    # plt.plot(pxh , T1tkp,label='T1t',c='orange')
    # plt.plot(pxh , T2tkp,label='T2t',c='c')
    # plt.plot(pxh , T3tkp,label='T3t',c='olive')
    # #plt.plot(pxh , T4tkp)
    # plt.plot(pxh , T5tkp,label='T5t',c='m')

    # plt.xlim(-50,0)
    # plt.grid(),plt.legend()
    
    # plt.xlabel('$x$ [km]') , plt.ylabel(r'$\frac{\partial \phi}{\partial t}$ [W m$^{-3}$]')
    
    # plt.show()
    
    plt.plot(pxh , 1e3*T1kp,ls=':',c='grey')
    plt.plot(pxh , 1e3*T2kp,ls=':',c='grey')
    plt.plot(pxh , 1e3*T3kp,label=r'$u^\prime_{st}\frac{\partial \bar{s}_{st}}{\partial x}$',ls=':',c='blue')
    plt.plot(pxh , 1e3*T5kp,ls=':',c='grey')
    plt.plot(pxh , 1e3*T6kp,label=r'$-K_{v,ti} \frac{\partial^2 s^\prime_{st}}{\partial z^2}$',ls=':',c='red')
    plt.plot(pxh , 1e3*T7kp,label=' other \n subtidal \n terms',ls=':',c='grey')
    plt.plot(pxh , 1e3*T1tkp,label=r'$(\bar{u}_{ti}\frac{\partial s^\prime_{ti}}{\partial x})_{st}$',c='orange')
    plt.plot(pxh , 1e3*T2tkp,label=r'$(u^\prime_{ti}\frac{\partial \bar{s}_{ti}}{\partial x})_{st}$',c='c')
    plt.plot(pxh , 1e3*T3tkp,label=r'$(u^\prime_{ti}\frac{\partial s^\prime_{ti}}{\partial x})_{st}$',c='olive')
    plt.plot(pxh , 1e3*T5tkp,label=r'$(w_{ti}\frac{\partial s^\prime_{ti}}{\partial z})_{st}$',c='m')

    plt.xlim(-50,0), plt.ylim(-1,1)
    plt.grid(),plt.legend(loc='center left' , bbox_to_anchor=(1.05,0.5), ncol=1)
    
    plt.xlabel('$x$ [km]') , plt.ylabel(r'$\frac{\partial \phi}{\partial t}$ [$10^{-3}$ W m$^{-3}$]')
    
    #plt.savefig('/Users/biemo004/Documents/UU phd Saltisolutions/Verslagen/Papers/Paper getijmodel/figs/KPA_GUA_v1.jpg',dpi=600,bbox_inches='tight')
    plt.show()
        
    '''
    #plt.plot(pxh, T3kp)
    #plt.plot(pxh, T3tkp)
    plt.plot(pxh, T3tkp /T3kp)
    plt.grid()
    plt.ylim(0,0.2)
    plt.show()
    
    
    print(pxh[158],pxh[241])
    plt.plot(pxh[158:241], (T3tkp /T3kp)[158:241])
    print(np.mean((T3tkp /T3kp)[158:241]))
    '''
    return
    
    
#plot_Kp(run, out4)
 
 
 
 
 
 
 
 
def plot_transport2(self, sss, version):
    # =============================================================================
    # plot transports
    # =============================================================================
    indi = self.ii_all
    ss2 = np.delete(sss , np.concatenate([indi['bnl_rgt'].flatten(),indi['bnl_lft'].flatten(),indi['bnl_bnd']]))

    #calculate salinity
    sb = np.reshape(ss2,(self.di[-1],self.M))[:,0]  * self.soc
    sn = np.reshape(ss2,(self.di[-1],self.M))[:,1:] * self.soc
    
    dsbdx, dsbdx2 = np.zeros(self.di[-1]), np.zeros(self.di[-1])
    for dom in range(self.ndom):
        dsbdx[self.di[dom]] = (-3*sb[self.di[dom]] + 4*sb[self.di[dom]+1] - sb[self.di[dom]+2] )/(2*self.dxn[dom])
        dsbdx[self.di[dom]+1:self.di[dom+1]-1] = (sb[self.di[dom]+2:self.di[dom+1]] - sb[self.di[dom]:self.di[dom+1]-2])/(2*self.dxn[dom])
        dsbdx[self.di[dom+1]-1] = (sb[self.di[dom+1]-3] - 4*sb[self.di[dom+1]-2] + 3*sb[self.di[dom+1]-1] )/(2*self.dxn[dom])

        dsbdx2[self.di[dom]] = (2*sb[self.di[dom]] - 5*sb[self.di[dom]+1] +4*sb[self.di[dom]+2] -1*sb[self.di[dom]+3] )/(self.dxn[dom]**2)
        dsbdx2[self.di[dom]+1:self.di[dom+1]-1] = (sb[self.di[dom]+2:self.di[dom+1]] - 2*sb[self.di[dom]+1:self.di[dom+1]-1] + sb[self.di[dom]:self.di[dom+1]-2])/(self.dxn[dom]**2)
        dsbdx2[self.di[dom+1]-1] = (-sb[self.di[dom+1]-4] + 4*sb[self.di[dom+1]-3] - 5*sb[self.di[dom+1]-2] + 2*sb[self.di[dom+1]-1] )/(self.dxn[dom]**2)

    #dspdz  = np.array([np.sum([sn[i,n-1]*np.pi*n/self.H*-np.sin(np.pi*n*self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    #dspdz2 = np.array([np.sum([sn[i,n-1]*np.pi**2*n**2/self.H**2*-np.cos(np.pi*n*self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    
    #some variables for plottting
    #vertical viscosity
    if self.choice_viscosityv_st == 'constant': Av_st = self.Av_st + np.zeros(self.di[-1]) #do nothing, value is specified
    elif self.choice_viscosityv_st == 'cuh': Av_st = self.cv_st * self.Ut * self.H
    else: print('ERROR: no valid op option for choice vertical viscosity subtidal')

    if self.choice_bottomslip_st == 'sf': rr = Av_st/(self.sf_st*self.H)
    elif self.choice_bottomslip_st == 'rr': rr = self.rr_st + np.zeros(self.di[-1])
    else: print('ERROR: no valid op option for choice bottomslip subtidal')   
    g1 = -1 + (1.5+3*rr) / (1+ 3 *rr)
    g2 =  -3 / (2+6*rr)
    g3 = (1+4*rr) / (1+3*rr) * (9+18*rr) - 8 - 24*rr
    g4 = -9 * (1+4*rr) / (1+3*rr)
    g5 = - 8
    nnp = np.linspace(1,self.N,self.N)*np.pi #n*pi
    #horizontal diffusivity
    if self.choice_diffusivityh_st == 'constant':             
        Kh = self.Kh_st + np.zeros(self.di[-1])
        Kh[self.di[-2]:]= self.Kh_st * self.b[self.di[-2]:]/self.b[self.di[-2]] #sea domain
    if self.choice_diffusivityh_st == 'cub':             
        Kh = self.ch_st * self.Ut * self.b
      
    pxh = self.px + self.Ln[-1]/1000

    T_Q  = self.Q*sb
    T_E  = self.b*self.H*np.sum(sn*(2*(self.Q/(self.b*self.H))[:,np.newaxis]*g2[:,np.newaxis]*np.cos(nnp)/nnp**2 + (self.g*self.Be*self.H**3/(48*Av_st))[:,np.newaxis]*dsbdx[:,np.newaxis]*(2*g4[:,np.newaxis]*np.cos(nnp)/nnp**2 + g5*((6*np.cos(nnp)-6)/nnp**4 - 3*np.cos(nnp)/nnp**2))),1)
    T_D  = self.b*self.H*(-Kh*dsbdx)
    T_tot = T_Q+T_E+T_D
    #tidal transports 
    T_T, T_Tb, T_Tp, T_Tc = {}, {}, {} , {}
    
    for i in range(len(self.tid_comp)):
        tid_set = self.tid_sets[self.tid_comp[i]]
        tid_geg = self.tid_gegs[self.tid_comp[i]]
        tid_inp = self.tidal_salinity(sss, tid_set, tid_geg)   
        
        T_T[self.tid_comp[i]]  = self.b*self.H*1/4 * np.real(tid_geg['ut']*np.conj(tid_inp['st'])   + np.conj(tid_geg['ut'])*tid_inp['st']).mean(1)     
        T_Tb[self.tid_comp[i]] = self.b*self.H*1/4 * np.real(tid_geg['utb']*np.conj(tid_inp['stb'][:,0]) + np.conj(tid_geg['utb'])*tid_inp['stb'][:,0])   
        T_Tp[self.tid_comp[i]] = self.b*self.H*1/4 * np.real(tid_geg['utp']*np.conj(tid_inp['stp']) + np.conj(tid_geg['utp'])*tid_inp['stp']).mean(1)     
    
        if version in ['A']: T_T[self.tid_comp[i]] , T_Tb[self.tid_comp[i]], T_Tp[self.tid_comp[i]] = T_T[self.tid_comp[i]] * 0 , T_Tb[self.tid_comp[i]] * 0 , T_Tp[self.tid_comp[i]] * 0 
        
        T_Tc[self.tid_comp[i]] = self.b*self.H*1/4 * np.real(tid_geg['ut']*np.conj(self.calc_cor(sss, indi, i))   + np.conj(tid_geg['ut'])*self.calc_cor(sss, indi, i)).mean(1)  
        
        T_tot += T_T[self.tid_comp[i]] + T_Tc[self.tid_comp[i]]

    # =============================================================================
    # approximate tidal transport
    # =============================================================================
    sp = np.array([np.sum([sn[i,n-1]*np.cos(np.pi*n *self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    T_Ta = self.b/2 * np.abs(tid_geg['utb']) * sp[:,0] * np.abs(tid_geg['eta'][0,:,0]) * np.cos(np.angle(tid_geg['eta'][0,:,0]) - np.angle(tid_geg['utb']))
    
    
    plt.figure(figsize=(6,3))
    plt.plot(pxh,T_Q,label='$T_Q$',lw = 2, c='g')
    plt.plot(pxh,T_E,label='$T_E$',lw = 2, c='orange')
    plt.plot(pxh,T_D,label='$T_D$',lw = 2, c='firebrick')
    #plt.plot(pxh,T_Tc,label='$T_{Tc}$',lw=2, c='olive')
    #plt.plot(pxh,T_tot,label='$T_{tot}$',lw=2,c='navy')
    #add tides
    ctides = ['skyblue','pink','silver']
    for i in range(len(self.tid_comp)): 
        #plt.plot(pxh,T_T[self.tid_comp[i]],label='$T_{'+self.tid_comp[i]+'}$',lw=2, c=ctides[i])
        #plt.plot(pxh,T_T[self.tid_comp[i]],label='$T_{T}$',lw=2, c=ctides[i])
        plt.plot(pxh,T_Tb[self.tid_comp[i]],label='$T_{Tb}$',lw=2, c='skyblue')#, ls = ':')
        plt.plot(pxh,T_Tp[self.tid_comp[i]],label='$T_{Tp}$',lw=2, c='darkblue')#, ls = "-.")
        #plt.plot(pxh,T_Tc[self.tid_comp[i]],lw=2, c='olive')#,label='$T_{Tc,'+self.tid_comp[i]+'}$')

    #plt.plot(pxh, T_Ta,c='black') #analytical expression for tidal salnity

    plt.legend(loc=2) #plt.legend(loc=2)#,
    plt.grid()
    #plt.xlim(-np.sum(Ln[:-1])/1000,0)
    plt.xlim(-100,5*0)
    plt.xlabel('$x$ [km]',fontsize=12) , plt.ylabel('$T$ [kg/s]',fontsize=12)
    plt.show()
    
    # =============================================================================
    # compare two analytical expressinos for T_Tb
    # =============================================================================
    TT1A = self.b*self.H /(2*tid_geg['omega']) * np.abs(tid_geg['utb'])**2 *dsbdx
    TT1P = np.angle(tid_inp['stb'][:,0]) - np.angle(tid_geg['utb'])    
    TT2A = self.b/2 * np.abs(tid_geg['utb']) * sp[:,0] * np.abs(tid_geg['eta'][0,:,0])
    TT2P = np.angle(tid_geg['eta'][0,:,0]) - np.angle(tid_geg['utb'])    
    
    #plt.plot(pxh,TT1P)
    #plt.plot(pxh,TT2P)
    
    #plt.plot(pxh,TT1A)
    #plt.plot(pxh,TT2A)
    plt.plot(pxh,T_Tb[self.tid_comp[0]],label='num',lw=2, c='black')#, ls = ':')
    plt.plot(pxh,TT1A * np.cos(TT1P), label = 'eq 16')
    plt.plot(pxh,TT2A * np.cos(TT2P), label = 'eq 17')
    plt.xlim(-100,5)
    plt.legend() , plt.grid()
    plt.xlabel('$x$ [km]',fontsize=12) , plt.ylabel('T [kg/s]',fontsize=12)
    plt.show()
    
    print(TT1A.shape, TT1P.shape , TT2A.shape, TT2P.shape)
    
    plt.plot(pxh,(TT2A * np.cos(TT2P) - T_Tb[self.tid_comp[0]]) / (TT2A * np.cos(TT2P)),label='num',lw=2, c='black')#, ls = ':')
    #plt.plot(pxh,TT2A * np.cos(TT2P), label = 'eq 17')
    plt.xlim(-50,5) , plt.ylim(0,1)
    plt.legend() , plt.grid()
    plt.xlabel('$x$ [km]',fontsize=12) , plt.ylabel('T [kg/s]',fontsize=12)
    plt.show()
    
    
    return 


#plot_transport2(run, out4,  'D')

def plot_transport3(self, sss, version):
    # =============================================================================
    # plot transports
    # =============================================================================
    indi = self.ii_all
    ss2 = np.delete(sss , np.concatenate([indi['bnl_rgt'].flatten(),indi['bnl_lft'].flatten(),indi['bnl_bnd']]))

    #calculate salinity
    sb = np.reshape(ss2,(self.di[-1],self.M))[:,0]  * self.soc
    sn = np.reshape(ss2,(self.di[-1],self.M))[:,1:] * self.soc
    
    dsbdx, dsbdx2 = np.zeros(self.di[-1]), np.zeros(self.di[-1])
    for dom in range(self.ndom):
        dsbdx[self.di[dom]] = (-3*sb[self.di[dom]] + 4*sb[self.di[dom]+1] - sb[self.di[dom]+2] )/(2*self.dxn[dom])
        dsbdx[self.di[dom]+1:self.di[dom+1]-1] = (sb[self.di[dom]+2:self.di[dom+1]] - sb[self.di[dom]:self.di[dom+1]-2])/(2*self.dxn[dom])
        dsbdx[self.di[dom+1]-1] = (sb[self.di[dom+1]-3] - 4*sb[self.di[dom+1]-2] + 3*sb[self.di[dom+1]-1] )/(2*self.dxn[dom])

        dsbdx2[self.di[dom]] = (2*sb[self.di[dom]] - 5*sb[self.di[dom]+1] +4*sb[self.di[dom]+2] -1*sb[self.di[dom]+3] )/(self.dxn[dom]**2)
        dsbdx2[self.di[dom]+1:self.di[dom+1]-1] = (sb[self.di[dom]+2:self.di[dom+1]] - 2*sb[self.di[dom]+1:self.di[dom+1]-1] + sb[self.di[dom]:self.di[dom+1]-2])/(self.dxn[dom]**2)
        dsbdx2[self.di[dom+1]-1] = (-sb[self.di[dom+1]-4] + 4*sb[self.di[dom+1]-3] - 5*sb[self.di[dom+1]-2] + 2*sb[self.di[dom+1]-1] )/(self.dxn[dom]**2)

    #dspdz  = np.array([np.sum([sn[i,n-1]*np.pi*n/self.H*-np.sin(np.pi*n*self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    #dspdz2 = np.array([np.sum([sn[i,n-1]*np.pi**2*n**2/self.H**2*-np.cos(np.pi*n*self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    #calculate total salinity 
    s_p = np.array([np.sum([sn[i,n-1]*np.cos(np.pi*n *self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    s = (sb[:,np.newaxis]+s_p)
    s[np.where((s<0) & (s>-0.0001))]= 1e-10 #remove negative values due to numerical precision
    
    #some variables for plottting
    #vertical viscosity
    if self.choice_viscosityv_st == 'constant': Av_st = self.Av_st + np.zeros(self.di[-1]) #do nothing, value is specified
    elif self.choice_viscosityv_st == 'cuh': Av_st = self.cv_st * self.Ut * self.H
    else: print('ERROR: no valid op option for choice vertical viscosity subtidal')

    if self.choice_bottomslip_st == 'sf': rr = Av_st/(self.sf_st*self.H)
    elif self.choice_bottomslip_st == 'rr': rr = self.rr_st + np.zeros(self.di[-1])
    else: print('ERROR: no valid op option for choice bottomslip subtidal')   
    g1 = -1 + (1.5+3*rr) / (1+ 3 *rr)
    g2 =  -3 / (2+6*rr)
    g3 = (1+4*rr) / (1+3*rr) * (9+18*rr) - 8 - 24*rr
    g4 = -9 * (1+4*rr) / (1+3*rr)
    g5 = - 8
    nnp = np.linspace(1,self.N,self.N)*np.pi #n*pi
    #horizontal diffusivity
    if self.choice_diffusivityh_st == 'constant':             
        Kh = self.Kh_st + np.zeros(self.di[-1])
        Kh[self.di[-2]:]= self.Kh_st * self.b[self.di[-2]:]/self.b[self.di[-2]] #sea domain
    if self.choice_diffusivityh_st == 'cub':             
        Kh = self.ch_st * self.Ut * self.b
      
    pxh = self.px + self.Ln[-1]/1000

    T_Q  = self.Q*sb
    T_E  = self.b*self.H*np.sum(sn*(2*(self.Q/(self.b*self.H))[:,np.newaxis]*g2[:,np.newaxis]*np.cos(nnp)/nnp**2 + (self.g*self.Be*self.H**3/(48*Av_st))[:,np.newaxis]*dsbdx[:,np.newaxis]*(2*g4[:,np.newaxis]*np.cos(nnp)/nnp**2 + g5*((6*np.cos(nnp)-6)/nnp**4 - 3*np.cos(nnp)/nnp**2))),1)
    T_D  = self.b*self.H*(-Kh*dsbdx)
    T_tot = T_Q+T_E+T_D
    #tidal transports 
    T_T, T_Tb, T_Tp, T_Tc = {}, {}, {} , {}
    
    for i in range(len(self.tid_comp)):
        tid_set = self.tid_sets[self.tid_comp[i]]
        tid_geg = self.tid_gegs[self.tid_comp[i]]
        tid_inp = self.tidal_salinity(sss, tid_set, tid_geg)   
        
        T_T[self.tid_comp[i]]  = self.b*self.H*1/4 * np.real(tid_geg['ut']*np.conj(tid_inp['st'])   + np.conj(tid_geg['ut'])*tid_inp['st']).mean(1)     
        T_Tb[self.tid_comp[i]] = self.b*self.H*1/4 * np.real(tid_geg['utb']*np.conj(tid_inp['stb'][:,0]) + np.conj(tid_geg['utb'])*tid_inp['stb'][:,0])   
        T_Tp[self.tid_comp[i]] = self.b*self.H*1/4 * np.real(tid_geg['utp']*np.conj(tid_inp['stp']) + np.conj(tid_geg['utp'])*tid_inp['stp']).mean(1)     
    
        if version in ['A']: T_T[self.tid_comp[i]] , T_Tb[self.tid_comp[i]], T_Tp[self.tid_comp[i]] = T_T[self.tid_comp[i]] * 0 , T_Tb[self.tid_comp[i]] * 0 , T_Tp[self.tid_comp[i]] * 0 
        
        T_Tc[self.tid_comp[i]] = self.b*self.H*1/4 * np.real(tid_geg['ut']*np.conj(self.calc_cor(sss, indi, i))   + np.conj(tid_geg['ut'])*self.calc_cor(sss, indi, i)).mean(1)  
        
        T_tot += T_T[self.tid_comp[i]] + T_Tc[self.tid_comp[i]]

    

    

       
    #'''
    #find negative stratificatie
    strat =  s[:,0]-s[:,-1]
    neg = np.where(strat<0)[0]
    #print(neg)
    neg = np.where(sb<0.33)[0]
    #print(neg)
    
    '''
    pxd = np.delete(self.px, neg)
    
    T_Qd = np.delete(T_Q, neg)
    T_Q = np.interp(self.px, pxd, T_Qd)
    
    T_Ed = np.delete(T_E, neg)
    T_E = np.interp(self.px, pxd, T_Ed)
    
    T_Dd = np.delete(T_D, neg)
    T_D = np.interp(self.px, pxd, T_Dd)
    
    T_totd = np.delete(T_tot, neg)
    T_tot = np.interp(self.px, pxd, T_totd)
    
    T_Td = np.delete(T_T[self.tid_comp[0]], neg)
    T_T[self.tid_comp[0]] = np.interp(self.px, pxd, T_Td)
 
    T_Tbd = np.delete(T_Tb[self.tid_comp[0]], neg)
    T_Tb[self.tid_comp[0]] = np.interp(self.px, pxd, T_Tbd)
        
    T_Tpd = np.delete(T_Tp[self.tid_comp[0]], neg)
    T_Tp[self.tid_comp[0]] = np.interp(self.px, pxd, T_Tpd)
    #'''
    if len(neg)>0:
        
        T_Q[:neg[-1]+1] = 0#np.nan
        T_E[:neg[-1]+1] = 0#np.nan
        T_D[:neg[-1]+1] = 0#np.nan
        T_tot[:neg[-1]+1] = 0#np.nan
        
        for i in range(len(self.tid_comp)): 
            T_T[self.tid_comp[i]][:neg[-1]+1] = 0#np.nan
            T_Tp[self.tid_comp[i]][:neg[-1]+1] = 0#np.nan
            T_Tb[self.tid_comp[i]][:neg[-1]+1] = 0#np.nan
        
    plt.figure(figsize=(6,3))
    plt.plot(pxh,T_Q,label='$T_Q$',lw = 2, c='g')
    plt.plot(pxh,T_E,label='$T_E$',lw = 2, c='orange')
    plt.plot(pxh,T_D,label='$T_D$',lw = 2, c='firebrick')
    #plt.plot(pxh,T_Tc,label='$T_{Tc}$',lw=2, c='olive')
    plt.plot(pxh,T_tot,label='$T_{tot}$',lw=2,c='black')
    #add tides
    ctides = ['skyblue','pink','silver']
    for i in range(len(self.tid_comp)): 
        #plt.plot(pxh,T_T[self.tid_comp[i]],label='$T_{'+self.tid_comp[i]+'}$',lw=2, c=ctides[i])
        #plt.plot(pxh,T_T[self.tid_comp[i]],label='$T_{T}$',lw=2, c=ctides[i])
        plt.plot(pxh,T_Tb[self.tid_comp[i]],label='$T_{Tb}$',lw=2, c='skyblue')#, ls = ':')
        plt.plot(pxh,T_Tp[self.tid_comp[i]],label='$T_{Tp}$',lw=2, c='darkblue')#, ls = "-.")
        #plt.plot(pxh,T_Tc[self.tid_comp[i]],lw=2, c='olive')#,label='$T_{Tc,'+self.tid_comp[i]+'}$')


    plt.legend(loc=2) #plt.legend(loc=2)#,
    plt.grid()
    #plt.xlim(-np.sum(Ln[:-1])/1000,0)
    plt.xlim(-100,5) #, plt.ylim(-1000,1000)
    plt.xlabel('$x$ [km]',fontsize=12) , plt.ylabel('$T$ [kg/s]',fontsize=12)
    plt.show()
        
    return T_Q, T_E, T_D, T_T,  T_tot,  T_Tb, T_Tp,

#plot_transport3(run, out4,  'D')
