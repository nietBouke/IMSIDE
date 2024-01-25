# =============================================================================
# plot functions  
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

def plot_sst(self,sss, indi):
    
     
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

#plot_sst(run , out4, run.ii_all)

def prt_numbers(self, sss, indi, prt = True):
    
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
    
def plot_transport(self, sss, indi, version):
    # =============================================================================
    # plot transports
    # =============================================================================
 
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
    if self.choice_bottomslip_st == 'sf': rr = self.Av_st/(self.sf_st*self.H)
    elif self.choice_bottomslip_st == 'rr': rr = self.rr_st + np.zeros(self.di[-1])
    else: print('ERROR: no valid op option for choice bottomslip subtidal')   
    g1 = -1 + (1.5+3*rr) / (1+ 3 *rr)
    g2 =  -3 / (2+6*rr)
    g3 = (1+4*rr) / (1+3*rr) * (9+18*rr) - 8 - 24*rr
    g4 = -9 * (1+4*rr) / (1+3*rr)
    g5 = - 8
    nnp = np.linspace(1,self.N,self.N)*np.pi #n*pi
    Kh = self.Kh_st + np.zeros(self.di[-1])
    Kh[self.di[-2]:]= self.Kh_st * self.b[self.di[-2]:]/self.b[self.di[-2]] #sea domain
    
    pxh = self.px + self.Ln[-1]/1000

    T_Q  = self.Q*sb
    T_E  = self.b*self.H*np.sum(sn*(2*(self.Q/(self.b*self.H))[:,np.newaxis]*g2[:,np.newaxis]*np.cos(nnp)/nnp**2 + (self.g*self.Be*self.H**3/(48*self.Av_st))[:,np.newaxis]*dsbdx[:,np.newaxis]*(2*g4[:,np.newaxis]*np.cos(nnp)/nnp**2 + g5*((6*np.cos(nnp)-6)/nnp**4 - 3*np.cos(nnp)/nnp**2))),1)
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
        

    '''
    plt.figure(figsize=(6,3))
    plt.plot(pxh,T_Q,label='$T_Q$',lw = 2, c='g')
    plt.plot(pxh,T_E,label='$T_E$',lw = 2, c='orange')
    plt.plot(pxh,T_D,label='$T_D$',lw = 2, c='firebrick')
    #plt.plot(pxh,T_Tc,label='$T_{Tc}$',lw=2, c='olive')
    plt.plot(pxh,T_tot,label='$T_{tot}$',lw=2,c='navy')
    #add tides
    ctides = ['skyblue','pink','silver']
    for i in range(len(self.tid_comp)):
        plt.plot(pxh,T_T[self.tid_comp[i]],label='$T_{'+self.tid_comp[i]+'}$',lw=2, c=ctides[i])
        #plt.plot(pxh,T_Tb,label='$T_{Tb}$',lw=2, c='pink')
        #plt.plot(pxh,T_Tp,label='$T_{Tp}$',lw=2, c='silver')
        plt.plot(pxh,T_Tc[self.tid_comp[i]],label='$T_{Tc,'+self.tid_comp[i]+'}$',lw=2, c='olive')

    plt.legend(loc=2) #plt.legend(loc=2)#,
    plt.grid()
    #plt.xlim(-np.sum(Ln[:-1])/1000,0)
    plt.xlim(-50,0)
    plt.xlabel('$x$ [km]') , plt.ylabel('$T$ [kg/s]')
    plt.show()
    '''
    
    plt.figure(figsize=(6,3))
    plt.plot(pxh,T_Q,label='$T_Q$',lw = 2, c='g')
    plt.plot(pxh,T_E,label='$T_E$',lw = 2, c='orange')
    plt.plot(pxh,T_D,label='$T_D$',lw = 2, c='firebrick')
    #plt.plot(pxh,T_Tc,label='$T_{Tc}$',lw=2, c='olive')
    plt.plot(pxh,T_tot,label='$T_{tot}$',lw=2,c='navy')
    #add tides
    ctides = ['skyblue','pink','silver']
    for i in range(len(self.tid_comp)):
        #plt.plot(pxh,T_T[self.tid_comp[i]],label='$T_{'+self.tid_comp[i]+'}$',lw=2, c=ctides[i])
        plt.plot(pxh,T_Tb[self.tid_comp[i]],label='$T_{Tb}$',lw=2, c=ctides[i], ls = ':')
        plt.plot(pxh,T_Tp[self.tid_comp[i]],label='$T_{Tp}$',lw=2, c=ctides[i], ls = "-.")
        #plt.plot(pxh,T_Tc[self.tid_comp[i]],lw=2, c='olive')#,label='$T_{Tc,'+self.tid_comp[i]+'}$')

    plt.legend(loc=2) #plt.legend(loc=2)#,
    plt.grid()
    #plt.xlim(-np.sum(Ln[:-1])/1000,0)
    plt.xlim(-50,5)
    plt.xlabel('$x$ [km]') , plt.ylabel('$T$ [kg/s]')
    plt.show()
    
    return T_Q, T_E, T_D, T_T,  T_tot #T_Tb, T_Tp,

#plot_transport(run, out4, run.ii_all, 'D')


def plot_next(self, sss, indi, version):
    # =============================================================================
    # plot transports
    # =============================================================================
    i=0
    tid_set = self.tid_sets[self.tid_comp[i]]
    tid_geg = self.tid_gegs[self.tid_comp[i]]
    tid_inp = self.tidal_salinity(sss, tid_set, tid_geg)
    ss2 = np.delete(sss , np.concatenate([indi['bnl_rgt'].flatten(),indi['bnl_lft'].flatten(),indi['bnl_bnd']]))

    #calculate total salinity 
    s_b = np.transpose([np.reshape(ss2,(self.di[-1],self.M))[:,0]]*self.nz)
    sn = np.reshape(ss2,(self.di[-1],self.M))[:,1:]
    s_p = np.array([np.sum([sn[i,n-1]*np.cos(np.pi*n *self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    s = (s_b+s_p)*self.soc
    s[np.where((s<0) & (s>-0.0001))]= 1e-10 #remove negative values due to numerical precision
    
    
    
    #plot stratificatie
    strat =  s[:,0]-s[:,-1]
    pxh = self.px + self.Ln[-1]/1000

    '''
    plt.plot(pxh , strat)
    plt.xlim(-110,0)
    plt.ylabel(r'$\Delta s$ [psu]') , plt.xlabel('$x$ [km]')
    plt.show()
    '''
    stb = tid_inp['stb']
    stp = tid_inp['stp']
    
    stb_pha = np.angle(stb[:,0])/np.pi*180
    utb_pha = np.angle(tid_geg['utb'])/np.pi*180
    dif_phab = stb_pha - utb_pha
    stb_amp = np.abs(stb[:,0])
    utb_amp = np.abs(tid_geg['utb'])
    
    fig , ax = plt.subplots(3,1,figsize=(3,7))
    ax[0].plot(pxh[1:] , dif_phab[1:])
    ax[1].plot(pxh[1:] , stb_amp[1:])
    ax[2].plot(pxh[1:] , utb_amp[1:])
    [ax[i].set_xlim(-60,0) for i in range(3)], ax[0].set_ylim(89,95) , [ax[i].grid() for i in range(3)]
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
    return pxh, strat, dif_phab
    

#plot_next(run, out4, run.ii_all, 'D')


def terms_tide(self, sss):
    # =============================================================================
    # function to calculate the phase difference generation in the depth-averaged salinity
    # variatio nin the tidal cylce
    # =============================================================================
    pxh = self.px + self.Ln[-1]/1000

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
        l1=ax.contourf(px_tide,pz_tide[:,:,0], stot[:,:,0].T, cmap='RdBu_r',levels=(np.linspace(0,35,15)),extend='both')
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
        ax.contourf(px_tide,pz_tide[:,:,t], stot[:,:,t].T, cmap='RdBu_r',levels=(np.linspace(0,35,15)),extend='both')
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
    


#animate_salt(run , out4, 'test_041223_v5')

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
