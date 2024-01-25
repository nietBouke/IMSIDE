# =============================================================================
# plot functions  
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

def plot_sst(self,sss, indi):
     
    ss2 = np.delete(sss , np.concatenate([indi['bnl_rgt'].flatten(),indi['bnl_lft'].flatten(),indi['bnl_bnd']]))
    pxh = np.repeat(self.px, self.nz).reshape((self.di[-1],self.nz)) + 25

    # =============================================================================
    # Plot salt field
    # =============================================================================
   
    #calculate and plot total salinity 
    s_b = np.transpose([np.reshape(ss2,(self.di[-1],self.M))[:,0]]*self.nz)
    sn  = np.reshape(ss2,(self.di[-1],self.M))[:,1:]
    s_p = np.array([np.sum([sn[i,n-1]*np.cos(np.pi*n *self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    s   = (s_b+s_p)*self.soc_sca
    s[np.where((s<0) & (s>-0.0001))]= 1e-10 #remove negative values due to numerical precision

    #make contourplot   
    fig,ax = plt.subplots(figsize=(10,7))
    l1=ax.contourf(pxh,self.pz,  s, cmap='RdBu_r',levels=(np.linspace(0,self.soc_sca,36)))
    #ax.quiver(qx,qz,qu.transpose(),qw.transpose(),color='white')
    cb0 = fig.colorbar(l1, ax=ax,orientation='horizontal', pad=0.16)
    cb0.set_label(label='Salinity [psu]',fontsize=16)
    ax.set_xlabel('$x$ [km]',fontsize=16) 
    ax.set_ylabel('$z$ [m]',fontsize=16)    
    ax.set_xlim(-30,0)
    ax.set_facecolor('black')
    plt.show()  

#plot_sst(run, out[10], run.ii_all)
#plot_sst(run, out[11], run.ii_all)
#plot_sst(run, out[12], run.ii_all)


def prt_numbers(self, sss, indi):
    
    sss = np.delete(sss , np.concatenate([indi['bnl_rgt'].flatten(),indi['bnl_lft'].flatten(),indi['bnl_bnd']]))

    #calculate total salinity 
    s_b = np.transpose([np.reshape(sss,(self.di[-1],self.M))[:,0]]*self.nz)
    sn = np.reshape(sss,(self.di[-1],self.M))[:,1:]
    s_p = np.array([np.sum([sn[i,n-1]*np.cos(np.pi*n *self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    s = (s_b+s_p)*self.soc_sca
    s[np.where((s<0) & (s>-0.0001))]= 1e-10 #remove negative values due to numerical precision


    print()
    print('The salt intrusion length is ',-self.px[np.where(s[:,0]>2)[0][0]]-self.Ln[-1]/1000,' km')
    print('The minimum salinity is ',np.min(s),' psu')
    print('The salinity at the sea boundary at the bottom is ', s[np.where(self.px==-self.Ln[-1]/1000)[0][0],0])
    
    
    
def plot_transport(self, sss, indi, t):
    # =============================================================================
    # plot transports
    # =============================================================================
 
    tid_inp = self.tidal_salinity(sss)
    ss2 = np.delete(sss , np.concatenate([indi['bnl_rgt'].flatten(),indi['bnl_lft'].flatten(),indi['bnl_bnd']]))


    #calculate salinity
    sb = np.reshape(ss2,(self.di[-1],self.M))[:,0]  * self.soc_sca
    sn = np.reshape(ss2,(self.di[-1],self.M))[:,1:] * self.soc_sca
    
    dsbdx, dsbdx2 = np.zeros(self.di[-1]), np.zeros(self.di[-1])
    for dom in range(self.ndom):
        dsbdx[self.di[dom]] = (-3*sb[self.di[dom]] + 4*sb[self.di[dom]+1] - sb[self.di[dom]+2] )/(2*self.dxn[dom])
        dsbdx[self.di[dom]+1:self.di[dom+1]-1] = (sb[self.di[dom]+2:self.di[dom+1]] - sb[self.di[dom]:self.di[dom+1]-2])/(2*self.dxn[dom])
        dsbdx[self.di[dom+1]-1] = (sb[self.di[dom+1]-3] - 4*sb[self.di[dom+1]-2] + 3*sb[self.di[dom+1]-1] )/(2*self.dxn[dom])

        dsbdx2[self.di[dom]] = (2*sb[self.di[dom]] - 5*sb[self.di[dom]+1] +4*sb[self.di[dom]+2] -1*sb[self.di[dom]+3] )/(self.dxn[dom]**2)
        dsbdx2[self.di[dom]+1:self.di[dom+1]-1] = (sb[self.di[dom]+2:self.di[dom+1]] - 2*sb[self.di[dom]+1:self.di[dom+1]-1] + sb[self.di[dom]:self.di[dom+1]-2])/(self.dxn[dom]**2)
        dsbdx2[self.di[dom+1]-1] = (-sb[self.di[dom+1]-4] + 4*sb[self.di[dom+1]-3] - 5*sb[self.di[dom+1]-2] + 2*sb[self.di[dom+1]-1] )/(self.dxn[dom]**2)

    dspdz  = np.array([np.sum([sn[i,n-1]*np.pi*n/self.H*-np.sin(np.pi*n*self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    dspdz2 = np.array([np.sum([sn[i,n-1]*np.pi**2*n**2/self.H**2*-np.cos(np.pi*n*self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    
    #some variables for plottting
    rr = 0.5 #self.Av_st/(self.sf_st*self.H)
    g1 = -1 + (1.5+3*rr) / (1+ 3 *rr)
    g2 =  -3 / (2+6*rr)
    g3 = (1+4*rr) / (1+3*rr) * (9+18*rr) - 8 - 24*rr
    g4 = -9 * (1+4*rr) / (1+3*rr)
    g5 = - 8
    nnp = np.linspace(1,self.N,self.N)*np.pi #n*pi
    Kh = self.Kh_st + np.zeros(self.di[-1])
    Kh[self.di[-2]:]= self.Kh_st * self.b[self.di[-2]:]/self.b[self.di[-2]] #sea domain
    
    pxh = self.px + self.Ln[-1]/1000

    T_Q = self.Q[t]*sb
    T_E = self.b*self.H*np.sum(sn*(2*(self.Q[t]/(self.b*self.H))[:,np.newaxis]*g2*np.cos(nnp)/nnp**2 + (self.g*self.Be*self.H**3/(48*self.Av_st))*dsbdx[:,np.newaxis]*(2*g4*np.cos(nnp)/nnp**2 + g5*((6*np.cos(nnp)-6)/nnp**4 - 3*np.cos(nnp)/nnp**2))),1)
    T_D = self.b*self.H*(-Kh*dsbdx)
    T_T = self.b*self.H*1/4 * np.real(self.ut*np.conj(tid_inp['st']) + np.conj(self.ut)*tid_inp['st']).mean(1)     
    #T_T1 = -b*H*(tide_module(sss_n, (H, Ln, b0, bs, dxn), inp_t, (51,121),'fluxes')[1])
    #T_T2 = -b*H*(tide_module(sss_n, (H, Ln, b0, bs, dxn), inp_t, (51,121),'fluxes')[2])
    #T_Ts = -b[di[-2]:]*H*(tide_module(sss, (H, Ln, b0, bs, dxn), inp_t, (51,121),'fluxes')[3])
    T_Tc = -self.b*self.H*self.calc_cor(sss, tid_inp, indi)[1]

    T_tot = T_Q+T_E+T_D+T_T+T_Tc


    
    plt.figure(figsize=(6,3))
    plt.plot(pxh,T_Q,label='$T_Q$',lw = 2, c='g')
    plt.plot(pxh,T_E,label='$T_E$',lw = 2, c='orange')
    plt.plot(pxh,T_D,label='$T_D$',lw = 2, c='firebrick')
    plt.plot(pxh,T_T,label='$T_{T}$',lw=2, c='skyblue')
    plt.plot(pxh,T_Tc,label='$T_{Tc}$',lw=2, c='olive')
    plt.plot(pxh,T_tot,label='$T_{tot}$',lw=2,c='navy')
    
    plt.legend(loc=2) #plt.legend(loc=2)#,
    plt.grid()
    #plt.xlim(-np.sum(Ln[:-1])/1000,0)
    plt.xlim(-75,0)
    plt.xlabel('$x$ [km]') , plt.ylabel('$T$ [kg/s]')
    plt.show()
    
    
def plot_X2(self, sss_sav, indi):
    # =============================================================================
    # plot the salt intrusion length as a function of time 
    # =============================================================================
    Lint = np.zeros(self.T)

    for t in range(self.T):
        ss2 = np.delete(sss_sav[t] , np.concatenate([indi['bnl_rgt'].flatten(),indi['bnl_lft'].flatten(),indi['bnl_bnd']]))
        
        #calculate salt intrusion length 
        sbot = self.soc_sca*(np.reshape(ss2,(self.di[-1],self.M))[:,0] + np.sum(np.reshape(ss2,(self.di[-1],self.M))[:,1:]*np.array([(-1)**n for n in range(1,self.M)]),1))
        Lint[t] = -self.px[np.where(sbot>2)[0][0]] - self.Ln[-1]/1000 



    fig, ax = plt.subplots(2,1,figsize = (7,7))
    ax[0].plot(self.Tvec[1:],self.Q,c='blue',lw=2)
    ax[1].plot(self.Tvec[1:],Lint,c='black',lw=2)
    
    ax[1].set_xlabel('Time [days]')
    ax[1].set_ylabel('$X_2$ [km]')
    ax[0].set_ylabel('$Q$ [m3/s]')
    
    ax[0].grid()
    ax[1].grid()
    plt.show() 
