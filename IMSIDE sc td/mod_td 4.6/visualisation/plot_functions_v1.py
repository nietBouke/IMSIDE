# =============================================================================
# plot functions  
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

def plot_sst(self,sss):    
    # =============================================================================
    # Plot salt field for a certain time step
    # =============================================================================

    #load
    indi= self.ii_all
    ss2 = np.delete(sss , np.concatenate([indi['bnl_rgt'].flatten(),indi['bnl_lft'].flatten(),indi['bnl_bnd']]))
   
    #calculate and plot total salinity 
    s_b = np.transpose([np.reshape(ss2,(self.di[-1],self.M))[:,0]]*self.nz)
    sn  = np.reshape(ss2,(self.di[-1],self.M))[:,1:]
    s_p = np.array([np.sum([sn[i,n-1]*np.cos(np.pi*n *self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    s   = (s_b+s_p)*self.soc_sca
    s[np.where((s<0) & (s>-0.0001))]= 1e-10 #remove negative values due to numerical precision

    #make contourplot
    pxh = np.repeat(self.px+25,self.nz).reshape(len(self.px),self.nz)
    
    fig,ax = plt.subplots(figsize=(10,7))
    l1=ax.contourf(pxh,self.pz,  s, cmap='RdBu_r',levels=(np.linspace(0,self.soc_sca,36)))
    #ax.quiver(qx,qz,qu.transpose(),qw.transpose(),color='white')
    cb0 = fig.colorbar(l1, ax=ax,orientation='horizontal', pad=0.16)
    cb0.set_label(label='Salinity [psu]',fontsize=16)
    ax.set_xlabel('$x$ [km]',fontsize=16) 
    ax.set_ylabel('$z$ [m]',fontsize=16)    
    ax.set_xlim(-30,0)
    plt.show()  

    
def plot_X2(self, sss_sav,):
    # =============================================================================
    # plot the salt intrusion length as a function of time 
    # =============================================================================
    Lint = np.zeros(self.T)
    indi= self.ii_all

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
