# =============================================================================
# Visulisation of the results from IMSIDE - canal
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

def calc_rawtofine(self):
    # =============================================================================
    # compute the salt field (t,x,z) from the model output
    # =============================================================================
    saltfield = np.zeros((self.T, self.di[-1],self.nz)) + np.nan
    sigma=np.linspace(-1, 0, self.nz)
    
    for t in range(self.T): #for every timestep
        sss=self.sss_save[t]
        #compute the salt field according to the spectral methods and stuff
        s_b = np.reshape(sss[:self.di[-1]*self.M],(self.di[-1],self.M))[:,0] * self.soc_sca
        sn = np.reshape(sss[:self.di[-1]*self.M],(self.di[-1],self.M))[:,1:] * self.soc_sca
        s_p = np.array([np.sum([sn[i,n-1]*np.cos(np.pi*n*sigma) for n in range(1,self.M)],0) for i in range(self.di[-1])])

        saltfield[t] = s_p+s_b[:,np.newaxis]
        
    return saltfield

def plot_timeseries_point(self, saltfield, xpt, zpt): 
    # =============================================================================
    #  plot the timeseries of salinity at a certain point xpt at depth zpt
    # =============================================================================
    #note that xpt=0 is the lock, and e.g. xpt = -10 is 10 km upstream from the lock
    # xpt is in km and zpt in m. sorry for the inconsistency

    #get salinity at this point
    xi_here = np.argmin(np.abs(self.px-xpt))
    zi_here = np.argmin(np.abs(self.pz[xi_here] -zpt))
    ss_here = saltfield[:,xi_here, zi_here]

    #plot
    plt.plot(self.tvec/86400,ss_here , lw=2, c='k')
    plt.ylabel('Salinity [g/kg]')
    plt.xlabel('Time [days]')
    plt.grid()
    plt.show()
    
    
    return 


def plot_timeseries_sil(self, saltfield, scrit = 2):
    # =============================================================================
    # plot time series of the salt intrusion length, based on depth-averaged salinity
    # =============================================================================
    
    sda = np.mean(saltfield,2) #depth-averaged salinity
    SIL = []
    for t in range(self.T):
        if np.max(sda[t]) < scrit:  SIL.append(0)
        elif np.min(sda[t]) > scrit: SIL.append(np.min(self.px))
        else: 
            isil = np.where(sda[t] > scrit)[0][0]
            SIL.append(self.px[isil])
            
    SIL = np.array(SIL)
    SIL[SIL>0] = 0
    #plot  
    plt.plot(self.tvec/86400, SIL, c= 'k',lw=2)
    plt.xlabel('Time [days]')
    plt.ylabel('Salt intrusion length [km]')
    plt.grid()
    plt.show()
        
    
    return 

def plot_saltcontour(self, saltfield, t):
    # =============================================================================
    # plot a x,z contour of the salt field at a given time t. 
    # =============================================================================
   
    
    #plot
    plt.figure(figsize = (8,4))
    plt.contourf(self.px.repeat(self.nz).reshape((self.di[-1],self.nz)).T, self.pz.T, saltfield[t].T , cmap = 'RdBu_r')
    plt.title("Salinity field at day " + str(self.tvec[t]/86400)) 
    plt.ylabel('$z$ [m]')
    plt.xlabel('$x$ [km]')
    plt.grid()
    plt.xlim(np.min(self.pz),0)
    cb = plt.colorbar()
    cb.set_label(label ='Salinity [g/kg]')
    plt.show()
    
    
    
    return 
    
    