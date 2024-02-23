# =============================================================================
# plotting functions for the Rhine-Meuse Delta
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt         
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.animation as ani       #make animations
import matplotlib as mpl



def calc_hourly_salinity(self, key, xi, zi):
    # =============================================================================
    # calculate an hourly timeseries of salinity at a given point.
    # key is the channel, xi is the index of the x coordinate in this channel, zi is the z coordinate
    # =============================================================================        
    dth = 3600 #seconds in an hour
    
    #hourly time series
    Tvec_new = np.arange(0,np.sum(self.DT[1:])+dth,dth)/86400
    #subtidal salinity
    s_st = self.ch_outp[key]['s_st'][:,xi,zi]
    #subtidal slainity interpolated 
    s_st_intp = np.interp(Tvec_new,self.Tvec,s_st)

    #tidal salinity
    #extract
    s_ti_raw = np.zeros(self.T , dtype=complex)
    for tt in range(self.T):
        sss = self.out[tt,self.ch_inds[key]['totx'][0]*self.M:self.ch_inds[key]['totx'][-1]*self.M]
        s_ti_raw[tt] = self.tidal_salinity(key,sss)['st'][xi,zi]
    s_ti_abs , s_ti_ang = np.abs(s_ti_raw) , np.angle(s_ti_raw)
    
    #interpolate the same timeseries 
    s_ti_abs_intp = np.interp(Tvec_new,self.Tvec,s_ti_abs)
    s_ti_ang_intp = np.interp(Tvec_new,self.Tvec,s_ti_ang)
    #build timeseries
    s_ti_series = np.real(s_ti_abs_intp * np.exp(1j*(self.omega*Tvec_new*86400 + s_ti_ang_intp)))
    '''
    #plot timeseries 
    plt.plot(Tvec_new , s_ti_series + s_st_intp )
    plt.grid()
    plt.show()
    '''
    
    return Tvec_new , s_st_intp , s_ti_series 
    

def plot_salt_pointRM(self,locE,locN,depth):
    # =============================================================================
    # plot the salinty timeseries at a certain location in the Rhine-Meuse Delta
    # I think this works in theory also for other locations
    # locE is the longitude of the point you want to consider, locN is the lattitude
    # depth is the depth in meters where you want the plot . 
    # this function finds the closest grid cell in the model and plots the salinty time series there
    # =============================================================================

    # =============================================================================
    # Subtract the model output at the location of the observations
    # =============================================================================
    #find closest point. Note that the difference is in degrees and not in meters. This should not be too much of a problem. 
    close1,close2 = [] , []
    for key in self.ch_keys:
        temp = ((self.ch_outp[key]['plot xs']-locE)**2+(self.ch_outp[key]['plot ys']-locN)**2)**0.5
        close1.append(np.min(temp))
        close2.append(np.argmin(temp))
    if np.min(close1)>1000:#0.01: 
        ind_ch = self.ch_keys[np.argmin(close1)] #name of the channel where the point is
        ind_co = close2[np.argmin(close1)]  #index of the point in this channel
        print('WARNING: point '+punten[pt]+' too far away from estuary, not plotted. Coordinates: '
              ,self.ch_outp[ind_ch]['plot ys'][ind_co],self.ch_outp[ind_ch]['plot xs'][ind_co])
        return
        
    ind_ch = self.ch_keys[np.argmin(close1)] #name of the channel where the point is
    ind_co = close2[np.argmin(close1)]  #index of the point in this channel
    
    #subtract salinity
    depth_ind = self.nz-int(depth/self.ch_pars[ind_ch]['H'][ind_co]*self.nz+1)
    #sb = self.ch_outp[ind_ch]['sb_st'][:,ind_co]
    #ss = self.ch_outp[ind_ch]['s_st'][:,ind_co,depth_ind]
    tvec, sst, sti = calc_hourly_salinity(self, ind_ch, ind_co, depth_ind)

    #plot
    plt.plot(tvec,sst+sti,'red',linewidth=2,label='mod')
    plt.grid(),plt.legend()
    plt.ylabel('Salinity [g/kg]')
    plt.show()
       
    return sst+sti

#plot_salt_pointRM(delta,4.2,51.95,1)
#plot_salt_pointRM(delta,4.4,51.9,1)

def anim_RM_st(self,savename):
    # =============================================================================
    # Function to animate subtidal salinity variations
    # Tidal variation is not added yet
    # =============================================================================

    print('Warning: making of this animation is probably quite slow, find something to do in the meantime')
    
    keys_here = self.ch_keys    
    norm1 = plt.Normalize(0,35)

    # =============================================================================
    # Plot
    # =============================================================================
    fig = plt.figure(figsize=(16,9))
    gs = fig.add_gridspec(11,12+2)
    ax_cen = fig.add_subplot(gs[4:7,4:9])
    #ax_leg = fig.add_subplot(gs[0:3,-2])
    
    ax0 = fig.add_subplot(gs[0:3,1:4])
    ax1 = fig.add_subplot(gs[0:3,5:8])
    ax2 = fig.add_subplot(gs[0:3,9:12])
    
    ax3 = fig.add_subplot(gs[4:7,0:3])
    ax4 = fig.add_subplot(gs[4:7,10:13])

    ax5 = fig.add_subplot(gs[8:11,1:4])
    ax6 = fig.add_subplot(gs[8:11,5:8])
    ax7 = fig.add_subplot(gs[8:11,9:12])
    
    keys_now = ['Breeddiep' , 'Nieuwe Waterweg v2', 'Nieuwe Maas 1 old', 'Hartelkanaal v2', 'Hollandse IJssel', 'Oude Maas 2', 'Spui', 'Oude Maas 3']
    axes = [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7]
    
    '''
    #make water level varying for tides
    coords_tides = {}
    
    for k in range(len(axes)):
        key = keys_now[k]

        #take water level into account for htis
        px_tide = np.tile(self.ch_outp[key]['px'],self.nz).reshape(self.nz,len(self.ch_outp[key]['px']))
        pz_tide = np.zeros((self.nz,len(self.ch_outp[key]['px']),self.nt)) + np.nan
        for t in range(self.nt):
            pz_1t = np.zeros((self.nz,len(self.ch_outp[key]['px']))) + np.nan
            for x in range(len(self.ch_outp[key]['px'])): pz_1t[:,x] = np.linspace(-self.ch_gegs[key]['H'],self.ch_outp[key]['eta_r'][x,t],self.nz)
            pz_tide[:,:,t] = pz_1t
                
        coords_tides[key] = (px_tide , pz_tide)
    '''
    def init():
        plot_t = 0
        ax_cen.cla(), ax0.cla(), ax1.cla(), ax2.cla(), ax3.cla(), ax4.cla(), ax5.cla(), ax6.cla(), ax7.cla()
        
        #overview plot
        for key in self.ch_keys:
            #plot salinity
            self.ch_outp[key]['lc1'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm1)
            self.ch_outp[key]['lc1'].set_array(self.ch_outp[key]['sb_st'][plot_t])
            self.ch_outp[key]['lc1'].set_linewidth(5)
            line1=ax_cen.add_collection(self.ch_outp[key]['lc1'])
            
            #plot the salt intrusion limit
            #subtidal 
            i0 = np.where(self.ch_outp[key]['sb_st'][plot_t]>2)[0]
            i1 = np.where(self.ch_outp[key]['sb_st'][plot_t]<2)[0]
            if len(i0)>1 and len(i1)>0: ax_cen.scatter(*self.ch_outp[key]['segments'][i0[0]][0],marker='|',color='red',linewidth=3,zorder=100)
            
        #layout overview salinity plot
        ax_cen.axis('scaled')
        ax_cen.set_xlim(4,4.6) , ax_cen.set_ylim(51.75,52.02) #zoom in on mouth RM
        
        ax_cen.set_ylabel('degrees N ')
        ax_cen.set_xlabel('degrees E ')
        ax_cen.set_facecolor('grey')
        
        cb1=plt.colorbar(line1, ax=ax_cen,orientation='vertical')
        cb1.set_label(label=r'$\bar{s}$ [g/kg]')    
        

        #Plot the salinity in 2DV                
        for k in range(len(axes)):
            key = keys_now[k]
            s_here = self.ch_outp[key]['s_st'][plot_t]
               
            cm = mpl.cm.get_cmap("RdBu_r").copy()
            a = axes[k].contourf(self.ch_outp[key]['px']*1e-3,self.z_nd, s_here.T,cmap = cm,levels = np.linspace(0,35,15),extend='both')
            
            axes[k].set_title(self.ch_gegs[key]['Name'])
            axes[k].set_xlabel('$x$ [km]'),axes[k].set_ylabel('$z/H$ [ ]')
            #axes[k].set_facecolor('lightgrey')
            if keys_now[k] != 'Spui': axes[k].invert_xaxis()
            
            '''
            import matplotlib
            cmap = matplotlib.cm.get_cmap('RdBu_r')
            cmin,cmax = cmap(0),cmap(1)
            a.cmap.set_under(cmin), a.cmap.set_over(cmax)
            #a.cmap.set_under('white'), a.cmap.set_over('white')
            a.set_clim(0,35)
            '''
            
        #add the arrows
        ax_cen.arrow(4.06,52,-0.1,0.1,clip_on = False,head_width=0.02,color='g',width=0.005)
        ax_cen.arrow(4.25,51.94,0.01,0.14,clip_on = False,head_width=0.02,color='g',width=0.005)
        ax_cen.arrow(4.47,51.93,0.2,0.22,clip_on = False,head_width=0.02,color='g',width=0.005)
        ax_cen.arrow(4.15,51.91,-0.27,0.01,clip_on = False,head_width=0.02,color='g',width=0.005)
        ax_cen.arrow(4.35,51.8,0.03,-0.12,clip_on = False,head_width=0.02,color='g',width=0.005)
        
        ax_cen.arrow(4.34,51.84,-0.2,0,clip_on = False,head_width=0,color ='g',width=0.005)
        ax_cen.arrow(4.14,51.84,-0.15,-0.2,clip_on = False,head_width=0.02,color ='g',width=0.005)
    
        ax_cen.arrow(4.55,51.81,0,-0.15,clip_on = False,head_width=0,color ='g',width=0.005)
        ax_cen.arrow(4.55,51.66,0.2,-0.04,clip_on = False,head_width=0.02,color ='g',width=0.005)
    
        ax_cen.arrow(4.58,51.94,0,0.07,clip_on = False,head_width=0,color ='g',width=0.005)
        ax_cen.arrow(4.58,52.01,0.25,0.01,clip_on = False,head_width=0.02,color ='g',width=0.005)


    def animate(plot_t):        
        ax_cen.cla(), ax0.cla(), ax1.cla(), ax2.cla(), ax3.cla(), ax4.cla(), ax5.cla(), ax6.cla(), ax7.cla()
        
        #overview plot
        for key in self.ch_keys:
            #plot salinity
            self.ch_outp[key]['lc1'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm1)
            self.ch_outp[key]['lc1'].set_array(self.ch_outp[key]['sb_st'][plot_t])
            self.ch_outp[key]['lc1'].set_linewidth(5)
            line1=ax_cen.add_collection(self.ch_outp[key]['lc1'])
            
            #plot the salt intrusion limit
            #subtidal 
            i0 = np.where(self.ch_outp[key]['sb_st'][plot_t]>2)[0]
            i1 = np.where(self.ch_outp[key]['sb_st'][plot_t]<2)[0]
            if len(i0)>1 and len(i1)>0: ax_cen.scatter(*self.ch_outp[key]['segments'][i0[0]][0],marker='|',color='red',linewidth=3,zorder=100)
            
        #layout overview salinity plot
        ax_cen.axis('scaled')
        ax_cen.set_xlim(4,4.6) , ax_cen.set_ylim(51.75,52.02) #zoom in on mouth RM
    
        ax_cen.set_ylabel('degrees N ')
        ax_cen.set_xlabel('degrees E ')
        ax_cen.set_facecolor('grey')
    
        #cb1=plt.colorbar(line1, ax=ax_cen,orientation='vertical')
        #cb1.set_label(label=r'$\bar{s}$ [g/kg]')    

        #Plot the salinity in the tidal cycle in 2DV
        for k in range(len(axes)):
            key = keys_now[k]
            s_here = self.ch_outp[key]['s_st'][plot_t]
               
            cm = mpl.cm.get_cmap("RdBu_r").copy()
            a = axes[k].contourf(self.ch_outp[key]['px']*1e-3,self.z_nd, s_here.T,cmap = cm,levels = np.linspace(0,35,15),extend='both')
            
            axes[k].set_title(self.ch_gegs[key]['Name'])
            axes[k].set_xlabel('$x$ [km]'),axes[k].set_ylabel('$z/H$ [ ]')
            #axes[k].set_facecolor('lightgrey')
            if keys_now[k] != 'Spui': axes[k].invert_xaxis()
            
        #add the arrows
        ax_cen.arrow(4.06,52,-0.1,0.1,clip_on = False,head_width=0.02,color='g',width=0.005)
        ax_cen.arrow(4.25,51.94,0.01,0.14,clip_on = False,head_width=0.02,color='g',width=0.005)
        ax_cen.arrow(4.47,51.93,0.2,0.22,clip_on = False,head_width=0.02,color='g',width=0.005)
        ax_cen.arrow(4.15,51.91,-0.27,0.01,clip_on = False,head_width=0.02,color='g',width=0.005)
        ax_cen.arrow(4.35,51.8,0.03,-0.12,clip_on = False,head_width=0.02,color='g',width=0.005)
        
        ax_cen.arrow(4.34,51.84,-0.2,0,clip_on = False,head_width=0,color ='g',width=0.005)
        ax_cen.arrow(4.14,51.84,-0.15,-0.2,clip_on = False,head_width=0.02,color ='g',width=0.005)
    
        ax_cen.arrow(4.55,51.81,0,-0.15,clip_on = False,head_width=0,color ='g',width=0.005)
        ax_cen.arrow(4.55,51.66,0.2,-0.04,clip_on = False,head_width=0.02,color ='g',width=0.005)
    
        ax_cen.arrow(4.58,51.94,0,0.07,clip_on = False,head_width=0,color ='g',width=0.005)
        ax_cen.arrow(4.58,52.01,0.25,0.01,clip_on = False,head_width=0.02,color ='g',width=0.005)


    anim = ani.FuncAnimation(fig,animate,self.T,init_func=init,blit=False)
    #frames per second is now nt/11. should be determined from input but ja geen zin 
    anim.save("/Users/biemo004/Documents/UU phd Saltisolutions/Output/Animations all/"+savename+".mp4", fps=int(7), extra_args=['-vcodec', 'libx264'])
    
    plt.show()

#anim_new_compRM(delta,'td_170124_v2')