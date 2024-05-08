# =============================================================================
# Visualisation, salinity and river discharge 
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt         
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.animation as ani       #make animations

def plot_procRM(self):
    # =============================================================================
    # Function to plot salt, discharge, stratification and the different processes 
    # in the network. Transport by tides not included. 
    # =============================================================================
    keys_here = self.ch_keys    
    norm1 = plt.Normalize(0,self.soc_sca)

    # =============================================================================
    # Plot
    # =============================================================================
    fig = plt.figure(figsize=(16,9))
    gs = fig.add_gridspec(11,14)

    ax_cen = fig.add_subplot(gs[4:7,4:9])
    ax_leg = fig.add_subplot(gs[0:3,-2])
    
    ax0 = fig.add_subplot(gs[0:3,1:4])
    ax1 = fig.add_subplot(gs[0:3,5:8])
    ax2 = fig.add_subplot(gs[0:3,9:12])
    
    ax3 = fig.add_subplot(gs[4:7,0:3])
    ax4 = fig.add_subplot(gs[4:7,10:13])

    ax5 = fig.add_subplot(gs[8:11,1:4])
    ax6 = fig.add_subplot(gs[8:11,5:8])
    ax7 = fig.add_subplot(gs[8:11,9:12])
    '''
    gs = fig.add_gridspec(11,15)

    ax_cen = fig.add_subplot(gs[4:7,5:10])
    ax_leg = fig.add_subplot(gs[0:3,0])
    
    ax0 = fig.add_subplot(gs[0:3,2:5])
    ax1 = fig.add_subplot(gs[0:3,6:9])
    ax2 = fig.add_subplot(gs[0:3,10:13])
    
    ax3 = fig.add_subplot(gs[4:7,1:4])
    ax4 = fig.add_subplot(gs[4:7,11:14])

    ax5 = fig.add_subplot(gs[8:11,2:5])
    ax6 = fig.add_subplot(gs[8:11,6:9])
    ax7 = fig.add_subplot(gs[8:11,10:13])
    '''

    
    
    #plot salinity
    for key in self.ch_keys:
        self.ch_outp[key]['lc1'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm1)
        self.ch_outp[key]['lc1'].set_array(self.ch_outp[key]['sb_st'])
        self.ch_outp[key]['lc1'].set_linewidth(5)
        line1=ax_cen.add_collection(self.ch_outp[key]['lc1'])
        
    #layout salinity plot
    ax_cen.axis('scaled')
    ax_cen.set_xlim(4,4.625) , ax_cen.set_ylim(51.75,52.02) #zoom in on mouth RM

    ax_cen.set_ylabel('degrees N ')
    ax_cen.set_xlabel('degrees E ')
    ax_cen.set_facecolor('grey')

    cb1=plt.colorbar(line1, ax=ax_cen,orientation='vertical')
    cb1.set_label(label=r'$\bar{s}$ [g/kg]')    

    #plot the salt intrusion limit
    for key in self.ch_keys:
        i0 = np.where(self.ch_outp[key]['sb_st']>2)[0]
        i1 = np.where(self.ch_outp[key]['sb_st']<2)[0]
        if len(i0)>1 and len(i1)>0: ax_cen.scatter(*self.ch_outp[key]['segments'][i0[0]][0],marker='|',color='red',linewidth=3,zorder=100)


    #Plot the transports for every channel
    keys_now = ['Breeddiep' , 'Nieuwe Waterweg v2', 'Nieuwe Maas 1 old', 'Hartelkanaal v2', 'Hollandse IJssel', 'Oude Maas 2', 'Spui', 'Oude Maas 3']
    axes = [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7]
    for k in range(len(axes)):
        key = keys_now[k]
        axes[k].set_title(self.ch_gegs[key]['Name'])
        axes[k].plot(self.ch_outp[key]['px']/1000,self.ch_outp[key]['TQ'],label='$T_Q$',lw = 2,c='g')
        axes[k].plot(self.ch_outp[key]['px']/1000,self.ch_outp[key]['TE'],label='$T_E$',lw = 2,c='orange')
        axes[k].plot(self.ch_outp[key]['px']/1000,self.ch_outp[key]['TT'],label='$T_T$',lw = 2,c='b')
        axes[k].plot(self.ch_outp[key]['px']/1000,self.ch_outp[key]['TD'],label='$T_D$',lw = 2,c='firebrick')
        axes[k].plot(self.ch_outp[key]['px']/1000,self.ch_outp[key]['TQ']+self.ch_outp[key]['TD']+self.ch_outp[key]['TT']+self.ch_outp[key]['TE'],label='$T_o$',lw = 2,c='black')

        axes[k].grid()
        axes[k].set_xlabel('$x$ [km]'),axes[k].set_ylabel('$T$ [kg/s]')
        axes[k].set_xlim(self.ch_outp[key]['px'][0]/1000,self.ch_outp[key]['px'][-1]/1000)
        
        if keys_now[k] != 'Spui':     axes[k].invert_xaxis()
        #    print('hi')
    #plt.xlim(-50,5)
    #ax0.legend()#,loc=2)

    #add the arrows
    ax_cen.arrow(4.06,52,-0.1,0.1,clip_on = False,head_width=0.02,color='black',width=0.005)
    ax_cen.arrow(4.25,51.94,0.01,0.14,clip_on = False,head_width=0.02,color='black',width=0.005)
    ax_cen.arrow(4.47,51.93,0.2,0.22,clip_on = False,head_width=0.02,color='black',width=0.005)
    ax_cen.arrow(4.15,51.91,-0.27,0.01,clip_on = False,head_width=0.02,color='black',width=0.005)
    ax_cen.arrow(4.35,51.8,0.03,-0.12,clip_on = False,head_width=0.02,color='black',width=0.005)
    
    ax_cen.arrow(4.34,51.84,-0.2,0,clip_on = False,head_width=0,color ='black',width=0.005)
    ax_cen.arrow(4.14,51.84,-0.15,-0.2,clip_on = False,head_width=0.02,color ='black',width=0.005)

    ax_cen.arrow(4.55,51.81,0,-0.15,clip_on = False,head_width=0,color ='black',width=0.005)
    ax_cen.arrow(4.55,51.66,0.2,-0.04,clip_on = False,head_width=0.02,color ='black',width=0.005)

    ax_cen.arrow(4.58,51.94,0,0.07,clip_on = False,head_width=0,color ='black',width=0.005)
    ax_cen.arrow(4.58,52.01,0.25,0.01,clip_on = False,head_width=0.02,color ='black',width=0.005)

    axes[0].text(0.02,0.9,'(a)',fontsize=13,transform = axes[0].transAxes)
    axes[1].text(0.02,0.9,'(b)',fontsize=13,transform = axes[1].transAxes)
    axes[2].text(0.02,0.9,'(c)',fontsize=13,transform = axes[2].transAxes)
    axes[3].text(0.02,0.9,'(d)',fontsize=13,transform = axes[3].transAxes)
    axes[4].text(0.02,0.9,'(e)',fontsize=13,transform = axes[4].transAxes)
    axes[5].text(0.02,0.9,'(f)',fontsize=13,transform = axes[5].transAxes)
    axes[6].text(0.02,0.9,'(g)',fontsize=13,transform = axes[6].transAxes)
    axes[7].text(0.02,0.9,'(h)',fontsize=13,transform = axes[7].transAxes)


    clr,lab  = [] , []
    legs = ['$T_Q$','$T_E$','$T_T$','$T_D$','$T_O$']
    colors = ['g','orange','b','firebrick', 'black']
    for k in range(len(legs)):
        clr.append(Line2D([0], [0], color=colors[k], lw=2)) 
        lab.append(legs[k])
    
    ax_leg.legend(clr ,lab)
    ax_leg.axis('off')#,ax_ti.axis('off')

    #delete axis I do not like 
    #for rem in [0,1,2,3,4,5,6,7]:
    #for rem in [0,1,2,4,5,6,7]:
    #    axes[rem].remove()
        #axes[rem].axis('off')
    #ax_leg.remove()

    fig.text(0.05,0.05,'hoi',c='white')    
    fig.text(0.85,0.91,'hoi',c='white')    

    #plt.savefig('/Users/biemo004/Documents/UU phd Saltisolutions/Verslagen/Papers/Paper bignetwork1/figs/proc_RM_v1.jpg', dpi = 600, bbox_inches='tight')

    plt.show()
  
#plot_procRM(delta)  



  
def plot_salt_compRM(self,plot_t):
    # =============================================================================
    # Function to plot salt, in the tidal cycle, at different locations
    # in the network. 
    # =============================================================================
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
    
    
    #plot salinity
    for key in self.ch_keys:
        self.ch_outp[key]['lc1'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm1)
        self.ch_outp[key]['lc1'].set_array(np.mean(self.ch_outp[key]['ss']+self.ch_outp[key]['st_cor_r'][:,:,plot_t] , 1))
        self.ch_outp[key]['lc1'].set_linewidth(5)
        line1=ax_cen.add_collection(self.ch_outp[key]['lc1'])
        
    #layout salinity plot
    ax_cen.axis('scaled')
    ax_cen.set_xlim(4,4.6) , ax_cen.set_ylim(51.75,52.02) #zoom in on mouth RM

    ax_cen.set_ylabel('degrees N ')
    ax_cen.set_xlabel('degrees E ')
    ax_cen.set_facecolor('grey')

    cb1=plt.colorbar(line1, ax=ax_cen,orientation='vertical')
    cb1.set_label(label=r'$\bar{s}$ [g/kg]')    

    #plot the salt intrusion limit
    for key in self.ch_keys:
        i0 = np.where(self.ch_outp[key]['sb']>2)[0]
        i1 = np.where(self.ch_outp[key]['sb']<2)[0]
        if len(i0)>1 and len(i1)>0: ax_cen.scatter(*self.ch_outp[key]['segments'][i0[0]][0],marker='|',color='red',linewidth=3,zorder=100)


    #Plot the transports for every channel
    keys_now = ['Breeddiep' , 'Nieuwe Waterweg v2', 'Nieuwe Maas 1 old', 'Hartelkanaal v2', 'Hollandse IJssel', 'Oude Maas 2', 'Spui', 'Oude Maas 3']
    axes = [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7]
    for k in range(len(axes)):
        key = keys_now[k]
        s_here = self.ch_outp[key]['ss']+self.ch_outp[key]['st_cor_r'][:,:,plot_t]
        
        axes[k].set_title(self.ch_gegs[key]['Name'])
        axes[k].contourf(self.ch_outp[key]['px']/1000 , self.z_nd, s_here.T,cmap = 'RdBu_r',levels = np.linspace(0,35,15))

        axes[k].set_xlabel('$x$ [km]'),axes[k].set_ylabel('$z/H$ [ ]')

    #plt.xlim(-50,5)
    #ax0.legend()#,loc=2)

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
    
    plt.show()
        
import matplotlib as mpl
       
def anim_new_compRM(self,savename):
    print('Warning: making of this animation is probably quite slow, find something to do in the meantime')
    
    # =============================================================================
    # Function to animate salinity in the tidal cycle
    # =============================================================================
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
    
    
    #make water level varying for tides
    coords_tides = {}
    
    for k in range(len(axes)):
        key = keys_now[k]

        #take water level into account for htis
        px_tide = np.tile(self.ch_outp[key]['px'],self.nz).reshape(self.nz,len(self.ch_outp[key]['px']))
        pz_tide = np.zeros((self.nz,len(self.ch_outp[key]['px']),self.nt)) + np.nan
        for t in range(self.nt):
            pz_1t = np.zeros((self.nz,len(self.ch_outp[key]['px']))) + np.nan
            for x in range(len(self.ch_outp[key]['px'])): pz_1t[:,x] = np.linspace(-self.ch_pars[key]['H'][x],self.ch_outp[key]['eta_r'][x,t],self.nz)
            pz_tide[:,:,t] = pz_1t
                
        coords_tides[key] = (px_tide , pz_tide)

    def init():
        plot_t = 0
        ax_cen.cla(), ax0.cla(), ax1.cla(), ax2.cla(), ax3.cla(), ax4.cla(), ax5.cla(), ax6.cla(), ax7.cla()
        
        #overview plot
        for key in self.ch_keys:
            #plot salinity
            self.ch_outp[key]['lc1'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm1)
            self.ch_outp[key]['lc1'].set_array(np.mean(self.ch_outp[key]['s_st']+self.ch_outp[key]['s_ti_r'][:,:,plot_t] , 1))
            self.ch_outp[key]['lc1'].set_linewidth(5)
            line1=ax_cen.add_collection(self.ch_outp[key]['lc1'])
            
            #plot the salt intrusion limit
            #subtidal 
            i0 = np.where(self.ch_outp[key]['sb_st']>2)[0]
            i1 = np.where(self.ch_outp[key]['sb_st']<2)[0]
            if len(i0)>1 and len(i1)>0: ax_cen.scatter(*self.ch_outp[key]['segments'][i0[0]][0],marker='|',color='red',linewidth=3,zorder=100, alpha = 0.5)
                        
            #tidal
            i0 = np.where(np.mean(self.ch_outp[key]['s_st']+self.ch_outp[key]['s_ti_r'][:,:,plot_t] , 1)>2)[0]
            i1 = np.where(np.mean(self.ch_outp[key]['s_st']+self.ch_outp[key]['s_ti_r'][:,:,plot_t] , 1)<2)[0]
            if len(i0)>1 and len(i1)>0: ax_cen.scatter(*self.ch_outp[key]['segments'][i0[0]][0],marker='|',color='red',linewidth=3,zorder=100)
        
            
        #layout overview salinity plot
        ax_cen.axis('scaled')
        ax_cen.set_xlim(4,4.6) , ax_cen.set_ylim(51.75,52.02) #zoom in on mouth RM
        
        ax_cen.set_ylabel('degrees N ')
        ax_cen.set_xlabel('degrees E ')
        ax_cen.set_facecolor('grey')
        
        cb1=plt.colorbar(line1, ax=ax_cen,orientation='vertical')
        cb1.set_label(label=r'$\bar{s}$ [g/kg]')    
        
        #Plot the salinity in the tidal cycle in 2DV                
        for k in range(len(axes)):
            key = keys_now[k]
            s_here = self.ch_outp[key]['s_st']+self.ch_outp[key]['s_ti_r'][:,:,plot_t]
               
            cm = mpl.cm.get_cmap("RdBu_r").copy()
            a = axes[k].contourf(coords_tides[key][0]*1e-3,coords_tides[key][1][:,:,plot_t], s_here.T,cmap = cm,levels = np.linspace(0,35,15),extend='both')
            
            axes[k].set_title(self.ch_gegs[key]['Name'])
            axes[k].set_xlabel('$x$ [km]'),axes[k].set_ylabel('$z/H$ [ ]')
            axes[k].set_ylim(-np.max(self.ch_pars[key]['H']),np.max(self.ch_outp[key]['eta_r'])*1.25*2)
            axes[k].set_facecolor('lightgrey')
            if keys_now[k] != 'Spui': axes[k].invert_xaxis()
            
            import matplotlib
            cmap = matplotlib.cm.get_cmap('RdBu_r')
            cmin,cmax = cmap(0),cmap(1)
            a.cmap.set_under(cmin), a.cmap.set_over(cmax)
            #a.cmap.set_under('white'), a.cmap.set_over('white')
            a.set_clim(0,35)

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
            self.ch_outp[key]['lc1'].set_array(np.mean(self.ch_outp[key]['s_st']+self.ch_outp[key]['s_ti_r'][:,:,plot_t] , 1))
            self.ch_outp[key]['lc1'].set_linewidth(5)
            line1=ax_cen.add_collection(self.ch_outp[key]['lc1'])
            
            #plot the salt intrusion limit
            #subtidal 
            i0 = np.where(self.ch_outp[key]['sb_st']>2)[0]
            i1 = np.where(self.ch_outp[key]['sb_st']<2)[0]
            if len(i0)>1 and len(i1)>0: ax_cen.scatter(*self.ch_outp[key]['segments'][i0[0]][0],marker='|',color='red',linewidth=3,zorder=100, alpha = 0.5)
            #tidal
            i0 = np.where(np.mean(self.ch_outp[key]['s_st']+self.ch_outp[key]['s_ti_r'][:,:,plot_t] , 1)>2)[0]
            i1 = np.where(np.mean(self.ch_outp[key]['s_st']+self.ch_outp[key]['s_ti_r'][:,:,plot_t] , 1)<2)[0]
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
            s_here = self.ch_outp[key]['s_st']+self.ch_outp[key]['s_ti_r'][:,:,plot_t]

            cm = mpl.cm.get_cmap("RdBu_r").copy()
            a = axes[k].contourf(coords_tides[key][0]*1e-3,coords_tides[key][1][:,:,plot_t], s_here.T,cmap = cm,levels = np.linspace(0,35,15), extend="both")
            axes[k].set_title(self.ch_gegs[key]['Name'])
            axes[k].set_xlabel('$x$ [km]'),axes[k].set_ylabel('$z/H$ [ ]')
            axes[k].set_ylim(-np.max(self.ch_pars[key]['H']),np.max(self.ch_outp[key]['eta_r'])*1.25*2)
            axes[k].set_facecolor('lightgrey')
            if keys_now[k] != 'Spui': axes[k].invert_xaxis()
            
            import matplotlib
            cmap = matplotlib.cm.get_cmap('RdBu_r')
            cmin,cmax = cmap(0),cmap(1)
            a.cmap.set_under(cmin), a.cmap.set_over(cmax)
            
            #a.cmap.set_under('white'), a.cmap.set_over('white')
            
            a.set_clim(0,35)

            
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


    anim = ani.FuncAnimation(fig,animate,self.nt,init_func=init,blit=False)
    #frames per second is now nt/11. should be determined from input but ja geen zin 
    anim.save("/Users/biemo004/Documents/UU phd Saltisolutions/Output/Animations all/"+savename+".mp4", fps=int(self.nt/11), extra_args=['-vcodec', 'libx264'])
    
    plt.show()

def plot_salt_compRM(self,var,title,smin,smax,tplot=None):
    # =============================================================================
    # Function to plot salt, in the tidal cycle, at different locations
    # in the network. 
    # =============================================================================
    keys_here = self.ch_keys    
    norm1 = plt.Normalize(smin,smax)

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
    
    
    #plot salinity
    for key in self.ch_keys:
        if var == 's_st':  varplot = np.mean(self.ch_outp[key][var],1)
        elif var == 's_ti': varplot = np.abs(np.mean(self.ch_outp[key][var],1))
        elif var == 's_ti_r': varplot = np.mean(self.ch_outp[key][var][:,:,tplot],1)
        elif var == 'ut_r': varplot = np.mean(self.ch_outp[key][var][:,:,tplot],1)
        elif var == 'stc': varplot = np.abs(np.mean(self.ch_outp[key]['st_cor']-self.ch_pars[key]['st'],1))
        else: print('Unkwon var')
        
        self.ch_outp[key]['lc1'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm1)
        self.ch_outp[key]['lc1'].set_array(varplot)
        self.ch_outp[key]['lc1'].set_linewidth(5)
        line1=ax_cen.add_collection(self.ch_outp[key]['lc1'])

    #layout salinity plot
    ax_cen.axis('scaled')
    ax_cen.set_xlim(4,4.6) , ax_cen.set_ylim(51.75,52.02) #zoom in on mouth RM

    ax_cen.set_ylabel('degrees N ')
    ax_cen.set_xlabel('degrees E ')
    ax_cen.set_facecolor('grey')

    cb1=plt.colorbar(line1, ax=ax_cen,orientation='vertical')
    cb1.set_label(label=r'$\bar{s}$ [g/kg]')    

    #plot the salt intrusion limit
    for key in self.ch_keys:
        i0 = np.where(self.ch_outp[key]['sb_st']>2)[0]
        i1 = np.where(self.ch_outp[key]['sb_st']<2)[0]
        if len(i0)>1 and len(i1)>0: ax_cen.scatter(*self.ch_outp[key]['segments'][i0[0]][0],marker='|',color='red',linewidth=3,zorder=100)


    #Plot the transports for every channel
    keys_now = ['Breeddiep' , 'Nieuwe Waterweg v2', 'Nieuwe Maas 1 old', 'Hartelkanaal v2', 'Hollandse IJssel', 'Oude Maas 2', 'Spui', 'Oude Maas 3']
    axes = [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7]
    for k in range(len(axes)):
        key = keys_now[k]
        if var == 's_st':  varplot = self.ch_outp[key][var]
        elif var == 's_ti': varplot = np.abs(self.ch_outp[key][var])
        elif var == 's_ti_r': varplot = self.ch_outp[key][var][:,:,tplot]
        elif var == 'ut_r': varplot = self.ch_outp[key][var][:,:,tplot]
        elif var == 'stc': varplot = np.abs(self.ch_outp[key]['st_cor']-self.ch_pars[key]['st'])
        else: print('Unkwon var')
        
        axes[k].set_title(self.ch_gegs[key]['Name'])
        axes[k].contourf(self.ch_outp[key]['px']/1000 , self.z_nd, varplot.T,cmap = 'RdBu_r',levels = np.linspace(smin,smax,15))
        
        #if keys_now[k] == 'Nieuwe Waterweg v2': axes[k].set_xlim(-2.5,0)
        if keys_now[k] != 'Spui': axes[k].invert_xaxis()
        
        
        axes[k].set_xlabel('$x$ [km]'),axes[k].set_ylabel('$z/H$ [ ]')

    #plt.xlim(-50,5)
    #ax0.legend()#,loc=2)

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
    
    fig.suptitle(title,fontsize=20)
    plt.show()

#plot_salt_compRM(delta,'s_st','Subtidal salinity',0,35)
#plot_salt_compRM(delta,'s_ti','Tidal salinity',0,5)
#plot_salt_compRM(delta,'s_ti_r','Tidal salinity at time t=0',-5,5,0)
#plot_salt_compRM(delta,'s_ti_r','Tidal salinity at time t=0',-5,5,25)
#plot_salt_compRM(delta,'s_ti_r','Tidal salinity at time t=0',-5,5,50)
#plot_salt_compRM(delta,'s_ti_r','Tidal salinity at time t=0',-5,5,75)
#plot_salt_compRM(delta,'s_ti_r','Tidal salinity at time t=0',-5,5,100)
#plot_salt_compRM(delta,'s_ti_r','Tidal salinity at time t=0',-5,5,100)
#plot_salt_compRM(delta,'ut_r','Tidal currents at time t=0',-0.5,0.5,100)
#plot_salt_compRM(delta,'stc','Tidal correction salinity')


#for key in delta.ch_keys: print(np.max(np.abs(delta.ch_outp[key]['st_cor']-delta.ch_pars[key]['st'])))

#plot_new_compRM(delta)


def plot_tide_pointRM(self):
    # =============================================================================
    # function to plot the tidal water level amplitude and phase at the observations
    # points in the Rhine-Meuse estuary, together with the observations. 
    # =============================================================================
    #location datafile observations M2 tide
    loc = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Water levels conv/'
    dat = pd.read_excel(loc+'results_harmonic_tides_v1.xlsx',skiprows=3) 
    
    # =============================================================================
    # load the observations
    # =============================================================================
    punten = np.array(dat['Punt'])
    amp_M2 = np.array(dat['amplitude M2'])/100
    amp_M2n= np.array(dat['error van harmo'])/100
    
    pha_M2 = np.array(dat['fase M2'])
    pha_M2n= np.array(dat['error van harmo.1'])
    
    lats=  np.array(dat['lat'])
    lons =  np.array(dat['lon'])
    
    # =============================================================================
    # subtract the model output at the location of the observations
    # =============================================================================
    aM2_mod = np.zeros(len(lats))+np.nan
    pM2_mod = np.zeros(len(lats))+np.nan
    
    for pt in range(len(lats)):
        locE,locN = lons[pt],lats[pt]
        #find closest point. Note that the difference is in degrees and not in meters. This should not be too much of a problem. 
        close1,close2 = [] , []
        for key in self.ch_keys:
            temp = ((self.ch_outp[key]['plot xs']-locE)**2+(self.ch_outp[key]['plot ys']-locN)**2)**0.5
            close1.append(np.min(temp))
            close2.append(np.argmin(temp))
        if np.min(close1)>0.02: 
            print('WARNING: point '+punten[pt]+' too far away from estuary, not plotted')
           
            ind_ch = self.ch_keys[np.argmin(close1)] #name of the channel where the point is
            ind_co = close2[np.argmin(close1)]  #index of the point in this channel
            print(self.ch_outp[ind_ch]['plot ys'][ind_co],self.ch_outp[ind_ch]['plot xs'][ind_co])
            continue 
        
        ind_ch = self.ch_keys[np.argmin(close1)] #name of the channel where the point is
        ind_co = close2[np.argmin(close1)]  #index of the point in this channel

        aM2_mod[pt] = np.abs(self.ch_outp[ind_ch]['eta'][ind_co])
        pM2_mod[pt] = 180-np.angle(self.ch_outp[ind_ch]['eta'][ind_co])/np.pi*180
        
        #reduce the phase
        if pM2_mod[pt]<0:
            pM2_mod[pt] = pM2_mod[pt]+360
        #tricks
        if pM2_mod[pt]>180:
            pM2_mod[pt] = 360-pM2_mod[pt]
        if pha_M2[pt]>180:
            pha_M2[pt] = 360-pha_M2[pt]
    # =============================================================================
    # Plot this
    # =============================================================================

    fig, ax = plt.subplots(2,1,figsize = (5,5))
    ax[0].scatter(punten[np.where(~np.isnan(aM2_mod))[0]],amp_M2[np.where(~np.isnan(aM2_mod))[0]],marker='D',label='obs',color='blue')
    ax[0].scatter(punten[np.where(~np.isnan(aM2_mod))[0]],aM2_mod[np.where(~np.isnan(aM2_mod))[0]],marker='D',label='mod',color='red')
    #plt.errorbar(punten[np.where(~np.isnan(aM2_mod))[0]],amp_M2[np.where(~np.isnan(aM2_mod))[0]], yerr=amp_M2n[np.where(~np.isnan(aM2_mod))[0]], fmt="o")
    #ax[0].set_xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
    ax[0].xaxis.set_ticklabels([])
    ax[0].grid()
    ax[0].set_ylim(0,1)
    ax[0].set_ylabel('Amplitude M$_2$ tide [m]')
    
    ax[1].scatter(punten[np.where(~np.isnan(aM2_mod))[0]],pha_M2[np.where(~np.isnan(aM2_mod))[0]],marker='D',label='obs',color='blue')
    ax[1].scatter(punten[np.where(~np.isnan(aM2_mod))[0]],pM2_mod[np.where(~np.isnan(aM2_mod))[0]],marker='D',label='mod',color='red')
    #plt.errorbar(punten[np.where(~np.isnan(aM2_mod))[0]],pha_M2[np.where(~np.isnan(aM2_mod))[0]], yerr=pha_M2n[np.where(~np.isnan(aM2_mod))[0]], fmt="o")
    #ax[1].xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
    ax[1].grid(),ax[1].set_ylim(0,180)
    ax[1].set_ylabel('Phase M$_2$ tide [deg]')
    ax[1].tick_params(labelrotation=90)
    
    ax[1].legend(bbox_to_anchor=(1., -.75))
    
    
    ax[0].text(0.02,0.9,'(a)',fontsize=13,transform = ax[0].transAxes)
    ax[1].text(0.02,0.9,'(b)',fontsize=13,transform = ax[1].transAxes)
    
    #plt.savefig('/Users/biemo004/Documents/UU phd Saltisolutions/Verslagen/Papers/Paper bignetwork1/figs/tun_tide_RM_v1.jpg', dpi = 600, bbox_inches='tight')

    plt.show()



#plot_tide_pointRM(delta)





