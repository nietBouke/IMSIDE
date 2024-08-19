# =============================================================================
# Visualisation, salinity and river discharge 
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt         
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.animation as ani       #make animations
import matplotlib as mpl

def plot_Qs_simple(self,show_inds=False,arrow_scale =0.01,arc = 'black'):
    # =============================================================================
    # function to plot the different channels, with the discharge through them, and 
    # the subtidal salinity in the complete network
    # =============================================================================
    
    #normalize
    norm = plt.Normalize(0,self.soc_sca)
    fig,ax = plt.subplots(2,1,figsize=(10,6))
    
    for key in self.ch_keys:
        self.ch_outp[key]['lc'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm)
        self.ch_outp[key]['lc'].set_array(self.ch_outp[key]['sb_st'])
        self.ch_outp[key]['lc'].set_linewidth(5)
        
        line=ax[1].add_collection(self.ch_outp[key]['lc'])
    
    count=0
    used_jun = []
    for key in self.ch_keys:
        ax[0].plot(self.ch_gegs[key]['plot x'],self.ch_gegs[key]['plot y'],c=self.ch_gegs[key]['plot color'],
                 label = self.ch_gegs[key]['Name']+': $Q$ = '+str(int(np.abs(self.ch_pars[key]['Q'])))+' m$^3$/s')

        if show_inds==True:
            if self.ch_gegs[key]['loc x=-L'] not in used_jun:
                ax[0].text(self.ch_gegs[key]['plot x'][0],self.ch_gegs[key]['plot y'][0],self.ch_gegs[key]['loc x=-L'])
                used_jun.append(self.ch_gegs[key]['loc x=-L'])
            if self.ch_gegs[key]['loc x=0'] not in used_jun:
                ax[0].text(self.ch_gegs[key]['plot x'][-1],self.ch_gegs[key]['plot y'][-1],self.ch_gegs[key]['loc x=0'])
                used_jun.append(self.ch_gegs[key]['loc x=0'])
        
        if self.ch_pars[key]['Q']>0:
            ax[0].arrow(self.ch_gegs[key]['plot x'].mean()+1*arrow_scale,
                      self.ch_gegs[key]['plot y'].mean()+0.5*arrow_scale,
                      (self.ch_gegs[key]['plot x'][-1]-self.ch_gegs[key]['plot x'][0])/4,
                      (self.ch_gegs[key]['plot y'][-1]-self.ch_gegs[key]['plot y'][0])/4,
                      head_width = 2*arrow_scale, width = 0.1*arrow_scale,color=arc)# ch_gegs[key]['plot color'])
        else:
            ax[0].arrow(self.ch_gegs[key]['plot x'].mean()+1*arrow_scale+(self.ch_gegs[key]['plot x'][-1]-self.ch_gegs[key]['plot x'][0])/4,
                      self.ch_gegs[key]['plot y'].mean()+0.5*arrow_scale+(self.ch_gegs[key]['plot y'][-1]-self.ch_gegs[key]['plot y'][0])/4,
                      -(self.ch_gegs[key]['plot x'][-1]-self.ch_gegs[key]['plot x'][0])/4,
                      -(self.ch_gegs[key]['plot y'][-1]-self.ch_gegs[key]['plot y'][0])/4,
                      head_width = 2*arrow_scale, width = 0.1*arrow_scale,color=arc)# ch_gegs[key]['plot color'])
        count = count+1
    
    ax[0].legend(loc='center left', bbox_to_anchor=(1, -0.44))#,ncol =np.max([1, int(len(ch_gegs)/7)]))
    
    ax[0].axis('scaled'),ax[1].axis('scaled')
    ax[1].set_xlabel('degrees E '),ax[0].set_ylabel('degrees N '),ax[1].set_ylabel('degrees N ')
    ax[0].set_facecolor('lightgrey'),ax[1].set_facecolor('lightgrey')
    
    #ax[1].set_xlim(4,4.6) , ax[1].set_ylim(51.75,52.05) #zoom in on mouth RM
    
    cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.033])
    cb=fig.colorbar(line, cax=cbar_ax,orientation='horizontal')
    cb.set_label(label='Depth-averaged salinity [g/kg]')    
    #cb.ax.tick_params(labelsize=15)
    plt.show()
    
    return 



def plot_s_gen(self):
    # =============================================================================
    # function to plot the different channels, with the discharge through them, and 
    # the subtidal salinity in the complete network
    # =============================================================================
    
    #normalize
    norm = plt.Normalize(0,self.soc_sca)
    fig,ax = plt.subplots(1,1,figsize=(6,4))
    
    for key in self.ch_keys:
        self.ch_outp[key]['lc'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm)
        self.ch_outp[key]['lc'].set_array(self.ch_outp[key]['sb_st'])
        self.ch_outp[key]['lc'].set_linewidth(5)
        
        line=ax.add_collection(self.ch_outp[key]['lc'])
    
    ax.axis('scaled')
    ax.set_xlabel('degrees E '),ax.set_ylabel('degrees N ')
    ax.set_facecolor('lightgrey')
    
    #ax.set_xlim(4,4.5) #, ax[1].set_ylim(51.75,52.05) #zoom in on mouth RM
    
    #plot the salt intrusion limit
    for key in self.ch_keys:
        i0 = np.where(self.ch_outp[key]['sb_st']>0.150*1.807)[0]
        i1 = np.where(self.ch_outp[key]['sb_st']<0.150*1.807)[0]
        if len(i0)>1 and len(i1)>0: ax.scatter(*self.ch_outp[key]['segments'][i0[0]][0],marker='|',color='red',linewidth=3,zorder=100)

    
    # cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.033])
    # cb=fig.colorbar(line, cax=cbar_ax,orientation='horizontal')
    # cb.set_label(label='Depth-averaged salinity [g/kg]')    
    
    cbar_ax = fig.add_axes([0.95, 0.1, 0.02, 0.8])
    cb=fig.colorbar(line, cax=cbar_ax,orientation='vertical')
    cb.set_label(label='Depth-averaged salinity [g/kg]')    
    
    #cb.ax.tick_params(labelsize=15)
    plt.show()

    return 

#plot_s_gen(delta)

  

def plot_proc_ch(self,channels = None):
    # =============================================================================
    # function to plot the salt transport processes in each channel seperately. 
    # =============================================================================
    if channels == None:
        keys_here = self.ch_keys
    else: 
        keys_here = channels

    for key in keys_here:
        plt.figure(figsize=(7,4))
        plt.title(self.ch_gegs[key]['Name'],fontsize=14)
        plt.plot(self.ch_outp[key]['px']/1000,self.ch_outp[key]['TQ'],label='$T_Q$',lw = 2,c='m')
        plt.plot(self.ch_outp[key]['px']/1000,self.ch_outp[key]['TE'],label='$T_E$',lw = 2,c='brown')
        plt.plot(self.ch_outp[key]['px']/1000,self.ch_outp[key]['TT'],label='$T_T$',lw = 2,c='red')
        plt.plot(self.ch_outp[key]['px']/1000,self.ch_outp[key]['TD'],label='$T_D$',lw = 2,c='c')
        plt.plot(self.ch_outp[key]['px']/1000,self.ch_outp[key]['TQ']+self.ch_outp[key]['TD']+self.ch_outp[key]['TT']+self.ch_outp[key]['TE'],label='$T_o$',lw = 2,c='navy')
        '''
        if self.ch_gegs[key]['loc x=0'][0] == 's': #plot sea domain
            plt.plot(self.ch_outp[key]['pxs']/1000,self.ch_outp[key]['TQs'],lw = 2,c='m')
            plt.plot(self.ch_outp[key]['pxs']/1000,self.ch_outp[key]['TEs'],lw = 2,c='brown')
            plt.plot(self.ch_outp[key]['pxs']/1000,self.ch_outp[key]['TTs'],lw = 2,c='red')
            plt.plot(self.ch_outp[key]['pxs']/1000,self.ch_outp[key]['TDs'],lw = 2,c='c')
            plt.plot(self.ch_outp[key]['pxs']/1000,self.ch_outp[key]['TQs']+self.ch_outp[key]['TDs']+self.ch_outp[key]['TTs']+self.ch_outp[key]['TEs'],lw = 2,c='navy')
        '''
        plt.grid()
        plt.xlabel('$x$ [km]',fontsize=14),plt.ylabel('$T$ [kg/s]',fontsize=14)
        plt.xlim(self.ch_outp[key]['px'][0]/1000,self.ch_outp[key]['px'][-1]/1000)
        #plt.xlim(-50,5)
        plt.legend(fontsize=14)#,loc=2)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()
        

def anim_new_compRM(self,savename):
    print('Warning: making of this animation is probably quite slow, find something to do in the meantime. \n It also only works for a 3 channel system')
    
    # =============================================================================
    # Function to animate salinity in the tidal cycle
    # =============================================================================
    keys_here = self.ch_keys    
    norm1 = plt.Normalize(0,35)

    # =============================================================================
    # Plot
    # =============================================================================
    fig, ax  = plt.subplots(3,1,figsize=(16,9))
    axes = [ax[0], ax[1], ax[2]]
    keys_now = self.ch_keys
    
    
    #make water level varying for tides
    coords_tides = {}
    
    for k in range(len(keys_now)):
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

    
    def animate(plot_t):        
        ax[0].cla(), ax[1].cla(),  ax[2].cla()
        

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



    anim = ani.FuncAnimation(fig,animate,self.nt,init_func=init,blit=False)
    #frames per second is now nt/11. should be determined from input but ja geen zin 
    anim.save("/Users/biemo004/Documents/UU phd Saltisolutions/Output/Animations all/"+savename+".mp4", fps=int(self.nt/11), extra_args=['-vcodec', 'libx264'])
    
    plt.show()
    
#anim_new_compRM(delta, 'tests_170624_v3')