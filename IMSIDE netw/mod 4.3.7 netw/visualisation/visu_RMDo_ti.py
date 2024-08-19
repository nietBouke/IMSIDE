# =============================================================================
# Visualisation, salinity and river discharge 
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt         
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.animation as ani       #make animations



def plot_s_RMDo(self):
    # =============================================================================
    # function to plot the different channels, with the discharge through them, and 
    # the subtidal salinity in the complete network
    # =============================================================================
    img = plt.imread("/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/RijnMaas_oud/NL-RtSA_4001_VI-74-01_lowres.jpg")

    #normalize
    norm = plt.Normalize(0,self.soc_sca)
    fig,ax = plt.subplots(1,1,figsize=(10,6))
    ax.imshow(img)

    for key in self.ch_keys:
        self.ch_outp[key]['lc'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm)
        self.ch_outp[key]['lc'].set_array(self.ch_outp[key]['sb_st'])
        self.ch_outp[key]['lc'].set_linewidth(5)
        
        line=ax.add_collection(self.ch_outp[key]['lc'])
    
    ax.axis('scaled')
    ax.set_xlabel('degrees E '),ax.set_ylabel('degrees N ')
    ax.set_facecolor('lightgrey')
    
    #ax[1].set_xlim(4,4.6) , ax[1].set_ylim(51.75,52.05) #zoom in on mouth RM

    cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.033])
    cb=fig.colorbar(line, cax=cbar_ax,orientation='horizontal')
    cb.set_label(label='Depth-averaged salinity [g/kg]')    
    #cb.ax.tick_params(labelsize=15)
    plt.show()
    


    return 

#plot_s_RMDo(delta)

  
def anim_tide_wl(self,savename):   
    # =============================================================================
    # Make an animation with only the tidal water level in the entire network
    # =============================================================================
    #normalize
    ext_eta, ext_utb, ext_stb = np.zeros((len(self.ch_keys),2)) , np.zeros((len(self.ch_keys),2)) , np.zeros((len(self.ch_keys),2))
    count = 0
    for key in self.ch_keys:
        ext_eta[count] = [np.nanmin(self.ch_outp[key]['eta_r']),np.nanmax(self.ch_outp[key]['eta_r'])]
        count +=1
    norm_p1 = plt.Normalize(np.round(np.min(ext_eta[:,0]),1),np.round(np.max(ext_eta[:,1]),1))

    img = plt.imread("/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/RijnMaas_oud/NL-RtSA_4001_VI-74-01_lowres.jpg")

    def init():
        plot_t = 0
        #ax_ti.text(0,0,'Time = '+str(plot_t)+' timesteps after start of the simulation',fontsize=12)    
        for key in self.ch_keys:
            self.ch_outp[key]['p1'] = LineCollection(self.ch_outp[key]['segments'], cmap='winter', norm=norm_p1)
            self.ch_outp[key]['p1'].set_array(self.ch_outp[key]['eta_r'][:,plot_t])
            self.ch_outp[key]['p1'].set_linewidth(5)

            line0=ax.add_collection(self.ch_outp[key]['p1'])

        ax.imshow(img)
        #ax.axis('scaled')
        #ax.set_xlabel('degrees E '),ax.set_ylabel('degrees N ')
        #ax.set_facecolor('lightgrey')
        
        #ax[1].set_xlim(4,4.6) , ax[1].set_ylim(51.75,52.05) #zoom in on mouth RM
        cb0=fig.colorbar(line0, ax=ax,orientation='vertical')
        cb0.set_label(label='Water level [m]') 
           
    def animate(plot_t):
        ax.cla() #, cb0.cla(),cb1.cla(),cb2.cla()
        
        for key in self.ch_keys:
            self.ch_outp[key]['p1'] = LineCollection(self.ch_outp[key]['segments'], cmap='winter', norm=norm_p1)
            self.ch_outp[key]['p1'].set_array(self.ch_outp[key]['eta_r'][:,plot_t])
            self.ch_outp[key]['p1'].set_linewidth(5)


            line0=ax.add_collection(self.ch_outp[key]['p1'])
            
        ax.imshow(img)
        #ax.axis('scaled')
        #ax.set_xlabel('degrees E '),ax.set_ylabel('degrees N ')
        #ax.set_facecolor('lightgrey')

        return ax 

    fig,ax = plt.subplots(1,1,figsize=(8,4))

    anim = ani.FuncAnimation(fig,animate,self.nt,init_func=init,blit=False)
    anim.save("/Users/biemo004/Documents/UU phd Saltisolutions/Output/Animations all/"+savename+".mp4", fps=21,extra_args=['-vcodec', 'libx264'])
    
    plt.show()


    return


#anim_tide_wl(delta,'RMDo_v3')


def plot_strat_RMDo(self):
    # =============================================================================
    # plot the subtidal salt field and stratification in the Rotterdam Waterway     
    # =============================================================================
    img = plt.imread("/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/RijnMaas_oud/NL-RtSA_4001_VI-74-01_lowres.jpg")
    
    fig = plt.figure(figsize=(10,15))
    gs = fig.add_gridspec(5,1)
    
    ax0 = fig.add_subplot(gs[0:3])
    ax1 = fig.add_subplot(gs[3])
    ax2 = fig.add_subplot(gs[4])
    
    #normalize
    norm = plt.Normalize(0,self.soc_sca)
    ax0.imshow(img)
    
    for key in self.ch_keys:
        self.ch_outp[key]['lc'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm)
        self.ch_outp[key]['lc'].set_array(self.ch_outp[key]['sb_st'])
        self.ch_outp[key]['lc'].set_linewidth(5)
        
        line=ax0.add_collection(self.ch_outp[key]['lc'])
    
    ax0.axis('scaled')
    #x[0].set_xlabel('degrees E '),ax[0].set_ylabel('degrees N ')
    ax0.set_facecolor('lightgrey')
    
    #ax[1].set_xlim(4,4.6) , ax[1].set_ylim(51.75,52.05) #zoom in on mouth RM
    
    cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.033])
    cb=fig.colorbar(line, cax=cbar_ax,orientation='horizontal')
    cb.set_label(label='Depth-averaged salinity [g/kg]')    
    #cb.ax.tick_params(labelsize=15)
    
    print(self.ch_keys)
    print(self.ch_outp['Scheur']['s_st'].shape)
    
    ax1.contourf(self.ch_outp['Scheur']['px']/1000, self.z_nd, self.ch_outp['Scheur']['s_st'].T, cmap='RdBu_r')
    ax2.plot(self.ch_outp['Scheur']['px']/1000,self.ch_outp['Scheur']['s_st'][:,0]-self.ch_outp['Scheur']['s_st'][:,-1],c='black',lw=2)
    ax2.set_xlim(self.ch_outp['Scheur']['px'][0]/1000 , self.ch_outp['Scheur']['px'][-1]/1000)
    ax1.invert_xaxis(),   ax2.invert_xaxis()
    
    ax1.set_ylabel('z/H')
    ax2.set_ylabel('$\Delta s$ [psu]')
    ax2.set_xlabel('$x$ [km]')
    
    plt.show()
    
#plot_strat_RMDo(delta)

    