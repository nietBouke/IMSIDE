# =============================================================================
# make plots for the time dependent model 
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt         
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.animation as ani       #make animations

def plot_s_gen_td(self,t):
     
    #normalize
    norm = plt.Normalize(0,self.soc_sca)
    fig,ax = plt.subplots(1,1,figsize=(10,4))
    
    for key in self.ch_keys:       
        self.ch_outp[key]['lc'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm)
        self.ch_outp[key]['lc'].set_array(self.ch_outp[key]['sb_st'][t])
        self.ch_outp[key]['lc'].set_linewidth(5)
        
        line=ax.add_collection(self.ch_outp[key]['lc'])

    #ax.legend(loc='center left', bbox_to_anchor=(1, -0.44))#,ncol =np.max([1, int(len(ch_gegs)/7)]))
    
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