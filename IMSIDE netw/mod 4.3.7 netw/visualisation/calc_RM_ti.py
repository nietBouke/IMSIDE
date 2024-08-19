# =============================================================================
# Calculation
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt         


def calc_tide_pointRM(self):
    # =============================================================================
    # function to plot the tidal water level amplitude and phase at the observations
    # points in the Rhine-Meuse estuary, together with the observations. 
    # =============================================================================
    #location datafile observations M2 tide
    loc = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Water levels conv/'
    dat = pd.read_excel(loc+'results_harmonic_tides_v2.xlsx',skiprows=3) 
    
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
    #tM2_mod = np.zeros(len(lats))+np.nan
    
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
        #tM2_mod[pt] = self.ch_outp[ind_ch]['eta'][ind_co]

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

    def score_WEI(am,pm,ao,po):
        f = np.mean(np.sqrt((ao-am)**2 + 2*ao*am*(1-np.cos((po-pm)* np.pi/180))))
        return f
    
    
    score = (score_WEI(aM2_mod[np.where(~np.isnan(aM2_mod))[0]],pM2_mod[np.where(~np.isnan(aM2_mod))[0]],amp_M2[np.where(~np.isnan(aM2_mod))[0]],pha_M2[np.where(~np.isnan(aM2_mod))[0]]))

    return score
    '''
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
    plt.show()
    '''

#calc_tide_pointRM(delta)


def calc_X2(self):
    # =============================================================================
    # X2 in Oude Maas
    # =============================================================================
    
    if np.max(self.ch_outp['Oude Maas 1']['sb_st'])<2: 
        print('No salt intrusion in the Oude Maas')
        L2_OM = np.nan
    else:  
        Ltot = 0
        for key in ['Oude Maas 1','Oude Maas 2','Oude Maas 3','Oude Maas 4']:
            if np.min(self.ch_outp[key]['sb_st']) >2: 
                Ltot += np.sum(self.ch_gegs[key]['L'])
                continue
            else: 
                L2_OM = Ltot-self.ch_outp[key]['px'][np.where(self.ch_outp[key]['sb_st']>2)[0]][0]
                print('The salt intrusion in the Oude Maas is ' ,L2_OM/1000, ' km')
                break
  
    # =============================================================================
    # X2 in Nieuwe Maas
    # =============================================================================
    if np.max(self.ch_outp['Nieuwe Maas 1 old']['sb_st'])<2: 
        print('No salt intrusion in the Nieuwe Maas')
        L2_NM = np.nan
    else:  
        Ltot = 0
        for key in ['Nieuwe Maas 1 old','Nieuwe Maas 2 old']:
            if np.min(self.ch_outp[key]['sb_st']) >2: 
                Ltot += np.sum(self.ch_gegs[key]['L'])
                continue
            else: 
                L2_NM = Ltot-self.ch_outp[key]['px'][np.where(self.ch_outp[key]['sb_st']>2)[0]][0]
                print('The salt intrusion in the Nieuwe Maas is ' ,L2_NM/1000, ' km')
                break
    
    # =============================================================================
    # X0.5 in Hollandse IJssel
    # =============================================================================
    if np.max(self.ch_outp['Hollandse IJssel']['sb_st'])<2: 
        print('No salt intrusion in the Hollandse IJssel')
        L2_HY = 0#np.nan
    else:  
        L2_HY = -self.ch_outp['Hollandse IJssel']['px'][np.where(self.ch_outp['Hollandse IJssel']['sb_st']>2)[0]][0]
        print('The salt intrusion in the Hollandse IJssel is ' ,L2_HY/1000, ' km')

    # =============================================================================
    # X0.5 in Hollandse IJssel
    # =============================================================================
    if np.max(self.ch_outp['Spui']['sb_st'])<0.5: 
        print('No salt intrusion in the Spui')
        L2_SP = 0 #np.nan
    else:  
        L2_SP = -self.ch_outp['Spui']['px'][np.where(self.ch_outp['Spui']['sb_st']>0.5)[0]][0]
        print('The salt intrusion in the Spui is ' ,L2_SP/1000, ' km')

    
    
    return L2_OM, L2_NM ,  L2_HY, L2_SP

#calc_X2(delta)