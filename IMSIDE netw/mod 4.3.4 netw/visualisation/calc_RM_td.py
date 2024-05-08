# =============================================================================
# calculate properties of RM for time-dependent simulations
# =============================================================================

import numpy as np

def calc_X2_td(self):
    # =============================================================================
    # calculate the salt intrusion length in the Oude and Nieuwe Maas    
    # =============================================================================
    L2_OM, L2_NM = np.zeros(self.T) , np.zeros(self.T) 
    for t in range(self.T):    
        
        if np.max(self.ch_outp['Oude Maas 1']['sb_st'][t])<2: 
            #print('No salt intrusion in the Oude Maas')
            L2_OM[t] = 0
        else:  
            Ltot = 0
            for key in ['Oude Maas 1','Oude Maas 2','Oude Maas 3','Oude Maas 4']:
                if np.min(self.ch_outp[key]['sb_st'][t]) >2: 
                    Ltot += np.sum(self.ch_gegs[key]['L'])
                    continue
                else: 
                    L2_OM[t] = Ltot-self.ch_outp[key]['px'][np.where(self.ch_outp[key]['sb_st'][t]>2)[0]][0]
                    #print('The salt intrusion in the Oude Maas is ' ,L2_OM[t]/1000, ' km')
                    break
      
        if np.max(self.ch_outp['Nieuwe Maas 1 old']['sb_st'])<2: 
            #print('No salt intrusion in the Nieuwe Maas')
            L2_NM[t] = 0 #np.nan
        else:  
            Ltot = 0
            for key in ['Nieuwe Maas 1 old','Nieuwe Maas 2 old']:
                if np.min(self.ch_outp[key]['sb_st'][t]) >2: 
                    Ltot += np.sum(self.ch_gegs[key]['L'])
                    continue
                else: 
                    L2_NM[t] = Ltot-self.ch_outp[key]['px'][np.where(self.ch_outp[key]['sb_st'][t]>2)[0]][0]
                    #print('The salt intrusion in the Nieuwe Maas is ' ,L2_NM[t]/1000, ' km')
                    break

    return L2_OM, L2_NM