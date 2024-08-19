# =============================================================================
# calculate properties of RM for time-dependent simulations
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

def split_opeen(list_ori):
    list_split=[]
    temp=[]
    for i in range(len(list_ori)-1):
        if list_ori[i+1]- list_ori[i] <2:
            temp.append(list_ori[i])
        else:
            temp.append(list_ori[i])
            list_split.append(temp)
            temp=[]
    temp.append(list_ori[-1])
    list_split.append(temp)
    return list_split

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
      
        if np.max(self.ch_outp['Nieuwe Maas 1 old']['sb_st'][t])<2: 
            #print('No salt intrusion in the Nieuwe Maas')
            L2_NM[t] = 0 #np.nan
        else:  
            Ltot = 0
            for key in ['Nieuwe Maas 1 old','Nieuwe Maas 2 old']:
                if np.min(self.ch_outp[key]['sb_st'][t]) >2: 
                    Ltot += np.sum(self.ch_gegs[key]['L'])
                    continue
                else: 
                    try: L2_NM[t] = Ltot-self.ch_outp[key]['px'][np.where(self.ch_outp[key]['sb_st'][t]>2)[0]][0]
                    except: L2_NM[t] = Ltot-self.ch_outp[key]['px'][np.where(self.ch_outp[key]['sb_st'][t]>2)[0]][0]
                    #print('The salt intrusion in the Nieuwe Maas is ' ,L2_NM[t]/1000, ' km')
                    break

    return L2_OM, L2_NM

#temp = calc_X2_td(delta)

'''
def calc_respons(self,ch_id):
    # =============================================================================
    # calculate the adjustment and recovery time for a specified channel
    # based on salt concentration (not on salt intrusion length)
    # =============================================================================
    t1, t2 = 5,55 #start and stop time of drought
    fac = 0.9 #after which factor we consider equilibrium reached
    s_tres = 0.150*1.807 #salinity values below this treshold are not considered
    day_div = 10# the resolution of the time vector, is day/day_div

    sb = self.ch_outp[ch_id]['sb_st']
    T_adj = []
    T_rec = []
    
    #if there is no change
    if np.max(np.abs(sb[0]-sb[-1])) < 1e-5: return 0,0

    for xi in range(len(sb[0])):
        sb_now = sb[:,xi]
        if np.max(sb_now) > s_tres:

            #interpolate
            sb_int = np.interp(np.linspace(0,len(sb),len(sb)*day_div),np.linspace(0,len(sb),len(sb)),sb_now)
            
            tres1 = fac*(np.max(sb_int)-np.min(sb_int)) + np.min(sb_int)
            tres2 = - fac*(np.max(sb_int)-np.min(sb_int)) + np.max(sb_int)
            
            T_adj.append((np.where(sb_int>tres1)[0][0]-(t1-1)*day_div)/day_div)
            T_rec.append((split_opeen(np.where(sb_int<tres2)[0])[1][0]-(t2-1)*day_div)/day_div)
            
            
        else:
            T_adj.append(np.nan)
            T_rec.append(np.nan)
            
    return np.nanmax(T_adj), np.nanmax(T_rec)
'''

def calc_Qs(self, t):
    #discharges
    #quantities for first timestep    
    Qr_inp = self.Qriv[:,t]  if len(self.Qriv)  >0 else []
    Qw_inp = self.Qweir[:,t] if len(self.Qweir) >0 else []
    Qh_inp = self.Qhar[:,t]  if len(self.Qhar)  >0 else []
    ns_inp = self.n_sea[:,t]  if len(self.n_sea)  >0 else []
    Qnow = self.Qdist_calc((Qr_inp , Qw_inp , Qh_inp, ns_inp))    
        
    return Qnow
        
def loc_iso(self, ch_id,  isoha, t):
    sb = self.ch_outp[ch_id]['sb_st']

    #function to find t he location of a certain isohaline
    if len(np.where(sb[t]>isoha)[0] ) > 0 and len(np.where(sb[t]<isoha)[0]) >0:
        return -self.ch_outp[ch_id]['px'][np.argmin(np.abs(sb[t]- isoha))]
        #print('Isohaline found')
    else: 
        return np.nan
        #print('Isohaline not in domain')
    


def calc_respons(self,ch_id):
    # =============================================================================
    # calculate the adjustment and recovery time for a specified channel
    # based on salt concentration (not on salt intrusion length)
    # =============================================================================
    t1, t2 = 5,55 #start and stop time of drought
    fac = 0.9 #after which factor we consider equilibrium reached
    s_tres = 0.150*1.807 #salinity values below this treshold are not considered
    day_div = 10# the resolution of the time vector, is day/day_div

    sb = self.ch_outp[ch_id]['sb_st']
    T_adj = []
    T_rec = []
    
    #calcualte the total amount of salt in the channel
    tot_salt = np.sum(sb*self.ch_outp[ch_id]['CS']*self.ch_outp[ch_id]['dl']*self.Lsc, 1)
    ave_salt = tot_salt / np.sum(self.ch_outp[ch_id]['CS']*self.ch_outp[ch_id]['dl']*self.Lsc)
    
    #if there is no change
    if np.max(np.max(tot_salt)-np.min(tot_salt)) < 1: return np.nan, np.nan, np.nan

    #interpolate
    salt_int = np.interp(np.linspace(0,len(tot_salt),len(tot_salt)*day_div),np.linspace(0,len(tot_salt),len(tot_salt)),tot_salt)
    
    #treshold
    tres1 = fac*(np.max(salt_int)-np.min(salt_int)) + np.min(salt_int)
    tres2 = - fac*(np.max(salt_int)-np.min(salt_int)) + np.max(salt_int)
    
    #change in salt content
    dSr = (np.max(salt_int)-np.min(salt_int)) / salt_int[0]
    dS2 = np.max(ave_salt) - np.min(ave_salt)
    
    #response times
    T_adj = (np.where(salt_int>tres1)[0][0]-(t1-1)*day_div)/day_div
    try:
        T_rec = (split_opeen(np.where(salt_int<tres2)[0])[1][0]-(t2-1)*day_div)/day_div
    except: 
        T_rec = np.nan
        
    '''
    # =============================================================================
    # expectations from FWP theory
    # this works not very well. 
    # =============================================================================
    Q_here = np.zeros(self.T)
    for t in range(self.T): Q_here[t] = calc_Qs(self,t)[ch_id]

    u_st_max = np.mean(np.max(Q_here) / self.ch_outp[ch_id]['CS'])

    #First: check if we can follow any isohaline. If we can, calculate dX from there
    
    Xx_ts = np.zeros(self.T) + np.nan
    isoha_step = 0.25
    isoha = 2-isoha_step
    while len(np.where(np.isnan(Xx_ts))[0]) > 0 :
        isoha += isoha_step
        for t in range(self.T): Xx_ts[t] = loc_iso(self, ch_id, isoha, t)
        
        if isoha > 35: 
            print('No usefull isohaline to trace in channel '+ ch_id)
            break
        
    T_adj_sca = 0.9 * (np.max(Xx_ts) - np.min(Xx_ts)) / u_st_max / 86400
        
    print(T_adj, T_adj_sca)
    
    #if len(np.where(np.min(sb,1) < 2)[0]) == self.T:
    #0#follow 2 psu isohaline
    
    #If this is not possible, follow the isohaline assocaited with the smallest salinity value
    #if np.min(sb,1)
    
    #if this is also not possible, take the time it took the lowest salinity value to leave the domain. 

    #print(np.min(sb[0]))

    
    
    # =============================================================================
    # look at transports in and out of a channel
    # =============================================================================
    
    fig, ax = plt.subplots(1,3,figsize = (10,3))
    
    xloc = 0
    ax[0].plot(self.ch_outp[ch_id]['TQ'][:,xloc])
    ax[0].plot(self.ch_outp[ch_id]['TE'][:,xloc])
    ax[0].plot(self.ch_outp[ch_id]['TD'][:,xloc])
    ax[0].plot(self.ch_outp[ch_id]['TT'][:,xloc])
    ax[0].plot((self.ch_outp[ch_id]['TQ']+self.ch_outp[ch_id]['TE']+self.ch_outp[ch_id]['TD']+self.ch_outp[ch_id]['TT'])[:,xloc], c= 'black')
    
    xloc = -1
    ax[1].plot(self.ch_outp[ch_id]['TQ'][:,xloc])
    ax[1].plot(self.ch_outp[ch_id]['TE'][:,xloc])
    ax[1].plot(self.ch_outp[ch_id]['TD'][:,xloc])
    ax[1].plot(self.ch_outp[ch_id]['TT'][:,xloc])
    ax[1].plot((self.ch_outp[ch_id]['TQ']+self.ch_outp[ch_id]['TE']+self.ch_outp[ch_id]['TD']+self.ch_outp[ch_id]['TT'])[:,xloc] , c='black')

    ax[2].plot(tot_salt)
    
    plt.tight_layout()
    plt.show()   
    '''
    TQb= np.min(self.ch_outp[ch_id]['TQ'][:,-1])
    dS = np.max(tot_salt) - np.min(tot_salt)
    
    #print()
    #
    #print(ch_id,dS/TQb/3600)#,dS, TQb)
    
    return T_adj, T_rec, dS2


def calc_responses(self):
    T_adj_all, T_rec_all, dSr_all = [] , [] , []
    
    for ch in self.ch_keys :
        temp = calc_respons(self,ch)
        T_adj_all.append(temp[0])      
        T_rec_all.append(temp[1])
        dSr_all.append(temp[2])
        
    return self.ch_keys, T_adj_all , T_rec_all, dSr_all

# calc_respons(delta,'Nieuwe Waterweg v2')
# calc_respons(delta,'Hartelkanaal v2')
# # print(calc_respons(delta,'Spui'))
# calc_respons(delta,'Waal')
# calc_respons(delta,'Nieuwe Maas 1 old')
# # print(calc_respons(delta,'Hollandse IJssel'))

#print(calc_responses(delta))


