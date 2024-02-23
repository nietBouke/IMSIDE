# =============================================================================
# load here the forcing
# for now: only Q
# =============================================================================
import numpy as np
import pandas as pd
import scipy as sp
import scipy.io

def UN_tosalt2(C,T): #conductivity, temperature
    a = [0.0080, -0.1692, 25.3851, 14.0941, -7.0261, 2.7081]
    b = [0.0005, -0.0056, -0.0066, -0.0375,  0.0636, -0.0144]
    c = [6.76609e-1, 2.00564e-2, 1.104259e-4, -6.9698e-7, 1.0031e-9]
    d= [None, 3.426e-2, 4.464e-4, 4.215e-1, -3.107e-3]
    e =[None, 2.070e-5, -6.370e-10, 3.989e-15]
   
    r = C/42914
    r = r/np.sum([c[i]*T**i for i in range(0,5)],0)
    #r = r/(c[0]+T*(c[1]+T*(c[2]+T*(c[3]+T*c[4]))))
    
    r2 = np.sqrt(r)
    
    ds = np.sum([b[i]*r2**i for i in range(0,6)],0)
    ds = ds*((T-15)/(1+0.0162*(T-15)))

    s = ds + np.sum([a[i]*r2**i for i in range(0,6)],0)

    return s


def forc_GUA1():
    #time
    T = 30
    dt = np.zeros(T) + 24*3600
    
    
    #subtidal
    Q   = np.linspace(10,100,T)
    soc = 35  + np.zeros(T)
    sri = 0.5 + np.zeros(T)
    
    #tidal 
    tid_comp = ['M2']
    tid_per = [44700]
    a_tide  = [0.95]
    p_tide  = [53]
    
    
    return (T, dt) , (Q, soc, sri) , (tid_comp, tid_per, a_tide, p_tide)

def forc_LOI1():
    #time
    T = 30
    dt = np.zeros(T) + 24*3600

    #subtidal
    Q   = 700 + np.zeros(T)
    soc = 35  + np.zeros(T)
    sri = 0.15 + np.zeros(T)
    
    #tidal 
    tid_comp = ['M2']
    tid_per = [44700]
    a_tide  = [1.85] #1.8
    p_tide  = [190]  #195
    
    return (T, dt) , (Q, soc, sri) , (tid_comp, tid_per, a_tide, p_tide)


def forc_DLW1():
    #time
    T = 15
    dt = np.zeros(T) + 24*3600
    
    #subtidal
    #Q   = np.linspace(300,500,T)
    Q   = np.concatenate([np.zeros(5)+500 , np.zeros(T-5)+400])
    soc = 35 + np.zeros(T)
    sri = 0.15 + np.zeros(T)
    
    #tidal 
    tid_comp = ['M2']
    tid_per = [44700]
    a_tide  = [0.68]
    p_tide  = [0]
    
    
    return (T, dt) , (Q, soc, sri) , (tid_comp, tid_per, a_tide, p_tide)


def forc_try1():
    #time
    T = 120
    dt = np.zeros(T) + 24*3600
    
    #subtidal
    #Q   = np.linspace(50,200,T)
    Q   = np.concatenate([np.linspace(10,1000,100) , np.zeros(20)+1000])
    soc = 35 + np.zeros(T)
    sri = 0.5 + np.zeros(T)
    
    #tidal 
    tid_comp = ['M2']
    tid_per = [44700]
    a_tide  = [1.]
    p_tide  = [0]
    
    
    return (T, dt) , (Q, soc, sri) , (tid_comp, tid_per, a_tide, p_tide)

def forc_DLW2(dat_start , dat_stop):
    
    #run Delaware with observed river discharge. 
    loc = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Delaware/'
    dat3 = pd.read_csv(loc+'raw_daily/discharge/Delaware River at Belvidere NJ - 01446500.txt',delimiter="\t",skiprows = 36,dtype=object) 
    tQ = np.array(dat3['datetime'][1:],dtype='datetime64')
    dQ = np.array(dat3['97355_00060_00003'][1:],dtype=float) * 0.0283168466 #to m3/s
    fQ = np.array(dat3['97355_00060_00003_cd'][1:],dtype=str)
    #dQf = np.zeros(dQ.shape) + np.nan #flagged numbers - does not make such a big difference here actually. Only most recent period is not approved yet. 
    #dQf[np.where(fQ == 'A')] = dQ[np.where(fQ == 'A')]
    
    i_start = np.where(tQ == np.datetime64(dat_start))[0][0]
    i_stop  = np.where(tQ == np.datetime64(dat_stop))[0][0]
 
    #time
    T  = i_stop - i_start 
    dt = np.zeros(T) + 24*3600
    
    #subtidal
    Q   = dQ[i_start : i_stop] 
    soc = 35 + np.zeros(T)
    sri = 0.15 + np.zeros(T)
    
    #tidal 
    tid_comp = ['M2']
    tid_per = [44700]
    a_tide  = [0.76]
    p_tide  = [0]
    
    return (T, dt) , (Q, soc, sri) , (tid_comp, tid_per, a_tide, p_tide)




def forc_GUA2(dat_start , dat_stop):
    #starting dates 
            
    disc = sp.io.loadmat('/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Quadalquivir/BBiemond/freshwater_discharges.mat')
    Q_gu = np.array(disc['Q']).flatten()
    Qt = np.array(disc['t']).flatten()
    i_start = np.where(Qt == pd.to_datetime(dat_start+' 00:00:00').value/(10**9*3600*24)+719529)[0]
    i_stop =  np.where(Qt == pd.to_datetime(dat_stop+' 00:00:00').value/(10**9*3600*24)+719529)[0]
    
    if len(i_start) ==0 or len(i_stop) == 0 : print('ERROR: chosen date not available ')
    
    Q_here = Q_gu[i_start[0]:i_stop[0]]

    #time
    T = len(Q_here)
    dt = np.zeros(T) + 24*3600
    
    #subtidal
    Q   = Q_here.copy()
    soc = 35  + np.zeros(T)
    sri = 0.5 + np.zeros(T)
    
    #tidal 
    tid_comp = ['M2']
    tid_per = [44700]
    a_tide  = [0.95]
    p_tide  = [53]
        
    return (T, dt) , (Q, soc, sri) , (tid_comp, tid_per, a_tide, p_tide)


def forc_LOI2(dat_start , dat_stop):
    
    #load discharge
    loc = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Loire/'
    #'raw' values
    tQ_1014 = np.array(pd.read_csv(loc+'discharge/'+'M530001010_Q1014.csv')['Date (TU)'].astype('datetime64[s]'), dtype = np.datetime64)
    qQ_1014 = np.array(pd.read_csv(loc+'discharge/'+'M530001010_Q1014.csv')['Valeur (en l/s)']/1e3)
    tQ_1418 = np.array(pd.read_csv(loc+'discharge/'+'M530001010_Q1418.csv')['Date (TU)'].astype('datetime64[s]'), dtype = np.datetime64)
    qQ_1418 = np.array(pd.read_csv(loc+'discharge/'+'M530001010_Q1418.csv')['Valeur (en l/s)']/1e3)
    tQ_1822 = np.array(pd.read_csv(loc+'discharge/'+'M530001010_Q1822.csv')['Date (TU)'].astype('datetime64[s]'), dtype = np.datetime64)
    qQ_1822 = np.array(pd.read_csv(loc+'discharge/'+'M530001010_Q1822.csv')['Valeur (en l/s)']/1e3)

    #interpolate to hourly values
    th_1014 = np.arange(np.datetime64('2010-01-01'), np.datetime64('2014-01-01'), np.timedelta64(1, 'h'))
    th_1418 = np.arange(np.datetime64('2014-01-01'), np.datetime64('2018-01-01'), np.timedelta64(1, 'h'))
    th_1822 = np.arange(np.datetime64('2018-01-01'), np.datetime64('2022-01-01'), np.timedelta64(1, 'h'))

    th_1022 = np.concatenate([th_1014,th_1418,th_1822])
    Qh_1022 = np.concatenate([np.interp(th_1014.astype('datetime64[s]').astype(np.float64), tQ_1014.astype('datetime64[s]').astype(np.float64), qQ_1014) , 
                              np.interp(th_1418.astype('datetime64[s]').astype(np.float64), tQ_1418.astype('datetime64[s]').astype(np.float64), qQ_1418) , 
                              np.interp(th_1822.astype('datetime64[s]').astype(np.float64), tQ_1822.astype('datetime64[s]').astype(np.float64), qQ_1822) ])

    #take daily average
    td_1022 = np.arange(np.datetime64('2010-01-01'), np.datetime64('2022-01-01'), np.timedelta64(24, 'h'))
    Qd_1022 = Qh_1022.reshape((len(td_1022),24)).mean(1)

    i_start = np.where(td_1022 == np.datetime64(dat_start))
    i_stop =  np.where(td_1022 == np.datetime64(dat_stop))
    
    if len(i_start[0]) ==0 or len(i_stop[0]) == 0 : print('ERROR: chosen date not available ')
        
    Q_here = Qd_1022[i_start[0][0]:i_stop[0][0]]

    #time
    T = len(Q_here)
    dt = np.zeros(T) + 24*3600
    
    #subtidal
    Q   = Q_here.copy()
    soc = 35  + np.zeros(T)
    sri = 0.15 + np.zeros(T)
    
    #tidal 
    tid_comp = ['M2']
    tid_per = [44700]
    a_tide  = [1.85] #1.8
    p_tide  = [190]  #195
    
    return (T, dt) , (Q, soc, sri) , (tid_comp, tid_per, a_tide, p_tide)


def forc_GUA3(dat_start , dat_stop):
    # =============================================================================
    # build forcing for Guadalquivir for the summer of 2009
    # =============================================================================
    if dat_start != '2009-04-01' : print('This forcing is only for the summer of 2009')
    if dat_stop  != '2009-11-01' : print('This forcing is only for the summer of 2009')   


    #starting dates 
            
    disc = sp.io.loadmat('/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Quadalquivir/BBiemond/freshwater_discharges.mat')
    Q_gu = np.array(disc['Q']).flatten()
    Qt = np.array(disc['t']).flatten()
    i_start = np.where(Qt == pd.to_datetime(dat_start+' 00:00:00').value/(10**9*3600*24)+719529)[0]
    i_stop =  np.where(Qt == pd.to_datetime(dat_stop+' 00:00:00').value/(10**9*3600*24)+719529)[0]
    
    if len(i_start) ==0 or len(i_stop) == 0 : print('ERROR: chosen date not available ')
    
    Q_here = Q_gu[i_start[0]:i_stop[0]]
    Q_here[:2] = [12,15] #adjust
    

    #time
    T = len(Q_here)
    dt = np.zeros(T) + 24*3600
    
    #subtidal
    Q   = Q_here.copy()
    soc = 35  + np.zeros(T)
    sri = 0.5 + np.zeros(T)
    
    #tidal 
    tid_comp = ['M2']
    tid_per = [44700]
    a_tide  = [0.95]
    p_tide  = [53]
        
    return (T, dt) , (Q, soc, sri) , (tid_comp, tid_per, a_tide, p_tide)













