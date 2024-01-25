# =============================================================================
# here the forcing files for the network are placed. 
# =============================================================================
import numpy as np
import pandas as pd
dms = 10**9*3600*24


def forc_RMD4():
    #time parameters
    T = 40
    DT = np.zeros(T) + 24*3600 # I usually work with subtidal time steps of one day 
    
    #forcing conditions
    Qriv   = np.array([600 + np.linspace(0,0,T),50+ np.zeros(T)]) #mind the sign! this is the discharge at r1,r2,...
    Qweir  = np.array([0+ np.zeros(T),125+ np.zeros(T)])#,0,0]
    Qhar   = np.array([0+ np.zeros(T)])
    n_sea  = np.array([0+ np.zeros(T)]) #this is the water level at s1,s2,...
    soc    = np.array([30+ np.zeros(T)])  #this is salinity at s1,s2, ...
    sri    = np.array([0+ np.zeros(T),0+ np.zeros(T)]) #this is the salinity of the river water at r1,r2, ...
    swe    = np.array([0+ np.zeros(T),0+ np.zeros(T)])
    tid_per= 44700
    a_tide = np.array([0.83]) #this is the amplitude of the tide at s1,s2, ...
    p_tide = np.array([70*np.pi/180]) #this is the phase of the tide at s1,s2, ...

    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, tid_per, a_tide, p_tide, T, DT

def forc_RMD5():
    #time parameters
    T = 30
    DT = np.zeros(T) + 24*3600 # I usually work with subtidal time steps of one day 
    
    QQQ = np.concatenate([[2000],np.zeros(T-1)+1000])
    
    #forcing conditions
    Qriv   = np.array([QQQ,500+ np.zeros(T)]) #mind the sign! this is the discharge at r1,r2,...
    Qweir  = np.array([5 + np.zeros(T),500+ np.zeros(T)])#,0,0]
    Qhar   = np.array([1000 + np.zeros(T)])
    n_sea  = np.array([0  + np.zeros(T)]) #this is the water level at s1,s2,...
    soc    = np.array([30 + np.zeros(T)])  #this is salinity at s1,s2, ...
    sri    = np.array([0  + np.zeros(T),0+ np.zeros(T)]) #this is the salinity of the river water at r1,r2, ...
    swe    = np.array([0  + np.zeros(T),0+ np.zeros(T)])
    tid_per= 44700
    a_tide = np.array([0.83]) #this is the amplitude of the tide at s1,s2, ...
    p_tide = np.array([70*np.pi/180]) #this is the phase of the tide at s1,s2, ...

    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, tid_per, a_tide, p_tide, T, DT

def forc_RMD_fromSOBEK(date_start, date_stop):
    #import the datasets from the SOBEK model as forcings.
    startnum, stopnum = pd.to_datetime(date_start,dayfirst=True).value/dms+719529 , pd.to_datetime(date_stop,dayfirst=True).value/dms+719529 
    loc = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/MATROOS discharge/'
    
    DT = np.zeros(int(stopnum-startnum)) + 24*3600 #timesteps of one day
    
    #Waal
    d = np.loadtxt(loc+'SOBEK_discharge_Waal.txt')
    startind, stopind = np.where(d[:,0]==startnum)[0][0],np.where(d[:,0]==stopnum)[0][0]
    Q1 = d[startind:stopind,1]
    #Maas
    d = np.loadtxt(loc+'SOBEK_discharge_Maas.txt')
    startind, stopind = np.where(d[:,0]==startnum)[0][0],np.where(d[:,0]==stopnum)[0][0]
    Q2 = d[startind:stopind,1]
    #Hollandse IJssel
    d = np.loadtxt(loc+'SOBEK_discharge_Hollandse IJssel.txt')
    startind, stopind = np.where(d[:,0]==startnum)[0][0],np.where(d[:,0]==stopnum)[0][0]
    Q3 = d[startind:stopind,1]
    if np.min(Q3)<-10: print('Strange values in Hollandse IJssel discharge')
    #Q3[np.where(Q3<0)[0]] = 0 #the Hollandse IJssel cannot have negative discharge because of the boudnary condition.
    #Lek
    d = np.loadtxt(loc+'SOBEK_discharge_Lek.txt')
    startind, stopind = np.where(d[:,0]==startnum)[0][0],np.where(d[:,0]==stopnum)[0][0]
    Q4 = d[startind:stopind,1]
    if np.min(Q4)<-10: print('Strange values in Lek discharge')
    #Q4[np.where(Q4<0)[0]] = 0 #the Lek cannot have negative discharge because of the boudnary condition.

    #Haringvliet
    d = np.loadtxt(loc+'SOBEK_discharge_Haringvliet.txt')
    startind, stopind = np.where(d[:,0]==startnum)[0][0],np.where(d[:,0]==stopnum)[0][0]
    Q5 = d[startind:stopind,1]

    #check if there are no nans in the discharge
    for q in [Q1,Q2,Q3,Q4,Q5]:
        if len(np.where(np.isnan(q)==True)[0]) >0:
            print('ERROR: no(t all) forcing discharge available for this period')
            sy.exit()

    #interpolate daily discharge to chosen time grid
    #make time grid
    T = len(DT)
    t_grid = np.zeros(T)
    for t in range(1,T): t_grid[t] = np.sum(DT[:t])
    t_ori = np.arange(0,stopnum-startnum)*3600*24

    Q1 = np.interp(t_grid,t_ori,Q1)
    Q2 = np.interp(t_grid,t_ori,Q2)
    Q3 = np.interp(t_grid,t_ori,Q3)
    Q4 = np.interp(t_grid,t_ori,Q4)
    Q5 = np.interp(t_grid,t_ori,Q5)

    Qriv = np.array([Q1,Q2]) #mind the sign! this is the discharge at r1,r2,...
    Qweir= np.array([Q3,Q4])
    Qhar = np.array([Q5])
    n_sea = np.array([np.zeros(T)]) #this is the water level at s1,s2,...
    soc = np.array([30+np.zeros(T)]) #this is salinity at s1,s2, ...
    sri = np.array([np.zeros(T)+0.17,np.zeros(T)+0.17]) #TODO: get this from data
    swe = np.array([np.zeros(T)+0.17,np.zeros(T)+0.17]) #TODO: get this from data
    
    #tide
    tid_per= 44700
    a_tide = np.array([0.83]) #this is the amplitude of the tide at s1,s2, ...
    p_tide = np.array([70*np.pi/180]) #this is the phase of the tide at s1,s2, ...

    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, tid_per, a_tide, p_tide , T,DT #, (date_start, date_stop)
    
    


def forc_test1():
    T = 70
    DT = np.zeros(T) + 24*3600 # I usually work with subtidal time steps of one day 

    Qriv = np.array([200 + np.linspace(0,200,T)]) #mind the sign! this is the discharge at r1,r2,...
    Qweir= np.array([])
    Qhar = np.array([])
    n_sea = np.array([0 + np.zeros(T),0 + np.zeros(T)]) #this is the water level at s1,s2,...
    soc = np.array([35+ np.zeros(T),35+ np.zeros(T)]) #this is salinity at s1,s2, ...
    sri = np.array([0+ np.zeros(T)]) #this is the salinity of the river water at r1,r2, ...
    swe = np.array([])
    tid_per = 44700
    a_tide = np.array([0.7,0.7]) #this is the amplitude of the tide at s1,s2, ...
    p_tide = np.array([0,0]) #this is the phase of the tide at s1,s2, ...


    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, tid_per, a_tide, p_tide, T, DT

def forc_test2():
    T = 30
    DT = np.zeros(T) + 24*3600 # I usually work with subtidal time steps of one day 
    
    QQQ = np.concatenate([[200],np.zeros(T-1)+400])
    
    Qriv = np.array([QQQ]) #mind the sign! this is the discharge at r1,r2,...
    Qweir= np.array([])
    Qhar = np.array([])
    n_sea = np.array([0 + np.zeros(T),0 + np.zeros(T)]) #this is the water level at s1,s2,...
    soc = np.array([35+ np.zeros(T),35+ np.zeros(T)]) #this is salinity at s1,s2, ...
    sri = np.array([0+ np.zeros(T)]) #this is the salinity of the river water at r1,r2, ...
    swe = np.array([])
    tid_per = 44700
    a_tide = np.array([0.7*0,0.7*0]) #this is the amplitude of the tide at s1,s2, ...
    p_tide = np.array([0,0]) #this is the phase of the tide at s1,s2, ...


    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, tid_per, a_tide, p_tide, T, DT
