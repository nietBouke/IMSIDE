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
    soc    = np.array([33+ np.zeros(T)])  #this is salinity at s1,s2, ...
    sri    = np.array([0+ np.zeros(T),0+ np.zeros(T)]) #this is the salinity of the river water at r1,r2, ...
    swe    = np.array([0+ np.zeros(T),0+ np.zeros(T)])
    tid_per= 44700
    a_tide = np.array([0.77]) #this is the amplitude of the tide at s1,s2, ...
    p_tide = np.array([70*np.pi/180]) #this is the phase of the tide at s1,s2, ...

    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, tid_per, a_tide, p_tide, T, DT



def forc_RMD5():
    #time parameters
    T = 50
    DT = np.zeros(T) + 24*3600 # I usually work with subtidal time steps of one day 
    
    Qa_W = 1500
    Qa_M = 230
    Qa_L = 430
    fac = 0.5
    
    Q_W = np.concatenate([np.zeros(5)+Qa_W,np.zeros(30)+Qa_W*fac,np.zeros(15)+Qa_W])
    Q_L = np.concatenate([np.zeros(5)+Qa_L,np.zeros(30)+Qa_L*fac,np.zeros(15)+Qa_L])
    Q_M = np.concatenate([np.zeros(5)+Qa_M,np.zeros(30)+Qa_M*fac,np.zeros(15)+Qa_M])
    
    #forcing conditions
    Qriv   = np.array([Q_W,Q_M]) #mind the sign! this is the discharge at r1,r2,...
    Qweir  = np.array([np.zeros(T),Q_L])#,0,0]
    Qhar   = np.array([np.zeros(T)])
    n_sea  = np.array([0  + np.zeros(T)]) #this is the water level at s1,s2,...
    soc    = np.array([33 + np.zeros(T)])  #this is salinity at s1,s2, ...
    sri    = np.array([0.15 + np.zeros(T) , 0.15 + np.zeros(T)]) #this is the salinity of the river water at r1,r2, ...
    swe    = np.array([0.15 + np.zeros(T) , 0.15 + np.zeros(T)])
    tid_per= 44700
    a_tide = np.array([0.77]) #this is the amplitude of the tide at s1,s2, ...
    p_tide = np.array([70]) #this is the phase of the tide at s1,s2, ...

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
    soc = np.array([33+np.zeros(T)]) #this is salinity at s1,s2, ...
    sri = np.array([np.zeros(T)+0.17,np.zeros(T)+0.17]) #TODO: get this from data
    swe = np.array([np.zeros(T)+0.17,np.zeros(T)+0.17]) #TODO: get this from data
    
    #tide
    tid_per= 44700
    a_tide = np.array([0.77]) #this is the amplitude of the tide at s1,s2, ...
    p_tide = np.array([70]) #this is the phase of the tide at s1,s2, ...

    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, tid_per, a_tide, p_tide , T,DT #, (date_start, date_stop)
    
    

def forc_RMD_fromcsv(loc,fil,dat_start, dat_stop, ssea = 33):
    #fil = 'forcing_'+str(sce)+'.csv'
    dat_raw = pd.read_csv(loc+fil ,sep = ' ')
    tvec = dat_raw['Time']
    #select the first and last timestep
    try:
        dstart = dat_start.replace('-','/')
        dstop  = dat_stop.replace('-','/')

        istart = np.where(tvec == dstart)[0][0]
        istop  = np.where(tvec == dstop )[0][0]
    except: 
        raise Exception('Selected period not available in forcing file')
        
    #time parameters
    T = istop - istart
    DT = np.zeros(T) + 24*3600 # I usually work with subtidal time steps of one day

    #forcing conditions
    Qriv   = np.array([dat_raw['QTielDailyMean'][istart:istop],dat_raw['QMegenDailyMean'][istart:istop]]) #mind the sign! this is the discharge at r1,r2,...
    Qweir  = np.array([dat_raw['QGoudaBrugDailyMean'][istart:istop],dat_raw['QHagesteinDailyMean'][istart:istop]])#,0,0]
    Qhar   = np.array([dat_raw['QHaringvlietDailyMean'][istart:istop]])
    n_sea  = np.array([0+ np.zeros(T)]) #this is the water level at s1,s2,...
    soc    = np.array([ssea + np.zeros(T)])  #this is salinity at s1,s2, ...
    sri    = np.array([0.15+ np.zeros(T),0.15+ np.zeros(T)]) #this is the salinity of the river water at r1,r2, ...
    swe    = np.array([0.15+ np.zeros(T),0.15+ np.zeros(T)])
    tid_per= 44700
    a_tide = np.array([0.8]) #this is the amplitude of the tide at s1,s2, ...
    p_tide = np.array([63]) #this is the phase of the tide at s1,s2, ...

    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, tid_per, a_tide, p_tide, T, DT


def forc_RMD_fromMo(loc,fil, ssea = 33):
    dat_raw = pd.read_csv(loc+fil )
    tvec = dat_raw['Time']

    #time parameters
    T = len(tvec)
    DT = np.zeros(T) + 24*3600 # I usually work with subtidal time steps of one day

    #forcing conditions
    Qriv   = np.array([dat_raw['QTielDailyMean'],dat_raw['QMegenDailyMean']]) #mind the sign! this is the discharge at r1,r2,...
    Qweir  = np.array([dat_raw['QGoudaBrugDailyMean'],dat_raw['QHagesteinDailyMean']])#,0,0]
    Qhar   = np.array([dat_raw['QHaringvlietDailyMean']])
    n_sea  = np.array([0+ np.zeros(T)]) #this is the water level at s1,s2,...
    soc    = np.array([ssea + np.zeros(T)])  #this is salinity at s1,s2, ...
    sri    = np.array([0.15+ np.zeros(T),0.15+ np.zeros(T)]) #this is the salinity of the river water at r1,r2, ...
    swe    = np.array([0.15+ np.zeros(T),0.15+ np.zeros(T)])
    tid_per= 44700
    a_tide = np.array([0.77]) #this is the amplitude of the tide at s1,s2, ...
    p_tide = np.array([70]) #this is the phase of the tide at s1,s2, ...

    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, tid_per, a_tide, p_tide, T, DT

def forc_RMD_drought(id_drought):

    #load discharges
    Q_W = pd.read_csv('/Users/biemo004/Documents/UU phd Saltisolutions/Output/RMD_drought/Q_input/Q_W_drought_scen1.csv')[id_drought] 
    Q_L = pd.read_csv('/Users/biemo004/Documents/UU phd Saltisolutions/Output/RMD_drought/Q_input/Q_L_drought_scen1.csv')[id_drought] 
    Q_M = pd.read_csv('/Users/biemo004/Documents/UU phd Saltisolutions/Output/RMD_drought/Q_input/Q_M_drought_scen1.csv')[id_drought] 
    Q_Ha= pd.read_csv('/Users/biemo004/Documents/UU phd Saltisolutions/Output/RMD_drought/Q_input/Q_Ha_drought_scen1.csv')[id_drought] 
    Q_HY= pd.read_csv('/Users/biemo004/Documents/UU phd Saltisolutions/Output/RMD_drought/Q_input/Q_HY_drought_scen1.csv')[id_drought] 

    #time parameters   
    T = len(Q_W)
    DT = np.zeros(T) + 24*3600 # I usually work with subtidal time steps of one day

    #forcing conditions
    Qriv   = np.array([Q_W,Q_M]) #mind the sign! this is the discharge at r1,r2,...
    Qweir  = np.array([Q_HY,Q_L])#,0,0]
    Qhar   = np.array([Q_Ha])
    n_sea  = np.array([0+ np.zeros(T)]) #this is the water level at s1,s2,...
    soc    = np.array([33 + np.zeros(T)])  #this is salinity at s1,s2, ...
    sri    = np.array([0.15+ np.zeros(T),0.15+ np.zeros(T)]) #this is the salinity of the river water at r1,r2, ...
    swe    = np.array([0.15+ np.zeros(T),0.15+ np.zeros(T)])
    tid_per= 44700
    a_tide = np.array([0.77]) #this is the amplitude of the tide at s1,s2, ...
    p_tide = np.array([70]) #this is the phase of the tide at s1,s2, ...

    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, tid_per, a_tide, p_tide, T, DT

def forc_RMD_drought2(id_drought):

    #load discharges
    Q_W = pd.read_csv('/Users/biemo004/Documents/UU phd Saltisolutions/Output/RMD_drought/Q_input2/Q_W_drought2_scen1.csv')[id_drought] 
    Q_L = pd.read_csv('/Users/biemo004/Documents/UU phd Saltisolutions/Output/RMD_drought/Q_input2/Q_L_drought2_scen1.csv')[id_drought] 
    Q_M = pd.read_csv('/Users/biemo004/Documents/UU phd Saltisolutions/Output/RMD_drought/Q_input2/Q_M_drought2_scen1.csv')[id_drought] 
    Q_Ha= pd.read_csv('/Users/biemo004/Documents/UU phd Saltisolutions/Output/RMD_drought/Q_input2/Q_Ha_drought2_scen1.csv')[id_drought] 
    Q_HY= pd.read_csv('/Users/biemo004/Documents/UU phd Saltisolutions/Output/RMD_drought/Q_input2/Q_HY_drought2_scen1.csv')[id_drought] 

    #time parameters   
    T = len(Q_W)
    DT = np.zeros(T) + 24*3600 # I usually work with subtidal time steps of one day

    #forcing conditions
    Qriv   = np.array([Q_W,Q_M]) #mind the sign! this is the discharge at r1,r2,...
    Qweir  = np.array([Q_HY,Q_L])#,0,0]
    Qhar   = np.array([Q_Ha])
    n_sea  = np.array([0+ np.zeros(T)]) #this is the water level at s1,s2,...
    soc    = np.array([33 + np.zeros(T)])  #this is salinity at s1,s2, ...
    sri    = np.array([0.15+ np.zeros(T),0.15+ np.zeros(T)]) #this is the salinity of the river water at r1,r2, ...
    swe    = np.array([0.15+ np.zeros(T),0.15+ np.zeros(T)])
    tid_per= 44700
    a_tide = np.array([0.77]) #this is the amplitude of the tide at s1,s2, ...
    p_tide = np.array([70]) #this is the phase of the tide at s1,s2, ...

    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, tid_per, a_tide, p_tide, T, DT

def forc_RMD_fromJesse(ssea = 33, i0=0, i1=-1):
    path = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/for_Jesse/'
    
    QTiel = np.array(pd.read_excel(path+'NorESM_SSP5-8.5_Tiel.xlsx'))[i0:i1,2]
    QMegen = np.array(pd.read_excel(path+'NorESM_SSP5-8.5_Megen.xlsx'))[i0:i1,2]
    
    #time parameters
    T = len(QTiel)
    DT = np.zeros(T) + 24*3600 # I usually work with subtidal time steps of one day

    #forcing conditions
    Qriv   = np.array([QTiel,QMegen]) #mind the sign! this is the discharge at r1,r2,...
    Qweir  = np.array([0 + np.zeros(T),0 + np.zeros(T)])#,0,0]
    Qhar   = np.array([0 + np.zeros(T)])
    n_sea  = np.array([0+ np.zeros(T)]) #this is the water level at s1,s2,...
    soc    = np.array([ssea + np.zeros(T)])  #this is salinity at s1,s2, ...
    sri    = np.array([0+ np.zeros(T),0+ np.zeros(T)]) #this is the salinity of the river water at r1,r2, ...
    swe    = np.array([0+ np.zeros(T),0+ np.zeros(T)])
    tid_per= 44700
    a_tide = np.array([0.77]) #this is the amplitude of the tide at s1,s2, ...
    p_tide = np.array([70]) #this is the phase of the tide at s1,s2, ...

    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, tid_per, a_tide, p_tide, T, DT


# out9 = forc_RMD_fromcsv('/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/RM_data/MO_Q2122/','Q_daily_mean_Hag-Tie-Meg-Har-Gou_2021-2022_Bouke030524.csv',
#                         '01-09-2022', '01-11-2022')
# import matplotlib.pyplot as plt
# print(out9[0][0])
# plt.plot(out9[0][0])
# plt.plot(out9[0][1])
# plt.plot(out9[1][0])
# plt.plot(out9[1][1])


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
