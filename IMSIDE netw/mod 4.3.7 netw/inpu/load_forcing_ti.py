# =============================================================================
# here the forcing files for the network are placed.
# =============================================================================
import numpy as np

def forc_RMD1():
    Qriv = [1834,327] #mind the sign! this is the discharge at r1,r2,...
    Qweir= [0,348]
    Qhar = [821]
    n_sea = [0] #this is the water level at s1,s2,...
    soc = [35] #this is salinity at s1,s2, ...
    sri = [0,0] #this is the salinity of the river water at r1,r2, ...
    swe = [0,0]
    tid_per = 44700
    a_tide = [0.83*0] #this is the amplitude of the tide at s1,s2, ...
    p_tide = [70] #this is the phase of the tide at s1,s2, ...

    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, tid_per, a_tide, p_tide

def forc_RMD2():
    Qriv = [10500,230] #mind the sign! this is the discharge at r1,r2,...
    Qweir= [0,490]
    Qhar = [0]
    n_sea = [0] #this is the water level at s1,s2,...
    soc = [33] #this is salinity at s1,s2, ...
    sri = [0.15,0.15] #this is the salinity of the river water at r1,r2, ...
    swe= [0.15,0.15]
    tid_per = 44700
    a_tide = [0.83] #this is the amplitude of the tide at s1,s2, ...
    p_tide = [70] #this is the phase of the tide at s1,s2, ...

    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, tid_per, a_tide, p_tide

def forc_RMD3():
    Qriv = [1504,264] #mind the sign! this is the discharge at r1,r2,...
    Qweir= [0,390]#,0,0]
    Qhar = [516]
    n_sea = [0] #this is the water level at s1,s2,...
    soc = [35] #this is salinity at s1,s2, ...
    sri = [0,0] #this is the salinity of the river water at r1,r2, ...
    swe = [0,0]
    tid_per = 44700
    a_tide = [0.83] #this is the amplitude of the tide at s1,s2, ...
    p_tide = [70] #this is the phase of the tide at s1,s2, ...

    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, tid_per, a_tide, p_tide

def forc_RMD4():
    Qriv = [650,50] #mind the sign! this is the discharge at r1,r2,...
    Qweir= [0,0]#,0,0]
    Qhar = [0]
    n_sea = [0] #this is the water level at s1,s2,...
    soc = [33]  #this is salinity at s1,s2, ...
    sri = [0,0] #this is the salinity of the river water at r1,r2, ...
    swe = [0,0]
    tid_per = 44700
    a_tide = [0.8] #this is the amplitude of the tide at s1,s2, ...
    p_tide = [63] #this is the phase of the tide at s1,s2, ...

    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, tid_per, a_tide, p_tide

def forc_test1():
    Qriv = [] #mind the sign! this is the discharge at r1,r2,...
    Qweir= [50,-20]
    Qhar = []
    n_sea = [0] #this is the water level at s1,s2,...
    soc = [35] #this is salinity at s1,s2, ...
    sri = [] #this is the salinity of the river water at r1,r2, ...
    swe = [0,0]
    tid_per = 44700
    a_tide = [1] #this is the amplitude of the tide at s1,s2, ...
    p_tide = [0] #this is the phase of the tide at s1,s2, ...

    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, tid_per, a_tide, p_tide

def forc_RR():
    Qriv = [500,100] #mind the sign! this is the discharge at r1,r2,...
    Qweir= []
    Qhar = []
    n_sea = [0,0,0,0] #this is the water level at s1,s2,...
    soc = [35,35,35,35] #this is salinity at s1,s2, ...
    swe = []
    sri = [0,0] #this is the salinity of the river water at r1,r2, ...
    tid_per = 44700
    a_tide = [1.7,1.5,1,1.2] #this is the amplitude of the tide at s1,s2, ...
    p_tide = [20,10,0,2] #this is the phase of the tide at s1,s2, ...

    return Qriv, Qweir, Qhar,n_sea, soc, sri, swe, tid_per, a_tide, p_tide

def forc_fun():
    Qriv = [50,300] #mind the sign! this is the discharge at r1,r2,...
    Qweir= []
    Qhar = []
    n_sea = [0,0] #this is the water level at s1,s2,...
    soc = [35,35] #this is salinity at s1,s2, ...
    sri = [0,0] #this is the salinity of the river water at r1,r2, ...
    swe = []
    tid_per = 44700
    a_tide = [0.5,1.5] #this is the amplitude of the tide at s1,s2, ...
    p_tide = [0,0] #this is the phase of the tide at s1,s2, ...

    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, tid_per, a_tide, p_tide

def forc_RMD_oldmap1():
    Qriv = [648,185] #mind the sign! this is the discharge at r1,r2,...
    Qweir= []#,0,0]
    Qhar = []
    n_sea = [0,0] #this is the water level at s1,s2,...
    soc = [35,35]  #this is salinity at s1,s2, ...
    sri = [0,0] #this is the salinity of the river water at r1,r2, ...
    swe = []
    tid_per = 44700
    a_tide = [0.83,0.83] #this is the amplitude of the tide at s1,s2, ...
    p_tide = [70,50] #this is the phase of the tide at s1,s2, ...

    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, tid_per, a_tide, p_tide


def forc_RMD_HVO_1():
    Qriv = [1500,230] #mind the sign! this is the discharge at r1,r2,...
    Qweir= [0,490]#,0,0]
    Qhar = []
    n_sea = [0,0] #this is the water level at s1,s2,...
    soc = [33,33]  #this is salinity at s1,s2, ...
    sri = [0.15,0.15] #this is the salinity of the river water at r1,r2, ...
    swe = [0.15,0.15]
    tid_per = 44700
    a_tide = [0.83,0.83] #this is the amplitude of the tide at s1,s2, ...
    p_tide = [70,50] #this is the phase of the tide at s1,s2, ...

    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, tid_per, a_tide, p_tide
