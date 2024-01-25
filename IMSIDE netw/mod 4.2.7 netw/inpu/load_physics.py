# =============================================================================
# In this file the physical constants are loaded 
# 
# =============================================================================


def phys_gen():
    g = 9.81 #gravitation
    Be= 7.6e-4 #isohaline contractie
    CD= 0.001 #wind drag coefficient
    r = 1.225/1000 #density air divided by density water   
    tol = 0.01
    return g , Be, CD, r, tol

def phys_RMD1():
    # an example of forcing

    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot
    theta = 0.5
    
    #physical parameters, subtidal
    Av_st = 0.0017
    Kv_st = Av_st/2.2
    sf_st = 2*Av_st/10
    Kh_st = 300
    
    #physical parameters, tidal
    Av_ti = 0.02
    Kv_ti = Av_ti/2.2
    sf_ti = 0.009
    Kh_ti = 33
    

    return (N, Lsc, nz, nt, theta) , (Av_st, Kv_st, sf_st, Kh_st) , (Av_ti, Kv_ti, sf_ti, Kh_ti)

def phys_test1():
    # an example of forcing

    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot
    theta = 0.5*2

    #physical parameters, subtidal
    Av_st = 0.002
    Kv_st = Av_st/2.2
    sf_st = 2*Av_st/10
    Kh_st = 33
    
    #physical parameters, tidal
    Av_ti = 0.02
    Kv_ti = Av_ti/2.2
    sf_ti = 0.01 
    Kh_ti = 33


    return (N, Lsc, nz, nt, theta) , (Av_st, Kv_st, sf_st, Kh_st) , (Av_ti, Kv_ti, sf_ti, Kh_ti)



#we should keep the option open to let parameters depend on width and depth and such. that has to be build in. 