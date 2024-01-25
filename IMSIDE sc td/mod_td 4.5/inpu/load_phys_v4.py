# =============================================================================
# In this file the physical constants are loaded 
# 
# =============================================================================


def phys_gen():
    g = 9.81 #gravitation
    Be= 7.6e-4 #isohaline contractie
    Sc= 2.2 #Schmidt getal 1#
    cv= 0.001/7.1 #7.28e-5 #empirische constante 1e-4#
    ch= 0.035 #empirische constante
    CD= 0.001 #wind drag coefficient
    r = 1.225/1000 #density air divided by density water   
    
    return g , Be, Sc, cv, ch, CD, r


def phys_LOI1():
    # an example of forcing

    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot
    theta = 0.5

    #physical parameters, subtidal
    Av_st = 0.005
    Kv_st = Av_st/2.2
    sf_st = None
    rr_st = 1/2
    Kh_st = 30
    
    #physical parameters, tidal
    Av_ti = [0.023]  #0.06
    Kv_ti = [0.056/2.2]
    sf_ti = [None]
    rr_ti = [1/2]
    Kh_ti = [20]
    
    #choices for parametrizations
    choice_bottomslip_st = 'rr'
    choice_bottomslip_ti = 'rr'

    return (N, Lsc, nz, nt, theta) , (Av_st, Kv_st, sf_st, rr_ti, Kh_st) , (Av_ti, Kv_ti, sf_ti, rr_ti, Kh_ti), (choice_bottomslip_st , choice_bottomslip_ti)

def phys_LOI_A1():
    # an example of forcing

    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot
    theta = 0.5

    #physical parameters, subtidal
    Av_st = 0.0026
    Kv_st = Av_st/2.3
    sf_st = None
    rr_st = 1/2
    Kh_st = 55
    
    #physical parameters, tidal
    Av_ti = [0.023]  #0.06
    Kv_ti = [0.023/2.2]
    sf_ti = [None] #+ np.inf
    rr_ti = [1/2] #+ np.inf
    Kh_ti = [20]

    
    #choiches regarding parametrizations
    #bottom slip
    choice_bottomslip_st = 'rr'
    choice_bottomslip_ti = 'rr'
    
    
    return (N, Lsc, nz, nt, theta) , (Av_st, Kv_st, sf_st, rr_st, Kh_st) , (Av_ti, Kv_ti, sf_ti, rr_ti, Kh_ti), (choice_bottomslip_st,choice_bottomslip_ti)

def phys_LOI_B1():
    # an example of forcing

    #numerical parameters
    N   = 1
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot
    theta = 0.5

    #physical parameters, subtidal
    Av_st = 1000
    Kv_st = Av_st/2.3
    sf_st = None
    rr_st = 1/2
    Kh_st = 180
    
    #physical parameters, tidal
    Av_ti = [0.023]  #0.06
    Kv_ti = [0.023/3.1]
    sf_ti = [None] #+ np.inf
    rr_ti = [1/2] #+ np.inf
    Kh_ti = [20]

    
    #choiches regarding parametrizations
    #bottom slip
    choice_bottomslip_st = 'rr'
    choice_bottomslip_ti = 'rr'
    
    
    return (N, Lsc, nz, nt, theta) , (Av_st, Kv_st, sf_st, rr_st, Kh_st) , (Av_ti, Kv_ti, sf_ti, rr_ti, Kh_ti), (choice_bottomslip_st,choice_bottomslip_ti)

def phys_LOI_C1():
    # an example of forcing

    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot
    theta = 0.5

    #physical parameters, subtidal
    Av_st = 0.0028
    Kv_st = Av_st/2.3
    sf_st = None
    rr_st = 1/2
    Kh_st = 55
    
    #physical parameters, tidal
    Av_ti = [0.023]  #0.06
    Kv_ti = [0.023/2.2]
    sf_ti = [None] #+ np.inf
    rr_ti = [1/2] #+ np.inf
    Kh_ti = [20]

    
    #choiches regarding parametrizations
    #bottom slip
    choice_bottomslip_st = 'rr'
    choice_bottomslip_ti = 'rr'
    
    
    return (N, Lsc, nz, nt, theta) , (Av_st, Kv_st, sf_st, rr_st, Kh_st) , (Av_ti, Kv_ti, sf_ti, rr_ti, Kh_ti), (choice_bottomslip_st,choice_bottomslip_ti)

def phys_LOI_D1():
    # an example of forcing

    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot
    theta = 0.5

    #physical parameters, subtidal
    Av_st = 0.0027
    Kv_st = Av_st/2.1
    sf_st = None
    rr_st = 1/2
    Kh_st = 55
    
    #physical parameters, tidal
    Av_ti = [0.023]  #0.06
    Kv_ti = [0.023/2.2]
    sf_ti = [None] #+ np.inf
    rr_ti = [1/2] #+ np.inf
    Kh_ti = [20]

    
    #choiches regarding parametrizations
    #bottom slip
    choice_bottomslip_st = 'rr'
    choice_bottomslip_ti = 'rr'
    
    
    return (N, Lsc, nz, nt, theta) , (Av_st, Kv_st, sf_st, rr_st, Kh_st) , (Av_ti, Kv_ti, sf_ti, rr_ti, Kh_ti), (choice_bottomslip_st,choice_bottomslip_ti)




