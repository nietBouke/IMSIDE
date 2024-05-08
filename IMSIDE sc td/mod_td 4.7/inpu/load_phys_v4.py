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
'''
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
'''

def phys_all3est():
    #lots of options here possible. 
    
    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot
    theta = 0.5

    #physical parameters, subtidal
    Ut    = 1
    
    Av_st = None #0.0027
    cv_st = 1.9e-4
    
    Kv_st = None 
    Sc_st = 10.8
    
    sf_st = None
    rr_st = 1/2
    
    Kh_st = 45
    ch_st = None

    #physical parameters, tidal
    Av_ti = [None]  #0.06
    cv_ti = [1.3e-3]  #0.06
    
    Kv_ti = [None]
    Sc_ti = [2.44]
    
    sf_ti = [None] #+ np.inf
    rr_ti = [1/2] #+ np.inf
    
    Kh_ti = [20]

    
    #choiches regarding parametrizations
    #bottom slip
    choice_bottomslip_st = 'rr'
    choice_bottomslip_ti = 'rr'
    choice_viscosityv_st = 'cuh'#'constant'
    choice_viscosityv_ti = 'cuh'#'cuh'
    choice_diffusivityv_st = 'as'
    choice_diffusivityv_ti = 'as'#'as'
    choice_diffusivityh_st = 'constant'#'cub'
    #choice_diffusivityh_ti = 'constant' #this has to be constant I guess. 
    
    
    return (N, Lsc, nz, nt, theta) , (Ut, Av_st, cv_st, Kv_st, Sc_st, sf_st, rr_st, Kh_st, ch_st) , (Av_ti, cv_ti, Kv_ti, Sc_ti, sf_ti, rr_ti, Kh_ti), \
        (choice_bottomslip_st,choice_bottomslip_ti,choice_viscosityv_st,choice_viscosityv_ti,choice_diffusivityv_st,choice_diffusivityv_ti,choice_diffusivityh_st)

def phys_all3est_v2():
    #lots of options here possible. 
    
    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot
    theta = 0.5

    #physical parameters, subtidal
    Ut    = 1
    
    Av_st = None #0.0027
    cv_st = 0.6e-4 #1.1e-4
    
    Kv_st = None 
    Sc_st = 2.2
    
    sf_st = None
    rr_st = 1/2
    
    Kh_st = 25
    ch_st = None

    #physical parameters, tidal
    Av_ti = [None]  #0.06
    cv_ti = [1.3e-3]  #0.06
    
    Kv_ti = [None]
    Sc_ti = [2.44]
    
    sf_ti = [None] #+ np.inf
    rr_ti = [1/2] #+ np.inf
    
    Kh_ti = [20]

    
    #choiches regarding parametrizations
    #bottom slip
    choice_bottomslip_st = 'rr'
    choice_bottomslip_ti = 'rr'
    choice_viscosityv_st = 'cuh'#'constant'
    choice_viscosityv_ti = 'cuh'#'cuh'
    choice_diffusivityv_st = 'as'
    choice_diffusivityv_ti = 'as'#'as'
    choice_diffusivityh_st = 'constant'#'cub'
    #choice_diffusivityh_ti = 'constant' #this has to be constant I guess. 
    
    
    return (N, Lsc, nz, nt, theta) , (Ut, Av_st, cv_st, Kv_st, Sc_st, sf_st, rr_st, Kh_st, ch_st) , (Av_ti, cv_ti, Kv_ti, Sc_ti, sf_ti, rr_ti, Kh_ti), \
        (choice_bottomslip_st,choice_bottomslip_ti,choice_viscosityv_st,choice_viscosityv_ti,choice_diffusivityv_st,choice_diffusivityv_ti,choice_diffusivityh_st)


def phys_all3est_tune(cv_st_t, Sc_st_t, Kh_st_t, Sc_ti_t):
    #lots of options here possible. 
    
    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot
    theta = 0.5

    #physical parameters, subtidal
    Ut    = 1
    
    Av_st = None #0.0027
    cv_st = cv_st_t
    
    Kv_st = None 
    Sc_st = Sc_st_t
    
    sf_st = None
    rr_st = 1/2
    
    Kh_st = Kh_st_t
    ch_st = None

    #physical parameters, tidal
    Av_ti = [None]  #0.06
    cv_ti = [1.3e-3]  #0.06
    
    Kv_ti = [None]
    Sc_ti = [Sc_ti_t]
    
    sf_ti = [None] #+ np.inf
    rr_ti = [1/2] #+ np.inf
    
    Kh_ti = [20]

    
    #choiches regarding parametrizations
    #bottom slip
    choice_bottomslip_st = 'rr'
    choice_bottomslip_ti = 'rr'
    choice_viscosityv_st = 'cuh'#'constant'
    choice_viscosityv_ti = 'cuh'#'cuh'
    choice_diffusivityv_st = 'as'
    choice_diffusivityv_ti = 'as'#'as'
    choice_diffusivityh_st = 'constant'#'cub'
    #choice_diffusivityh_ti = 'constant' #this has to be constant I guess. 
    
    
    return (N, Lsc, nz, nt, theta) , (Ut, Av_st, cv_st, Kv_st, Sc_st, sf_st, rr_st, Kh_st, ch_st) , (Av_ti, cv_ti, Kv_ti, Sc_ti, sf_ti, rr_ti, Kh_ti), \
        (choice_bottomslip_st,choice_bottomslip_ti,choice_viscosityv_st,choice_viscosityv_ti,choice_diffusivityv_st,choice_diffusivityv_ti,choice_diffusivityh_st)


def phys_DLW_tune(cv_ti_t, cv_st_t, Sc_st_t, Kh_st_t, Sc_ti_t):
    #lots of options here possible. 
    
    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot
    theta = 0.5

    #physical parameters, subtidal
    Ut    = 1
    
    Av_st = None #0.0027
    cv_st = cv_st_t
    
    Kv_st = None 
    Sc_st = Sc_st_t
    
    sf_st = None
    rr_st = 1/2
    
    Kh_st = Kh_st_t
    ch_st = None

    #physical parameters, tidal
    Av_ti = [None]  #0.06
    cv_ti = [cv_ti_t]  #1.4e-4
    
    Kv_ti = [None]
    Sc_ti = [Sc_ti_t]
    
    sf_ti = [None] #+ np.inf
    rr_ti = [1/2] #+ np.inf
    
    Kh_ti = [20]

    
    #choiches regarding parametrizations
    #bottom slip
    choice_bottomslip_st = 'rr'
    choice_bottomslip_ti = 'rr'
    choice_viscosityv_st = 'cuh'#'constant'
    choice_viscosityv_ti = 'cuh'#'cuh'
    choice_diffusivityv_st = 'as'
    choice_diffusivityv_ti = 'as'#'as'
    choice_diffusivityh_st = 'constant'#'cub'
    #choice_diffusivityh_ti = 'constant' #this has to be constant I guess. 
    
    
    return (N, Lsc, nz, nt, theta) , (Ut, Av_st, cv_st, Kv_st, Sc_st, sf_st, rr_st, Kh_st, ch_st) , (Av_ti, cv_ti, Kv_ti, Sc_ti, sf_ti, rr_ti, Kh_ti), \
        (choice_bottomslip_st,choice_bottomslip_ti,choice_viscosityv_st,choice_viscosityv_ti,choice_diffusivityv_st,choice_diffusivityv_ti,choice_diffusivityh_st)



def phys_GUA_tune(cv_ti_t, cv_st_t, Sc_st_t, Kh_st_t, Sc_ti_t):
    #lots of options here possible. 
    
    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot
    theta = 0.5

    #physical parameters, subtidal
    Ut    = 1
    
    Av_st = None #0.0027
    cv_st = cv_st_t
    
    Kv_st = None 
    Sc_st = Sc_st_t
    
    sf_st = None
    rr_st = 1/2
    
    Kh_st = Kh_st_t
    ch_st = None

    #physical parameters, tidal
    Av_ti = [None]  #0.06
    cv_ti = [cv_ti_t]  #1.2e-3
    
    Kv_ti = [None]
    Sc_ti = [Sc_ti_t]
    
    sf_ti = [None] #+ np.inf
    rr_ti = [1/2] #+ np.inf
    
    Kh_ti = [20]

    
    #choiches regarding parametrizations
    #bottom slip
    choice_bottomslip_st = 'rr'
    choice_bottomslip_ti = 'rr'
    choice_viscosityv_st = 'cuh'#'constant'
    choice_viscosityv_ti = 'cuh'#'cuh'
    choice_diffusivityv_st = 'as'
    choice_diffusivityv_ti = 'as'#'as'
    choice_diffusivityh_st = 'constant'#'cub'
    #choice_diffusivityh_ti = 'constant' #this has to be constant I guess. 
    
    
    return (N, Lsc, nz, nt, theta) , (Ut, Av_st, cv_st, Kv_st, Sc_st, sf_st, rr_st, Kh_st, ch_st) , (Av_ti, cv_ti, Kv_ti, Sc_ti, sf_ti, rr_ti, Kh_ti), \
        (choice_bottomslip_st,choice_bottomslip_ti,choice_viscosityv_st,choice_viscosityv_ti,choice_diffusivityv_st,choice_diffusivityv_ti,choice_diffusivityh_st)




def phys_LOI_tune(cv_ti_t, cv_st_t, Sc_st_t, Kh_st_t, Sc_ti_t):
    #lots of options here possible. 
    
    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot
    theta = 0.5

    #physical parameters, subtidal
    Ut    = 1
    
    Av_st = None #0.0027
    cv_st = cv_st_t
    
    Kv_st = None 
    Sc_st = Sc_st_t
    
    sf_st = None
    rr_st = 1/2
    
    Kh_st = Kh_st_t
    ch_st = None

    #physical parameters, tidal
    Av_ti = [None]  #0.06
    cv_ti = [cv_ti_t]  #2.3e-3
    
    Kv_ti = [None]
    Sc_ti = [Sc_ti_t]
    
    sf_ti = [None] #+ np.inf
    rr_ti = [1/2] #+ np.inf
    
    Kh_ti = [20]

    
    #choiches regarding parametrizations
    #bottom slip
    choice_bottomslip_st = 'rr'
    choice_bottomslip_ti = 'rr'
    choice_viscosityv_st = 'cuh'#'constant'
    choice_viscosityv_ti = 'cuh'#'cuh'
    choice_diffusivityv_st = 'as'
    choice_diffusivityv_ti = 'as'#'as'
    choice_diffusivityh_st = 'constant'#'cub'
    #choice_diffusivityh_ti = 'constant' #this has to be constant I guess. 
    
    
    return (N, Lsc, nz, nt, theta) , (Ut, Av_st, cv_st, Kv_st, Sc_st, sf_st, rr_st, Kh_st, ch_st) , (Av_ti, cv_ti, Kv_ti, Sc_ti, sf_ti, rr_ti, Kh_ti), \
        (choice_bottomslip_st,choice_bottomslip_ti,choice_viscosityv_st,choice_viscosityv_ti,choice_diffusivityv_st,choice_diffusivityv_ti,choice_diffusivityh_st)




def phys_DLW():
    #lots of options here possible. 
    
    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot
    theta = 0.5

    #physical parameters, subtidal
    Ut    = 1
    
    Av_st = None #0.0027
    cv_st = 5.9e-5
    
    Kv_st = None 
    Sc_st = 2.1
    
    sf_st = None
    rr_st = 1/2
    
    Kh_st = 36
    ch_st = None

    #physical parameters, tidal
    Av_ti = [None]  #0.06
    cv_ti = [1.3e-3]  #1.4e-4
    
    Kv_ti = [None]
    Sc_ti = [2.2]
    
    sf_ti = [None] #+ np.inf
    rr_ti = [1/2] #+ np.inf
    
    Kh_ti = [20]

    
    #choiches regarding parametrizations
    #bottom slip
    choice_bottomslip_st = 'rr'
    choice_bottomslip_ti = 'rr'
    choice_viscosityv_st = 'cuh'#'constant'
    choice_viscosityv_ti = 'cuh'#'cuh'
    choice_diffusivityv_st = 'as'
    choice_diffusivityv_ti = 'as'#'as'
    choice_diffusivityh_st = 'constant'#'cub'
    #choice_diffusivityh_ti = 'constant' #this has to be constant I guess. 
    
    
    return (N, Lsc, nz, nt, theta) , (Ut, Av_st, cv_st, Kv_st, Sc_st, sf_st, rr_st, Kh_st, ch_st) , (Av_ti, cv_ti, Kv_ti, Sc_ti, sf_ti, rr_ti, Kh_ti), \
        (choice_bottomslip_st,choice_bottomslip_ti,choice_viscosityv_st,choice_viscosityv_ti,choice_diffusivityv_st,choice_diffusivityv_ti,choice_diffusivityh_st)


def phys_GUA():
    #lots of options here possible. 
    
    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot
    theta = 0.5

    #physical parameters, subtidal
    Ut    = 1
    
    Av_st = None #0.0027
    cv_st = 1e-4
    
    Kv_st = None 
    Sc_st = 2.2
    
    sf_st = None
    rr_st = 1/2
    
    Kh_st = 22
    ch_st = None

    #physical parameters, tidal
    Av_ti = [None]  #0.06
    cv_ti = [1.2e-3]  #1.2e-3
    
    Kv_ti = [None]
    Sc_ti = [1.9]
    
    sf_ti = [None] #+ np.inf
    rr_ti = [1/2] #+ np.inf
    
    Kh_ti = [20]

    
    #choiches regarding parametrizations
    #bottom slip
    choice_bottomslip_st = 'rr'
    choice_bottomslip_ti = 'rr'
    choice_viscosityv_st = 'cuh'#'constant'
    choice_viscosityv_ti = 'cuh'#'cuh'
    choice_diffusivityv_st = 'as'
    choice_diffusivityv_ti = 'as'#'as'
    choice_diffusivityh_st = 'constant'#'cub'
    #choice_diffusivityh_ti = 'constant' #this has to be constant I guess. 
    
    
    return (N, Lsc, nz, nt, theta) , (Ut, Av_st, cv_st, Kv_st, Sc_st, sf_st, rr_st, Kh_st, ch_st) , (Av_ti, cv_ti, Kv_ti, Sc_ti, sf_ti, rr_ti, Kh_ti), \
        (choice_bottomslip_st,choice_bottomslip_ti,choice_viscosityv_st,choice_viscosityv_ti,choice_diffusivityv_st,choice_diffusivityv_ti,choice_diffusivityh_st)




def phys_LOI():
    #lots of options here possible. 
    
    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot
    theta = 0.5

    #physical parameters, subtidal
    Ut    = 1
    
    Av_st = None #0.0027
    cv_st = 2.2e-4
    
    Kv_st = None 
    Sc_st = 2.2
    
    sf_st = None
    rr_st = 1/2
    
    Kh_st = 44
    ch_st = None

    #physical parameters, tidal
    Av_ti = [None]  #0.06
    cv_ti = [2.3e-3]  #2.3e-3
    
    Kv_ti = [None]
    Sc_ti = [2.2]
    
    sf_ti = [None] #+ np.inf
    rr_ti = [1/2] #+ np.inf
    
    Kh_ti = [20]

    
    #choiches regarding parametrizations
    #bottom slip
    choice_bottomslip_st = 'rr'
    choice_bottomslip_ti = 'rr'
    choice_viscosityv_st = 'cuh'#'constant'
    choice_viscosityv_ti = 'cuh'#'cuh'
    choice_diffusivityv_st = 'as'
    choice_diffusivityv_ti = 'as'#'as'
    choice_diffusivityh_st = 'constant'#'cub'
    #choice_diffusivityh_ti = 'constant' #this has to be constant I guess. 
    
    
    return (N, Lsc, nz, nt, theta) , (Ut, Av_st, cv_st, Kv_st, Sc_st, sf_st, rr_st, Kh_st, ch_st) , (Av_ti, cv_ti, Kv_ti, Sc_ti, sf_ti, rr_ti, Kh_ti), \
        (choice_bottomslip_st,choice_bottomslip_ti,choice_viscosityv_st,choice_viscosityv_ti,choice_diffusivityv_st,choice_diffusivityv_ti,choice_diffusivityh_st)






