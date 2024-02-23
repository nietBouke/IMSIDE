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



def phys_DLW():
    #lots of options here possible. 
    
    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot

    #physical parameters, subtidal
    Ut    = 1
    
    Av_st = None #0.0027
    cv_st = 5.5e-5
    
    Kv_st = None 
    Sc_st = 2.1
    
    sf_st = None
    rr_st = 1/2
    
    Kh_st = 25
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
    
    
    return (N, Lsc, nz, nt) , (Ut, Av_st, cv_st, Kv_st, Sc_st, sf_st, rr_st, Kh_st, ch_st) , (Av_ti, cv_ti, Kv_ti, Sc_ti, sf_ti, rr_ti, Kh_ti), \
        (choice_bottomslip_st,choice_bottomslip_ti,choice_viscosityv_st,choice_viscosityv_ti,choice_diffusivityv_st,choice_diffusivityv_ti,choice_diffusivityh_st)


def phys_GUA():
    #lots of options here possible. 
    
    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot

    #physical parameters, subtidal
    Ut    = 1
    
    Av_st = None #0.0027
    cv_st = 1e-4
    
    Kv_st = None 
    Sc_st = 2.2
    
    sf_st = None
    rr_st = 1/2
    
    Kh_st = 22#+10
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
    
    
    return (N, Lsc, nz, nt) , (Ut, Av_st, cv_st, Kv_st, Sc_st, sf_st, rr_st, Kh_st, ch_st) , (Av_ti, cv_ti, Kv_ti, Sc_ti, sf_ti, rr_ti, Kh_ti), \
        (choice_bottomslip_st,choice_bottomslip_ti,choice_viscosityv_st,choice_viscosityv_ti,choice_diffusivityv_st,choice_diffusivityv_ti,choice_diffusivityh_st)




def phys_LOI():
    #lots of options here possible. 
    
    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot

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
    
    
    return (N, Lsc, nz, nt) , (Ut, Av_st, cv_st, Kv_st, Sc_st, sf_st, rr_st, Kh_st, ch_st) , (Av_ti, cv_ti, Kv_ti, Sc_ti, sf_ti, rr_ti, Kh_ti), \
        (choice_bottomslip_st,choice_bottomslip_ti,choice_viscosityv_st,choice_viscosityv_ti,choice_diffusivityv_st,choice_diffusivityv_ti,choice_diffusivityh_st)


def phys_try1():
    #lots of options here possible. 
    
    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot

    #physical parameters, subtidal
    Ut    = 1
    
    Av_st = 0.002
    cv_st = None
    
    Kv_st = None 
    Sc_st = 2.2
    
    sf_st = None
    rr_st = 1/2
    
    Kh_st = 33
    ch_st = None

    #physical parameters, tidal
    Av_ti = [0.01]  #0.06
    cv_ti = [None]  #2.3e-3
    
    Kv_ti = [None]
    Sc_ti = [2.2]
    
    sf_ti = [None] #+ np.inf
    rr_ti = [1/2] #+ np.inf
    
    Kh_ti = [33]

    
    #choiches regarding parametrizations
    #bottom slip
    choice_bottomslip_st = 'rr'
    choice_bottomslip_ti = 'rr'
    choice_viscosityv_st = 'constant'#'constant'
    choice_viscosityv_ti = 'constant'#'cuh'
    choice_diffusivityv_st = 'as'
    choice_diffusivityv_ti = 'as'#'as'
    choice_diffusivityh_st = 'constant'#'cub'
    #choice_diffusivityh_ti = 'constant' #this has to be constant I guess. 
    
    
    return (N, Lsc, nz, nt) , (Ut, Av_st, cv_st, Kv_st, Sc_st, sf_st, rr_st, Kh_st, ch_st) , (Av_ti, cv_ti, Kv_ti, Sc_ti, sf_ti, rr_ti, Kh_ti), \
        (choice_bottomslip_st,choice_bottomslip_ti,choice_viscosityv_st,choice_viscosityv_ti,choice_diffusivityv_st,choice_diffusivityv_ti,choice_diffusivityh_st)

def phys_try2():
    #lots of options here possible. 
    
    #numerical parameters
    N   = 2
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot

    #physical parameters, subtidal
    Ut    = 1
    
    Av_st = 0.002+9e99*0
    cv_st = None
    
    Kv_st = None 
    Sc_st = 2.2
    
    sf_st = None
    rr_st = 1/2
    
    Kh_st = 50
    ch_st = None

    #physical parameters, tidal
    Av_ti = [0.01]  #0.06
    cv_ti = [None]  #2.3e-3
    
    Kv_ti = [None]
    Sc_ti = [2.2]
    
    sf_ti = [None] #+ np.inf
    rr_ti = [1/2] #+ np.inf
    
    Kh_ti = [33]

    
    #choiches regarding parametrizations
    #bottom slip
    choice_bottomslip_st = 'rr'
    choice_bottomslip_ti = 'rr'
    choice_viscosityv_st = 'constant'#'constant'
    choice_viscosityv_ti = 'constant'#'cuh'
    choice_diffusivityv_st = 'as'
    choice_diffusivityv_ti = 'as'#'as'
    choice_diffusivityh_st = 'constant'#'cub'
    #choice_diffusivityh_ti = 'constant' #this has to be constant I guess. 
    
    
    return (N, Lsc, nz, nt) , (Ut, Av_st, cv_st, Kv_st, Sc_st, sf_st, rr_st, Kh_st, ch_st) , (Av_ti, cv_ti, Kv_ti, Sc_ti, sf_ti, rr_ti, Kh_ti), \
        (choice_bottomslip_st,choice_bottomslip_ti,choice_viscosityv_st,choice_viscosityv_ti,choice_diffusivityv_st,choice_diffusivityv_ti,choice_diffusivityh_st)



