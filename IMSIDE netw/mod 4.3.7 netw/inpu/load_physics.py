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
    theta = 0.5*2
    
    #physical parameters, subtidal
    #vertical viscosity: constant, cH, or cuH
    choice_viscosityv_st = 'cH'
    Av_st = 0.0024
    Av_st_cH = None
    Av_st_cuH = None
    
    #vertical diffusivity: constant or Schmidt
    choice_diffusivityv_st = 'Schmidt'
    Kv_st = None
    Kv_st_Sc = 2.2
    
    #bottom slip: rr or constant
    choice_bottomslip_st = 'rr'
    sf_st = None
    sf_st_rr = 1/2
    
    #horizontal diffusivity: constant or cb or cub
    choice_diffusivityh_st = 'constant'#'cub'
    Kh_st = 275
    Kh_st_cb = None
    Kh_st_cub = None
    
    #physical parameters, tidal - TODO
    Av_ti = 0.025 #0.02
    Kv_ti = Av_ti/2.2
    sf_ti = 'rr1/2'
    Kh_ti = 25
    
    #choiches regarding parametrizations
    choice_bottomslip_ti = 'rr'
    choice_viscosityv_ti = 'cuh'#'cuh'
    choice_diffusivityv_ti = 'as'#'as'
    
    
    return (N, Lsc, nz, nt, theta) , \
        (choice_viscosityv_st, Av_st, Av_st_cH, Av_st_cuH)  , \
        (choice_diffusivityv_st, Kv_st, Kv_st_Sc)  , \
        (choice_bottomslip_st, sf_st, sf_st_rr)  , \
        (choice_diffusivityh_st, Kh_st, Kh_st_cb, Kh_st_cub)  , \
        (choice_viscosityv_ti, Av_ti)  , \
        (choice_diffusivityv_ti, Kv_ti)  , \
        (choice_bottomslip_ti, sf_ti)  , \
        Kh_ti 

def phys_RMD2():
    

    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot
    theta = 0.5*2
    
    #physical parameters, subtidal
    #vertical viscosity: constant, cH, or cuH
    choice_viscosityv_st = 'cH'
    Av_st = None
    Av_st_cH = 1.75e-4
    Av_st_cuH = None
    
    #vertical diffusivity: constant or Schmidt
    choice_diffusivityv_st = 'Schmidt'
    Kv_st = None
    Kv_st_Sc = 2.2
    
    #bottom slip: rr or constant
    choice_bottomslip_st = 'rr'
    sf_st = None
    sf_st_rr = 1/2
    
    #horizontal diffusivity: constant or cb or cub
    choice_diffusivityh_st = 'cb'
    Kh_st = None
    Kh_st_cb = 0.7
    Kh_st_cub = None
    Kh_st_tres= 25
    
    #physical parameters, tidal 
    #vertical viscosity: constant, cH, or cuH
    choice_viscosityv_ti = 'cH'
    Av_ti = None 
    Av_ti_cH = 2.6e-3
    Av_ti_cuH = None
    
    #vertical diffusivity: constant or Schmidt
    choice_diffusivityv_ti = 'Schmidt'
    Kv_ti = None
    Kv_ti_Sc = 2.2
    
    #bottom slip: rr or constant
    choice_bottomslip_ti = 'rr'
    sf_ti = None
    sf_ti_rr = 1/2
    
    #horizontal tidal diffusivity is always constant
    Kh_ti = 25
        
    
    return (N, Lsc, nz, nt, theta) , \
        (choice_viscosityv_st, Av_st, Av_st_cH, Av_st_cuH)  , \
        (choice_diffusivityv_st, Kv_st, Kv_st_Sc)  , \
        (choice_bottomslip_st, sf_st, sf_st_rr)  , \
        (choice_diffusivityh_st, Kh_st, Kh_st_cb, Kh_st_cub, Kh_st_tres)  , \
        (choice_viscosityv_ti, Av_ti, Av_ti_cH,  Av_ti_cuH)  , \
        (choice_diffusivityv_ti, Kv_ti, Kv_ti_Sc)  , \
        (choice_bottomslip_ti, sf_ti, sf_ti_rr)  , \
        Kh_ti 

def phys_RMD_tune(Av_st_cH = 1.75e-4, Kv_st_Sc = 2.2, Kh_st_cb = 0.7, Av_ti_cH = 2.6e-3, Kv_ti_Sc = 2.2, sf_ti_rr = 1/2, Kh_ti = 25):
    # Tune the physics of the RMD
  
    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot
    theta = 0.5*2
    
    #physical parameters, subtidal
    #vertical viscosity: constant, cH, or cuH
    choice_viscosityv_st = 'cH'
    Av_st = None
    #Av_st_cH = 2e-4
    Av_st_cuH = None
    
    #vertical diffusivity: constant or Schmidt
    choice_diffusivityv_st = 'Schmidt'
    Kv_st = None
    #Kv_st_Sc = 2.2
    
    #bottom slip: rr or constant
    choice_bottomslip_st = 'rr'
    sf_st = None
    sf_st_rr = 1/2
    
    #horizontal diffusivity: constant or cb or cub
    choice_diffusivityh_st = 'cb'
    Kh_st = None
    #Kh_st_cb = 0.5
    Kh_st_cub = None
    Kh_st_tres= 25
    
    #physical parameters, tidal 
    #vertical viscosity: constant, cH, or cuH
    choice_viscosityv_ti = 'cH'
    Av_ti = None 
    #Av_ti_cH = 2e-3
    Av_ti_cuH = None
    
    #vertical diffusivity: constant or Schmidt
    choice_diffusivityv_ti = 'Schmidt'
    Kv_ti = None
    #Kv_ti_Sc = 2.2
    
    #bottom slip: rr or constant
    choice_bottomslip_ti = 'rr'
    sf_ti = None
    #sf_ti_rr = 1/2
    
    #horizontal tidal diffusivity is always constant
    #Kh_ti = 25
        
    
    return (N, Lsc, nz, nt, theta) , \
        (choice_viscosityv_st, Av_st, Av_st_cH, Av_st_cuH)  , \
        (choice_diffusivityv_st, Kv_st, Kv_st_Sc)  , \
        (choice_bottomslip_st, sf_st, sf_st_rr)  , \
        (choice_diffusivityh_st, Kh_st, Kh_st_cb, Kh_st_cub, Kh_st_tres)  , \
        (choice_viscosityv_ti, Av_ti, Av_ti_cH,  Av_ti_cuH)  , \
        (choice_diffusivityv_ti, Kv_ti, Kv_ti_Sc)  , \
        (choice_bottomslip_ti, sf_ti, sf_ti_rr)  , \
        Kh_ti 


def phys_test1():
    # an example of forcing

    #numerical parameters
    N   = 5
    Lsc = 1000
    nz  = 51 #vertical step 
    nt  = 121 #time step iu tidal cylce - only for plot
    theta = 0.5

    #physical parameters, subtidal
    Av_st = 0.002
    Kv_st = Av_st/2.2
    sf_st = 2*Av_st/10
    Kh_st = 33
    
    #physical parameters, tidal
    Av_ti = 0.02
    Kv_ti = Av_ti/2.2
    sf_ti = 2*Av_ti/10
    Kh_ti = 33


    return (N, Lsc, nz, nt, theta) , (Av_st, Kv_st, sf_st, Kh_st) , (Av_ti, Kv_ti, sf_ti, Kh_ti)



#we should keep the option open to let parameters depend on width and depth and such. that has to be build in. 
