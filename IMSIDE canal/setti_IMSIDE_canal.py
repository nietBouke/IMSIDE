# =============================================================================
# In this file, you can adjust the parameters, initial conditions, geometry etc 
# for running IMSIDE for the canal case (with shipping lock)
# For questions, contact Bouke 
# =============================================================================

import numpy as np

# =============================================================================
# the functions below are used for some conversions and should not be changed
# =============================================================================
def compute_nx(Ln,dxn): return int(np.sum(Ln/dxn)+len(Ln))

def set_to_init(inp_bc, inp_num, inp_geo, inp_mix, inp_t, inp_lk):
    #convert the input from the model to input to find the equilibrium solution for t=0
    out_num = inp_num # no changes to numerical settings
    T=2
    out_t = (T,np.inf+np.zeros(T)) #one timestep with infinite duration to find equilibrium
    out_geo = inp_geo #no changes
    out_lk = inp_lk # no changes
    out_mix, out_bc = [],[]
    for v in inp_mix: 
        if type(v[0]) == float or type(v[0]) == np.float64: out_mix.append(np.array([v[0]]*T)) #only the first timestep
        else: out_mix.append(np.tile(v[0] , T).reshape((T,len(v[0])))) #only the first timestep
    for v in inp_bc: 
        if type(v[0]) == float or type(v[0]) == np.float64: out_bc.append(np.array([v[0]]*T)) #only the first timestep
        else: out_bc.append(np.tile(v[0] , T).reshape((T,len(v[0]))))  #only the first timestep
    
    return out_bc, out_num, out_geo, out_mix, out_t, out_lk


# =============================================================================
# the functions below can be changed to change the input
# =============================================================================

def set_par_num1(): 
    #set the numerical parameters. 
    N = 5 #number of Fourier modes in vertical. 
    theta = 0.5 #parameter for time integration. Standard  = 0.5 for Crank-Nicolson Scheme. Note that 0.5<theta<1.0
    Lsc = 1000 # horizontal length scale in m
    soc_sca = 30 #salinity scale in g/kg
    
    return N, theta, Lsc, soc_sca

def set_par_time1():
    #set the parameters for time 
    T = 131  #number of timesteps to run. Note that the first timestep is the intial state, so to run 20 timesteps you have to put 21 here. 
    dt = 24*3600 + np.zeros(T) # timestep, in seconds. Standard is 1 day = 24*3600 s. Timestep can vary in time
    
    return T, dt

def set_par_geo1():
    #set the geometry
    
    Ln = np.array([200,27000]) #length of the sections in m. Two are used for KGT
    b0 = 150 
    bn = np.array([b0,b0,b0]) #width in m at the edges of the sections. In between, the width varies exponentially
    dxn = np.array([20,250]) #horizontal grid size for sections, in m. 
    H0 = 13.5
    Hn = [H0,H0,H0] #depth of the edges of the sections in m. 13.5 for KGT. In between, the depth varies exponentially 
    
    return Ln, bn, dxn, Hn

def set_par_lock1():
    #parameters for the lock parametrisation as detailed in the article 
    Qch=20
    c1=0.035
    c2=0.042
    smooth=5
    
    return Qch, c1, c2, smooth
    

def set_par_phys1(nt, nx):
    #note that all of these can vary in space and time. Should work, but is not tested extensively. Be carefull with the edges of the domains. 
    Av = np.zeros((nt,nx)) + 6.75e-4 #vertical eddy viscosity m2/s, standard value from article
    Kv = np.zeros((nt,nx)) + 6.75e-4/2.2 #vertical eddy diffusivity m2/s, standard value from article
    Kh = np.zeros((nt,nx)) + 25 #horizontal eddy viscosity m2/s, standard value from article
    sf = np.zeros((nt,nx)) + 1e-4 #partial slip coefficient, standard value from article. Note that it should vary with depth

    return Av, Kv, Kh, sf

def set_par_bc1(nt,nx):
    #set the boundary conditions for the simulation
    sri = np.zeros(nt) + np.nan #salinity of the upstream boundary is assumed to be zero. Can easily be added
    Qi = np.linspace(25,25,nt) #discharge in m3/s as a function of time. 
    # Qi[10:] = 25
    
    #add the possibility of spatial variation in discharge, to include tributaries and onttrekkingen. If not present, just repeat Qi for every point of x 
    Q = np.zeros((nt,nx))
    for i in range(nt): Q[i] = Qi[i]  #no spatial variation in discharge

    soc = np.zeros(nt) + 18 #salinity outside the locks in g/kg
    soc[10:] = 28

    return soc, sri, Q
    
