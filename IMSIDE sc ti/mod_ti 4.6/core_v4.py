# =============================================================================
# Defition of the object, to run the model 
# =============================================================================
import numpy as np
import copy 

class mod1c_g4:
    def __init__(self, inp_constants, inp_phys, inp_geo, inp_forc):

        # =============================================================================
        # unzip input        
        # =============================================================================
        self.g , self.Be, self.Sc, self.cv, self.ch, self.CD, self.r = copy.deepcopy(inp_constants)
        
        self.N, self.Lsc, self.nz, self.nt = copy.deepcopy(inp_phys[0])
        self.Ut, self.Av_st, self.cv_st, self.Kv_st, self.Sc_st, self.sf_st, self.rr_st, self.Kh_st, self.ch_st = copy.deepcopy(inp_phys[1])
        self.Av_ti, self.cv_ti, self.Kv_ti, self.Sc_ti, self.sf_ti, self.rr_ti, self.Kh_ti = copy.deepcopy(inp_phys[2])
        self.choice_bottomslip_st, self.choice_bottomslip_ti, self.choice_viscosityv_st, self.choice_viscosityv_ti, self.choice_diffusivityv_st, \
            self.choice_diffusivityv_ti, self.choice_diffusivityh_st = copy.deepcopy(inp_phys[3])
        self.M = self.N+1
        self.Hn, self.Ln, self.bsn, self.dxn = copy.deepcopy(inp_geo)
         
        self.Q, self.soc, self.sri = copy.deepcopy(inp_forc[0])
        self.tid_comp, self.tid_per, self.a_tide, self.p_tide = copy.deepcopy(inp_forc[1])
         
        # =============================================================================
        # checks
        # =============================================================================
        
        if np.any(self.Ln%self.dxn!=0): print( 'WARNING: L/dx is not an integer')#check numerical parameters
        if len(self.Ln) != len(self.bsn)-1 or len(self.Ln)!=len(self.dxn) : print('ERROR: number of domains is not correct')


        # =============================================================================
        # calculate parameters related to geometry
        # =============================================================================
    
        self.dln  = self.dxn/self.Lsc
        self.nxn  = np.array(self.Ln/self.dxn+1,dtype=int)
        self.di   = np.zeros(len(self.Ln)+1,dtype=int) #starting indices of the domains
        self.ndom = len(self.Ln)
        for i in range(1,self.ndom): self.di[i] = np.sum(self.nxn[:i])
        self.di[-1] = np.sum(self.nxn)
        
        self.dl = np.zeros(np.sum(self.nxn))
        self.dl[0:self.nxn[0]] = self.dln[0]
        for i in range(1,self.ndom): self.dl[np.sum(self.nxn[:i]):np.sum(self.nxn[:i+1])] = self.dln[i]
        
        #build parameters: convergence length
        self.bn = np.zeros(self.ndom)
        for i in range(self.ndom): self.bn[i] = np.inf if self.bsn[i+1] == self.bsn[i] else self.Ln[i]/np.log(self.bsn[i+1]/self.bsn[i])
            
        
        #build width
        self.b, self.bex = np.zeros(self.di[-1]),np.zeros(self.di[-1])
        for i in range(self.ndom): self.bex[np.sum(self.nxn[:i]):np.sum(self.nxn[:i+1])] = [self.bn[i]]*self.nxn[i]
        for i in range(self.ndom): self.b[np.sum(self.nxn[:i]):np.sum(self.nxn[:i+1])] = self.bsn[i] * np.exp(self.bn[i]**(-1)*(np.linspace(-self.Ln[i],0,self.nxn[i])+self.Ln[i]))
            
        #build depth        
        self.H = np.zeros(self.di[-1])
        for i in range(self.ndom): self.H[self.di[i]:self.di[i+1]] = self.Hn[i]
        
        #plotting parameters
        self.px = np.zeros(np.sum(self.nxn))
        self.px[0:self.nxn[0]] = -np.linspace(np.sum(self.Ln[0:]), np.sum(self.Ln[0+1:]), self.nxn[0]) #here i can use the di list
        for i in range(1,self.ndom): self.px[np.sum(self.nxn[:i]):np.sum(self.nxn[:i+1])] = -np.linspace(np.sum(self.Ln[i:]), np.sum(self.Ln[i+1:]), self.nxn[i])
        self.px = self.px/1000
        self.pz = np.linspace(-self.H,0,self.nz).T
        
        #other
        self.z_nd = np.linspace(-1,0,self.nz)
        self.nn = np.arange(1,self.M)[:,np.newaxis,np.newaxis] 
        self.zlist = np.linspace(-self.H,0,self.nz).T[np.newaxis]
        self.nph = self.nn*np.pi/self.H[np.newaxis,:,np.newaxis] 
        self.A_lft = self.b[self.di[1:-1]] * self.H[self.di[1:-1]]
        self.A_rgt = self.b[self.di[1:-1]-1] * self.H[self.di[1:-1]-1]
        
        #for boundary layer correection
        self.z_inds = np.linspace(0,self.nz-1,self.M,dtype=int) # at which vertical levels we evaluate the expression
        self.tol = 0.01 #maybe intialise from somewhere else
        
    # =============================================================================
    # Add functions and dicitionaries from other files 
    # =============================================================================
        
    from physics.subtidal_v3 import subtidal_module , solu_subtidal, jaco_subtidal 
        
    from physics.indices_v1 import indices
        
    #from boundaries_v3 import solu_boundaries, jaco_boundaries#, calc_cor #, corsol , corjac, corsol_bnd , corjac_bnd, corsol_ins , corjac_ins
     
    from physics.bnd_subtidal_v4 import solu_bnd_subtidal, jaco_bnd_subtidal
    
    from physics.bnd_tidal_v4 import solu_bnd_tidal, jaco_bnd_tidal, calc_cor
    
    from physics.solve_v4 import NewtonRaphson , solve_eqs
    from physics.solve_v4_check import NewtonRaphson_check , check_eqs
     
    from physics.tidal_v5 import tidal_module , tidal_salinity , solu_tidal, jaco_tidal #, tidsol_inda , tidjac_inda , tidsol_indv , tidjac_indv , tidsol_bnd , tidjac_bnd ,
    
    from visualisation.plot_functions_v1 import plot_sst , prt_numbers , plot_transport, plot_next, terms_vert