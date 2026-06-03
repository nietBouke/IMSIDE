# =============================================================================
# the definition of the class for IMSIDE in case of a closed canal
# the structure of this file is slightly incompatible with the versions for 
# networks and open canals. But the physics is the same.
# Of course no tides in the canal.  
# =============================================================================
#libraries
import numpy as np
import warnings 

class mod_SIcan:
    # =============================================================================
    # the IMSIDE model for closed canals with a lock
    # =============================================================================
    
    def __init__(self, inp_bc, inp_num, inp_geo, inp_mix, inp_t, inp_lk, par_seadom = (25000,10,250)):
        #meaning input:
            #inp_p: physical parameters: 
            #inp_n: domain paramters: 
            #inp_t: time parameters
            #init: initial guess for solutions, in the right format 
            
        #load the input into the variables of the object
        self.soc, self.sri, self.Q = inp_bc 
        self.N, self.theta, self.Lsc, self.soc_sca = inp_num
        self.Ln, self.bn, self.dxn, self.Hn = inp_geo
        self.Av,self.Kv,self.Kh, self.sf = inp_mix
        self.T, self.dt = inp_t 
        #parameters for the lock do not need to be decomposed here 
            
        #add sea domain
        self.Ln = np.append(self.Ln,par_seadom[0])
        self.bn = np.append(self.bn,self.bn[-1]*np.exp(par_seadom[1]))
        self.dxn = np.append(self.dxn,par_seadom[2])
        self.Hn = np.append(self.Hn,self.Hn[-1])
        #extend the x dependent parameters
        self.Q = np.concatenate([self.Q , np.zeros((self.T,int(par_seadom[0]/par_seadom[2]+1))) + self.Q[:,-1,np.newaxis]] , 1)        
        self.Av = np.concatenate([self.Av , np.zeros((self.T,int(par_seadom[0]/par_seadom[2]+1))) + self.Av[:,-1,np.newaxis]] , 1)        
        self.Kv = np.concatenate([self.Kv , np.zeros((self.T,int(par_seadom[0]/par_seadom[2]+1))) + self.Kv[:,-1,np.newaxis]] , 1)        
        self.sf = np.concatenate([self.sf , np.zeros((self.T,int(par_seadom[0]/par_seadom[2]+1))) + self.sf[:,-1,np.newaxis]] , 1)        
        #put in the Kh in the sea domain
        Kh_sea = self.Kh[:,-1,np.newaxis] * np.exp((par_seadom[0]/par_seadom[1])**(-1) * np.linspace(0,par_seadom[0],int(par_seadom[0]/par_seadom[2]+1)) )
        self.Kh = np.concatenate([self.Kh , Kh_sea] , 1)        

        
        # =============================================================================
        # convert some variables and some pre-computations
        # below: grid-related quantities
        # =============================================================================
        self.M=self.N+1 #for later use 
        self.nseg = len(self.Ln)
        
        #do a check
        if np.any(self.Ln%self.dxn!=0): raise Warning( 'WARNING: L/dx is not an integer')#check numerical parameters
        
        # variables for grid and indexing
        self.dln = self.dxn/self.Lsc
        self.nxn = np.array(self.Ln/self.dxn+1,dtype=int)
        self.di = np.zeros(len(self.Ln)+1,dtype=int) #starting indices of the domains
        for i in range(1,len(self.Ln)): self.di[i] = np.sum(self.nxn[:i])
        self.di[-1] = np.sum(self.nxn)
        
        #dx as funciton of x
        self.DX = np.zeros(self.di[-1])
        for i in range(len(self.nxn)): self.DX[self.di[i]:self.di[i+1]] = self.dxn[i]
        #normalized dx
        self.dl = np.zeros(np.sum(self.nxn))
        self.dl[0:self.nxn[0]] = self.dln[0]
        for i in range(1,len(self.nxn)): self.dl[np.sum(self.nxn[:i]):np.sum(self.nxn[:i+1])] = self.dln[i]
        
        #time step scaled
        # self.dt_sca = (self.dt*np.sum(self.Ln)) / (self.Lsc*self.Q[:,0]/(self.bn[0]*self.Hn[0])) 
        self.dt_sca = self.dt / self.Lsc
        
        #construct depth vector. 
        self.H = np.zeros((self.di[-1]))
        for i in range(self.nseg): self.H[self.di[i]:self.di[i+1]] = np.linspace(self.Hn[i],self.Hn[i+1],self.nxn[i])
        
        #construct width vector 
        #width convergence 
        self.wc = np.zeros(self.nseg)
        for i in range(self.nseg): self.wc[i] = np.inf if self.bn[i] == self.bn[i+1] else self.Ln[i]/np.log(self.bn[i+1]/self.bn[i])
        #build width
        self.b = np.zeros(np.sum(self.nxn))
        for i in range(len(self.nxn)): 
            self.b[np.sum(self.nxn[:i]):np.sum(self.nxn[:i+1])] = self.bn[i] * np.exp(self.wc[i]**(-1)*(np.linspace(-self.Ln[i],0,self.nxn[i])+self.Ln[i]))
            
            
        #gradients/derivatives of depth and width of the canal. 
        #derivatives of H
        self.dHdx= np.zeros(len(self.H))+np.nan
        self.dHdx[1:self.di[1]-1] =  (self.H[2:self.di[1]]-self.H[:self.di[1]-2]) / (2*self.DX[1:self.di[1]-1])
        for i in range(1,len(self.nxn)): self.dHdx[self.di[i]+1:self.di[i+1]-1] = \
            (self.H[self.di[i]+2:self.di[i+1]]-self.H[self.di[i]:self.di[i+1]-2])/(2*self.DX[self.di[i]+1:self.di[i+1]-1])
        #same for b
        self.dbdx= np.zeros(len(self.b))+np.nan
        self.dbdx[1:self.di[1]-1] =  (self.b[2:self.di[1]]-self.b[:self.di[1]-2])/(2*self.DX[1:self.di[1]-1])
        for i in range(1,len(self.nxn)): self.dbdx[self.di[i]+1:self.di[i+1]-1] = \
            (self.b[self.di[i]+2:self.di[i+1]]-self.b[self.di[i]:self.di[i+1]-2])/(2*self.DX[self.di[i]+1:self.di[i+1]-1])

        #plotting parameters
        self.nz=51 #vertical step - only for plot
        #horizontal space vector
        self.px = np.zeros(self.di[-1])
        for i in range(self.nseg): self.px[self.di[i]:self.di[i+1]] = - np.linspace(np.sum(self.Ln[i:]), np.sum(self.Ln[i+1:]), self.nxn[i])
        self.px = (self.px+par_seadom[0])/1000 
        #vertical space vector
        self.pz =np.zeros((self.di[-1],self.nz))
        for i in range(self.di[-1]): self.pz[i] = np.linspace(-self.H[i],0,self.nz)
        
        #time vector 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.tvec = np.arange(self.T) * self.dt   # suppress warning only for this line
        
        
        # =============================================================================
        # computation of the lock parametrisation
        # =============================================================================
        def co_slu_func(Q,Qch=20,c1=0.035,c2=0.042,smooth=5):
            return ((c1-c2)*np.tanh((Q-Qch)/smooth) + (c1+c2))/2
        self.co_slu = co_slu_func(self.Q[:,-1], *inp_lk)
        
        
    # =============================================================================
    # import functions from the other files
    # =============================================================================
    from eqs_IMSIDE_canal import build_indices, build_coeffs, build_sol, build_jac
    from solv_IMSIDE_canal import do_time_integration, run_func
    
    
    