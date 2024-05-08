# =============================================================================
# Code to calculate everything about the tides in a general channel network. 
# So water level, currents, salinity, contribution to subtidal balance, fluxes, boundary layer
# A bit slow, if we have a good idea how to solve that we should implement it. 
# There should be options to make this faster, because some calculations are repeated quite a few times. 
# =============================================================================
#libraries
import numpy as np
import matplotlib.pyplot as plt  
import time                              #measure time of operation

#the functions to calculate the tidal salinity
from physics.tide_funcs5 import sti_coefs

def tide_calc(self):
    # =============================================================================
    # This function calculates tidal properties, and, based on that, defines functions
    # to calculate tidal salinty and such, given a subtidal salinty
    # =============================================================================
    
    #add properties which will  be used for the tides
    for key in self.ch_keys:  
        # =============================================================================
        #   add physical quantities
        # =============================================================================
        #We have two possibilities for Av and bottom fricition:
        #Av depends on depth or does not depend on depth
        #we specify sf or we specify r 
        '''
        #calculate Av       
        if type(self.cv_t) == float or type(self.cv_t) == np.float64: self.ch_pars[key]['Av_t'] = self.cv_t*self.ch_gegs[key]['H']
        elif type(self.Av_t) == float or type(self.Av_t) == np.float64: self.ch_pars[key]['Av_t'] = self.Av_t
        else: print('ERROR, Av_t not well specified')        
        
        #Calcualte Kv
        self.ch_pars[key]['Kv_t'] = self.Kv_t #cv_t*ch_gegs[key]['Ut']*ch_gegs[key]['H']/Sc
        
        #Calculate r
        if type(self.sf_t) == float or type(self.sf_t) == np.float64: self.ch_pars[key]['r_t'] = (self.ch_pars[key]['Av_t']/(self.sf_t*self.ch_gegs[key]['H']))
        elif type(self.r_t) == float or type(self.r_t) == np.float64: self.ch_pars[key]['r_t'] = self.r_t
        else: print('ERROR, sf_t well specified')        
        '''
        self.ch_pars[key]['Av_ti'] = self.Av_ti
        self.ch_pars[key]['Kv_ti'] = self.Kv_ti
        self.ch_pars[key]['sf_ti'] = self.sf_ti
        
        if self.ch_pars[key]['sf_ti'] == 'rr1/2': self.ch_pars[key]['r_t'] = 1/2 + np.zeros(len(self.ch_gegs[key]['Hn']))#[np.newaxis,:,np.newaxis]
        else: self.ch_pars[key]['r_t'] = (self.ch_pars[key]['Av_ti'] / (self.ch_pars[key]['sf_ti'] * self.ch_gegs[key]['Hn']))
        
        #parameters for equations - see Wang et al. (2021) for an explanation what they are
        self.ch_pars[key]['deA'] = (1+1j)*self.ch_gegs[key]['Hn']/np.sqrt(2*self.ch_pars[key]['Av_ti']/self.omega)
        self.ch_pars[key]['deK'] = (1+1j)*self.ch_gegs[key]['Hn']/np.sqrt(2*self.ch_pars[key]['Kv_ti']/self.omega)
        
        self.ch_pars[key]['B'] = (np.cosh(self.ch_pars[key]['deA']) +  self.ch_pars[key]['r_t'] * self.ch_pars[key]['deA'] * np.sinh(self.ch_pars[key]['deA']))**-1
        self.ch_pars[key]['ka'] = np.sqrt(1/4*self.ch_pars[key]['bn']**-2 + self.omega**2/(self.g*self.ch_gegs[key]['Hn']) * (self.ch_pars[key]['B']/self.ch_pars[key]['deA']*np.sinh(self.ch_pars[key]['deA'])-1)**-1)
        
        #for different depth in different domains: extended versions of these parameters
        if self.ch_pars[key]['sf_ti'] == 'rr1/2': self.ch_pars[key]['r_t_dom'] = 1/2 + np.zeros(len(self.ch_pars[key]['H']))[np.newaxis,:,np.newaxis]
        else: self.ch_pars[key]['r_t_dom'] = (self.ch_pars[key]['Av_ti'] / (self.ch_pars[key]['sf_ti'] * self.ch_pars[key]['H']))[np.newaxis,:,np.newaxis]
        self.ch_pars[key]['deA_dom'] = ((1+1j)*self.ch_pars[key]['H']/np.sqrt(2*self.ch_pars[key]['Av_ti']/self.omega))[np.newaxis,:,np.newaxis]
        self.ch_pars[key]['B_dom'  ] = ((np.cosh(self.ch_pars[key]['deA_dom']) +  self.ch_pars[key]['r_t_dom'] * self.ch_pars[key]['deA_dom'] * np.sinh(self.ch_pars[key]['deA_dom']))**-1)

        
    # =============================================================================
    # build matrix equation to calculate tdal water levels in each channel
    # basically solving analytical system of equations Ax = b
    # equations look a bit weird because of the change from x to x' late in the process. 
    # but it should be correct 
    # =============================================================================

    # =============================================================================
    # internal parts of channels, solution vectors are empty here, but the matrix A has values, due to segments
    # =============================================================================
    ch_matr  = {} 
    for key in self.ch_keys:  
        #create empty vector
        if self.ch_gegs[key]['loc x=0'][0] == 's' or self.ch_gegs[key]['loc x=-L'][0] == 's': 
            ch_matr[key] = np.zeros((2*(len(self.ch_pars[key]['nxn']))+1,2*(len(self.ch_pars[key]['nxn']))+1),dtype=complex)
        else: 
            ch_matr[key] = np.zeros((2*(len(self.ch_pars[key]['nxn'])),2*(len(self.ch_pars[key]['nxn']))),dtype=complex)
            
            
        for dom in range(len(self.ch_pars[key]['nxn'])-1): #for every segment
            #water level equal
            ch_matr[key][dom*2+1, dom*2+0] = 1
            ch_matr[key][dom*2+1, dom*2+1] = 1
            ch_matr[key][dom*2+1, dom*2+2] = -np.exp(self.ch_gegs[key]['L'][dom+1]/(2*self.ch_pars[key]['bn'][dom+1])) * np.exp(self.ch_gegs[key]['L'][dom+1]*self.ch_pars[key]['ka'][dom+1])
            ch_matr[key][dom*2+1, dom*2+3] = -np.exp(self.ch_gegs[key]['L'][dom+1]/(2*self.ch_pars[key]['bn'][dom+1])) * np.exp(-self.ch_gegs[key]['L'][dom+1]*self.ch_pars[key]['ka'][dom+1])
            #discharge equal, i.e. water level gradient
            ch_matr[key][dom*2+2, dom*2+0] = -1/(2*self.ch_pars[key]['bn'][dom]) - self.ch_pars[key]['ka'][dom]
            ch_matr[key][dom*2+2, dom*2+1] = -1/(2*self.ch_pars[key]['bn'][dom]) + self.ch_pars[key]['ka'][dom]
            ch_matr[key][dom*2+2, dom*2+2] = -np.exp(self.ch_gegs[key]['L'][dom+1]/(2*self.ch_pars[key]['bn'][dom+1])) * np.exp( self.ch_gegs[key]['L'][dom+1]*self.ch_pars[key]['ka'][dom+1]) \
                * (-1/(2*self.ch_pars[key]['bn'][dom+1]) - self.ch_pars[key]['ka'][dom+1])
            ch_matr[key][dom*2+2, dom*2+3] = -np.exp(self.ch_gegs[key]['L'][dom+1]/(2*self.ch_pars[key]['bn'][dom+1])) * np.exp(-self.ch_gegs[key]['L'][dom+1]*self.ch_pars[key]['ka'][dom+1]) \
                * (-1/(2*self.ch_pars[key]['bn'][dom+1]) + self.ch_pars[key]['ka'][dom+1])
        
    #now build complete matrices
    sol_tot = np.zeros(np.sum([len(ch_matr[key]) for key in self.ch_keys]),dtype=complex)
    matr_tot = np.zeros((np.sum([len(ch_matr[key]) for key in self.ch_keys]),np.sum([len(ch_matr[key]) for key in self.ch_keys])),dtype=complex)
    
    ind = 0
    count = 0
    ind_jun = np.zeros((self.n_j,1),dtype=int)
    
    for key in self.ch_keys:
        #first add matrix for inner part
        matr_tot[ind:ind+len(ch_matr[key]),ind:ind+len(ch_matr[key])] = ch_matr[key]

        #river boundaries
        if self.ch_gegs[key]['loc x=-L'][0] == 'r' :
            matr_tot[ind,ind] =   np.exp(self.ch_gegs[key]['L'][0]/(2*self.ch_pars[key]['bn'][0])) * np.exp( self.ch_gegs[key]['L'][0]*self.ch_pars[key]['ka'][0]) * (-1/(2*self.ch_pars[key]['bn'][0]) - self.ch_pars[key]['ka'][0])
            matr_tot[ind,ind+1] = np.exp(self.ch_gegs[key]['L'][0]/(2*self.ch_pars[key]['bn'][0])) * np.exp(-self.ch_gegs[key]['L'][0]*self.ch_pars[key]['ka'][0]) * (-1/(2*self.ch_pars[key]['bn'][0]) + self.ch_pars[key]['ka'][0])        
        
        #weir and har boundaries
        if self.ch_gegs[key]['loc x=-L'][0] == 'w' or self.ch_gegs[key]['loc x=-L'][0] == 'h': 
            matr_tot[ind,ind] =   np.exp(self.ch_gegs[key]['L'][0]/(2*self.ch_pars[key]['bn'][0])) * np.exp( self.ch_gegs[key]['L'][0]*self.ch_pars[key]['ka'][0]) * (-1/(2*self.ch_pars[key]['bn'][0]) - self.ch_pars[key]['ka'][0])
            matr_tot[ind,ind+1] = np.exp(self.ch_gegs[key]['L'][0]/(2*self.ch_pars[key]['bn'][0])) * np.exp(-self.ch_gegs[key]['L'][0]*self.ch_pars[key]['ka'][0]) * (-1/(2*self.ch_pars[key]['bn'][0]) + self.ch_pars[key]['ka'][0])        
        
        #sea boundary
        if self.ch_gegs[key]['loc x=-L'][0] == 's': 
            print('SERIOUS WARNING: SEA AT X=0 DOES NOT WORK PROBABLY')
            matr_tot[ind,ind] = np.exp(self.ch_gegs[key]['L'][1]/(2*self.ch_pars[key]['bn'][1])) * np.exp( self.ch_gegs[key]['L'][1]*self.ch_pars[key]['ka'][1])
            matr_tot[ind,ind+1] = np.exp(self.ch_gegs[key]['L'][1]/(2*self.ch_pars[key]['bn'][1])) * np.exp(-self.ch_gegs[key]['L'][1]*self.ch_pars[key]['ka'][1]) 

            sol_tot[ind] = self.a_tide[count] * np.exp(-1j*self.p_tide[count])
            count+=1
        
        #update ind, and go to the x=0 boundary
        ind += len(ch_matr[key])
        
        #river
        if self.ch_gegs[key]['loc x=0'][0] == 'r' :
            matr_tot[ind-1,ind-2] = -1/(2*self.ch_pars[key]['bn'][-1]) - self.ch_pars[key]['ka'][-1]
            matr_tot[ind-1,ind-1] = -1/(2*self.ch_pars[key]['bn'][-1]) + self.ch_pars[key]['ka'][-1]
        
        #weir and har boundaries
        if self.ch_gegs[key]['loc x=0'][0] == 'w' or self.ch_gegs[key]['loc x=0'][0] == 'h': 
            matr_tot[ind-1,ind-2] = -1/(2*self.ch_pars[key]['bn'][-1]) - self.ch_pars[key]['ka'][-1]  
            matr_tot[ind-1,ind-1] = -1/(2*self.ch_pars[key]['bn'][-1]) + self.ch_pars[key]['ka'][-1]    
        
        #sea boundaries
        elif self.ch_gegs[key]['loc x=0'][0] == 's': 
            
            #first condition: at the sea boundary, the level is equal to a to be determined level
            matr_tot[ind-2,ind-3] = 1
            matr_tot[ind-2,ind-2] = 1
            matr_tot[ind-2,ind-1] = -1
            
            #second condition: the water level difference in the sea domain equals the difference between the prescribed sea level and the to be determined level
            matr_tot[ind-1,ind-3] = 1 - np.exp(self.ch_gegs[key]['L'][-1]/(2*self.ch_pars[key]['bn'][-1])) * np.exp(self.ch_pars[key]['ka'][-1]*self.ch_gegs[key]['L'][-1])
            matr_tot[ind-1,ind-2] = 1 - np.exp(self.ch_gegs[key]['L'][-1]/(2*self.ch_pars[key]['bn'][-1])) * np.exp(-self.ch_pars[key]['ka'][-1]*self.ch_gegs[key]['L'][-1])
            matr_tot[ind-1,ind-1] = -1 
            sol_tot[ind-1] = self.a_tide[count] * np.exp(-1j*self.p_tide[count]/180*np.pi)   
                        
            count+=1
    
    #finally add the junctions 
    #create a new index array
    ind = 0
    for key in self.ch_keys:
        ind2 = len(ch_matr[key])+ind
        self.ch_pars[key]['ind_wl'] = [ind,ind2-1]
        ind = ind2
           
    for j in range(self.n_j):
        #find the connections
        ju_geg = []
        for key in self.ch_keys: 
            if self.ch_gegs[key]['loc x=-L'] == 'j'+str(j+1): 
                ju_geg.append([key,'x=-L',self.ch_pars[key]['ind_wl'][0]])
            elif self.ch_gegs[key]['loc x=0'] == 'j'+str(j+1): 
                ju_geg.append([key,'x=0',self.ch_pars[key]['ind_wl'][1]])
        
        #three conditions: 
        #first:n1=n2
        if ju_geg[0][1] == 'x=-L':
            matr_tot[ju_geg[0][2],ju_geg[0][2]] = np.exp(self.ch_gegs[ju_geg[0][0]]['L'][0]/(2*self.ch_pars[ju_geg[0][0]]['bn'][0])) * np.exp( self.ch_gegs[ju_geg[0][0]]['L'][0]*self.ch_pars[ju_geg[0][0]]['ka'][0])
            matr_tot[ju_geg[0][2],ju_geg[0][2]+1] = np.exp(self.ch_gegs[ju_geg[0][0]]['L'][0]/(2*self.ch_pars[ju_geg[0][0]]['bn'][0])) * np.exp(-self.ch_gegs[ju_geg[0][0]]['L'][0]*self.ch_pars[ju_geg[0][0]]['ka'][0]) 
        elif ju_geg[0][1] == 'x=0':
            matr_tot[ju_geg[0][2],ju_geg[0][2]-1] = 1
            matr_tot[ju_geg[0][2],ju_geg[0][2]] = 1
            
        if ju_geg[1][1] == 'x=-L':
            matr_tot[ju_geg[0][2],ju_geg[1][2]] = -np.exp(self.ch_gegs[ju_geg[1][0]]['L'][0]/(2*self.ch_pars[ju_geg[1][0]]['bn'][0])) * np.exp( self.ch_gegs[ju_geg[1][0]]['L'][0]*self.ch_pars[ju_geg[1][0]]['ka'][0])
            matr_tot[ju_geg[0][2],ju_geg[1][2]+1] = -np.exp(self.ch_gegs[ju_geg[1][0]]['L'][0]/(2*self.ch_pars[ju_geg[1][0]]['bn'][0])) * np.exp(-self.ch_gegs[ju_geg[1][0]]['L'][0]*self.ch_pars[ju_geg[1][0]]['ka'][0]) 
        elif ju_geg[1][1] == 'x=0':
            matr_tot[ju_geg[0][2],ju_geg[1][2]-1] = -1
            matr_tot[ju_geg[0][2],ju_geg[1][2]] = -1
            
        
        #second: n2=n3
        if ju_geg[1][1] == 'x=-L':
            matr_tot[ju_geg[1][2],ju_geg[1][2]] = np.exp(self.ch_gegs[ju_geg[1][0]]['L'][0]/(2*self.ch_pars[ju_geg[1][0]]['bn'][0])) * np.exp( self.ch_gegs[ju_geg[1][0]]['L'][0]*self.ch_pars[ju_geg[1][0]]['ka'][0])
            matr_tot[ju_geg[1][2],ju_geg[1][2]+1] = np.exp(self.ch_gegs[ju_geg[1][0]]['L'][0]/(2*self.ch_pars[ju_geg[1][0]]['bn'][0])) * np.exp(-self.ch_gegs[ju_geg[1][0]]['L'][0]*self.ch_pars[ju_geg[1][0]]['ka'][0]) 
        elif ju_geg[1][1] == 'x=0':
            matr_tot[ju_geg[1][2],ju_geg[1][2]-1] = 1
            matr_tot[ju_geg[1][2],ju_geg[1][2]] = 1
            
        if ju_geg[2][1] == 'x=-L':
            matr_tot[ju_geg[1][2],ju_geg[2][2]] =  -np.exp(self.ch_gegs[ju_geg[2][0]]['L'][0]/(2*self.ch_pars[ju_geg[2][0]]['bn'][0])) * np.exp( self.ch_gegs[ju_geg[2][0]]['L'][0]*self.ch_pars[ju_geg[2][0]]['ka'][0])
            matr_tot[ju_geg[1][2],ju_geg[2][2]+1] = -np.exp(self.ch_gegs[ju_geg[2][0]]['L'][0]/(2*self.ch_pars[ju_geg[2][0]]['bn'][0])) * np.exp(-self.ch_gegs[ju_geg[2][0]]['L'][0]*self.ch_pars[ju_geg[2][0]]['ka'][0]) 
        elif ju_geg[2][1] == 'x=0':
            matr_tot[ju_geg[1][2],ju_geg[2][2]-1] = -1
            matr_tot[ju_geg[1][2],ju_geg[2][2]] = -1
            
        #third: sum Q = 0 - updated wrt previous version! now also valid for H not constant. 
        if ju_geg[0][1] == 'x=-L':
            matr_tot[ju_geg[2][2],ju_geg[0][2]] = self.ch_gegs[ju_geg[0][0]]['b'][0]*self.ch_gegs[ju_geg[0][0]]['Hn'][0]*(-self.ch_pars[ju_geg[0][0]]['ka'][0]-1/(2*self.ch_pars[ju_geg[0][0]]['bn'][0]))\
                * np.exp(self.ch_pars[ju_geg[0][0]]['ka'][0]*self.ch_gegs[ju_geg[0][0]]['L'][0])*np.exp(self.ch_gegs[ju_geg[0][0]]['L'][0]/(2*self.ch_pars[ju_geg[0][0]]['bn'][0])) \
                * (self.ch_pars[ju_geg[0][0]]['B'][0]/self.ch_pars[ju_geg[0][0]]['deA'][0] * np.sinh(self.ch_pars[ju_geg[0][0]]['deA'][0]) - 1)
            matr_tot[ju_geg[2][2],ju_geg[0][2]+1] = self.ch_gegs[ju_geg[0][0]]['b'][0]*self.ch_gegs[ju_geg[0][0]]['Hn'][0]*(self.ch_pars[ju_geg[0][0]]['ka'][0]-1/(2*self.ch_pars[ju_geg[0][0]]['bn'][0]))\
                * np.exp(-self.ch_pars[ju_geg[0][0]]['ka'][0]*self.ch_gegs[ju_geg[0][0]]['L'][0])*np.exp(self.ch_gegs[ju_geg[0][0]]['L'][0]/(2*self.ch_pars[ju_geg[0][0]]['bn'][0]))      \
                * (self.ch_pars[ju_geg[0][0]]['B'][0]/self.ch_pars[ju_geg[0][0]]['deA'][0] * np.sinh(self.ch_pars[ju_geg[0][0]]['deA'][0]) - 1)     
        elif ju_geg[0][1] == 'x=0':
            matr_tot[ju_geg[2][2],ju_geg[0][2]-1] =-self.ch_gegs[ju_geg[0][0]]['b'][-1]*self.ch_gegs[ju_geg[0][0]]['Hn'][-1]*(-self.ch_pars[ju_geg[0][0]]['ka'][-1]-1/(2*self.ch_pars[ju_geg[0][0]]['bn'][-1])) \
                * (self.ch_pars[ju_geg[0][0]]['B'][-1]/self.ch_pars[ju_geg[0][0]]['deA'][-1] * np.sinh(self.ch_pars[ju_geg[0][0]]['deA'][-1]) - 1)
            matr_tot[ju_geg[2][2],ju_geg[0][2]] = - self.ch_gegs[ju_geg[0][0]]['b'][-1]*self.ch_gegs[ju_geg[0][0]]['Hn'][-1]*(self.ch_pars[ju_geg[0][0]]['ka'][-1]-1/(2*self.ch_pars[ju_geg[0][0]]['bn'][-1]))  \
                * (self.ch_pars[ju_geg[0][0]]['B'][-1]/self.ch_pars[ju_geg[0][0]]['deA'][-1] * np.sinh(self.ch_pars[ju_geg[0][0]]['deA'][-1]) - 1)
        
        if ju_geg[1][1] == 'x=-L':
            matr_tot[ju_geg[2][2],ju_geg[1][2]] = self.ch_gegs[ju_geg[1][0]]['b'][0]*self.ch_gegs[ju_geg[1][0]]['Hn'][0]*(-self.ch_pars[ju_geg[1][0]]['ka'][0]-1/(2*self.ch_pars[ju_geg[1][0]]['bn'][0]))\
                * np.exp(self.ch_pars[ju_geg[1][0]]['ka'][0]*self.ch_gegs[ju_geg[1][0]]['L'][0])*np.exp(self.ch_gegs[ju_geg[1][0]]['L'][0]/(2*self.ch_pars[ju_geg[1][0]]['bn'][0])) \
                * (self.ch_pars[ju_geg[1][0]]['B'][0]/self.ch_pars[ju_geg[1][0]]['deA'][0] * np.sinh(self.ch_pars[ju_geg[1][0]]['deA'][0]) - 1)
            matr_tot[ju_geg[2][2],ju_geg[1][2]+1] = self.ch_gegs[ju_geg[1][0]]['b'][0]*self.ch_gegs[ju_geg[1][0]]['Hn'][0]*(self.ch_pars[ju_geg[1][0]]['ka'][0]-1/(2*self.ch_pars[ju_geg[1][0]]['bn'][0]))\
                * np.exp(-self.ch_pars[ju_geg[1][0]]['ka'][0]*self.ch_gegs[ju_geg[1][0]]['L'][0])*np.exp(self.ch_gegs[ju_geg[1][0]]['L'][0]/(2*self.ch_pars[ju_geg[1][0]]['bn'][0])) \
                * (self.ch_pars[ju_geg[1][0]]['B'][0]/self.ch_pars[ju_geg[1][0]]['deA'][0] * np.sinh(self.ch_pars[ju_geg[1][0]]['deA'][0]) - 1)  
        elif ju_geg[1][1] == 'x=0':
            matr_tot[ju_geg[2][2],ju_geg[1][2]-1] = -self.ch_gegs[ju_geg[1][0]]['b'][-1]*self.ch_gegs[ju_geg[1][0]]['Hn'][-1]*(-self.ch_pars[ju_geg[1][0]]['ka'][-1]-1/(2*self.ch_pars[ju_geg[1][0]]['bn'][-1])) \
                * (self.ch_pars[ju_geg[1][0]]['B'][-1]/self.ch_pars[ju_geg[1][0]]['deA'][-1] * np.sinh(self.ch_pars[ju_geg[1][0]]['deA'][-1]) - 1)
            matr_tot[ju_geg[2][2],ju_geg[1][2]] = - self.ch_gegs[ju_geg[1][0]]['b'][-1]*self.ch_gegs[ju_geg[1][0]]['Hn'][-1]*(self.ch_pars[ju_geg[1][0]]['ka'][-1]-1/(2*self.ch_pars[ju_geg[1][0]]['bn'][-1])) \
                * (self.ch_pars[ju_geg[1][0]]['B'][-1]/self.ch_pars[ju_geg[1][0]]['deA'][-1] * np.sinh(self.ch_pars[ju_geg[1][0]]['deA'][-1]) - 1)
            
        if ju_geg[2][1] == 'x=-L':
            matr_tot[ju_geg[2][2],ju_geg[2][2]] = self.ch_gegs[ju_geg[2][0]]['b'][0]*self.ch_gegs[ju_geg[2][0]]['Hn'][0]*(-self.ch_pars[ju_geg[2][0]]['ka'][0]-1/(2*self.ch_pars[ju_geg[2][0]]['bn'][0]))\
                * np.exp(self.ch_pars[ju_geg[2][0]]['ka'][0]*self.ch_gegs[ju_geg[2][0]]['L'][0])*np.exp(self.ch_gegs[ju_geg[2][0]]['L'][0]/(2*self.ch_pars[ju_geg[2][0]]['bn'][0])) \
                * (self.ch_pars[ju_geg[2][0]]['B'][0]/self.ch_pars[ju_geg[2][0]]['deA'][0] * np.sinh(self.ch_pars[ju_geg[2][0]]['deA'][0]) - 1)
            matr_tot[ju_geg[2][2],ju_geg[2][2]+1] = self.ch_gegs[ju_geg[2][0]]['b'][0]*self.ch_gegs[ju_geg[2][0]]['Hn'][0]*(self.ch_pars[ju_geg[2][0]]['ka'][0]-1/(2*self.ch_pars[ju_geg[2][0]]['bn'][0]))\
                * np.exp(-self.ch_pars[ju_geg[2][0]]['ka'][0]*self.ch_gegs[ju_geg[2][0]]['L'][0])*np.exp(self.ch_gegs[ju_geg[2][0]]['L'][0]/(2*self.ch_pars[ju_geg[2][0]]['bn'][0]))  \
                * (self.ch_pars[ju_geg[2][0]]['B'][0]/self.ch_pars[ju_geg[2][0]]['deA'][0] * np.sinh(self.ch_pars[ju_geg[2][0]]['deA'][0]) - 1)
        elif ju_geg[2][1] == 'x=0':
            matr_tot[ju_geg[2][2],ju_geg[2][2]-1] = -self.ch_gegs[ju_geg[2][0]]['b'][-1]*self.ch_gegs[ju_geg[2][0]]['Hn'][-1]*(-self.ch_pars[ju_geg[2][0]]['ka'][-1]-1/(2*self.ch_pars[ju_geg[2][0]]['bn'][-1]))\
                * (self.ch_pars[ju_geg[2][0]]['B'][-1]/self.ch_pars[ju_geg[2][0]]['deA'][-1] * np.sinh(self.ch_pars[ju_geg[2][0]]['deA'][-1]) - 1)
            matr_tot[ju_geg[2][2],ju_geg[2][2]] = - self.ch_gegs[ju_geg[2][0]]['b'][-1]*self.ch_gegs[ju_geg[2][0]]['Hn'][-1]*(self.ch_pars[ju_geg[2][0]]['ka'][-1]-1/(2*self.ch_pars[ju_geg[2][0]]['bn'][-1]))\
                * (self.ch_pars[ju_geg[2][0]]['B'][-1]/self.ch_pars[ju_geg[2][0]]['deA'][-1] * np.sinh(self.ch_pars[ju_geg[2][0]]['deA'][-1]) - 1)



    #solve the matrix equation
    oplossing = np.linalg.solve(matr_tot,sol_tot)


    # =============================================================================
    # Build eta from the raw output 
    # =============================================================================

    ind = 0
    for key in self.ch_keys:
        #prepare some indices
        di_here = self.ch_pars[key]['di']
        ind2 = len(ch_matr[key])+ind
        
        #deal with sea domain
        if self.ch_gegs[key]['loc x=0'][0] == 's' or self.ch_gegs[key]['loc x=-L'][0] == 's': 
            coef_eta= oplossing[ind:ind2-1].reshape((int(len(ch_matr[key])/2),2))
        else: 
            coef_eta= oplossing[ind:ind2].reshape((int(len(ch_matr[key])/2),2))

        ind =ind2 

        #save coefficients for single channel code
        self.ch_pars[key]['coef_eta'] = coef_eta

        #create empty vectors 
        eta = np.zeros(di_here[-1],dtype=complex)
        etar = np.zeros(di_here[-1],dtype=complex)
        detadx = np.zeros(di_here[-1],dtype=complex)
        detadx2 = np.zeros(di_here[-1],dtype=complex)
        detadx3 = np.zeros(di_here[-1],dtype=complex)
        
        #calculate eta and its derivatives. 
        for dom in range(len(self.ch_pars[key]['nxn'])):
            x_here = np.linspace(-self.ch_gegs[key]['L'][dom],0,self.ch_pars[key]['nxn'][dom])
            eta[di_here[dom]:di_here[dom+1]]  = np.exp(-x_here/(2*self.ch_pars[key]['bn'][dom])) * ( coef_eta[dom,0]*np.exp(-x_here*self.ch_pars[key]['ka'][dom]) + coef_eta[dom,1]*np.exp(x_here*self.ch_pars[key]['ka'][dom]) )
            etar[di_here[dom]:di_here[dom+1]] = np.exp(-x_here/(2*self.ch_pars[key]['bn'][dom])) * (-coef_eta[dom,0]*np.exp(-x_here*self.ch_pars[key]['ka'][dom]) + coef_eta[dom,1]*np.exp(x_here*self.ch_pars[key]['ka'][dom]) )
            
            detadx[di_here[dom]:di_here[dom+1]] = - eta[di_here[dom]:di_here[dom+1]]/(2*self.ch_pars[key]['bn'][dom]) \
                + self.ch_pars[key]['ka'][dom] * etar[di_here[dom]:di_here[dom+1]]
            
            detadx2[di_here[dom]:di_here[dom+1]] = eta[di_here[dom]:di_here[dom+1]]/(4*self.ch_pars[key]['bn'][dom]**2) \
                - self.ch_pars[key]['ka'][dom]/self.ch_pars[key]['bn'][dom] * etar[di_here[dom]:di_here[dom+1]] \
                + self.ch_pars[key]['ka'][dom]**2 * eta[di_here[dom]:di_here[dom+1]]
            
            detadx3[di_here[dom]:di_here[dom+1]] = etar[di_here[dom]:di_here[dom+1]]*self.ch_pars[key]['ka'][dom]**3 \
                                                    - eta[di_here[dom]:di_here[dom+1]]/(8*self.ch_pars[key]['bn'][dom]**3) \
                                                    + 3*etar[di_here[dom]:di_here[dom+1]] * self.ch_pars[key]['ka'][dom]/(4*self.ch_pars[key]['bn'][dom]**2) \
                                                    - 3*eta[di_here[dom]:di_here[dom+1]] * self.ch_pars[key]['ka'][dom]**2/(2*self.ch_pars[key]['bn'][dom])
                                                    

        #save in the right format
        self.ch_pars[key]['eta']     = eta[np.newaxis,:,np.newaxis]
        self.ch_pars[key]['detadx']  = detadx[np.newaxis,:,np.newaxis]
        self.ch_pars[key]['detadx2'] = detadx2[np.newaxis,:,np.newaxis]
        self.ch_pars[key]['detadx3'] = detadx3[np.newaxis,:,np.newaxis]
        
        #calculate also long-channel velocity
        self.ch_pars[key]['ut'] = self.g/(1j*self.omega) * self.ch_pars[key]['detadx']  * (self.ch_pars[key]['B_dom']*np.cosh(self.ch_pars[key]['deA_dom']*self.z_nd) - 1)
        self.ch_pars[key]['dutdx'] = self.g/(1j*self.omega) * self.ch_pars[key]['detadx2']  * (self.ch_pars[key]['B_dom']*np.cosh(self.ch_pars[key]['deA_dom']*self.z_nd) - 1)
        self.ch_pars[key]['wt'] = 1j*self.omega*self.ch_pars[key]['eta'] - self.g/(1j*self.omega) * (self.ch_pars[key]['detadx2'] + self.ch_pars[key]['detadx']/self.ch_pars[key]['bex'][np.newaxis,:,np.newaxis]) \
            * (self.ch_pars[key]['B_dom']*self.ch_pars[key]['H'][np.newaxis,:,np.newaxis]/self.ch_pars[key]['deA_dom']*np.sinh(self.ch_pars[key]['deA_dom']*self.z_nd) - self.z_nd*self.ch_pars[key]['H'][np.newaxis,:,np.newaxis])        
        
        self.ch_pars[key]['utb'] = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * (self.ch_pars[key]['B_dom']/self.ch_pars[key]['deA_dom']*np.sinh(self.ch_pars[key]['deA_dom'])-1))
        self.ch_pars[key]['utp'] = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * self.ch_pars[key]['B_dom'] * (np.cosh(self.ch_pars[key]['deA_dom']*self.z_nd) - np.sinh(self.ch_pars[key]['deA_dom'])/self.ch_pars[key]['deA_dom']))
            
        # =============================================================================
        # coefficients for tidal salinty - new in 4.2.7
        # =============================================================================
        c1 = - self.Kv_ti/(1j*self.omega)
        st_coef = sti_coefs(self.ch_pars[key]['zlist'],c1,(self.ch_pars[key]['deA_dom'],self.ch_pars[key]['H'][np.newaxis,:,np.newaxis],self.ch_pars[key]['B_dom'],self.mm[:,np.newaxis,np.newaxis] ))
        #st_coef = sti_coefs(self.ch_pars[key]['zlist'],c1,(self.ch_pars[key]['deA'][np.newaxis,:,np.newaxis],self.ch_gegs[key]['H'][np.newaxis,:,np.newaxis],self.ch_pars[key]['B'][np.newaxis,:,np.newaxis],self.mm[:,np.newaxis,np.newaxis] ))
        self.ch_pars[key]['c2c'], self.ch_pars[key]['c3c'], self.ch_pars[key]['c4c']  = st_coef[0] , st_coef[1] , st_coef[2] 
        self.ch_pars[key]['c2pc'], self.ch_pars[key]['c3pc'], self.ch_pars[key]['c4pc'] = st_coef[3] , st_coef[4] , st_coef[5] 
        self.ch_pars[key]['c2c_z'], self.ch_pars[key]['c3c_z'], self.ch_pars[key]['c4c_z']= st_coef[6] , st_coef[7] , st_coef[8] 

        # =============================================================================
        # boundary layer correction     
        # =============================================================================
        dstc_xL , dstc_xLi , dstc_x_xLi = [] , [] , []
        dstcp_x_xLi, dstc_z_xLi = [] , []
        
        dstc_x0, dstc_x0i , dstc_x_x0i = [] , [] , []
        dstcp_x_x0i, dstc_z_x0i = [] , []
        
        for dom in range(self.ch_pars[key]['n_seg']):
            
            # =============================================================================
            # first at x=-L
            # =============================================================================
            #calculate x 
            x_all = np.arange(di_here[dom+1]-di_here[dom])*self.ch_gegs[key]['dx'][dom] 
            x_bnd = x_all[np.where(x_all<(-np.sqrt(self.epsL)*np.log(self.tol)))[0]]
            if x_bnd[-1] >= self.ch_gegs[key]['L'][dom]: print('ERROR: boundary layer too large. Can be solved...')
                        
            #derivatives for jacobian
            dstc   = np.cos(np.pi*self.m0[:,np.newaxis,np.newaxis]*self.z_nd) * self.soc_sca
            dstcp  = np.zeros(dstc.shape)
            dstcp[1:]  = np.cos(np.pi*self.m0[1:,np.newaxis,np.newaxis]*self.z_nd) * self.soc_sca
            dstc_z = -self.m0[:,np.newaxis,np.newaxis]*np.pi/self.ch_gegs[key]['Hn'][dom] * np.sin(np.pi*self.m0[:,np.newaxis,np.newaxis]*self.z_nd) * self.soc_sca
                    
            dstci   = np.exp(-x_bnd[:,np.newaxis]/np.sqrt(self.epsL)) * dstc
            dstcpi  = np.exp(-x_bnd[:,np.newaxis]/np.sqrt(self.epsL)) * dstcp
            dstc_zi = np.exp(-x_bnd[:,np.newaxis]/np.sqrt(self.epsL)) * dstc_z

            dstc_xL.append(dstc)
            dstc_xLi.append(dstci)
            dstc_x_xLi.append(dstci * -1/np.sqrt(self.epsL))
            dstcp_x_xLi.append(dstcpi * -1/np.sqrt(self.epsL))
            dstc_z_xLi.append(dstc_zi)
            
            # =============================================================================
            # then at x=0        
            # =============================================================================
            #calculate x 
            x_all = np.arange(di_here[dom]-di_here[dom+1],1)*self.ch_gegs[key]['dx'][dom] 
            x_bnd = x_all[np.where(x_all>(np.sqrt(self.epsL)*np.log(self.tol)))[0]]
            if -x_bnd[0] > self.ch_gegs[key]['L'][dom]: print('ERROR: boundary layer too large. Can be solved...', key, -x_bnd[0] , self.ch_gegs[key]['L'][dom] )
           
            #derivatives for jacobian
            dstc = np.cos(np.pi*self.m0[:,np.newaxis,np.newaxis]*self.z_nd) * self.soc_sca
            dstcp  = np.zeros(dstc.shape)
            dstcp[1:]  = np.cos(np.pi*self.m0[1:,np.newaxis,np.newaxis]*self.z_nd) * self.soc_sca
            dstc_z = -self.m0[:,np.newaxis,np.newaxis]*np.pi/self.ch_gegs[key]['Hn'][dom] * np.sin(np.pi*self.m0[:,np.newaxis,np.newaxis]*self.z_nd) * self.soc_sca

            dstci   = np.exp(x_bnd[:,np.newaxis]/np.sqrt(self.epsL)) * dstc
            dstcpi  = np.exp(x_bnd[:,np.newaxis]/np.sqrt(self.epsL)) * dstcp
            dstc_zi = np.exp(x_bnd[:,np.newaxis]/np.sqrt(self.epsL)) * dstc_z
            
            dstc_x0.append(dstc)
            dstc_x0i.append(dstci)
            dstc_x_x0i.append(dstci * 1/np.sqrt(self.epsL))
            dstcp_x_x0i.append(dstcpi * 1/np.sqrt(self.epsL))
            dstc_z_x0i.append(dstc_zi)
        
        
        #save variables which I need - not in object, since they change during the calculation
        self.ch_pars[key]['dstc_x=-L'] = dstc_xL  ,
        self.ch_pars[key]['dstci_x=-L'] = dstc_xLi ,
        self.ch_pars[key]['dstci_x_x=-L'] = dstc_x_xLi ,
        
        self.ch_pars[key]['dstcpi_x_x=-L'] = dstcp_x_xLi ,
        self.ch_pars[key]['dstc_zi_x=-L'] = dstc_z_xLi ,
                     
        self.ch_pars[key]['dstc_x=0'] = dstc_x0  ,
        self.ch_pars[key]['dstci_x=0'] = dstc_x0i ,
        self.ch_pars[key]['dstci_x_x=0'] = dstc_x_x0i ,
                                
        self.ch_pars[key]['dstcpi_x_x=0'] = dstcp_x_x0i ,
        self.ch_pars[key]['dstc_zi_x=0'] = dstc_z_x0i ,
        
        
        
    return
        
    
            
