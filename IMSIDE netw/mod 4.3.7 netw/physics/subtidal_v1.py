# =============================================================================
# module to solve the subtidal salinity balance in a general estuarine network
# model includes tides, with vertical advection, those are all taken into account 
# in the subtidal depth-averaged balance, but not in the the depth-perturbed balance
# at the junctions a boundary layer correction is applied, and in that manner salinity
# matches also at the tidal timescale. 
# for performance: we could try to remove the for loops. 
# =============================================================================

#import libraries
import numpy as np
import scipy as sp          #do fits
from scipy import optimize   , interpolate 
import time                              #measure time of operation
import matplotlib.pyplot as plt         


def subtidal_module(self):
    # =============================================================================
    # The function to calculate the salinity in the network.      
    # =============================================================================

    #create helpful dictionaries
    ch_parja = {} # parameters for solution vector/jacobian

    #lists of ks and ns and pis
    kkp = np.linspace(1,self.N,self.N)*np.pi #k*pi
    nnp = np.linspace(1,self.N,self.N)*np.pi #n*pi
    pkn = np.array([kkp]*self.N) + np.transpose([nnp]*self.N) #pi*(n+k)
    pnk = np.array([nnp]*self.N) - np.transpose([kkp]*self.N) #pi*(n-k)
    np.fill_diagonal(pkn,None),np.fill_diagonal(pnk,None)
     
    # =============================================================================
    # parameters for solution vector/jacobian
    # =============================================================================
    for key in self.ch_keys:
        ch_parja[key] = {}
        g1,g2,g3,g4,g5 = self.ch_pars[key]['g1'],self.ch_pars[key]['g2'],self.ch_pars[key]['g3'],self.ch_pars[key]['g4'],self.ch_pars[key]['g5']
        
        # =============================================================================
        # Vertical subtidal salt balance 
        # =============================================================================
        #term 1
        ch_parja[key]['C1a'] = self.ch_pars[key]['bH_1']/2
        
        #term 2        
        ch_parja[key]['C2a'] = self.ch_pars[key]['bH_1'][:,np.newaxis] * (g1[:,np.newaxis]/2 + g2[:,np.newaxis]/6 + g2[:,np.newaxis]/(4*kkp**2))
        ch_parja[key]['C2b'] = self.ch_pars[key]['alf'][:,np.newaxis]*self.soc_sca/self.Lsc * (g3[:,np.newaxis]/2 + g4[:,np.newaxis]*(1/6 + 1/(4*kkp**2)) -g5*(1/8 + 3/(8*kkp**2)) )
        ch_parja[key]['C2c'] = self.ch_pars[key]['bH_1'][:,np.newaxis,np.newaxis]*g2[:,np.newaxis,np.newaxis]* ( np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2 ) 
        ch_parja[key]['C2d'] = self.soc_sca/self.Lsc*self.ch_pars[key]['alf'][:,np.newaxis,np.newaxis] * (g4[:,np.newaxis,np.newaxis]*(np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2) \
                                                + g5* ((3*np.cos(pkn)-3)/pkn**4 + (3*np.cos(pnk)-3)/pnk**4 - 3/2*np.cos(pkn)/pkn**2 - 3/2*np.cos(pnk)/pnk**2) )
        ch_parja[key]['C2c'][np.where(np.isnan(ch_parja[key]['C2c']))] = 0 
        ch_parja[key]['C2d'][np.where(np.isnan(ch_parja[key]['C2d']))] = 0 
            
        #term 3
        ch_parja[key]['C3a'] = 2*self.ch_pars[key]['bH_1'][:,np.newaxis]*g2[:,np.newaxis]/(kkp**2)*np.cos(kkp) 
        ch_parja[key]['C3b'] = self.ch_pars[key]['alf'][:,np.newaxis]*self.soc_sca/self.Lsc * ( 2*g4[:,np.newaxis]/kkp**2 * np.cos(kkp) - g5/kkp**4 *(6-6*np.cos(kkp) +3*kkp**2*np.cos(kkp)) )
        
        #term 4 does not exist 
        
        #term 5
        ch_parja[key]['C5a'] = self.ch_pars[key]['alf'][:,np.newaxis]*self.soc_sca/self.Lsc *kkp* ( -9*g5+6*g4[:,np.newaxis]+kkp**2*(-12*g3[:,np.newaxis]-4*g4[:,np.newaxis]+3*g5) ) / (48*kkp**3)
        ch_parja[key]['C5b'] = self.ch_pars[key]['alf'][:,np.newaxis]*self.soc_sca*self.ch_pars[key]['bex'][:,np.newaxis]**(-1)*kkp * ( -9*g5+6*g4[:,np.newaxis]+kkp**2*(-12*g3[:,np.newaxis]-4*g4[:,np.newaxis]+3*g5) ) / (48*kkp**3)    
        ch_parja[key]['C5c'] = self.ch_pars[key]['alf'][:,np.newaxis,np.newaxis]*self.soc_sca/self.Lsc*nnp* ( (3*g5*np.cos(pkn)-3*g5)/pkn**5 + np.cos(pkn)/pkn**3 * (g4[:,np.newaxis,np.newaxis]-1.5*g5) + np.cos(pkn)/pkn * (g5/8 - g4[:,np.newaxis,np.newaxis]/6 - g3[:,np.newaxis,np.newaxis]/2) 
                                + (3*g5*np.cos(pnk)-3*g5)/pnk**5 + np.cos(pnk)/pnk**3 * (g4[:,np.newaxis,np.newaxis]-1.5*g5) + np.cos(pnk)/pnk * (g5/8 - g4[:,np.newaxis,np.newaxis]/6 - g3[:,np.newaxis,np.newaxis]/2))
        ch_parja[key]['C5d'] = self.ch_pars[key]['alf'][:,np.newaxis,np.newaxis]*self.soc_sca*nnp*self.ch_pars[key]['bex'][:,np.newaxis,np.newaxis]**(-1)*( (3*g5*np.cos(pkn)-3*g5)/pkn**5 + np.cos(pkn)/pkn**3 * (g4[:,np.newaxis,np.newaxis]-1.5*g5) 
                                    + np.cos(pkn)/pkn * (g5/8 - g4[:,np.newaxis,np.newaxis]/6 - g3[:,np.newaxis,np.newaxis]/2)  + (3*g5*np.cos(pnk)-3*g5)/pnk**5 + np.cos(pnk)/pnk**3 * (g4[:,np.newaxis,np.newaxis]-1.5*g5) + np.cos(pnk)/pnk * (g5/8 - g4[:,np.newaxis,np.newaxis]/6 - g3[:,np.newaxis,np.newaxis]/2))
        #no wind
        ch_parja[key]['C5e'] = np.zeros((np.sum(self.ch_pars[key]['nxn']),self.N))
        ch_parja[key]['C5f'] = np.zeros((np.sum(self.ch_pars[key]['nxn']),self.N,self.N))
        
        ch_parja[key]['C5c'][np.where(np.isnan(ch_parja[key]['C5c']))] = 0 
        ch_parja[key]['C5d'][np.where(np.isnan(ch_parja[key]['C5d']))] = 0 
        #ch_parja[key]['C5f'][np.where(np.isnan(ch_parja[key]['C5f']))] = 0  #no wind
        
        #term 6
        ch_parja[key]['C6a'] = self.Lsc*self.ch_pars[key]['Kv'][:,np.newaxis]*kkp**2/(2*self.ch_pars[key]['H'][:,np.newaxis]**2)        
        
        #term 7
        ch_parja[key]['C7a'] = -self.ch_pars[key]['bex']**-1 * self.ch_pars[key]['Kh'] / 2 - self.ch_pars[key]['Kh_x']/2
        ch_parja[key]['C7b'] = -self.ch_pars[key]['Kh']/(2*self.Lsc)
        

        #term 8 and 9 do not exist
        
        # =============================================================================
        # Horizontal subtidal salt balance
        # =============================================================================
        ch_parja[key]['C10a'] = (-self.ch_pars[key]['bex']**(-1)*self.ch_pars[key]['Kh'] - self.ch_pars[key]['Kh_x'])
        ch_parja[key]['C10b'] = -self.ch_pars[key]['Kh']/self.Lsc
        ch_parja[key]['C10c'] = self.ch_pars[key]['bex'][:,np.newaxis]**(-1)*self.ch_pars[key]['alf'][:,np.newaxis]*self.soc_sca * ( 2*g4[:,np.newaxis]/nnp**2*np.cos(nnp) - g5/nnp**4 * (6-6*np.cos(nnp)+3*nnp**2*np.cos(nnp)) )
        ch_parja[key]['C10d'] = self.ch_pars[key]['alf'][:,np.newaxis]*self.soc_sca/self.Lsc * ( 2*g4[:,np.newaxis]/nnp**2*np.cos(nnp) - g5/nnp**4 * (6-6*np.cos(nnp)+3*nnp**2*np.cos(nnp)) )
        ch_parja[key]['C10e'] = (2*self.ch_pars[key]['bH_1'][:,np.newaxis]*g2[:,np.newaxis]) / nnp**2 * np.cos(nnp) 
        ch_parja[key]['C10f'] = self.ch_pars[key]['alf'][:,np.newaxis]*self.soc_sca/self.Lsc * ( 2*g4[:,np.newaxis]/nnp**2*np.cos(nnp) - g5/nnp**4 * (6-6*np.cos(nnp)+3*nnp**2*np.cos(nnp)) )
        ch_parja[key]['C10g'] = np.zeros((np.sum(self.ch_pars[key]['nxn']),self.N)) #no wind
        ch_parja[key]['C10h'] = self.ch_pars[key]['bH_1']
        
        # =============================================================================
        # Boundaries: transport calculations
        # =============================================================================
        #vertical
        if self.ch_gegs[key]['loc x=-L'][0] == "j":  
            #calculate local (using average depth) parameters
            alfb = self.g*self.Be*self.junc_gegs[self.ch_gegs[key]['loc x=-L']]['Ha']/(48*self.ch_pars[key]['Av'][0])
            bHb = 1/(self.junc_gegs[self.ch_gegs[key]['loc x=-L']]['Ha']*self.ch_pars[key]['b'][0])
            
            ch_parja[key]['C12a_x=-L'] = self.ch_pars[key]['H'][0]*self.ch_pars[key]['b'][0]*-0.5*self.ch_pars[key]['Kh'][0]
            ch_parja[key]['C12b_x=-L'] = self.ch_pars[key]['H'][0]*self.ch_pars[key]['b'][0]*alfb * self.soc_sca * (1/2 * g3[0] + (1/6 + 1/(4*kkp**2)) * g4[0] + (-1/8 - 3/(8*kkp**2)) * g5 )
            ch_parja[key]['C12c_x=-L'] = self.ch_pars[key]['H'][0]*self.ch_pars[key]['b'][0]*alfb * self.soc_sca * ((np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2) * g4[0] \
                                                                    + ((3*np.cos(pkn)-3)/pkn**4 - 3/2*np.cos(pkn)/pkn**2 + (3*np.cos(pnk)-3)/pnk**4 - 3/2*np.cos(pnk)/pnk**2 ) * g5)
            ch_parja[key]['C12d_x=-L'] = self.ch_pars[key]['H'][0]*self.ch_pars[key]['b'][0]* self.Lsc * bHb * (1/2 + 1/2 * g1[0] + (1/6 + 1/(4*kkp**2)) * g2[0])
            ch_parja[key]['C12e_x=-L'] = self.ch_pars[key]['H'][0]*self.ch_pars[key]['b'][0]* self.Lsc * bHb * g2[0] * (np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2)

            ch_parja[key]['C12c_x=-L'][np.where(np.isnan(ch_parja[key]['C12c_x=-L']))] = 0 
            ch_parja[key]['C12e_x=-L'][np.where(np.isnan(ch_parja[key]['C12e_x=-L']))] = 0 
        
        if self.ch_gegs[key]['loc x=0'][0] == "j": 
            #calculate local (using average depth) parameters
            alfb = self.g*self.Be*self.junc_gegs[self.ch_gegs[key]['loc x=0']]['Ha']/(48*self.ch_pars[key]['Av'][-1])
            bHb = 1/(self.junc_gegs[self.ch_gegs[key]['loc x=0']]['Ha']*self.ch_pars[key]['b'][-1])
                
            ch_parja[key]['C12a_x=0'] = self.ch_pars[key]['H'][-1]*self.ch_pars[key]['b'][-1]*-0.5*self.ch_pars[key]['Kh'][-1]
            ch_parja[key]['C12b_x=0'] = self.ch_pars[key]['H'][-1]*self.ch_pars[key]['b'][-1]*alfb*self.soc_sca * (1/2 * g3[-1] + (1/6 + 1/(4*kkp**2)) * g4[-1] + (-1/8 - 3/(8*kkp**2)) * g5 )
            ch_parja[key]['C12c_x=0'] = self.ch_pars[key]['H'][-1]*self.ch_pars[key]['b'][-1]*alfb * self.soc_sca * ((np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2) * g4[-1] \
                                                                    + ((3*np.cos(pkn)-3)/pkn**4 - 3/2*np.cos(pkn)/pkn**2 + (3*np.cos(pnk)-3)/pnk**4 - 3/2*np.cos(pnk)/pnk**2 ) * g5)
            ch_parja[key]['C12d_x=0'] = self.ch_pars[key]['H'][-1]*self.ch_pars[key]['b'][-1]* self.Lsc * bHb * (1/2 + 1/2 * g1[-1] + (1/6 + 1/(4*kkp**2)) * g2[-1])
            ch_parja[key]['C12e_x=0'] = self.ch_pars[key]['H'][-1]*self.ch_pars[key]['b'][-1]* self.Lsc * bHb * g2[-1] * (np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2)

            ch_parja[key]['C12c_x=0'][np.where(np.isnan(ch_parja[key]['C12c_x=0']))]   = 0 
            ch_parja[key]['C12e_x=0'][np.where(np.isnan(ch_parja[key]['C12e_x=0']))]   = 0 
            
               
        # depth-averaged 
        ch_parja[key]['C13a_x=-L'] = self.ch_pars[key]['H'][0]*self.ch_pars[key]['b'][0]*self.ch_pars[key]['bH_1'][0]
        ch_parja[key]['C13b_x=-L'] = self.ch_pars[key]['H'][0]*self.ch_pars[key]['b'][0]*self.ch_pars[key]['bH_1'][0]*2*g2[0]*np.cos(nnp)/nnp**2
        ch_parja[key]['C13c_x=-L'] = self.ch_pars[key]['H'][0]*self.ch_pars[key]['b'][0]*self.ch_pars[key]['alf'][0]*self.soc_sca/self.Lsc * (2*g4[0]*np.cos(nnp)/nnp**2 + g5*(6*np.cos(nnp)-6)/nnp**4 -g5*3*np.cos(nnp)/nnp**2 )
        ch_parja[key]['C13d_x=-L'] = self.ch_pars[key]['H'][0]*self.ch_pars[key]['b'][0]*-self.ch_pars[key]['Kh'][0]/self.Lsc
    
        ch_parja[key]['C13a_x=0'] = self.ch_pars[key]['H'][-1]*self.ch_pars[key]['b'][-1]*self.ch_pars[key]['bH_1'][-1]
        ch_parja[key]['C13b_x=0'] = self.ch_pars[key]['H'][-1]*self.ch_pars[key]['b'][-1]*self.ch_pars[key]['bH_1'][-1]*2*g2[-1]*np.cos(nnp)/nnp**2
        ch_parja[key]['C13c_x=0'] = self.ch_pars[key]['H'][-1]*self.ch_pars[key]['b'][-1]*self.ch_pars[key]['alf'][-1]*self.soc_sca/self.Lsc * (2*g4[-1]*np.cos(nnp)/nnp**2 + g5*(6*np.cos(nnp)-6)/nnp**4 -g5*3*np.cos(nnp)/nnp**2 )
        ch_parja[key]['C13d_x=0'] = self.ch_pars[key]['H'][-1]*self.ch_pars[key]['b'][-1]*-self.ch_pars[key]['Kh'][-1]/self.Lsc
        
        # =============================================================================
        # Inner boundaries: transport calculations
        # TODO: if the formulations of Av and such change, this might not be correct anymore,. 
        # =============================================================================
        i0 = self.ch_pars[key]['di'][1:-1]
        i_1 = self.ch_pars[key]['di'][1:-1]-1

        Hbnd = np.array([(self.ch_gegs[key]['Hn'][i]+self.ch_gegs[key]['Hn'][i+1])/2 for i in range(self.ch_pars[key]['n_seg']-1)])
        alf0b = self.g*self.Be*Hbnd/(48*self.ch_pars[key]['Av'][i0])
        alf_1b = self.g*self.Be*Hbnd/(48*self.ch_pars[key]['Av'][i_1])
        bH0b = 1/(Hbnd*self.ch_pars[key]['b'][i0])
        bH_1b= 1/(Hbnd*self.ch_pars[key]['b'][i_1])
        
        ch_parja[key]['C14a_x=-L'] = self.ch_pars[key]['H'][i0]*self.ch_pars[key]['b'][i0]*-0.5*self.ch_pars[key]['Kh'][i0]
        ch_parja[key]['C14b_x=-L'] = self.ch_pars[key]['H'][i0,np.newaxis]*self.ch_pars[key]['b'][i0,np.newaxis]*alf0b[:,np.newaxis] * self.soc_sca * (1/2 * g3[i0,np.newaxis] + (1/6 + 1/(4*kkp**2)) * g4[i0,np.newaxis] + (-1/8 - 3/(8*kkp**2)) * g5 )
        ch_parja[key]['C14c_x=-L'] = self.ch_pars[key]['H'][i0,np.newaxis,np.newaxis]*self.ch_pars[key]['b'][i0,np.newaxis,np.newaxis] * alf0b[:,np.newaxis,np.newaxis] * self.soc_sca * ((np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2) * g4[i0,np.newaxis,np.newaxis] \
                                                                + ((3*np.cos(pkn)-3)/pkn**4 - 3/2*np.cos(pkn)/pkn**2 + (3*np.cos(pnk)-3)/pnk**4 - 3/2*np.cos(pnk)/pnk**2 ) * g5)
        ch_parja[key]['C14d_x=-L'] = self.ch_pars[key]['H'][i0,np.newaxis]*self.ch_pars[key]['b'][i0,np.newaxis]* self.Lsc * bH0b[:,np.newaxis] * (1/2 + 1/2 * g1[i0,np.newaxis] + (1/6 + 1/(4*kkp**2)) * g2[i0,np.newaxis])
        ch_parja[key]['C14e_x=-L'] = self.ch_pars[key]['H'][i0,np.newaxis,np.newaxis]*self.ch_pars[key]['b'][i0,np.newaxis,np.newaxis]* self.Lsc * bH0b[:,np.newaxis,np.newaxis] * g2[i0,np.newaxis,np.newaxis] * (np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2)

        ch_parja[key]['C14a_x=0'] = self.ch_pars[key]['H'][i_1]*self.ch_pars[key]['b'][i_1]*-0.5*self.ch_pars[key]['Kh'][i_1]
        ch_parja[key]['C14b_x=0'] = self.ch_pars[key]['H'][i_1,np.newaxis]*self.ch_pars[key]['b'][i_1,np.newaxis]*alf_1b[:,np.newaxis]*self.soc_sca * (1/2 * g3[i_1,np.newaxis] + (1/6 + 1/(4*kkp**2)) * g4[i_1,np.newaxis] + (-1/8 - 3/(8*kkp**2)) * g5 )
        ch_parja[key]['C14c_x=0'] = self.ch_pars[key]['H'][i_1,np.newaxis,np.newaxis]*self.ch_pars[key]['b'][i_1,np.newaxis,np.newaxis]*alf_1b[:,np.newaxis,np.newaxis] * self.soc_sca * ((np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2) * g4[i_1,np.newaxis,np.newaxis] \
                                                                + ((3*np.cos(pkn)-3)/pkn**4 - 3/2*np.cos(pkn)/pkn**2 + (3*np.cos(pnk)-3)/pnk**4 - 3/2*np.cos(pnk)/pnk**2 ) * g5)
        ch_parja[key]['C14d_x=0'] = self.ch_pars[key]['H'][i_1,np.newaxis]*self.ch_pars[key]['b'][i_1,np.newaxis]* self.Lsc * bH_1b[:,np.newaxis] * (1/2 + 1/2 * g1[i_1,np.newaxis] + (1/6 + 1/(4*kkp**2)) * g2[i_1,np.newaxis])
        ch_parja[key]['C14e_x=0'] = self.ch_pars[key]['H'][i_1,np.newaxis,np.newaxis]*self.ch_pars[key]['b'][i_1,np.newaxis,np.newaxis]* self.Lsc * bH_1b[:,np.newaxis,np.newaxis] * g2[i_1,np.newaxis,np.newaxis] * (np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2)

        ch_parja[key]['C14c_x=-L'][np.where(np.isnan(ch_parja[key]['C14c_x=-L']))] = 0 
        ch_parja[key]['C14c_x=0'][np.where(np.isnan(ch_parja[key]['C14c_x=0']))]   = 0 
        ch_parja[key]['C14e_x=-L'][np.where(np.isnan(ch_parja[key]['C14e_x=-L']))] = 0 
        ch_parja[key]['C14e_x=0'][np.where(np.isnan(ch_parja[key]['C14e_x=0']))]   = 0 
                
        ch_parja[key]['C14a_x=-L_rep'] = np.repeat(ch_parja[key]['C14a_x=-L'],self.N).reshape((len(ch_parja[key]['C14a_x=-L']),self.N))
        ch_parja[key]['C14a_x=0_rep'] = np.repeat(ch_parja[key]['C14a_x=0'],self.N).reshape((len(ch_parja[key]['C14a_x=0']),self.N))
                
        # depth-averaged 
        ch_parja[key]['C15a_x=-L'] = self.ch_pars[key]['H'][i0]*self.ch_pars[key]['b'][i0]*self.ch_pars[key]['bH_1'][i0]
        ch_parja[key]['C15b_x=-L'] = self.ch_pars[key]['H'][i0,np.newaxis]*self.ch_pars[key]['b'][i0,np.newaxis]*self.ch_pars[key]['bH_1'][i0,np.newaxis]*2*g2[i0,np.newaxis]*np.cos(nnp)/nnp**2
        ch_parja[key]['C15c_x=-L'] = self.ch_pars[key]['H'][i0,np.newaxis]*self.ch_pars[key]['b'][i0,np.newaxis]*self.ch_pars[key]['alf'][i0,np.newaxis]*self.soc_sca/self.Lsc * (2*g4[i0,np.newaxis]*np.cos(nnp)/nnp**2 + g5*(6*np.cos(nnp)-6)/nnp**4 -g5*3*np.cos(nnp)/nnp**2 )
        ch_parja[key]['C15d_x=-L'] = self.ch_pars[key]['H'][i0]*self.ch_pars[key]['b'][i0]*-self.ch_pars[key]['Kh'][i0]/self.Lsc
    
        ch_parja[key]['C15a_x=0'] = self.ch_pars[key]['H'][i_1]*self.ch_pars[key]['b'][i_1]*self.ch_pars[key]['bH_1'][i_1]
        ch_parja[key]['C15b_x=0'] = self.ch_pars[key]['H'][i_1,np.newaxis]*self.ch_pars[key]['b'][i_1,np.newaxis]*self.ch_pars[key]['bH_1'][i_1,np.newaxis]*2*g2[i_1,np.newaxis]*np.cos(nnp)/nnp**2
        ch_parja[key]['C15c_x=0'] = self.ch_pars[key]['H'][i_1,np.newaxis]*self.ch_pars[key]['b'][i_1,np.newaxis]*self.ch_pars[key]['alf'][i_1,np.newaxis]*self.soc_sca/self.Lsc * (2*g4[i_1,np.newaxis]*np.cos(nnp)/nnp**2 + g5*(6*np.cos(nnp)-6)/nnp**4 -g5*3*np.cos(nnp)/nnp**2 )
        ch_parja[key]['C15d_x=0'] = self.ch_pars[key]['H'][i_1]*self.ch_pars[key]['b'][i_1]*-self.ch_pars[key]['Kh'][i_1]/self.Lsc    
    
    self.ch_parja = ch_parja
    #return ch_parja
    

# =============================================================================
# define functions to solve in the Newton-Raphson algoritm
# =============================================================================
    
def sol_subtidal(self, key, ans, pars_Q):
    # =============================================================================
    # function to build the internal part of the solution vector 
    # =============================================================================
    
    #create empty vector
    so = np.zeros(self.ch_inds[key]['di3'][-1]*self.M)
    #local variables, for shorter notation
    inds = self.ch_inds[key].copy()
    dl = self.ch_pars[key]['dl'].copy()
    pars = self.ch_parja[key].copy()
          
    #some calculations to increase speed 
    sb_x = (ans[inds['xnrp_m']]-ans[inds['xnrm_m']])/(2*dl[inds['xr']])
    sb_xx= (ans[inds['xnrp_m']] - 2*ans[inds['xnr_m']] + ans[inds['xnrm_m']])/(dl[inds['xr']]**2) 

    sk_x = (ans[inds['xnrp_mj']]-ans[inds['xnrm_mj']])/(2*dl[inds['xr']])    
    sk_xx = (ans[inds['xnrp_mj']] - 2*ans[inds['xnr_mj']] + ans[inds['xnrm_mj']])/(dl[inds['xr']]**2) 

    # =============================================================================
    # subtidal part    
    # =============================================================================
    #vertical
    #term 1
    so[inds['xnr_mj']] += pars['C1a'][inds['xr']] * pars_Q[key] * sk_x

    #term 2
    so[inds['xnr_mj']] += (pars['C2a'][inds['xr'],inds['j1']]*pars_Q[key]*sk_x
                         + pars['C2b'][inds['xr'],inds['j1']] * sk_x*sb_x 
                         + np.sum(pars['C2c'][inds['xr'],inds['j1']] * (ans[inds['xnrp_mj4']]-ans[inds['xnrm_mj4']]) , 1)/(2*dl[inds['xr']]) * pars_Q[key]
                         + np.sum(pars['C2d'][inds['xr'],inds['j1']] * (ans[inds['xnrp_mj4']]-ans[inds['xnrm_mj4']]) , 1)/(2*dl[inds['xr']]) * sb_x
                         )
    
    #term 3
    so[inds['xnr_mj']] +=  pars['C3a'][inds['xr'],inds['j1']] * pars_Q[key] * sb_x + pars['C3b'][inds['xr'],inds['j1']] * sb_x**2
    
    #term 4 is absent
    
    #term 5
    so[inds['xnr_mj']] += (pars['C5a'][inds['xr'],inds['j1']]*sb_xx*ans[inds['xnr_mj']] 
                        + pars['C5b'][inds['xr'],inds['j1']]*sb_x * ans[inds['xnr_mj']]  
                        + np.sum(pars['C5c'][inds['xr'],inds['j1']]*ans[inds['xnr_mj4']] , 1) * sb_xx
                        + np.sum(pars['C5d'][inds['xr'],inds['j1']]*ans[inds['xnr_mj4']] , 1) * sb_x
                        + pars['C5e'][inds['xr'],inds['j1']]*ans[inds['xnr_mj']] + np.sum(pars['C5f'][inds['xr'],inds['j1']]*ans[inds['xnr_mj4']] , 1)  )
    
    #term 6
    so[inds['xnr_mj']] += pars['C6a'][inds['xr'],inds['j1']]*ans[inds['xnr_mj']]

    #term 7    
    so[inds['xnr_mj']] += pars['C7a'][inds['xr']]*sk_x + pars['C7b'][inds['xr']]*sk_xx 

    #depth - averaged
    so[inds['xn_m']] +=  (pars['C10a'][inds['x']]*(ans[inds['xnp_m']]-ans[inds['xnm_m']])/(2*dl[inds['x']]) 
                        + pars['C10b'][inds['x']]*(ans[inds['xnp_m']] - 2*ans[inds['xn_m']] + ans[inds['xnm_m']])/(dl[inds['x']]**2) 
                        + (ans[inds['xnp_m']]-ans[inds['xnm_m']])/(2*dl[inds['x']]) * np.sum(pars['C10c'][inds['x']]*ans[inds['xnr_mj3']],1) 
                        + (ans[inds['xnp_m']] - 2*ans[inds['xn_m']] + ans[inds['xnm_m']])/(dl[inds['x']]**2) * np.sum(pars['C10d'][inds['x']]*ans[inds['xnr_mj3']] , 1) 
                        + np.sum(pars['C10e'][inds['x']]*(ans[inds['xnrp_mj3']]-ans[inds['xnrm_mj3']]) , 1) / (2*dl[inds['x']]) * pars_Q[key] 
                        + (ans[inds['xnp_m']]-ans[inds['xnm_m']])/(2*dl[inds['x']]) * np.sum(pars['C10f'][inds['x']]*(ans[inds['xnrp_mj3']]-ans[inds['xnrm_mj3']]) , 1)/(2*dl[inds['x']])
                        + np.sum(pars['C10g'][inds['x']]*ans[inds['xnr_mj3']],1) + pars['C10h'][inds['x']]*pars_Q[key]*(ans[inds['xnp_m']]-ans[inds['xnm_m']])/(2*dl[inds['x']]) )

    return so
    
    
      
def jac_subtidal_fix(self, key, pars_Q):
    # =============================================================================
    # function to build the Jacobian for the internal part
    # =============================================================================
    
    #create empty matrix
    jac = np.zeros((self.ch_inds[key]['di3'][-1]*self.M,self.ch_inds[key]['di3'][-1]*self.M))

    #local variables, for short notation
    inds = self.ch_inds[key].copy()
    dl   = self.ch_pars[key]['dl'].copy()
    pars = self.ch_parja[key].copy()

    # =============================================================================
    # Subtidal part
    # =============================================================================
    #vertical
    
    #term 1
    jac[inds['xnr_mj'],inds['xnrm_mj']] += - pars['C1a'][inds['xr']]*pars_Q[key]/(2*dl[inds['xr']]) 
    jac[inds['xnr_mj'],inds['xnrp_mj']] += pars['C1a'][inds['xr']]*pars_Q[key]/(2*dl[inds['xr']]) 
    
    #term 2
    jac[inds['xnr_mj'],inds['xnrm_mj']] += - pars['C2a'][inds['xr'],inds['j1']]*pars_Q[key]/(2*dl[inds['xr']]) 
    jac[inds['xnr_mj'],inds['xnrp_mj']] += pars['C2a'][inds['xr'],inds['j1']]*pars_Q[key]/(2*dl[inds['xr']]) 
    jac[inds['xnr_mj2'],inds['xnrm_mk']] +=- pars['C2c'][inds['xr2'],inds['j12'],inds['k1']]*pars_Q[key]/(2*dl[inds['xr2']]) 
    jac[inds['xnr_mj2'],inds['xnrp_mk']] +=  pars['C2c'][inds['xr2'],inds['j12'],inds['k1']]*pars_Q[key]/(2*dl[inds['xr2']]) 
    
    #term 3
    jac[inds['xnr_mj'],inds['xnrm_m']] +=- pars['C3a'][inds['xr'],inds['j1']]*pars_Q[key]/(2*dl[inds['xr']]) 
    jac[inds['xnr_mj'],inds['xnrp_m']] +=  pars['C3a'][inds['xr'],inds['j1']]*pars_Q[key]/(2*dl[inds['xr']]) 
    
    #term 6
    jac[inds['xnr_mj'],inds['xnr_mj']] += pars['C6a'][inds['xr'],inds['j1']]

    #term 7 
    jac[inds['xnr_mj'],inds['xnrm_mj']] += -pars['C7a'][inds['xr']]/(2*dl[inds['xr']]) + pars['C7b'][inds['xr']] / (dl[inds['xr']]**2)
    jac[inds['xnr_mj'],inds['xnr_mj']]  +=  - 2*pars['C7b'][inds['xr']]/(dl[inds['xr']]**2)
    jac[inds['xnr_mj'],inds['xnrp_mj']] +=  pars['C7a'][inds['xr']]/(2*dl[inds['xr']]) + pars['C7b'][inds['xr']] / (dl[inds['xr']]**2)

    #depth-averaged salt
    jac[inds['xn_m'], inds['xnm_m']] += (  - pars['C10a'][inds['x']]/(2*dl[inds['x']]) + pars['C10b'][inds['x']]/(dl[inds['x']]**2) - pars['C10h'][inds['x']]*pars_Q[key]/(2*dl[inds['x']]) )
    jac[inds['xnr_m'], inds['xnrm_mj']]+=  - pars['C10e'][inds['xr'],inds['j1']]*pars_Q[key]/(2*dl[inds['xr']])
    jac[inds['xn_m'],inds['xn_m']]     +=  - 2/(dl[inds['x']]**2)*pars['C10b'][inds['x']]  
    jac[inds['xnr_m'], inds['xnr_mj']] +=  pars['C10g'][inds['xr'],inds['j1']] 
    jac[inds['xn_m'], inds['xnp_m']] += ( pars['C10a'][inds['x']]/(2*dl[inds['x']]) + pars['C10b'][inds['x']]/(dl[inds['x']]**2) + pars['C10h'][inds['x']]*pars_Q[key]/(2*dl[inds['x']]) )
    jac[inds['xnr_m'], inds['xnrp_mj']] += pars['C10e'][inds['xr'],inds['j1']]*pars_Q[key]/(2*dl[inds['xr']])
   
    return jac




def jac_subtidal_vary(self, key, ans, pars_Q):
    # =============================================================================
    # function to build the Jacobian for the internal part
    # =============================================================================
    
    #create empty matrix
    jac = np.zeros((self.ch_inds[key]['di3'][-1]*self.M,self.ch_inds[key]['di3'][-1]*self.M))

    #local variables, for short notation
    inds = self.ch_inds[key].copy()
    dl   = self.ch_pars[key]['dl'].copy()
    pars = self.ch_parja[key].copy()

    # =============================================================================
    # Subtidal part
    # =============================================================================
    #vertical
    
    #term 2
    jac[inds['xnr_mj'],inds['xnrm_mj']] += - pars['C2b'][inds['xr'],inds['j1']]/(2*dl[inds['xr']]) * (ans[inds['xnrp_m']]-ans[inds['xnrm_m']])/(2*dl[inds['xr']]) 
    jac[inds['xnr_mj'],inds['xnrp_mj']] += pars['C2b'][inds['xr'],inds['j1']]/(2*dl[inds['xr']]) * (ans[inds['xnrp_m']]-ans[inds['xnrm_m']])/(2*dl[inds['xr']]) 
    jac[inds['xnr_mj'],inds['xnrm_m']] += ( - pars['C2b'][inds['xr'],inds['j1']]/(2*dl[inds['xr']]) * (ans[inds['xnrp_mj']]-ans[inds['xnrm_mj']])/(2*dl[inds['xr']])
                          - np.sum(pars['C2d'][inds['xr'],inds['j1']] * (ans[inds['xnrp_mj4']]-ans[inds['xnrm_mj4']]) , 1)/(4*dl[inds['xr']]**2))
    jac[inds['xnr_mj'],inds['xnrp_m']] += ( pars['C2b'][inds['xr'],inds['j1']]/(2*dl[inds['xr']]) * (ans[inds['xnrp_mj']]-ans[inds['xnrm_mj']])/(2*dl[inds['xr']]) 
                          + np.sum(pars['C2d'][inds['xr'],inds['j1']] * (ans[inds['xnrp_mj4']]-ans[inds['xnrm_mj4']]) , 1)/(4*dl[inds['xr']]**2))
    jac[inds['xnr_mj2'],inds['xnrm_mk']] +=- pars['C2d'][inds['xr2'],inds['j12'],inds['k1']]*(ans[inds['xnrp_m2']]-ans[inds['xnrm_m2']])/(4*dl[inds['xr2']]**2)
    jac[inds['xnr_mj2'],inds['xnrp_mk']] +=  pars['C2d'][inds['xr2'],inds['j12'],inds['k1']]*(ans[inds['xnrp_m2']]-ans[inds['xnrm_m2']])/(4*dl[inds['xr2']]**2)
    
    
    #term 3
    jac[inds['xnr_mj'],inds['xnrm_m']] += - pars['C3b'][inds['xr'],inds['j1']]/dl[inds['xr']] * (ans[inds['xnrp_m']]-ans[inds['xnrm_m']])/(2*dl[inds['xr']])
    jac[inds['xnr_mj'],inds['xnrp_m']] +=   pars['C3b'][inds['xr'],inds['j1']]/dl[inds['xr']] * (ans[inds['xnrp_m']]-ans[inds['xnrm_m']])/(2*dl[inds['xr']])

    #term 4 does not exist
    #term 5
    jac[inds['xnr_mj'], inds['xnrm_m']] += ( pars['C5a'][inds['xr'],inds['j1']]*ans[inds['xnr_mj']]/(dl[inds['xr']]**2) - pars['C5b'][inds['xr'],inds['j1']]/(2*dl[inds['xr']])*ans[inds['xnr_mj']]
                                   + np.sum(pars['C5c'][inds['xr'],inds['j1']]*ans[inds['xnr_mj4']] , 1) /(dl[inds['xr']]**2)
                                   - np.sum(pars['C5d'][inds['xr'],inds['j1']]*ans[inds['xnr_mj4']] , 1)/(2*dl[inds['xr']]) )
    jac[inds['xnr_mj'], inds['xnr_m']]  += - 2*pars['C5a'][inds['xr'],inds['j1']]*ans[inds['xnr_mj']]/(dl[inds['xr']]**2) -2* np.sum(pars['C5c'][inds['xr'],inds['j1']]*ans[inds['xnr_mj4']] , 1)/(dl[inds['xr']]**2)
    jac[inds['xnr_mj'], inds['xnrp_m']] += (pars['C5a'][inds['xr'],inds['j1']]*ans[inds['xnr_mj']]/(dl[inds['xr']]**2) + pars['C5b'][inds['xr'],inds['j1']]/(2*dl[inds['xr']])*ans[inds['xnr_mj']]
                                   + np.sum(pars['C5c'][inds['xr'],inds['j1']]*ans[inds['xnr_mj4']] , 1)/(dl[inds['xr']]**2)
                                   + np.sum(pars['C5d'][inds['xr'],inds['j1']]*ans[inds['xnr_mj4']] , 1)/(2*dl[inds['xr']]) )
    jac[inds['xnr_mj'],inds['xnr_mj']] += ( pars['C5a'][inds['xr'],inds['j1']]*(ans[inds['xnrp_m']]-2*ans[inds['xnr_m']]+ans[inds['xnrm_m']])/(dl[inds['xr']]**2) 
                                     + pars['C5b'][inds['xr'],inds['j1']]*(ans[inds['xnrp_m']]-ans[inds['xnrm_m']])/(2*dl[inds['xr']]) + pars['C5e'][inds['xr'],inds['j1']] )
    jac[inds['xnr_mj2'], inds['xnr_mk']] += ( pars['C5c'][inds['xr2'],inds['j12'],inds['k1']]*(ans[inds['xnrp_m2']]-2*ans[inds['xnr_m2']]+ans[inds['xnrm_m2']])/(dl[inds['xr2']]**2)
                                         + pars['C5d'][inds['xr2'],inds['j12'],inds['k1']]*(ans[inds['xnrp_m2']]-ans[inds['xnrm_m2']])/(2*dl[inds['xr2']]) +pars['C5f'][inds['xr2'],inds['j12'],inds['k1']] )


    #depth-averaged salt
    jac[inds['xn_m'], inds['xnm_m']] += ( - 1/(2*dl[inds['x']])*np.sum(pars['C10c'][inds['x']]*ans[inds['xnr_mj3']],1) 
                                   + 1/(dl[inds['x']]**2)*np.sum(pars['C10d'][inds['x']]*ans[inds['xnr_mj3']] , 1) 
                                   - 1/(2*dl[inds['x']]) * np.sum(pars['C10f'][inds['x']]*(ans[inds['xnrp_mj3']]-ans[inds['xnrm_mj3']]) , 1)/(2*dl[inds['x']]) )
    jac[inds['xnr_m'], inds['xnrm_mj']]+= - pars['C10f'][inds['xr'],inds['j1']]/(2*dl[inds['xr']])*(ans[inds['xnrp_m']]-ans[inds['xnrm_m']])/(2*dl[inds['xr']])
    jac[inds['xn_m'],inds['xn_m']]     += -2/(dl[inds['x']]**2)*np.sum(pars['C10d'][inds['x']]*ans[inds['xnr_mj3']] , 1)  
    jac[inds['xnr_m'], inds['xnr_mj']] += ((ans[inds['xnrp_m']]-ans[inds['xnrm_m']])/(2*dl[inds['xr']]) * pars['C10c'][inds['xr'],inds['j1']] 
                                        + (ans[inds['xnrp_m']]-2*ans[inds['xnr_m']]+ans[inds['xnrm_m']])/(dl[inds['xr']]**2) * pars['C10d'][inds['xr'],inds['j1']]  )
    jac[inds['xn_m'], inds['xnp_m']] += ( 1/(2*dl[inds['x']])*np.sum(pars['C10c'][inds['x']]*ans[inds['xnr_mj3']],1) 
                                   + 1/(dl[inds['x']]**2)*np.sum(pars['C10d'][inds['x']]*ans[inds['xnr_mj3']] , 1) 
                                   + 1/(2*dl[inds['x']]) * np.sum(pars['C10f'][inds['x']]*(ans[inds['xnrp_mj3']]-ans[inds['xnrm_mj3']]) , 1)/(2*dl[inds['x']]) )
    jac[inds['xnr_m'], inds['xnrp_mj']] +=  pars['C10f'][inds['xr'],inds['j1']]/(2*dl[inds['xr']])*(ans[inds['xnrp_m']]-ans[inds['xnrm_m']])/(2*dl[inds['xr']])

   
    return jac









