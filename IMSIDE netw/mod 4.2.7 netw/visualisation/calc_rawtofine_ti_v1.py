import numpy as np
import matplotlib.pyplot as plt

def test_instability(dat,notes):
    # =============================================================================
    # instability test 
    # =============================================================================
    #assumes that dat goes with increasing z
    flag = 0
    dmax = []
    for i in range(1,len(dat)):
        if dat[i]>np.min(dat[:i]):
            if flag ==0: 
                dmin = np.min(dat[:i])
                #print('Unstable stratification!')
            flag=1
            dmax.append(dat[i])
    
    if flag == 0: 0#print('Stably stratified')
    
    else:
        dmax = np.max(dmax)
        if dmax-dmin > 1e-5:   
            print('There is unstable stratification in channel '+notes)
            print('The magnitude of the unstable stratification is ',dmax-dmin, 
                  ' normalized with the variation it is ', (dmax-dmin)/(np.max(dat)-np.min(dat)))


def calc_output(self):
    # =============================================================================
    #     function to convert output of model to other quantities
    # =============================================================================
    
    #starting indices of the channels in the big matrix, beetje omslachtig hier. 
    ind = [0]
    for key in self.ch_keys: ind.append(ind[-1]+self.ch_pars[key]['di'][-1]*self.M) 
    nnp = np.arange(1,self.M)*np.pi #n*pi

    #quantities f
    Qnow = self.Qdist_calc((self.Qriv, self.Qweir, self.Qhar, self.n_sea))
    #snow = (self.sri , self.swe , self.soc)

    #do the operations for every channel
    count = 1
    for key in self.ch_keys:
        # =============================================================================
        # some quanttities
        # =============================================================================
        self.ch_outp[key]['px'] = np.zeros(np.sum(self.ch_pars[key]['nxn']))
        self.ch_outp[key]['px'][0:self.ch_pars[key]['nxn'][0]] = -np.linspace(np.sum(self.ch_gegs[key]['L'][0:]), np.sum(self.ch_gegs[key]['L'][0+1:]), self.ch_pars[key]['nxn'][0])
        for i in range(1,len(self.ch_pars[key]['nxn'])): self.ch_outp[key]['px'][np.sum(self.ch_pars[key]['nxn'][:i]):np.sum(self.ch_pars[key]['nxn'][:i+1])] =\
            -np.linspace(np.sum(self.ch_gegs[key]['L'][i:]), np.sum(self.ch_gegs[key]['L'][i+1:]), self.ch_pars[key]['nxn'][i])
        tot_L = np.sum(self.ch_gegs[key]['L'])

        self.ch_outp[key]['sss'] = self.out[self.ch_inds[key]['totx'][0]*self.M:self.ch_inds[key]['totx'][-1]*self.M]
        # =============================================================================
        # first calcualte subtidal salinities
        # =============================================================================
        #salinity, if the salinity module did not run, this cannot be calculated
        #try:    self.ch_outp[key]['sss'] = self.sss_n[ind[count-1]:ind[count]]
        #except: self.ch_outp[key]['sss'] = np.zeros(ind[count]-ind[count-1])
        #self.ch_outp[key]['ss_st'] = self.ch_outp[key]['sss'].reshape((self.ch_pars[key]['di'][-1],self.M))
        
        self.ch_outp[key]['sb_st'] = self.ch_outp[key]['sss'][self.ch_inds[key]['isb']]*self.soc_sca
        self.ch_outp[key]['sn_st'] = self.ch_outp[key]['sss'][self.ch_inds[key]['isn']].reshape((self.ch_pars[key]['di'][-1],self.N))*self.soc_sca
        
        self.ch_outp[key]['sp_st'] = np.sum(self.ch_outp[key]['sn_st'][:,:,np.newaxis] * np.cos(nnp[:,np.newaxis]*self.z_nd),1)
        self.ch_outp[key]['s_st'] = self.ch_outp[key]['sp_st'] + self.ch_outp[key]['sb_st'][:,np.newaxis]
        
        #calculate derivative of depth-averaged subtidal salinity 
        self.ch_outp[key]['sb_st_x'] = np.zeros(self.ch_pars[key]['di'][-1]) + np.nan #
        for dom in range(len(self.ch_pars[key]['di'])-1):
            self.ch_outp[key]['sb_st_x'][self.ch_pars[key]['di'][dom]+1:self.ch_pars[key]['di'][dom+1]-1] = (self.ch_outp[key]['sb_st'][self.ch_pars[key]['di'][dom]+2:self.ch_pars[key]['di'][dom+1]]\
                 -self.ch_outp[key]['sb_st'][self.ch_pars[key]['di'][dom]:self.ch_pars[key]['di'][dom+1]-2])/(2*self.ch_pars[key]['dl'][self.ch_pars[key]['di'][dom]+1:self.ch_pars[key]['di'][dom+1]-1]*self.Lsc)   
            #also do the boundaries
            self.ch_outp[key]['sb_st_x'][self.ch_pars[key]['di'][dom]] = (-3*self.ch_outp[key]['sb_st'][self.ch_pars[key]['di'][dom]] +4*self.ch_outp[key]['sb_st'][self.ch_pars[key]['di'][dom]+1] -self.ch_outp[key]['sb_st'][self.ch_pars[key]['di'][dom]+2]) / (2*self.Lsc*self.ch_pars[key]['dl'][self.ch_pars[key]['di'][dom]])
            self.ch_outp[key]['sb_st_x'][self.ch_pars[key]['di'][dom+1]-1] =(3*self.ch_outp[key]['sb_st'][self.ch_pars[key]['di'][dom+1]-1] -4*self.ch_outp[key]['sb_st'][self.ch_pars[key]['di'][dom+1]-2] +self.ch_outp[key]['sb_st'][self.ch_pars[key]['di'][dom+1]-3]) / (2*self.Lsc*self.ch_pars[key]['dl'][self.ch_pars[key]['di'][dom+1]-1])
            
        # =============================================================================
        # do checks on unphysical behaviour
        # =============================================================================
        #negative salinity
        if np.min(self.ch_outp[key]['s_st']) < -1e-5: #actually salinity should not be smaller than the river slainity
            print('WARNING: negative salinity simulated: ', np.min(self.ch_outp[key]['s_st']) ,' in canal', key, '. ')
            import matplotlib.pyplot as plt
            plt.contourf(self.ch_outp[key]['s_st'].T,levels=np.linspace(0,35,11),cmap='RdBu_r')
            plt.colorbar()
            plt.title('Negative salinity! channel = '+str(key))
            plt.show()
            
        #inverse stratification
        #another check for inverse stratification
        #for i in range(self.ch_pars[key]['di'][-1]):
        #    test_instability(self.ch_outp[key]['s_st'][i],self.ch_gegs[key]['Name'])
            
            
        # =============================================================================
        # then calculate salinity in tidal cycle, include boundary layer correction
        # =============================================================================
        tid_out = self.tidal_salinity(key , self.ch_outp[key]['sss'])
        
        self.ch_outp[key]['s_ti'] = tid_out['st'].copy()
        self.ch_outp[key]['s_ti_cor'] = []
        self.ch_outp[key]['s_ti_cor_r'] = []
        #add boundary layer to tidal salinity
        for dom in range(self.ch_pars[key]['n_seg']):
            #self.ch_outp[key]['s_ti_cor'].append(tid_out['stci_x=-L'][dom])
            #self.ch_outp[key]['s_ti_cor_r'].append(np.real(tid_out['stci_x=-L'][dom][:,:,np.newaxis] * np.exp(1j*self.omega*ttide)))
            self.ch_outp[key]['s_ti'][self.ch_pars[key]['di'][dom] : self.ch_pars[key]['di'][dom] + len(tid_out['stci_x=-L'][dom])]     += tid_out['stci_x=-L'][dom]
            self.ch_outp[key]['s_ti'][self.ch_pars[key]['di'][dom+1] - len(tid_out['stci_x=0'][dom]) : self.ch_pars[key]['di'][dom+1] ] += tid_out['stci_x=0'][dom]

        ttide= np.linspace(0,2*np.pi/self.omega,self.nt)
        self.ch_outp[key]['s_ti_r'] = np.real(self.ch_outp[key]['s_ti'][:,:,np.newaxis] * np.exp(1j*self.omega*ttide))
        
        # =============================================================================
        # some other tidal quantities
        # =============================================================================
        self.ch_outp[key]['eta'] = self.ch_pars[key]['eta'][0,:,0]
        self.ch_outp[key]['ut'] = self.ch_pars[key]['ut'][0,:]
        self.ch_outp[key]['wt'] = self.ch_pars[key]['wt'][0,:]
        
        self.ch_outp[key]['eta_r'] = np.real(self.ch_pars[key]['eta'][0] * np.exp(1j*self.omega*ttide))
        self.ch_outp[key]['ut_r'] = np.real(self.ch_pars[key]['ut'][0,:,:,np.newaxis] * np.exp(1j*self.omega*ttide))
        self.ch_outp[key]['wt_r'] = np.real(self.ch_pars[key]['wt'][0,:,:,np.newaxis] * np.exp(1j*self.omega*ttide))
        
        # =============================================================================
        # subtidal transports and stuff
        # =============================================================================
        #calcualte subtidal velocities. 
        self.ch_outp[key]['u_st'] = Qnow[key] * self.ch_pars[key]['bH_1'][:,np.newaxis] * (self.ch_pars[key]['g1'] + 1 + self.ch_pars[key]['g2'] * self.z_nd**2) \
            + self.ch_pars[key]['alf'] * self.ch_outp[key]['sb_st_x'][:,np.newaxis] * (self.ch_pars[key]['g3'] + self.ch_pars[key]['g4'] * self.z_nd**2 + self.ch_pars[key]['g5'] * self.z_nd**3)
            
        #Calculate transports      
        self.ch_outp[key]['TQ'] = Qnow[key]*self.ch_outp[key]['sb_st'] #transport by mean current
        self.ch_outp[key]['TE'] = np.sum(self.ch_outp[key]['sn_st']*(2*Qnow[key]*self.ch_pars[key]['g2']*np.cos(nnp)/nnp**2 \
                              + self.ch_gegs[key]['H']*self.ch_pars[key]['b'][:,np.newaxis]*self.ch_pars[key]['alf']*self.ch_outp[key]['sb_st_x'][:,np.newaxis]*(2*self.ch_pars[key]['g4']\
                              * np.cos(nnp)/nnp**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp)-6)/nnp**4 - 3*np.cos(nnp)/nnp**2))),1) #transport by vertically sheared current
        self.ch_outp[key]['TD'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b']*- self.ch_pars[key]['Kh']*self.ch_outp[key]['sb_st_x'] #transport by horizontal diffusion
        self.ch_outp[key]['TT'] = 1/4 * np.real(np.mean(np.conj(self.ch_outp[key]['s_ti'])*self.ch_outp[key]['ut'] + self.ch_outp[key]['s_ti'] * np.conj(self.ch_outp[key]['ut']),1)) * self.ch_gegs[key]['H']*self.ch_pars[key]['b'] #transport by tides
                              
        # =============================================================================
        # throw away sea and river domains
        # =============================================================================
        #remove sea domain
        if self.ch_gegs[key]['loc x=0'][0] == 's':           
            self.ch_outp[key]['px'] = self.ch_outp[key]['px'][:-self.nx_sea]+self.length_sea
            self.ch_outp[key]['eta'] = self.ch_outp[key]['eta'][:-self.nx_sea]
            self.ch_outp[key]['s_st'] = self.ch_outp[key]['s_st'][:-self.nx_sea]
            #self.ch_outp[key]['ss_st'] = self.ch_outp[key]['ss_st'][:-self.nx_sea]
            self.ch_outp[key]['sb_st'] = self.ch_outp[key]['sb_st'][:-self.nx_sea] 
            self.ch_outp[key]['sn_st'] = self.ch_outp[key]['sn_st'][:-self.nx_sea]
            self.ch_outp[key]['u_st'] = self.ch_outp[key]['u_st'][:-self.nx_sea]
            self.ch_outp[key]['sb_st_x'] = self.ch_outp[key]['sb_st_x'][:-self.nx_sea]
            self.ch_outp[key]['eta_r'] = self.ch_outp[key]['eta_r'][:-self.nx_sea]
            self.ch_outp[key]['ut_r'] = self.ch_outp[key]['ut_r'][:-self.nx_sea]
            self.ch_outp[key]['wt_r'] = self.ch_outp[key]['wt_r'][:-self.nx_sea]
            self.ch_outp[key]['s_ti'] = self.ch_outp[key]['s_ti'][:-self.nx_sea]
            self.ch_outp[key]['s_ti_r'] = self.ch_outp[key]['s_ti_r'][:-self.nx_sea]

            self.ch_outp[key]['TQ'] = self.ch_outp[key]['TQ'][:-self.nx_sea]
            self.ch_outp[key]['TE'] = self.ch_outp[key]['TE'][:-self.nx_sea]
            self.ch_outp[key]['TD'] = self.ch_outp[key]['TD'][:-self.nx_sea]
            self.ch_outp[key]['TT'] = self.ch_outp[key]['TT'][:-self.nx_sea]

            tot_L = np.sum(self.ch_gegs[key]['L'][:-1])

        elif self.ch_gegs[key]['loc x=-L'][0] == 's':
            self.ch_outp[key]['px'] = self.ch_outp[key]['px'][self.nx_sea:]
            self.ch_outp[key]['eta'] = self.ch_outp[key]['eta'][self.nx_sea:]
            self.ch_outp[key]['s_st'] = self.ch_outp[key]['s_st'][self.nx_sea:]
            #self.ch_outp[key]['ss_st'] = self.ch_outp[key]['ss_st'][self.nx_sea:]
            self.ch_outp[key]['sb_st'] = self.ch_outp[key]['sb_st'][self.nx_sea:] 
            self.ch_outp[key]['sn_st'] = self.ch_outp[key]['sn_st'][self.nx_sea:]
            self.ch_outp[key]['u_st'] = self.ch_outp[key]['u_st'][self.nx_sea:]
            self.ch_outp[key]['sb_st_x'] = self.ch_outp[key]['sb_st_x'][self.nx_sea:]
            self.ch_outp[key]['eta_r'] = self.ch_outp[key]['eta_r'][self.nx_sea:]
            self.ch_outp[key]['ut_r'] = self.ch_outp[key]['ut_r'][self.nx_sea:]
            self.ch_outp[key]['wt_r'] = self.ch_outp[key]['wt_r'][self.nx_sea:]
            self.ch_outp[key]['s_ti'] = self.ch_outp[key]['s_ti'][self.nx_sea:]
            self.ch_outp[key]['s_ti_r'] = self.ch_outp[key]['s_ti_r'][self.nx_sea:]

            self.ch_outp[key]['TQ'] = self.ch_outp[key]['TQ'][self.nx_sea:]
            self.ch_outp[key]['TE'] = self.ch_outp[key]['TE'][self.nx_sea:]
            self.ch_outp[key]['TD'] = self.ch_outp[key]['TD'][self.nx_sea:]
            self.ch_outp[key]['TT'] = self.ch_outp[key]['TT'][self.nx_sea:]

            tot_L = np.sum(self.ch_gegs[key]['L'][1:])

                    
        #remove river domain
        if self.ch_gegs[key]['loc x=0'][0] == 'r':
            self.ch_outp[key]['px'] = self.ch_outp[key]['px'][:-self.nx_riv]+self.length_riv
            self.ch_outp[key]['eta'] = self.ch_outp[key]['eta'][:-self.nx_riv]
            self.ch_outp[key]['s_st'] = self.ch_outp[key]['s_st'][:-self.nx_riv]
            #self.ch_outp[key]['ss_st'] = self.ch_outp[key]['ss_st'][:-self.nx_riv]
            self.ch_outp[key]['sb_st'] = self.ch_outp[key]['sb_st'][:-self.nx_riv] 
            self.ch_outp[key]['sn_st'] = self.ch_outp[key]['sn_st'][:-self.nx_riv]
            self.ch_outp[key]['u_st'] = self.ch_outp[key]['u_st'][:-self.nx_riv]
            self.ch_outp[key]['sb_st_x'] = self.ch_outp[key]['sb_st_x'][:-self.nx_riv]
            self.ch_outp[key]['eta_r'] = self.ch_outp[key]['eta_r'][:-self.nx_riv]
            self.ch_outp[key]['ut_r'] = self.ch_outp[key]['ut_r'][:-self.nx_riv]
            self.ch_outp[key]['wt_r'] = self.ch_outp[key]['wt_r'][:-self.nx_riv]
            self.ch_outp[key]['s_ti'] = self.ch_outp[key]['s_ti'][:-self.nx_riv]
            self.ch_outp[key]['s_ti_r'] = self.ch_outp[key]['s_ti_r'][:-self.nx_riv]

            self.ch_outp[key]['TQ'] = self.ch_outp[key]['TQ'][:-self.nx_riv]
            self.ch_outp[key]['TE'] = self.ch_outp[key]['TE'][:-self.nx_riv]
            self.ch_outp[key]['TD'] = self.ch_outp[key]['TD'][:-self.nx_riv]
            self.ch_outp[key]['TT'] = self.ch_outp[key]['TT'][:-self.nx_riv]

            tot_L = np.sum(self.ch_gegs[key]['L'][:-1])

        elif self.ch_gegs[key]['loc x=-L'][0] == 'r':
            self.ch_outp[key]['px'] = self.ch_outp[key]['px'][self.nx_riv:]
            self.ch_outp[key]['eta'] = self.ch_outp[key]['eta'][self.nx_riv:]
            self.ch_outp[key]['s_st'] = self.ch_outp[key]['s_st'][self.nx_riv:]
            #self.ch_outp[key]['ss_st'] = self.ch_outp[key]['ss_st'][self.nx_riv:]
            self.ch_outp[key]['sb_st'] = self.ch_outp[key]['sb_st'][self.nx_riv:] 
            self.ch_outp[key]['sn_st'] = self.ch_outp[key]['sn_st'][self.nx_riv:]
            self.ch_outp[key]['u_st'] = self.ch_outp[key]['u_st'][self.nx_riv:]
            self.ch_outp[key]['sb_st_x'] = self.ch_outp[key]['sb_st_x'][self.nx_riv:]
            self.ch_outp[key]['eta_r'] = self.ch_outp[key]['eta_r'][self.nx_riv:]
            self.ch_outp[key]['ut_r'] = self.ch_outp[key]['ut_r'][self.nx_riv:]
            self.ch_outp[key]['wt_r'] = self.ch_outp[key]['wt_r'][self.nx_riv:]
            self.ch_outp[key]['s_ti'] = self.ch_outp[key]['s_ti'][self.nx_riv:]
            self.ch_outp[key]['s_ti_r'] = self.ch_outp[key]['s_ti_r'][self.nx_riv:]

            self.ch_outp[key]['TQ'] = self.ch_outp[key]['TQ'][self.nx_riv:]
            self.ch_outp[key]['TE'] = self.ch_outp[key]['TE'][self.nx_riv:]
            self.ch_outp[key]['TD'] = self.ch_outp[key]['TD'][self.nx_riv:]
            self.ch_outp[key]['TT'] = self.ch_outp[key]['TT'][self.nx_riv:]

            tot_L = np.sum(self.ch_gegs[key]['L'][1:])          
        
        # =============================================================================
        # prepare some plotting quantities, x and y coordinates in map plots
        # =============================================================================
        self.ch_outp[key]['plot d'] = np.zeros(len(self.ch_gegs[key]['plot x']))
        for i in range(1,len(self.ch_gegs[key]['plot x'])): self.ch_outp[key]['plot d'][i] = \
            self.ch_outp[key]['plot d'][i-1]+ ((self.ch_gegs[key]['plot x'][i]-self.ch_gegs[key]['plot x'][i-1])**2 + (self.ch_gegs[key]['plot y'][i]-self.ch_gegs[key]['plot y'][i-1])**2)**0.5
        self.ch_outp[key]['plot d'] = (self.ch_outp[key]['plot d']-self.ch_outp[key]['plot d'][-1])/self.ch_outp[key]['plot d'][-1]*tot_L
        self.ch_outp[key]['plot xs'] = np.interp(self.ch_outp[key]['px'],self.ch_outp[key]['plot d'],self.ch_gegs[key]['plot x'])
        self.ch_outp[key]['plot ys'] = np.interp(self.ch_outp[key]['px'],self.ch_outp[key]['plot d'],self.ch_gegs[key]['plot y'])
            
        self.ch_outp[key]['points'] = np.array([self.ch_outp[key]['plot xs'],self.ch_outp[key]['plot ys']]).T.reshape(-1, 1, 2)
        self.ch_outp[key]['segments'] = np.concatenate([self.ch_outp[key]['points'][:-1], self.ch_outp[key]['points'][1:]], axis=1)

        # =============================================================================
        # go to next channel
        # =============================================================================
        count+=1

#calc_output(delta)