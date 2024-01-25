# =============================================================================
# this file converts the 'raw' output of the model to physical quantities like salinity
# for the output of the time-dependent model
# we keep it here relatively simple, so only salinity is calculated
# TODO: build in checks that the output is realistic. 
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt


def calc_output(self):
    
    self.Tvec = np.zeros(self.T)
    for t in range(self.T-1): self.Tvec[t+1] = np.sum(self.DT[:t+1])/(3600*24)
    
    #do the operations for every channel
    for key in self.ch_keys:
        # =============================================================================
        # x coordinates for ploting
        # =============================================================================
        self.ch_outp[key]['px'] = np.zeros(np.sum(self.ch_pars[key]['nxn']))
        self.ch_outp[key]['px'][0:self.ch_pars[key]['nxn'][0]] = -np.linspace(np.sum(self.ch_gegs[key]['L'][0:]), np.sum(self.ch_gegs[key]['L'][0+1:]), self.ch_pars[key]['nxn'][0])
        for i in range(1,len(self.ch_pars[key]['nxn'])): self.ch_outp[key]['px'][np.sum(self.ch_pars[key]['nxn'][:i]):np.sum(self.ch_pars[key]['nxn'][:i+1])] =\
            -np.linspace(np.sum(self.ch_gegs[key]['L'][i:]), np.sum(self.ch_gegs[key]['L'][i+1:]), self.ch_pars[key]['nxn'][i])
        tot_L = np.sum(self.ch_gegs[key]['L'])

        # =============================================================================
        # subtidal salinity      
        # =============================================================================
        self.ch_outp[key]['sss'] = self.out[:,self.ch_inds[key]['totx'][0]*self.M:self.ch_inds[key]['totx'][-1]*self.M]
        self.ch_outp[key]['sb_st'] = self.ch_outp[key]['sss'][:,self.ch_inds[key]['isb']] * self.soc_sca
        self.ch_outp[key]['sn_st'] = self.ch_outp[key]['sss'][:,self.ch_inds[key]['isn']].reshape((self.T,self.ch_pars[key]['di'][-1],self.N)) * self.soc_sca
        self.ch_outp[key]['sp_st'] = np.sum(self.ch_outp[key]['sn_st'][:,:,:,np.newaxis] * np.cos(np.pi*self.mm[:,np.newaxis]*self.z_nd),2)
        #subtidal salinity
        self.ch_outp[key]['s_st'] = self.ch_outp[key]['sp_st'] + self.ch_outp[key]['sb_st'][:,:,np.newaxis]

        
        # =============================================================================
        # remove sea and river domain
        # =============================================================================
        #remove sea domain
        if self.ch_gegs[key]['loc x=0'][0] == 's':           
            self.ch_outp[key]['px'] = self.ch_outp[key]['px'][:-self.nx_sea]+self.length_sea
            #self.ch_outp[key]['eta'] = self.ch_outp[key]['eta'][:-self.nx_sea]
            self.ch_outp[key]['s_st'] = self.ch_outp[key]['s_st'][:,:-self.nx_sea]
            #self.ch_outp[key]['ss_st'] = self.ch_outp[key]['ss_st'][:-self.nx_sea]
            self.ch_outp[key]['sb_st'] = self.ch_outp[key]['sb_st'][:,:-self.nx_sea] 
            self.ch_outp[key]['sn_st'] = self.ch_outp[key]['sn_st'][:,:-self.nx_sea]
            #self.ch_outp[key]['u_st'] = self.ch_outp[key]['u_st'][:-self.nx_sea]
            #self.ch_outp[key]['sb_st_x'] = self.ch_outp[key]['sb_st_x'][:-self.nx_sea]
            #self.ch_outp[key]['eta_r'] = self.ch_outp[key]['eta_r'][:-self.nx_sea]
            #self.ch_outp[key]['ut_r'] = self.ch_outp[key]['ut_r'][:-self.nx_sea]
            #self.ch_outp[key]['wt_r'] = self.ch_outp[key]['wt_r'][:-self.nx_sea]
            #self.ch_outp[key]['s_ti'] = self.ch_outp[key]['s_ti'][:-self.nx_sea]
            #self.ch_outp[key]['s_ti_r'] = self.ch_outp[key]['s_ti_r'][:-self.nx_sea]
            tot_L = np.sum(self.ch_gegs[key]['L'][:-1])

        elif self.ch_gegs[key]['loc x=-L'][0] == 's':
            self.ch_outp[key]['px'] = self.ch_outp[key]['px'][self.nx_sea:]
            #self.ch_outp[key]['eta'] = self.ch_outp[key]['eta'][self.nx_sea:]
            self.ch_outp[key]['s_st'] = self.ch_outp[key]['s_st'][:,self.nx_sea:]
            #self.ch_outp[key]['ss_st'] = self.ch_outp[key]['ss_st'][self.nx_sea:]
            self.ch_outp[key]['sb_st'] = self.ch_outp[key]['sb_st'][:,self.nx_sea:] 
            self.ch_outp[key]['sn_st'] = self.ch_outp[key]['sn_st'][:,self.nx_sea:]
            #self.ch_outp[key]['u_st'] = self.ch_outp[key]['u_st'][self.nx_sea:]
            #self.ch_outp[key]['sb_st_x'] = self.ch_outp[key]['sb_st_x'][self.nx_sea:]
            #self.ch_outp[key]['eta_r'] = self.ch_outp[key]['eta_r'][self.nx_sea:]
            #self.ch_outp[key]['ut_r'] = self.ch_outp[key]['ut_r'][self.nx_sea:]
            #self.ch_outp[key]['wt_r'] = self.ch_outp[key]['wt_r'][self.nx_sea:]
            #self.ch_outp[key]['s_ti'] = self.ch_outp[key]['s_ti'][self.nx_sea:]
            #self.ch_outp[key]['s_ti_r'] = self.ch_outp[key]['s_ti_r'][self.nx_sea:]

            tot_L = np.sum(self.ch_gegs[key]['L'][1:])

                    
        #remove river domain
        if self.ch_gegs[key]['loc x=0'][0] == 'r':
            self.ch_outp[key]['px'] = self.ch_outp[key]['px'][:-self.nx_riv]+self.length_riv
            #self.ch_outp[key]['eta'] = self.ch_outp[key]['eta'][:-self.nx_riv]
            self.ch_outp[key]['s_st'] = self.ch_outp[key]['s_st'][:,:-self.nx_riv]
            #self.ch_outp[key]['ss_st'] = self.ch_outp[key]['ss_st'][:-self.nx_riv]
            self.ch_outp[key]['sb_st'] = self.ch_outp[key]['sb_st'][:,:-self.nx_riv] 
            self.ch_outp[key]['sn_st'] = self.ch_outp[key]['sn_st'][:,:-self.nx_riv]
            #self.ch_outp[key]['u_st'] = self.ch_outp[key]['u_st'][:-self.nx_riv]
            #self.ch_outp[key]['sb_st_x'] = self.ch_outp[key]['sb_st_x'][:-self.nx_riv]
            #self.ch_outp[key]['eta_r'] = self.ch_outp[key]['eta_r'][:-self.nx_riv]
            #self.ch_outp[key]['ut_r'] = self.ch_outp[key]['ut_r'][:-self.nx_riv]
            #self.ch_outp[key]['wt_r'] = self.ch_outp[key]['wt_r'][:-self.nx_riv]
            #self.ch_outp[key]['s_ti'] = self.ch_outp[key]['s_ti'][:-self.nx_riv]
            #self.ch_outp[key]['s_ti_r'] = self.ch_outp[key]['s_ti_r'][:-self.nx_riv]
            tot_L = np.sum(self.ch_gegs[key]['L'][:-1])

        elif self.ch_gegs[key]['loc x=-L'][0] == 'r':
            self.ch_outp[key]['px'] = self.ch_outp[key]['px'][self.nx_riv:]
            #self.ch_outp[key]['eta'] = self.ch_outp[key]['eta'][self.nx_riv:]
            self.ch_outp[key]['s_st'] = self.ch_outp[key]['s_st'][:,self.nx_riv:]
            #self.ch_outp[key]['ss_st'] = self.ch_outp[key]['ss_st'][self.nx_riv:]
            self.ch_outp[key]['sb_st'] = self.ch_outp[key]['sb_st'][:,self.nx_riv:] 
            self.ch_outp[key]['sn_st'] = self.ch_outp[key]['sn_st'][:,self.nx_riv:]
            #self.ch_outp[key]['u_st'] = self.ch_outp[key]['u_st'][self.nx_riv:]
            #self.ch_outp[key]['sb_st_x'] = self.ch_outp[key]['sb_st_x'][self.nx_riv:]
            #self.ch_outp[key]['eta_r'] = self.ch_outp[key]['eta_r'][self.nx_riv:]
            #self.ch_outp[key]['ut_r'] = self.ch_outp[key]['ut_r'][self.nx_riv:]
            #self.ch_outp[key]['wt_r'] = self.ch_outp[key]['wt_r'][self.nx_riv:]
            #self.ch_outp[key]['s_ti'] = self.ch_outp[key]['s_ti'][self.nx_riv:]
            #self.ch_outp[key]['s_ti_r'] = self.ch_outp[key]['s_ti_r'][self.nx_riv:]
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

        





















