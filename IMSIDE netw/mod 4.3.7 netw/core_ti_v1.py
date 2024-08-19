# =============================================================================
# The class is defined here.
# Init function adds a lot of variables to the class
# at the end the relevant functions are selected.
# =============================================================================
import numpy as np
import copy


class mod42_netw:
    def __init__(self, inp_gen, inp_geo, inp_forc, inp_phys, pars_seadom = (25000,250,10), pars_rivdom = (200000,10000,0) ):

        # =============================================================================
        # Initialisation function
        # =============================================================================
        #add variables to object
        self.g , self.Be, self.CD, self.r, self.tol = copy.deepcopy(inp_gen)
        self.ch_gegs = copy.deepcopy(inp_geo)
        self.Qriv, self.Qweir, self.Qhar, self.n_sea, self.soc, self.sri, self.swe, self.tid_per, self.a_tide, self.p_tide = copy.deepcopy(inp_forc)

        #load physics
        self.N, self.Lsc, self.nz, self.nt, self.theta = copy.deepcopy(inp_phys[0]) #theta not used.
        self.choice_viscosityv_st, self.Av_st, self.Av_st_cH, self.Av_st_cuH = copy.deepcopy(inp_phys[1])
        self.choice_diffusivityv_st, self.Kv_st, self.Kv_st_Sc = copy.deepcopy(inp_phys[2])
        self.choice_bottomslip_st, self.sf_st, self.sf_st_rr = copy.deepcopy(inp_phys[3])
        self.choice_diffusivityh_st, self.Kh_st, self.Kh_st_cb, self.Kh_st_cub, self.Kh_st_tres = copy.deepcopy(inp_phys[4])
        self.choice_viscosityv_ti, self.Av_ti, self.Av_ti_cH, self.Av_ti_cuH = copy.deepcopy(inp_phys[5])
        self.choice_diffusivityv_ti, self.Kv_ti, self.Kv_ti_Sc = copy.deepcopy(inp_phys[6])
        self.choice_bottomslip_ti, self.sf_ti, self.sf_ti_rr = copy.deepcopy(inp_phys[7])
        self.Kh_ti = copy.deepcopy(inp_phys[8])

        self.length_sea, self.dx_sea, self.exp_sea = pars_seadom
        self.nx_sea = int(self.length_sea/self.dx_sea+1)
        self.length_riv, self.dx_riv, self.exp_riv = pars_rivdom
        self.nx_riv = int(self.length_riv/self.dx_riv+1)

        # =============================================================================
        # build commonly used parameters
        # =============================================================================
        #create some dictionaries
        self.ch_pars = {}
        self.ch_outp = {}
        self.ch_tide = {}
        self.ends = []
        self.ch_keys = list(self.ch_gegs.keys())

        #other parameters
        self.soc_sca = np.max(self.soc)
        self.M = self.N+1
        self.z_nd = np.linspace(-1,0,self.nz)
        self.omega = 2*np.pi/self.tid_per
        self.m0 = np.arange(self.M)
        self.n0 = np.arange(self.N)
        self.mm = np.arange(1,self.M)
        self.nn = np.arange(1,self.M)[:,np.newaxis,np.newaxis]
        self.go2 = self.g/(self.omega**2)
        self.npzh = np.pi*np.arange(self.M)[:,np.newaxis,np.newaxis]*self.z_nd

        #for boundary layer correection
        self.z_inds = np.linspace(0,self.nz-1,self.M,dtype=int) # at which vertical levels we evaluate the expression
        self.tol = 0.01 #maybe intialise from somewhere else, for boundary layer stuff



        #add properties
        for key in self.ch_keys:
            self.ch_pars[key] = {}
            self.ch_outp[key] = {}
            self.ch_tide[key] = {}

            #add adjacent sea domain to sea channels
            if self.ch_gegs[key]['loc x=-L'][0] == 's':
                self.ch_gegs[key]['L'] = np.concatenate(([self.length_sea],self.ch_gegs[key]['L']))
                self.ch_gegs[key]['Hn'] = np.concatenate(([self.ch_gegs[key]['Hn'][0]],self.ch_gegs[key]['Hn']))
                self.ch_gegs[key]['b'] = np.concatenate(([[self.ch_gegs[key]['b'][0]*np.exp(self.exp_sea)],self.ch_gegs[key]['b']]))
                self.ch_gegs[key]['dx'] = np.concatenate(([self.dx_sea],self.ch_gegs[key]['dx']))
            if self.ch_gegs[key]['loc x=0'][0] == 's':
                self.ch_gegs[key]['L'] = np.concatenate((self.ch_gegs[key]['L'],[self.length_sea]))
                self.ch_gegs[key]['Hn'] = np.concatenate((self.ch_gegs[key]['Hn'],[self.ch_gegs[key]['Hn'][-1]]))
                self.ch_gegs[key]['b'] = np.concatenate((self.ch_gegs[key]['b'],[self.ch_gegs[key]['b'][-1]*np.exp(self.exp_sea)]))
                self.ch_gegs[key]['dx'] = np.concatenate((self.ch_gegs[key]['dx'],[self.dx_sea]))

            #add river domain to river channels
            if self.ch_gegs[key]['loc x=-L'][0] == 'r':
                self.ch_gegs[key]['L'] = np.concatenate(([self.length_riv],self.ch_gegs[key]['L']))
                self.ch_gegs[key]['Hn'] = np.concatenate(([self.ch_gegs[key]['Hn'][0]],self.ch_gegs[key]['Hn']))
                self.ch_gegs[key]['b'] = np.concatenate(([[self.ch_gegs[key]['b'][0]*np.exp(self.exp_riv)],self.ch_gegs[key]['b']]))
                self.ch_gegs[key]['dx'] = np.concatenate(([self.dx_riv],self.ch_gegs[key]['dx']))
            if self.ch_gegs[key]['loc x=0'][0] == 'r':
                self.ch_gegs[key]['L'] = np.concatenate((self.ch_gegs[key]['L'],[self.length_riv]))
                self.ch_gegs[key]['Hn'] = np.concatenate(([self.ch_gegs[key]['Hn']],self.ch_gegs[key]['Hn'][-1]))
                self.ch_gegs[key]['b'] = np.concatenate((self.ch_gegs[key]['b'],[self.ch_gegs[key]['b'][-1]*np.exp(self.exp_riv)]))
                self.ch_gegs[key]['dx'] = np.concatenate((self.ch_gegs[key]['dx'],[self.dx_riv]))

            #make list with all channel endings
            self.ends.append(self.ch_gegs[key]['loc x=0'])
            self.ends.append(self.ch_gegs[key]['loc x=-L'])

            #check here if the geometry input is in the right shape and such. More extensive tests are at the end of this file
            if len(self.ch_gegs[key]['L']) != len(self.ch_gegs[key]['Hn']) or \
                len(self.ch_gegs[key]['L']) != len(self.ch_gegs[key]['dx']) or \
                len(self.ch_gegs[key]['L']) != len(self.ch_gegs[key]['b'])-1 :
                print('WARNING: the geometry input of channel '+self.ch_gegs[key]['Name']+' is not in the right format.')

            # =============================================================================
            # indices, making lists of inputs, etc
            # =============================================================================
            self.ch_pars[key]['n_seg'] = len(self.ch_gegs[key]['L']) #nubmer of segments
            self.ch_pars[key]['dln'] = self.ch_gegs[key]['dx']/self.Lsc #normalised dx
            self.ch_pars[key]['nxn'] = np.array(self.ch_gegs[key]['L']/self.ch_gegs[key]['dx']+1,dtype=int) #number of points in segments

            self.ch_pars[key]['di'] = np.zeros(self.ch_pars[key]['n_seg']+1,dtype=int) #starting indices of segments
            for i in range(1,self.ch_pars[key]['n_seg']):   self.ch_pars[key]['di'][i] = np.sum(self.ch_pars[key]['nxn'][:i])
            self.ch_pars[key]['di'][-1] = np.sum(self.ch_pars[key]['nxn'])

            self.ch_pars[key]['dl'] = np.zeros(self.ch_pars[key]['nxn'].sum()) #normalised dx, per point
            self.ch_pars[key]['dl'][0:self.ch_pars[key]['nxn'][0]] = self.ch_pars[key]['dln'][0]
            for i in range(1,len(self.ch_pars[key]['nxn'])): self.ch_pars[key]['dl'][np.sum(self.ch_pars[key]['nxn'][:i]):np.sum(self.ch_pars[key]['nxn'][:i+1])] = self.ch_pars[key]['dln'][i]

            self.ch_pars[key]['bn'] = np.zeros(self.ch_pars[key]['n_seg']) #convergene length
            for i in range(self.ch_pars[key]['n_seg']): self.ch_pars[key]['bn'][i] = np.inf if self.ch_gegs[key]['b'][i+1] == self.ch_gegs[key]['b'][i] \
                else self.ch_gegs[key]['L'][i]/np.log(self.ch_gegs[key]['b'][i+1]/self.ch_gegs[key]['b'][i])

            self.ch_pars[key]['b'] = np.zeros(self.ch_pars[key]['nxn'].sum()) #width
            self.ch_pars[key]['b'][0:self.ch_pars[key]['nxn'][0]] = self.ch_gegs[key]['b'][0] * np.exp(self.ch_pars[key]['bn'][0]**(-1) \
                  * (np.linspace(-self.ch_gegs[key]['L'][0],0,self.ch_pars[key]['nxn'][0])+self.ch_gegs[key]['L'][0]))
            for i in range(1,self.ch_pars[key]['n_seg']): self.ch_pars[key]['b'][np.sum(self.ch_pars[key]['nxn'][:i]):np.sum(self.ch_pars[key]['nxn'][:i+1])] \
                = self.ch_pars[key]['b'][np.sum(self.ch_pars[key]['nxn'][:i])-1] * np.exp(self.ch_pars[key]['bn'][i]**(-1) \
                    * (np.linspace(-self.ch_gegs[key]['L'][i],0,self.ch_pars[key]['nxn'][i])+self.ch_gegs[key]['L'][i]))

            #build depth
            self.ch_pars[key]['H'] = np.zeros(self.ch_pars[key]['di'][-1])
            for i in range(self.ch_pars[key]['n_seg']): self.ch_pars[key]['H'][self.ch_pars[key]['di'][i]:self.ch_pars[key]['di'][i+1]] = self.ch_gegs[key]['Hn'][i]

            #stuff derived from width and depth
            self.ch_pars[key]['bex'] = np.zeros(self.ch_pars[key]['nxn'].sum()) #Lb
            self.ch_pars[key]['bex'][0:self.ch_pars[key]['nxn'][0]] = [self.ch_pars[key]['bn'][0]]*self.ch_pars[key]['nxn'][0]
            for i in range(1,len(self.ch_pars[key]['nxn'])): self.ch_pars[key]['bex'][np.sum(self.ch_pars[key]['nxn'][:i]):np.sum(self.ch_pars[key]['nxn'][:i+1])] \
                = [self.ch_pars[key]['bn'][i]]*self.ch_pars[key]['nxn'][i]
            self.ch_pars[key]['zlist'] = np.linspace(-self.ch_pars[key]['H'],0,self.nz).T[np.newaxis]

            # =============================================================================
            # physical parameters
            # Maybe it is better to define this elsewhere. Or define functions elsewhere which will be used here.
            # =============================================================================
            #mixing parametrizations , subtidal 
            
            #horizontal eddy diffusivity
            if self.choice_diffusivityh_st == 'constant':
                self.ch_pars[key]['Kh'] = self.Kh_st + np.zeros(self.ch_pars[key]['di'][-1]) 
                #add the increase of Kh in the adjacent sea domain
                if self.ch_gegs[key]['loc x=-L'][0] == 's': self.ch_pars[key]['Kh'][:self.ch_pars[key]['di'][1]] = self.ch_pars[key]['Kh'][self.ch_pars[key]['di'][1]] * self.ch_pars[key]['b'][:self.ch_pars[key]['di'][1]]/self.ch_pars[key]['b'][self.ch_pars[key]['di'][1]]
                if self.ch_gegs[key]['loc x=0'][0] == 's': self.ch_pars[key]['Kh'][self.ch_pars[key]['di'][-2]:] = self.ch_pars[key]['Kh'][self.ch_pars[key]['di'][-2]] * self.ch_pars[key]['b'][self.ch_pars[key]['di'][-2]:]/self.ch_pars[key]['b'][self.ch_pars[key]['di'][-2]]
            elif self.choice_diffusivityh_st == 'cb':
                self.ch_pars[key]['Kh'] = self.Kh_st_cb * self.ch_pars[key]['b']
                self.ch_pars[key]['Kh'][np.where(self.ch_pars[key]['Kh']<self.Kh_st_tres)[0]] = self.Kh_st_tres #remove very small values
            elif self.choice_diffusivityh_st == 'cub':
                raise Exception('The choice to have Kh depend on the tidal current amplitude is not possible in the current model formulation. If to implement, please check the model thoroughly')                
            else: 
                raise Exception('No right choice for diffusivity st') 
            
            #vertical viscosity
            if self.choice_viscosityv_st == 'constant':
                self.ch_pars[key]['Av'] = self.Av_st + np.zeros(self.ch_pars[key]['di'][-1]) 
            elif self.choice_viscosityv_st == 'cH':
                self.ch_pars[key]['Av'] = self.Av_st_cH * self.ch_pars[key]['H']
            elif self.choice_viscosityv_st == 'cuH':
                raise Exception('The choice to have Av depend on the tidal current amplitude is not possible in the current model formulation. If to implement, please check the model thoroughly')                
            else:
                raise Exception('No right choice for viscosity st') 

            #vertical diffusivity
            if self.choice_diffusivityv_st == 'Schmidt':
                self.ch_pars[key]['Kv'] = self.ch_pars[key]['Av'] / self.Kv_st_Sc
            elif self.choice_diffusivityv_st == 'constant':
                self.ch_pars[key]['Kv'] = self.Kv_st + np.zeros(self.ch_pars[key]['di'][-1]) 
            else:
                raise Exception('No right choice for diffusivity st')   

            #bottom friction parameter
            if self.choice_bottomslip_st == 'constant': 
                rr = self.ch_pars[key]['Av']/(self.sf_st*self.ch_pars[key]['H'])
            elif self.choice_bottomslip_st == 'rr':  
                rr = self.sf_st_rr + np.zeros(self.ch_pars[key]['di'][-1])
            else:  
                raise Exception('No right choice for bottom friction st') 

            
            #other subtidal parameters
            self.ch_pars[key]['alf'] = self.g*self.Be*self.ch_pars[key]['H']**3/(48*self.ch_pars[key]['Av']) #strength of exchange flow
            self.ch_pars[key]['bH_1'] = 1/(self.ch_pars[key]['H']*self.ch_pars[key]['b']) #1 over cross section
            self.ch_pars[key]['Kh_x'] = np.concatenate([[0],(self.ch_pars[key]['Kh'][2:]-self.ch_pars[key]['Kh'][:-2])/(2*self.ch_pars[key]['dl'][1:-1]*self.Lsc),[0]])    #derivative of Kh
            self.ch_pars[key]['Kh_x'][[0,-1]] ,  self.ch_pars[key]['Kh_x'][self.ch_pars[key]['di'][1:-1]] ,  self.ch_pars[key]['Kh_x'][self.ch_pars[key]['di'][1:-1]-1] = None , None , None
            #coefficients in u' solution
            self.ch_pars[key]['g1'] = 0.5/(1+3*rr)
            self.ch_pars[key]['g2'] = -1.5/(1+3*rr)
            self.ch_pars[key]['g3'] = (1+6*rr)/(1+3*rr)
            self.ch_pars[key]['g4'] = (-9-36*rr)/(1+3*rr)
            self.ch_pars[key]['g5'] = -8


            #mixing parametrizations , tidal 

            #vertical viscosity
            if self.choice_viscosityv_ti == 'constant':
                self.ch_pars[key]['Av_ti'] = self.Av_ti + np.zeros(len(self.ch_gegs[key]['Hn'])) 
            elif self.choice_viscosityv_ti == 'cH':
                self.ch_pars[key]['Av_ti'] = self.Av_ti_cH * self.ch_gegs[key]['Hn']
            elif self.choice_viscosityv_ti == 'cuH':
                raise Exception('The choice to have Av depend on the tidal current amplitude is not possible in the current model formulation. If to implement, please check the model thoroughly') 
            else:
                raise Exception('No right choice for viscosity ti') 
                    
            #vertical diffusivity
            if self.choice_diffusivityv_ti == 'Schmidt':
                self.ch_pars[key]['Kv_ti'] = self.ch_pars[key]['Av_ti'] / self.Kv_ti_Sc
            elif self.choice_diffusivityv_ti == 'constant':
                self.ch_pars[key]['Kv_ti'] = self.Kv_ti + np.zeros(len(self.ch_gegs[key]['Hn'])) 
            else:
                raise Exception('No right choice for diffusivity ti')   

            #bottom friction parameter
            if self.choice_bottomslip_ti == 'constant': 
                self.ch_pars[key]['rr_ti'] = (self.ch_pars[key]['Av_ti'] / (self.sf_ti * self.ch_gegs[key]['Hn']))
            elif self.choice_bottomslip_ti == 'rr':  
                self.ch_pars[key]['rr_ti'] = self.sf_st_rr + np.zeros(len(self.ch_gegs[key]['Hn']))
            else:  
                raise Exception('No right choice for bottom friction ti')      
                
            #normalized Kh_ti and derived quantities, for boundary layer correction
            self.ch_pars[key]['eps'] , self.ch_pars[key]['epsL'] = np.zeros(self.ch_pars[key]['n_seg']) + np.nan , np.zeros(self.ch_pars[key]['n_seg']) + np.nan
            for d in range(self.ch_pars[key]['n_seg']):
                #the maximum Kh which does not exceed the channel length
                Kh_cor = self.omega * (self.ch_gegs[key]['L'][d] - self.ch_gegs[key]['dx'][d])**2/(np.log(self.tol)**2)

                if Kh_cor < self.Kh_ti: Kh_here = Kh_cor
                else: Kh_here = self.Kh_ti

                self.ch_pars[key]['eps'][d]  = Kh_here/(self.omega*self.Lsc**2)       #epsilon, the normalised horizontal diffusion
                self.ch_pars[key]['epsL'][d] = Kh_here/(self.omega)        #epsilon, the normalised horizontal diffusion



        # =============================================================================
        # Do checks and define some parameters
        # =============================================================================
        #Checks on numbers: no double river or sea endings
        ends_r, ends_s, ends_j, ends_w, ends_h = [] , [] , [] , [] , []
        for i in range(len(self.ends)):
            if self.ends[i][0] == 'r' :  ends_r.append(self.ends[i])
            elif self.ends[i][0] == 's' :  ends_s.append(self.ends[i])
            elif self.ends[i][0] == 'j' :  ends_j.append(self.ends[i])
            elif self.ends[i][0] == 'w' :  ends_w.append(self.ends[i])
            elif self.ends[i][0] == 'h' :  ends_h.append(self.ends[i])
            else:print('Unregcognised channel end')

        if ends_r != list(dict.fromkeys(ends_r)) : print('ERROR: Duplicates in river points!')
        if ends_s != list(dict.fromkeys(ends_s)) : print('ERROR: Duplicates in sea points!')
        if ends_w != list(dict.fromkeys(ends_w)) : print('ERROR: Duplicates in weir points!')
        if ends_h != list(dict.fromkeys(ends_h)) : print('ERROR: Duplicates in har points!')

        #remove duplicate channel endings
        self.ends = list(dict.fromkeys(self.ends))

        #calculate number of channels, sea vertices, river vertices, junctions
        self.n_ch = len(self.ch_keys)
        self.n_s, self.n_r, self.n_j, self.n_w, self.n_h = 0 , 0 , 0 , 0 , 0
        for i in range(len(self.ends)):
            if self.ends[i][0]=='r': self.n_r = self.n_r + 1
            elif self.ends[i][0]=='s': self.n_s = self.n_s + 1
            elif self.ends[i][0]=='j': self.n_j = self.n_j + 1
            elif self.ends[i][0]=='w': self.n_w = self.n_w + 1
            elif self.ends[i][0]=='h': self.n_h = self.n_h + 1
            else: print('ERROR: unrecognised channel ending')
        self.n_unk = self.n_ch+self.n_s+self.n_r+self.n_j+self.n_w+self.n_h #number of unknowns

        #variable where we keep track of how often the junctions are used in the solution vector e.g.
        self.junc_track = {}
        for j in range(self.n_j): self.junc_track['j'+str(j+1)] = 0

        # =============================================================================
        # do checks
        # =============================================================================
        if len(self.Qriv) != self.n_r: print('WARNING:Something goes wrong with the number of river channels')
        if len(self.Qweir) != self.n_w: print('WARNING:Something goes wrong with the number of weir channels')
        if len(self.Qhar) != self.n_h: print('WARNING:Something goes wrong with the number of har channels')
        if len(self.n_sea) != self.n_s: print('WARNING:Something goes wrong with the number of sea channels')
        if len(self.soc) != self.n_s: print('WARNING:Something goes wrong with the number of sea channels')
        if len(self.sri) != self.n_r: print('WARNING:Something goes wrong with the number of river channels')
        if len(self.swe) != self.n_w: print('WARNING:Something goes wrong with the number of weir channels')
        if len(self.a_tide) != self.n_s: print('WARNING:Something goes wrong with the number of sea channels')
        if len(self.p_tide) != self.n_s: print('WARNING:Something goes wrong with the number of sea channels')

        #check location of sea boundary
        for key in self.ch_keys:
            if self.ch_gegs[key]['loc x=-L'][0] == 's' :
                print('SERIOUS WARNING: please do not use a sea boundary at x=-L.')
                print('Expect the code to crash')

        #check labelling of vertices
        ends_r2,ends_s2,ends_j2,ends_w2,ends_h2 = [],[],[],[],[]
        for i in range(self.n_r):  ends_r2.append(int(ends_r[i][1:]))
        for i in range(self.n_s):  ends_s2.append(int(ends_s[i][1:]))
        for i in range(self.n_w):  ends_w2.append(int(ends_w[i][1:]))
        for i in range(self.n_h):  ends_h2.append(int(ends_h[i][1:]))
        for i in range(self.n_j*3):  ends_j2.append(int(ends_j[i][1:]))
        ends_j2 = np.sort(ends_j2).reshape((self.n_j,3))[:,0] #every junction point has to appear three times

        if not np.array_equal(np.sort(ends_r2) , np.arange(self.n_r)+1) :print('ERROR: something wrong with the labelling of the river points')
        if not np.array_equal(np.sort(ends_s2) , np.arange(self.n_s)+1) :print('ERROR: something wrong with the labelling of the sea points')
        if not np.array_equal(np.sort(ends_w2) , np.arange(self.n_w)+1) :print('ERROR: something wrong with the labelling of the weir points')
        if not np.array_equal(np.sort(ends_h2) , np.arange(self.n_h)+1) :print('ERROR: something wrong with the labelling of the har points')
        if not np.array_equal(np.sort(ends_j2) , np.arange(self.n_j)+1) :print('ERROR: something wrong with the labelling of the junction points')

        if np.max(np.abs(self.Qriv),initial=0) > 50000  : print('WARNING: values of river discharge are unlikely high, please check')
        if np.max(np.abs(self.Qweir),initial=0) > 50000 : print('WARNING: values of weir discharge are unlikely high, please check')
        if np.max(np.abs(self.Qhar),initial=0) > 50000  : print('WARNING: values of har discharge are unlikely high, please check')
        if np.max(np.abs(self.n_sea),initial=0) > 1 : print('WARNING: values of sea surface height are unlikely high, please check')
        if np.max(self.soc,initial=0) > 40 or np.min(self.soc,initial=0) < 0 : print('WARNING: values of sea salinity are unlikely, please check')
        if np.max(self.sri,initial=0) > 40 or np.min(self.sri,initial=0) < 0 : print('WARNING: values of river salinity are unlikely, please check')
        if np.max(self.swe,initial=0) > 40 or np.min(self.swe,initial=0) < 0 : print('WARNING: values of weir salinity are unlikely, please check')
        if np.max(self.a_tide) > 5: print('WARNING: values of tidal surface amplitude are unlikely, please check')

        for key in self.ch_keys:
            #unlikely or unphysical values for the geometry input
            if np.min(self.ch_pars[key]['H']) < 0 or np.max(self.ch_pars[key]['H']) > 50: print('WARNING: values of depth are unlikely, please check')
            if np.min(self.ch_gegs[key]['b']) < 0 : print('WARNING: width can not be negative')
            if np.min(self.ch_gegs[key]['dx']) < 0 or np.max(self.ch_gegs[key]['dx']) > 50000 : print('WARNING: values for spatial grid step are unlikely, please check')
            if np.min(self.ch_pars[key]['nxn']) < 4: print("WARNING: in the channel "+key+' is a channel which has less than 4 horizontal elements, which can give problems')


        #TODO: more advanced checks can be added, e.g.: jump in depth not too large, maybe something with the mixing parameters.


    # =============================================================================
    # Import the functions for the different calculations
    # =============================================================================
    #calculation of river discharge
    from physics.discharge_distribution_v1 import Qdist_calc
    #calculation of tidal properties
    from physics.tidal_hydro_v1 import tide_calc
    #subtidal modules
    from physics.subtidal_v1 import subtidal_module , sol_subtidal, jac_subtidal_fix,  jac_subtidal_vary
    #solution
    from physics.solve_ti_v1 import run_model
    from physics.solve_ti_v1_check import run_check

    from physics.indices_v1 import indices

    from physics.tidal_salt_v2 import tidal_salinity, sol_tidal, jac_tidal_fix, jac_tidal_vary

    from physics.boundaries_v1 import sol_bound, jac_bound_fix, jac_bound_vary

    from physics.junctions_v2 import prep_junc, sol_junc_tot, jac_junc_tot_fix, jac_junc_tot_vary

    from visualisation.calc_rawtofine_ti_v1 import calc_output


    #visualisation
    from visualisation.visu_RM_ti import plot_procRM , plot_salt_compRM, anim_new_compRM, plot_tide_pointRM, plot2_tide_pointRM
    from visualisation.visu_RMDo_ti import plot_s_RMDo
    from visualisation.visu_gen_ti import plot_s_gen , plot_proc_ch
    from visualisation.calc_RM_ti import calc_X2, calc_tide_pointRM
