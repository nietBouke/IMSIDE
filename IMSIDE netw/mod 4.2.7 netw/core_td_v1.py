# =============================================================================
# The class is defined here.
# Init function adds a lot of variables to the class
# at the end the relevant functions are selected.
# =============================================================================
import numpy as np

class mod42_netw:
    def __init__(self, inp_gen, inp_geo, inp_forc, inp_phys, pars_seadom = (25000,250,10), pars_rivdom = (200000,10000,0) ):

        # =============================================================================
        # Initialisation function
        # =============================================================================
        #add variables to object
        self.g , self.Be, self.CD, self.r, self.tol = inp_gen
        self.ch_gegs = inp_geo
        self.N, self.Lsc, self.nz, self.nt, self.theta  = inp_phys[0]
        self.Av_st, self.Kv_st, self.sf_st, self.Kh_st = inp_phys[1]
        self.Av_ti, self.Kv_ti, self.sf_ti, self.Kh_ti = inp_phys[2]
        self.Qriv, self.Qweir, self.Qhar, self.n_sea, self.soc, self.sri, self.swe, self.tid_per, self.a_tide, self.p_tide, self.T, self.DT = inp_forc

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
        self.mm = np.arange(1,self.M)
        self.m0 = np.arange(self.M)
        self.nn = np.arange(1,self.M)[:,np.newaxis,np.newaxis]
        self.go2 = self.g/(self.omega**2)
        self.npzh = np.pi*np.arange(self.M)[:,np.newaxis,np.newaxis]*self.z_nd
        self.dt  = self.DT/self.Lsc # I think this is the only scaling required, rest is unnessecary. 

        #for boundary layer correection
        self.z_inds = np.linspace(0,self.nz-1,self.M,dtype=int) # at which vertical levels we evaluate the expression
        self.eps = self.Kh_ti/(self.omega*self.Lsc**2)       #epsilon, the normalised horizontal diffusion
        self.epsL = self.Kh_ti/(self.omega)        #epsilon, the normalised horizontal diffusion
        self.tol = 0.01 #maybe intialise from somewhere else

        #add properties
        for key in self.ch_keys:
            self.ch_pars[key] = {}
            self.ch_outp[key] = {}
            self.ch_tide[key] = {}

            #add adjacent sea domain to sea channels
            if self.ch_gegs[key]['loc x=-L'][0] == 's':
                self.ch_gegs[key]['L'] = np.concatenate(([self.length_sea],self.ch_gegs[key]['L']))
                self.ch_gegs[key]['b'] = np.concatenate(([[self.ch_gegs[key]['b'][0]*np.exp(self.exp_sea)],self.ch_gegs[key]['b']]))
                self.ch_gegs[key]['dx'] = np.concatenate(([self.dx_sea],self.ch_gegs[key]['dx']))
            if self.ch_gegs[key]['loc x=0'][0] == 's':
                self.ch_gegs[key]['L'] = np.concatenate((self.ch_gegs[key]['L'],[self.length_sea]))
                self.ch_gegs[key]['b'] = np.concatenate((self.ch_gegs[key]['b'],[self.ch_gegs[key]['b'][-1]*np.exp(self.exp_sea)]))
                self.ch_gegs[key]['dx'] = np.concatenate((self.ch_gegs[key]['dx'],[self.dx_sea]))

            #add river domain to river channels
            if self.ch_gegs[key]['loc x=-L'][0] == 'r':
                self.ch_gegs[key]['L'] = np.concatenate(([self.length_riv],self.ch_gegs[key]['L']))
                self.ch_gegs[key]['b'] = np.concatenate(([[self.ch_gegs[key]['b'][0]*np.exp(self.exp_riv)],self.ch_gegs[key]['b']]))
                self.ch_gegs[key]['dx'] = np.concatenate(([self.dx_riv],self.ch_gegs[key]['dx']))
            if self.ch_gegs[key]['loc x=0'][0] == 'r':
                self.ch_gegs[key]['L'] = np.concatenate((self.ch_gegs[key]['L'],[self.length_riv]))
                self.ch_gegs[key]['b'] = np.concatenate((self.ch_gegs[key]['b'],[self.ch_gegs[key]['b'][-1]*np.exp(self.exp_riv)]))
                self.ch_gegs[key]['dx'] = np.concatenate((self.ch_gegs[key]['dx'],[self.dx_riv]))

            #make list with all channel endings
            self.ends.append(self.ch_gegs[key]['loc x=0'])
            self.ends.append(self.ch_gegs[key]['loc x=-L'])

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

            self.ch_pars[key]['bex'] = np.zeros(self.ch_pars[key]['nxn'].sum()) #Lb
            self.ch_pars[key]['bex'][0:self.ch_pars[key]['nxn'][0]] = [self.ch_pars[key]['bn'][0]]*self.ch_pars[key]['nxn'][0]
            for i in range(1,len(self.ch_pars[key]['nxn'])): self.ch_pars[key]['bex'][np.sum(self.ch_pars[key]['nxn'][:i]):np.sum(self.ch_pars[key]['nxn'][:i+1])] \
                = [self.ch_pars[key]['bn'][i]]*self.ch_pars[key]['nxn'][i]

            # =============================================================================
            # physical parameters
            # Maybe it is better to define this elsewhere. Or define functions elsewhere which will be used here.
            # =============================================================================
            #mixing parametrizations -  for now simple

            self.ch_pars[key]['Kh'] = self.Kh_st + np.zeros(self.ch_pars[key]['di'][-1]) #horizontal mixing coefficient
            #add the increase of Kh in the adjacent sea domain
            if self.ch_gegs[key]['loc x=-L'][0] == 's': self.ch_pars[key]['Kh'][:self.ch_pars[key]['di'][1]] = self.ch_pars[key]['Kh'][self.ch_pars[key]['di'][1]] * self.ch_pars[key]['b'][:self.ch_pars[key]['di'][1]]/self.ch_pars[key]['b'][self.ch_pars[key]['di'][1]]
            if self.ch_gegs[key]['loc x=0'][0] == 's': self.ch_pars[key]['Kh'][self.ch_pars[key]['di'][-2]:] = self.ch_pars[key]['Kh'][self.ch_pars[key]['di'][-2]] * self.ch_pars[key]['b'][self.ch_pars[key]['di'][-2]:]/self.ch_pars[key]['b'][self.ch_pars[key]['di'][-2]]
            #ch_pars[key]['Kh'][np.where(ch_pars[key]['Kh']<20.)] = 20. #remove very small values
            #else: print('ERROR: no valid value for ch')

            self.ch_pars[key]['Av'] = self.Av_st #vertical viscosity
            self.ch_pars[key]['Kv'] = self.Kv_st #vertical diffusivity
            self.ch_pars[key]['sf'] = self.sf_st #bottom slip
            self.ch_pars[key]['alf'] = self.g*self.Be*self.ch_gegs[key]['H']**3/(48*self.ch_pars[key]['Av']) #strength of exchange flow
            self.ch_pars[key]['bH_1'] = 1/(self.ch_gegs[key]['H']*self.ch_pars[key]['b']) #1 over cross section
            self.ch_pars[key]['Kh_x'] = np.concatenate([[0],(self.ch_pars[key]['Kh'][2:]-self.ch_pars[key]['Kh'][:-2])/(2*self.ch_pars[key]['dl'][1:-1]*self.Lsc),[0]])    #derivative of Kh
            self.ch_pars[key]['Kh_x'][[0,-1]] ,  self.ch_pars[key]['Kh_x'][self.ch_pars[key]['di'][1:-1]] ,  self.ch_pars[key]['Kh_x'][self.ch_pars[key]['di'][1:-1]-1] = None , None , None

            rr = self.ch_pars[key]['Av']/(self.ch_pars[key]['sf']*self.ch_gegs[key]['H'])#bottom friction parameter
            #coefficients in u' solution
            self.ch_pars[key]['g1'] = 0.5/(1+3*rr)
            self.ch_pars[key]['g2'] = -1.5/(1+3*rr)
            self.ch_pars[key]['g3'] = (1+6*rr)/(1+3*rr)
            self.ch_pars[key]['g4'] = (-9-36*rr)/(1+3*rr)
            self.ch_pars[key]['g5'] = -8

            self.ch_pars[key]['zlist'] = np.linspace(-self.ch_gegs[key]['H'],0,self.nz)[np.newaxis,np.newaxis,:]

            '''
            self.ch_pars[key]['Av'] = self.cv*self.ch_gegs[key]['H'] #vertical viscosity
            self.ch_pars[key]['Kv'] = self.ch_pars[key]['Av']/self.Sc #vertical diffusivity
            self.ch_pars[key]['sf'] = 2*self.cv #bottom slip
            self.ch_pars[key]['alf'] = self.g*self.Be*self.ch_gegs[key]['H']**3/(48*self.ch_pars[key]['Av']) #strength of exchange flow
            self.ch_pars[key]['bH_1'] = 1/(self.ch_gegs[key]['H']*self.ch_pars[key]['b']) #1 over cross section
            self.ch_pars[key]['Kh_x'] = np.concatenate([[0],(self.ch_pars[key]['Kh'][2:]-self.ch_pars[key]['Kh'][:-2])/(2*self.ch_pars[key]['dl'][1:-1]*self.Lsc),[0]])    #derivative of Kh
            self.ch_pars[key]['Kh_x'][[0,-1]] ,  self.ch_pars[key]['Kh_x'][self.ch_pars[key]['di'][1:-1]] ,  self.ch_pars[key]['Kh_x'][self.ch_pars[key]['di'][1:-1]-1] = None , None , None

            rr = self.ch_pars[key]['Av']/(self.ch_pars[key]['sf']*self.ch_gegs[key]['H'])#bottom friction parameter
            #coefficients in u' solution
            self.ch_pars[key]['g1'] = 0.5/(1+3*rr)
            self.ch_pars[key]['g2'] = -1.5/(1+3*rr)
            self.ch_pars[key]['g3'] = (1+6*rr)/(1+3*rr)
            self.ch_pars[key]['g4'] = (-9-36*rr)/(1+3*rr)
            self.ch_pars[key]['g5'] = -8
            #no wind for now...
            '''

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
        if len(self.Qriv) != self.n_r: print('Something goes wrong with the number of river channels')
        if len(self.Qweir) != self.n_w: print('Something goes wrong with the number of weir channels')
        if len(self.Qhar) != self.n_h: print('Something goes wrong with the number of har channels')
        if len(self.n_sea) != self.n_s: print('Something goes wrong with the number of sea channels')
        if len(self.soc) != self.n_s: print('Something goes wrong with the number of sea channels')
        if len(self.sri) != self.n_r: print('Something goes wrong with the number of river channels')
        if len(self.swe) != self.n_w: print('Something goes wrong with the number of weir channels')
        if len(self.a_tide) != self.n_s: print('Something goes wrong with the number of sea channels')
        if len(self.p_tide) != self.n_s: print('Something goes wrong with the number of sea channels')

        #check location of sea boundary
        for key in self.ch_keys:
            if self.ch_gegs[key]['loc x=-L'][0] == 's' :
                print('SERIOUS WARNING: please do not use a sea boundary at x=-L. There is some error, I cannot find out where it is. To repair later.  ')
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
            if self.ch_gegs[key]['H'] < 0 or self.ch_gegs[key]['H'] > 50: print('WARNING: values of depth are unlikely, please check')
            if np.min(self.ch_gegs[key]['b']) < 0 : print('WARNING: width can not be negative')
            if np.min(self.ch_gegs[key]['dx']) < 0 or np.max(self.ch_gegs[key]['dx']) > 50000 : print('WARNING: values for spatial grid step are unlikely, please check')

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
    from physics.solve_td_v1 import run_model
    #from solve_ti_v1_check import run_model

    from physics.indices_v1 import indices

    from physics.tidal_salt_v2 import tidal_salinity, sol_tidal, jac_tidal_fix, jac_tidal_vary

    from physics.inertiaterm_v1 import sol_inertia, sol_inertia_o, sol_inertia_n , jac_inertia

    from physics.boundaries_v1 import sol_bound, jac_bound_fix, jac_bound_vary

    from physics.junctions_v2 import prep_junc, sol_junc_tot, jac_junc_tot_fix, jac_junc_tot_vary

    from visualisation.calc_rawtofine_td_v1 import calc_output



    #visualisation
    from visualisation.visu_RM_td import plot_salt_pointRM , anim_RM_st
    from visualisation.visu_gen_td import plot_s_gen_td

    '''
    #calculation of salinity
    from saltmodule_ti_v1 import run_model
    #calculation of salinity
    from saltmodule_ti_notide import run_model_nt
    #convert raw output to plottable stuff
    from calc_rawtofine_v2 import calc_output
    # =============================================================================
    # Import the functions for the different visualisations
    # =============================================================================
    #visualisation of tidal properties
    from visu_tide_v1 import plot_tide , plot_salt_ti_point, plot_tide_timeseries
    #overview plot of salinity and river discharge
    from visu_Qssimple_v1 import plot_Qs_simple , plot_Qs_comp, plot_proc_ch, plot_junc_tid, plot_jinyang, plot_contours
    #animation of the tides
    from visu_tideani_v1 import anim_tide, anim_tide_wl
    #detailed plot of salinity and river discharge
    from visu_HO import plot_new_compHO
    from visu_RM import plot_new_compRM, plot_salt_compRM, anim_new_compRM ,  plot_tide_pointRM
    from plot_verts import plot_proc_vt
    # =============================================================================
    #  error calculation, for tuning
    # =============================================================================
    from calc_tide_v1 import calc_tide_pointRM, max_min_salts
    from calcs_afterwards import calc_X2
    '''
