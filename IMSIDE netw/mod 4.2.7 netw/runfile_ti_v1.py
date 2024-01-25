# =============================================================================
# Runfile general network in equilibrium
# =============================================================================
# Bouke, December 2023
# =============================================================================
# import functions
# =============================================================================
import settings_ti_v1
from core_ti_v1 import mod42_netw
#from inputfile_v1 import input_network
#from functions_all_v1 import network_funcs

delta = mod42_netw(settings_ti_v1.constants, settings_ti_v1.geo_pars, settings_ti_v1.forc_pars, settings_ti_v1.phys_pars)#, pars_seadom = (25000,100,10), pars_rivdom = (200000,2000,0))

#calculate river discharge distribution
delta.run_model()
 
#calculate output
delta.calc_output()


#visualisation
delta.plot_s_gen()
#delta.plot_proc_ch()

#for Rhine-Meuse
#delta.plot_salt_compRM('s_st','Subtidal salinity',0,35)
delta.plot_procRM()
delta.plot_tide_pointRM()
#delta.anim_new_compRM('newcode1201_v4')



