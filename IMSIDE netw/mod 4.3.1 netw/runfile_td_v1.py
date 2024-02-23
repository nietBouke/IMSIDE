# =============================================================================
# Runfile general network in equilibrium
# =============================================================================
# Bouke, December 2023
# =============================================================================
# import functions
# =============================================================================
import settings_td_v1
from core_td_v1 import mod42_netw
#from inputfile_v1 import input_network
#from functions_all_v1 import network_funcs

delta = mod42_netw(settings_td_v1.constants, settings_td_v1.geo_pars, settings_td_v1.forc_pars, settings_td_v1.phys_pars)#, pars_seadom = (25000,100,10), pars_rivdom = (200000,2000,0))

#calculate river discharge distribution
delta.run_model()
 
#
delta.calc_output()


#visualisation
delta.plot_s_gen_td(0)
delta.plot_s_gen_td(-1)

#for Rhine-Meuse
out427 = delta.plot_salt_pointRM(4.2,51.95,1)
#delta.plot_salt_pointRM(4.5,51.9,1)
#delta.anim_RM_st('td_220224_v3')
