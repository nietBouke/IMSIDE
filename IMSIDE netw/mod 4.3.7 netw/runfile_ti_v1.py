# =============================================================================
# Runfile general network in equilibrium
# =============================================================================
# Bouke, December 2023
# =============================================================================
# import functions
# =============================================================================
import numpy as np
import settings_ti_v1
from core_ti_v1 import mod42_netw
#from inputfile_v1 import input_network
#from functions_all_v1 import network_funcs

delta = mod42_netw(settings_ti_v1.constants, settings_ti_v1.geo_pars, settings_ti_v1.forc_pars, settings_ti_v1.phys_pars)#, pars_seadom = (25000,5000,10), pars_rivdom = (200000,20000,0))

#calculate river discharge distribution
#delta.run_check(np.loadtxt('/Users/biemo004/Documents/UU phd Saltisolutions/Output/temporary/out0102.txt' ))
#delta.run_check()
delta.run_model()
 
#calculate output
delta.calc_output()

#visualisation
# delta.plot_s_gen()
# delta.plot_proc_ch()

#for Rhine-Meuse
delta.plot_salt_compRM('s_st','Subtidal salinity',0,35)
delta.plot_procRM()
delta.plot2_tide_pointRM()
#delta.anim_new_compRM('newcode2202_v1')


#delta.plot_s_RMDo()
delta.calc_X2()

# #%%%

# print(delta.Qdist_calc((delta.Qriv, delta.Qweir, delta.Qhar, delta.n_sea)))
# #print(2007/(16*600*np.sqrt(9.81*7.6e-4*16*33)))
# #print(251/(6.4*250*np.sqrt(9.81*7.6e-4*6.4*33)))

# print()
# #Froude number NM1, high discharge
# print(843/(11*450*np.sqrt(9.81*7.6e-4*11*33)))
# #Froude number NM1, low discharge
# print(843/5/(11*450*np.sqrt(9.81*7.6e-4*11*33)))
# #Froude number OM2, high discharge
# print(1377/(14*317*np.sqrt(9.81*7.6e-4*14*33)))
# #Froude number OM2, low discharge
# print(1377/5/(14*317*np.sqrt(9.81*7.6e-4*14*33)))


