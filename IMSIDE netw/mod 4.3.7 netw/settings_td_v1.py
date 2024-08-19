# =============================================================================
# code where I group the physical and geometric input for the delta in a class
# =============================================================================
# =============================================================================
# load general constants
# =============================================================================
from inpu.load_physics import phys_gen

# =============================================================================
# load physical constants
# =============================================================================
from inpu.load_physics import phys_RMD1 , phys_test1, phys_RMD2

# =============================================================================
# load forcing conditions
# =============================================================================
from inpu.load_forcing_td import forc_RMD4, forc_RMD5,  forc_RMD_fromSOBEK, forc_RMD_fromcsv, forc_RMD_fromMo, forc_RMD_fromJesse
from inpu.load_forcing_td import forc_test1, forc_test2

# =============================================================================
# load geometry
# =============================================================================
from inpu.load_geo_RMD import geo_RMD9, geo_RMD10
from inpu.load_geo_test import geo_test1
from inpu.funn import geo_fun
# =============================================================================
# choose final numbers
# =============================================================================

#date_start, date_stop = '2008-01-01' , '2010-11-01'
'''
#choose physical constants
constants = phys_gen()
#choose geometry
geo_pars = geo_fun()
#choose forcing.
forc_pars = forc_fun()
#choose physical constants
phys_pars = phys_test1()
'''
#choose physical constants
constants = phys_gen()
#choose geometry
geo_pars = geo_RMD9()
#choose forcing.
#forc_pars = forc_RMD5()
forc_pars = forc_RMD_fromJesse(33,18700,18750)
#forc_pars = forc_RMD_fromSOBEK('01-07-2021' , '01-08-2021')
# forc_pars = forc_RMD_fromcsv('/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/RM_data/MO_Q2122/','Q_daily_mean_Hag-Tie-Meg-Har-Gou_2021-2022_Bouke030524.csv',
#                         '01-01-2022', '31-12-2022')
# orc_pars = forc_RMD_fromMo('/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/RM_data/MO_Q2122/','Q_daily_mean_Hag-Tie-Meg-Har-Gou_2020.csv' )
#choose physical constants
phys_pars = phys_RMD2()

'''
#choose physical constants
constants = phys_gen()
#choose geometry
geo_pars = geo_test1()
#choose forcing.
forc_pars = forc_test2()
#choose physical constants
phys_pars = phys_test1()
#'''