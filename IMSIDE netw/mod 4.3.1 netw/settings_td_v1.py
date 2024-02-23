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
from inpu.load_physics import phys_RMD1 , phys_test1

# =============================================================================
# load forcing conditions
# =============================================================================
from inpu.load_forcing_td import forc_RMD4, forc_RMD5,  forc_RMD_fromSOBEK
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
forc_pars = forc_RMD5()
#forc_pars = forc_RMD_fromSOBEK('01-06-2021' , '01-08-2021')
#choose physical constants
phys_pars = phys_RMD1()

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