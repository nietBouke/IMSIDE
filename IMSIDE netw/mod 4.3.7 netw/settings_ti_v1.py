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
from inpu.load_physics import phys_RMD1 , phys_test1, phys_RMD2, phys_RMD_tune

# =============================================================================
# load forcing conditions
# =============================================================================
from inpu.load_forcing_ti import forc_RMD1,forc_RMD2,forc_RMD3,forc_RMD4, forc_test1, forc_fun, forc_RR, forc_RMD_oldmap1, forc_RMD_HVO_1

# =============================================================================
# load geometry
# =============================================================================
from inpu.load_geo_RMD import geo_RMD9, geo_RMD10, geo_RMD9_hr, geo_RMD_HVO_1
from inpu.load_geo_RR import geo_RR2
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
#geo_pars = geo_RMD9()
geo_pars = geo_RMD9_hr(1)
#geo_pars = geo_RMD_HVO_1()
#choose forcing.
forc_pars = forc_RMD4()
#forc_pars = forc_RMD_HVO_1()
#choose physical constants
phys_pars = phys_RMD_tune()

'''
#choose physical constants
constants = phys_gen()
#choose geometry
geo_pars = geo_test1()
#choose forcing.
forc_pars = forc_test1()
#choose physical constants
phys_pars = phys_test1()


#choose physical constants
constants = phys_gen()
#choose geometry
geo_pars = geo_RR2()
#choose forcing.
forc_pars = forc_RR()
#choose physical constants
phys_pars = phys_RMD1()

#choose physical constants
constants = phys_gen()
#choose geometry
geo_pars = geo_RMD_oldmap1()
#choose forcing.
forc_pars = forc_RMD_oldmap1()
#choose physical constants
phys_pars = phys_RMD_oldmap1()
'''