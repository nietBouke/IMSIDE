# =============================================================================
# The idea of this file: run everything from one file. 
# So: choose here everything, but no ugly numbers
# Also plotting from this file
# Also that e.g. a gradient descent can be easily made from this file. 
# =============================================================================

#load functions
from inpu.load_phys_v4 import phys_gen, phys_try3, phys_GUA1, phys_LOI1
from inpu.load_phys_v4 import phys_LOI_A1, phys_LOI_B1, phys_LOI_C1, phys_LOI_D1
from inpu.load_geo_v3  import geo_try3, geo_LOI1, geo_GUA1, geo_DLW1
from inpu.load_forc_v4 import forc_try4, forc_GUA1, forc_LOI1
from core_v4      import mod1c_g4

import numpy as np

#choose universal constants
constants = phys_gen()
#choose geometry
geo_pars = geo_LOI1()
#choose forcing. Also here: equilibrium or time-dependent
forc_pars = forc_LOI1()
#choose physical constants
phys_pars = phys_LOI_D1()

#set up model environment
run = mod1c_g4(constants, phys_pars, geo_pars, forc_pars)

#solve equations
vers = 'D' #choose version here. 'D' is the full model . 
out4 = run.solve_eqs(vers)

#plot 
run.plot_sst(out4 , run.ii_all)
run.prt_numbers(out4 , run.ii_all)
run.plot_transport(out4 , run.ii_all, vers)
