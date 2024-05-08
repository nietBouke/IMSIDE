# =============================================================================
# The idea of this file: run everything from one file. 
# So: choose here everything, but no ugly numbers
# Also plotting from this file
# Also that e.g. a gradient descent can be easily made from this file. 
# =============================================================================

#load functions
from inpu.load_phys_v4 import phys_gen, phys_GUA, phys_DLW, phys_LOI, phys_try1, phys_try2
from inpu.load_geo_v3  import geo_try1, geo_try2, geo_LOI1, geo_GUA1, geo_DLW1, geo_GUA2
from inpu.load_forc_v4 import forc_try1, forc_GUA1, forc_LOI1, forc_DLW1, forc_GUA1Q
from core_v4      import mod1c_g4

import numpy as np

#choose universal constants
constants = phys_gen()
#choose geometry
geo_pars = geo_LOI1()
#choose forcing. Also here: equilibrium or time-dependent
forc_pars = forc_LOI1()
#choose physical constants
phys_pars = phys_LOI()

#set up model environment
run = mod1c_g4(constants, phys_pars, geo_pars, forc_pars)

#solve equations
vers = 'D' #choose version here. 'D' is the full model . 
out4 = run.solve_eqs(vers)
#out4 = run.check_eqs(vers)

#'''
#plot 
run.plot_sst(out4 )
#run.prt_numbers(out4 )
run.plot_transport(out4 , vers)
#run.terms_vert(out4, -12)
#run.terms_vert(out4, -21)
#run.plot_next(out4, vers)

