# =============================================================================
# The idea of this file: run everything from one file. 
# So: choose here everything, but no ugly numbers
# Also plotting from this file
# Also that e.g. a gradient descent can be easily made from this file. 
# =============================================================================

#load functions
from core_v4      import mod1c_g4
import numpy as np

#choosing right modules and stuff happens in settings
import settings

#set up model environment
runD = mod1c_g4(settings.constants, settings.phys_pars, settings.geo_pars, settings.forc_pars)

#solve equations
outD = runD.solve_eqs('D')

#plot 
#run.plot_sst(out[0] , run.ii_all)
#run.plot_sst(out[7] , run.ii_all)

runD.plot_X2(outD , runD.ii_all )

#run.plot_transport(out[0] , run.ii_all, 0)
