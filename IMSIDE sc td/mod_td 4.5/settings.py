# =============================================================================
# choose here which files to use - not sure if I am happy with this setup
# =============================================================================


from inpu.load_phys_v4 import phys_gen, phys_LOI1
from inpu.load_phys_v4 import phys_LOI_A1, phys_LOI_B1, phys_LOI_C1, phys_LOI_D1
from inpu.load_geo_v4  import geo_GUA1, geo_DLW1, geo_DLW2, geo_try1, geo_LOI1, geo_LOI2
from inpu.load_forc_td_v4 import forc_GUA1, forc_DLW1, forc_try1, forc_DLW2, forc_GUA2, forc_LOI1, forc_LOI2


date_start, date_stop = '2011-06-01' , '2011-07-01'

#choose physical constants
constants = phys_gen()
#choose geometry
geo_pars = geo_LOI2()
#choose forcing. 
forc_pars = forc_LOI1()#date_start, date_stop)
#choose physical constants
phys_pars = phys_LOI_D1()
