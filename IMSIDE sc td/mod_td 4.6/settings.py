# =============================================================================
# choose here which files to use - not sure if I am happy with this setup
# =============================================================================


from inpu.load_phys_v4 import phys_gen, phys_all3est, phys_all3est_v2, phys_DLW, phys_GUA, phys_LOI
from inpu.load_phys_v4 import phys_GUA
from inpu.load_geo_v4  import geo_GUA2, geo_DLW2, geo_LOI2
from inpu.load_forc_td_v4 import forc_GUA1, forc_DLW1, forc_try1, forc_DLW2, forc_GUA2, forc_LOI1, forc_LOI2
'''

# =============================================================================
# Delaware
# =============================================================================
date_start, date_stop = '2023-04-01' , '2023-11-01'

#choose physical constants
constants = phys_gen()
#choose geometry
geo_pars = geo_DLW2()
#choose forcing.
forc_pars = forc_DLW2(date_start, date_stop)
#choose physical constants
phys_pars = phys_DLW()
'''
# =============================================================================
# Guadalquivir
# =============================================================================
date_start, date_stop = '2009-04-01' , '2009-11-01'

#choose physical constants
constants = phys_gen()
#choose geometry
geo_pars = geo_GUA2()
#choose forcing.
forc_pars = forc_GUA1()
#forc_pars = forc_GUA2(date_start, date_stop)
#choose physical constants
phys_pars = phys_GUA()

'''
# =============================================================================
# Loire
# =============================================================================
#date_start, date_stop = '2013-06-01' , '2013-07-01'
date_start, date_stop = '2015-04-01' , '2015-11-01'

#choose physical constants
constants = phys_gen()
#choose geometry
geo_pars = geo_LOI2()
#choose forcing.
forc_pars = forc_LOI2(date_start, date_stop)
#choose physical constants
phys_pars = phys_LOI()



# =============================================================================
# test
# =============================================================================
#date_start, date_stop = '2013-06-01' , '2013-07-01'
#date_start, date_stop = '2013-06-01' , '2017-06-01'

#choose physical constants
constants = phys_gen()
#choose geometry
geo_pars = geo_GUA2()
#choose forcing.
forc_pars = forc_try1()
#choose physical constants
phys_pars = phys_GUA()
'''
