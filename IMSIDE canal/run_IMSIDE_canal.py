# =============================================================================
# script to run IMSIDE for the canal case 
# as in Biemond et al. (2024) https://doi.org/10.1016/j.ecss.2024.108654
# =============================================================================
#import the model from the other files 
from core_IMSIDE_canal import mod_SIcan
import setti_IMSIDE_canal
from visu_IMSIDE_canal import calc_rawtofine, plot_timeseries_point, plot_saltcontour, plot_timeseries_sil

#set up the parameters for the model. You can change the functions in the setti file to change this, or make new functions of course
pnum = setti_IMSIDE_canal.set_par_num1()
ptim = setti_IMSIDE_canal.set_par_time1()
pgeo = setti_IMSIDE_canal.set_par_geo1()
plok = setti_IMSIDE_canal.set_par_lock1()
pbc = setti_IMSIDE_canal.set_par_bc1(ptim[0], setti_IMSIDE_canal.compute_nx(pgeo[0],pgeo[2]))
pmix = setti_IMSIDE_canal.set_par_phys1(ptim[0], setti_IMSIDE_canal.compute_nx(pgeo[0],pgeo[2]))

#initialisation
IMSIDE_init = mod_SIcan(*setti_IMSIDE_canal.set_to_init(pbc, pnum, pgeo, pmix, ptim, plok))
IMSIDE_init.run_func(None, setup = True)
stzero = IMSIDE_init.sss_save[-1]

#load the model
IMSIDE_canal = mod_SIcan(pbc, pnum, pgeo, pmix, ptim, plok)
#run the model
IMSIDE_canal.run_func(stzero)

#do some visualisation
saltfield = calc_rawtofine(IMSIDE_canal)
plot_timeseries_point(IMSIDE_canal, saltfield, -5, 0)
plot_timeseries_point(IMSIDE_canal, saltfield, -25, -10)
# #%%
for t in [0,10,20,50]:
    plot_saltcontour(IMSIDE_canal, saltfield, t)

# plot_timeseries_sil(IMSIDE_canal, saltfield)
