README network model 4.2 and beyond
Bouke

########################################
Version 4.2.6:  17-01-2024
This is the first working version 

Version 4.2.7:  25-01-2024
Changes: different setup of the analytical equations for tidal salinity
Also tried to make the code faster, but this did not result in much improvement. 
It seems like I fixed a small error in the time-dependent simulation, but I am not so sure which I changed. Anyway, small numerical instabilities have disappeared. 

Version 4.3.1: 23-02-2024
Include different depth for different segments
Changed (corrected) minus signs for vertical balance, in accordance with the changes done to the single channel code. 
Also, use average depth at junctions and segment boundaries for the transport at depth. Is an improvement in stability

Version 4.3.4: 08-05-2024
Corrected problems with transport at depth at junctions and boundaries. 
Also, made an option to set bottom friction to the formulation used in the single channel models 
Standard choice for theta set to 1. (Backward Euler time integration instead of Crank-Nicolson). This is an improvement in numerical stability. 

Version 4.3.7: 19-08-2024
Introduced options for turbulence closure. 
Avoid the problematic behaviour encountered in earlier versions when a channel segment is very short. However, one has to be careful when a channel segment becomes very short. 
For some turbulence closures, mixing coefficients vary per channel or segment. 
Model is re-calibrated for the RMD. Agreement in terms of tidal water levels is better, agreement in terms of salinity is comparable to Version 4.3.4


########################################


Here I explain how to use the code to run the model for the network

There are two options: time-independent code and time-dependent code
Time-independent is an equilibrium simulation (but variations on the tidal timescale are present)
This is particularly useful to look at processes and for example effects of variation in the geometry

With the time-dependent code variations in the forcing conditions (discharge) can be taken into account
Note that wind and water level variations at the mouth are not taken into account

########################################
The time-independent code

To run this, select the file 'runfile_ti_v1.py' and run it

Options: to change the input, you can change the files 'settings_ti_v1.py', and everything in the folder 'inpu'
The file 'settings_ti_v1.py' calls functions from the folder 'inpu', where the input is defined.

There are three input files:
1. 'load_physics.py'. Here the physcial parameters are specified. They are:
  N   = number of Fourier modes for vertical decomposition (do not change)
  Lsc = length scale (do not change)
  nz  = vertical step for plotting (do not change)
  nt  = time step for plotting (do not change)
  theta = for theta method in time integration (do not change)

  Av_st = subtidal vertical viscosity (tuning parameter)
  Kv_st = subtidal vertical diffusivity (tuning parameter)
  sf_st = subtidal bottom slip (tuning parameter)
  Kh_st = subtidal horizontal diffusivity (tuning parameter)

  Av_ti = tidal vertical viscosity (tuning parameter)
  Kv_ti = tidal vertical diffusivity (tuning parameter)
  sf_ti = tidal bottom slip (tuning parameter)
  Kh_ti = tidal horizontal diffusivity (do not change)

  So for the model calibration, these parameters can be changed.
  It are now all constants in the entire domain (except Kh in the sea domain)
  It might be a good idea to change this at some point (i.e. make Av a function of H)
  I advise to use Av_ti and sf_ti for calibration for the tidal water levels
  and Av_st (with Kv_st = Av_st/2.2) and Kh_st for calibration of the subtidal salinity
  I think I placed here already the values wich I got from calibration
  Note that soc also has an effect.

2. load_geo_XXX
  In these files, the geometry of the delta under consideration is defined
  For the RMD system, I advise to use geo_RMD9.
  For each channel, you have to define what the length, width, and depth are, the horizontal step, and where they are connected to
  abbreviations:
  r = river
  w = weir
  h = haringvliet (outflow boundary)
  s = sea
  j = junction
  Also the coordinates from the channels are loaded.
  I think small adjustments to the channels are obvious to do. Larger adjustments to the network are also possible,
  if you want to do this but do not know how, please discuss it with me, I am happy to explain this.


3. 'load_forcing_ti'. Here the forcing conditions, i.e. discharge, salinity at the boundaries, and strength of the tides
The parameters mean
  Qriv = the discharge at rivers r1,r2,...
  Qweir= the discharge at weirs w1,w2,...
  Qhar = the discharge at outflow boundaries h1,h2,...
  n_sea = the subtidal water level at sea boundaries s1,s2,...
  soc = salinity at sea boundaries s1,s2, .... Suggest to choose a value between 30-35
  sri = salinity at river boundaries r1,r2, ...
  swe = salinity at weir boundaries w1,w2, ...
  tid_per = the period of the tide in seconds, 44700 for M2 tide
  a_tide =  the amplitude of the tide at sea boundaries s1,s2, ...
  p_tide = the phase of the tide at sea boundaries s1,s2, ...

for visualisation options we have:
'plot_s_gen()': plot subtidal depth-averaged salinity in all channels
'plot_procRM()': plot processes in the Rhine-Meuse delta
'plot_tide_pointRM()': plot the tidal water levels, amplitude and phase, in the Rhine-Meuse, and compare with observations.
Note: the agreement of amplitudes is quite good, however there is a shift in phases. I did not have time to figure out where it came from
'anim_new_compRM('name')': make an animation of the tides in the subtidal model.


######################################################
The time-dependent model

To run this, select the file 'runfile_td_v1.py' and run it

Options: to change the input, you can change the files 'settings_td_v1.py', and everything in the folder 'inpu'
The file 'settings_td_v1.py' calls functions from the folder 'inpu', where the input is defined

Everything is similar as in the time-independent code, except that the input files are lists of forcing conditions.
You have also a few new variabls in 'load_forcing_td':
  T  = the number of timesteps
  DT = the timesteps in seconds. I usually work with timesteps of one day, and recommend to keep it this way.

Visualisation option are:
- Timeseries of salinity at a certain point. For this, use the function plot_salt_pointRM. Inputs are: longitude, lattitude, and depth.
  The code will look for the closest point in the model domain for your geographic coordinates
  Note that it is possible for this code to simulate negative salinity (on tidal timescales), which might appear in your results.
  You might want to set the negative values to zero

- Animation of time simulation. For this, run the function anim_RM_st. Input is only the name of the file you are creating.
  Tidal effects are not visualised here.
