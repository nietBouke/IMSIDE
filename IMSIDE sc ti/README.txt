README single channel model 4.5
Bouke

########################################
Version 4.5:  25-01-2024
This is the first working version published online

Note that in this version in theory multiple tidal constituents can be simulated.
However, the use of multiple tidal constituents in this model has never been tested
and I am afraid they are not implemented in a physical appropriate way.

Version 4.6: 23-02-2024
Updated matching conditions at the boundaries of the segments 

Version 4.7: 08-05-2024
Corrected errors in 4.6. 

########################################
Here I explain how to use the code to run the model for the single channel

There are two options: time-independent code and time-dependent code
Time-independent is an equilibrium simulation (but variations on the tidal timescale are present)
This is particularly useful to look at processes and for example effects of variation in the geometry

With the time-dependent code variations in the forcing conditions (discharge) can be taken into account
Note that wind and water level variations at the mouth are not taken into account

In this folder only the time-independent model is present.

########################################
The time-independent code

To run this, select the file 'runfile_ti_v4.py' and run it

Options: to change the input, you can change the files in the folder 'inpu'.
Functions from this folder are imported at the top of the runfile.

There are three input files:
1. 'load_phy'. Here the physcial parameters are specified. They are:
  N   = number of Fourier modes for vertical decomposition (do not change)
  Lsc = length scale (do not change)
  nz  = vertical step for plotting (do not change)
  nt  = time step for plotting (do not change)

  Av_st = subtidal vertical viscosity (tuning parameter)
  Kv_st = subtidal vertical diffusivity (tuning parameter)
  sf_st = subtidal bottom slip (tuning parameter)
  Kh_st = subtidal horizontal diffusivity (tuning parameter)

  These values are lists, as they are different for different tidal constituents
  Av_ti = tidal vertical viscosity (tuning parameter)
  Kv_ti = tidal vertical diffusivity (tuning parameter)
  sf_ti = tidal bottom slip (tuning parameter)
  Kh_ti = tidal horizontal diffusivity (do not change)

  So for the model calibration, these parameters can be changed.
  It are now all constants in the entire domain (except Kh in the sea domain)
  It might be a good idea to change this at some point (i.e. make Av a function of H)
  I advise to use Av_ti and sf_ti for calibration for the tidal water levels
  and Av_st (with Kv_st = Av_st/2.2) and Kh_st for calibration of the subtidal salinity
  The model is calibrated to some extent for the Delaware, Guadalquivir and Loire estuaries
  Note that soc also has an effect.

2. 'load_geo'.  Here the geometry of the estuary is specified
  Ln  = length of the channel segments
  bsn = width of the channel segments (actually, of the connections between the segments)
  dxn = horizontal grid size segments
  Hn  = depth segments


3. 'load_forc'. Here the forcing conditions, i.e. discharge, salinity at the boundaries, and strength of the tides
The parameters mean
  Qriv = the river discharge
  soc = salinity at the sea boundary. Suggest to choose a value between 30-35
  sri = salinity at the river boundary. Typically 0.0-0.5
  tid_comp = the names of the tidal constituents you are including.
  tid_per = the periods of the tide in seconds, 44700 for M2 tide
  a_tide =  the amplitudes of the tide at sea boundaries in m
  p_tide = the phases of the tide at sea boundaries in degrees

########################################

for visualisation options we have:
'plot_sst()': plot subtidal salinity as a function of horizontal and vertical coordinate
'prt_numbers()': print some values like salt intrusion length.
'plot_transport()': plot the different components of the salt transport
