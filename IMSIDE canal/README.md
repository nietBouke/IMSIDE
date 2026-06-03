This is the code to simulate salt intrusion in a channel that is closed off with a lock. Note that the dynamics around the shipping lock are represented by a simple formulation. 

To run the code, run the file run_IMSIDE_canal.py

You can adjust the parameters in setti_IMSIDE_canal.py
The function set_par_num1 determines some numerical settings. In general, these are fine and there is no need to change them. 
The function set_par_time1 determines the number of timesteps (T) and the timestep in seconds (dt). Note that dt should be a list with length T; this means that you can have a variable timestep
The function set_par_geo1 sets the geometry of your canal. The canal is divided in section, which all have a length (set in Ln), a horizontal grid size (dxn) and a depth (Hn). The width is determined by setting the width at the beginning of the segment, and the width at the end of the segment, which is also the value for the width at the beginning of the next segment. This implies that the length of the vector bn is 1 longer than the number of segments. The width is assumed to vary exponentially within the sections. 
The function set_par_lock1 sets the parameters that are used for the formulation of the shippng lock. The given values follow from calibration with the Terneuzen shipping lock system (before the Nieuwe Sluis Terneuzen became operational). 
The function set_par_phys1 gives the values of the parameters in the turbulence closure and the bottom friction. 
The function set_par_bc1 gives the boundary conditions, that is, timeseries for discharge (Qi) and exterior salinity (soc). Note that tributaries can be incorporated by setting a spatially varying discharge. 

The file visu_IMSIDE_canal.py gives some functions to visualise the results. 

The rest of the files describe the system of equations that is solved, and the solution procedure. They should not be changed in general. 



