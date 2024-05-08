# =============================================================================
#  properties of the delta 
# =============================================================================
import numpy as np
from bs4 import BeautifulSoup
import os
import pyproj 

def kml_subtract_latlon(infile):
    with open(infile, 'r') as f:
        s = BeautifulSoup(f, 'xml')
        for coords in s.find_all('coordinates'):
            # Take coordinate string from KML and break it up into [Lat,Lon,Lat,Lon...] to get CSV row
            space_splits = coords.string.split(" ")
            row = []

            for split in space_splits[:-1]:
                # Note: because of the space between <coordinates>" "-80.123, we slice [1:]
                comma_split = split.split(',')
                # longitude,lattitude,
                row.append([comma_split[0],comma_split[1]])
    row[0][0] = row[0][0][5:]
    return row


    
def geo_RR1():
    #TODO: add width, depth, tidal current in this system
    # =============================================================================
    #     Input: geometry
    # =============================================================================
    ch_gegs = {}

    #get the length later from kml
    ch_gegs['Day 1'] = { 
                'loc x=0' : 's4' ,
               'loc x=-L': 'j4'
               } 

    ch_gegs['Day 2'] = { 
               'loc x=0' : 'j4' ,
               'loc x=-L': 'r2'
               } 
    
    ch_gegs['Red River 1'] = { 

               'loc x=0' : 's2' ,
               'loc x=-L': 'j3'
               } 
    
    ch_gegs['Red River 2'] = { 
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j2'
               } 
    
    ch_gegs['Red River 3'] = { 
               'loc x=0' : 'j2' ,
               'loc x=-L': 'j1'
               } 
    
    ch_gegs['Red River 4'] = { 
               'loc x=0' : 'j1' ,
               'loc x=-L': 'r1'
               } 
    
    ch_gegs['Song Nam Dinh 1'] = { 
               'loc x=0' : 'j4' ,
               'loc x=-L': 'j2'
               } 
    
    ch_gegs['Ninh Co 1'] = { 
               'loc x=0' : 's3' ,
               'loc x=-L': 'j3'
               } 
    
    ch_gegs['Song Tra Ly 1'] = { 
               'loc x=0' : 's1' ,
               'loc x=-L': 'j1'
               } 
    
    #add the properties I do not know
    for key in list(ch_gegs.keys()):
        ch_gegs[key]['Name'] = key
        ch_gegs[key]['H'] = 5 
        ch_gegs[key]['b'] = np.array([1000,1000], dtype=float)
        ch_gegs[key]['dx']= np.array([500], dtype=float)
        ch_gegs[key]['Ut']= 1 
        
    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================
    path_RR = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Geo Red River/'
    list_channels = [f for f in os.listdir(path_RR) if f.endswith('.kml')]
    RR_coords = {}
    for i in range(len(list_channels)):
        RR_coords[list_channels[i][:-4]] = np.array(kml_subtract_latlon(path_RR+list_channels[i]),dtype=float)
        
    cs = ['r','olive','b','black','c','orange','g','m','y']
    count=0
    for key in list(ch_gegs.keys()):
        ch_gegs[key]['plot x'] = np.flip(RR_coords[key][:,0])
        ch_gegs[key]['plot y'] = np.flip(RR_coords[key][:,1])
        ch_gegs[key]['plot color'] = cs[count]
        count+=1
        
    # =============================================================================
    #     #try to use kml files for length
    # =============================================================================
    p = pyproj.Proj(proj='utm',zone=49,ellps='WGS84') #to convert degrees to km. for appropriate zone see https://proj.org/operations/projections/utm.html#cmdoption-arg-zone
    for key in list(ch_gegs.keys()):
        #length of the channel
        dist = np.zeros(len(ch_gegs[key]['plot x']))
        for i in range(1,len(ch_gegs[key]['plot x'])):
            x0,y0 = p(longitude = ch_gegs[key]['plot x'][i],latitude=ch_gegs[key]['plot y'][i])
            x1,y1 = p(longitude = ch_gegs[key]['plot x'][i-1],latitude=ch_gegs[key]['plot y'][i-1])
            dist[i] = dist[i-1]+ ((x0-x1)**2 + (y0-y1)**2)**0.5
        ch_gegs[key]['L'] = np.array([np.round(dist[-1],-3)]) #to remove 
        #Name parameter
        ch_gegs[key]['Name'] = key

    return ch_gegs

    
def geo_RR2():
    #TODO: add width, depth, tidal current in this system
    # =============================================================================
    #     Input: geometry
    # =============================================================================
    ch_gegs = {}

    #get the length later from kml
    ch_gegs['Day 1'] = { 
                'L' : np.array([10000], dtype=float),
                'loc x=0' : 's4' ,
               'loc x=-L': 'j4'
               } 

    ch_gegs['Day 2'] = { 
                'L' : np.array([100000], dtype=float),
                'loc x=0' : 'j4' ,
               'loc x=-L': 'r2'
               } 
    
    ch_gegs['Red River 1'] = { 
                'L' : np.array([10000], dtype=float),
               'loc x=0' : 's2' ,
               'loc x=-L': 'j3'
               } 
    
    ch_gegs['Red River 2'] = { 
                'L' : np.array([10000], dtype=float),
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j2'
               } 
    
    ch_gegs['Red River 3'] = { 
                'L' : np.array([10000], dtype=float),
               'loc x=0' : 'j2' ,
               'loc x=-L': 'j1'
               } 
    
    ch_gegs['Red River 4'] = { 
                'L' : np.array([100000], dtype=float),
               'loc x=0' : 'j1' ,
               'loc x=-L': 'r1'
               } 
    
    ch_gegs['Song Nam Dinh 1'] = {      
               'L' : np.array([10000], dtype=float),
               'loc x=0' : 'j4' ,
               'loc x=-L': 'j2'
               } 
    
    ch_gegs['Ninh Co 1'] = {
                'L' : np.array([10000], dtype=float),
               'loc x=0' : 's3' ,
               'loc x=-L': 'j3'
               } 
    
    ch_gegs['Song Tra Ly 1'] = { 
                'L' : np.array([10000], dtype=float),
               'loc x=0' : 's1' ,
               'loc x=-L': 'j1'
               } 
    
    #add the properties I do not know
    for key in list(ch_gegs.keys()):
        ch_gegs[key]['Name'] = key
        ch_gegs[key]['Hn'] = np.array([5], dtype=float)
        ch_gegs[key]['b'] = np.array([1000,1000], dtype=float)
        ch_gegs[key]['dx']= np.array([500], dtype=float)
        ch_gegs[key]['Ut']= 1 
        
    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================
    path_RR = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Geo Red River/'
    list_channels = [f for f in os.listdir(path_RR) if f.endswith('.kml')]
    RR_coords = {}
    for i in range(len(list_channels)):
        RR_coords[list_channels[i][:-4]] = np.array(kml_subtract_latlon(path_RR+list_channels[i]),dtype=float)
        
    cs = ['r','olive','b','black','c','orange','g','m','y']
    count=0
    for key in list(ch_gegs.keys()):
        ch_gegs[key]['plot x'] = np.flip(RR_coords[key][:,0])
        ch_gegs[key]['plot y'] = np.flip(RR_coords[key][:,1])
        ch_gegs[key]['plot color'] = cs[count]
        count+=1

    '''
    # =============================================================================
    # do some adjustments to see if the code breaks
    # =============================================================================
    
    key = 'Ninh Co 1'
    ch_gegs[key]['L']= np.array([20000], dtype=float)
    ch_gegs[key]['H'] = 9 
    ch_gegs[key]['b'] = np.array([1000,2000], dtype=float)
    ch_gegs[key]['dx']= np.array([200], dtype=float)
    
    
    key = 'Ninh Co 1'
    ch_gegs[key]['L']= np.array([2000,5000,2000], dtype=float)
    ch_gegs[key]['H'] = 8 
    ch_gegs[key]['b'] = np.array([1000,500,1000], dtype=float)
    ch_gegs[key]['dx']= np.array([200,500,100], dtype=float)
    
    
    key = 'Ninh Co 1'
    ch_gegs[key]['L']= np.array([10000,1000], dtype=float)
    ch_gegs[key]['H'] = 7 
    ch_gegs[key]['b'] = np.array([1000,2000,3000], dtype=float)
    ch_gegs[key]['dx']= np.array([500,100], dtype=float)
    '''


    return ch_gegs


 
# =============================================================================
# physical properties of the delta
# =============================================================================
# def inp_RedRiver1_phys():
#     Qriv = [200,100] #mind the sign! this is the discharge at r1,r2,...
#     Qweir= []
#     Qhar = []
#     n_sea = [0,0,0,0] #this is the water level at s1,s2,...
#     soc = [35,35,35,35] #this is salinity at s1,s2, ...
#     swe = []
#     sri = [0,0] #this is the salinity of the river water at r1,r2, ...
#     a_tide = [1.7,1.5,1,1.2] #this is the amplitude of the tide at s1,s2, ...
#     p_tide = [0,0,0,0] #this is the phase of the tide at s1,s2, ...

#     return Qriv, Qweir, Qhar,n_sea, soc, sri, swe, a_tide, p_tide

# def inp_RedRiver2_phys():
#     Qriv = [500,100] #mind the sign! this is the discharge at r1,r2,...
#     Qweir= []
#     Qhar = []
#     n_sea = [0,0,0,0] #this is the water level at s1,s2,...
#     soc = [35,35,35,35] #this is salinity at s1,s2, ...
#     swe = []
#     sri = [0,0] #this is the salinity of the river water at r1,r2, ...
#     a_tide = [1.7,1.5,1,1.2] #this is the amplitude of the tide at s1,s2, ...
#     p_tide = [20,10,0,2] #this is the phase of the tide at s1,s2, ...

#     return Qriv, Qweir, Qhar,n_sea, soc, sri, swe, a_tide, p_tide




