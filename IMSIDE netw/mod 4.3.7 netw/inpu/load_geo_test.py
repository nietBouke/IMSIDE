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

def geo_test1():
    # =============================================================================
    #     Input: geometry
    # =============================================================================
    ch_gegs = {}
    ch_gegs['C1'] = { 'Name' : 'C1' ,
               'Hn' :  np.array([10,10], dtype=float) ,
               'L'  :  np.array([100000,10000], dtype=float),
               'b'  :  np.array([1000,1000,1000], dtype=float), #one longer than L
               'dx' :  np.array([1000,100], dtype=float)*1, #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'w1'
               }

    ch_gegs['C2'] = { 'Name' : 'C2' ,
               'Hn' : np.array([10], dtype=float),
               'L'  : np.array([200], dtype=float),
               'b'  : np.array([1000,1000], dtype=float), #one longer than L
               'dx' : np.array([50], dtype=float)*1, #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'w2'
               }

    ch_gegs['C3'] = { 'Name' : 'C3' ,
               'Hn' : np.array([10,8,10], dtype=float),
               'L'  : np.array([5000,1000,10000], dtype=float),
               'b'  : np.array([1000,1000,1000,1000], dtype=float), #one longer than L
               'dx' : np.array([100,50,100], dtype=float)*1, #same length as L
               'Ut' : 1 ,
               'loc x=0' : 's1' ,
               'loc x=-L': 'j1'
               }
    '''
    ch_gegs['C3'] = { 'Name' : 'C3' ,
               'Hn' : 10 ,
               'L' : np.array([10000,12000,9000], dtype=float),
               'b' : np.array([500,500,500,500], dtype=float), #one longer than L
               'dx' : np.array([1000/5,1000/5,1000/5], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 's2' ,
               'loc x=-L': 'j1'
               }
    '''
    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================
    ch_gegs['C1']['plot x'] = np.linspace(-ch_gegs['C1']['L'].sum(),0,101)
    ch_gegs['C1']['plot y'] = np.zeros(101)
    ch_gegs['C1']['plot color'] = 'r'

    ch_gegs['C2']['plot x'] = np.linspace(ch_gegs['C2']['L'].sum()+100,100,101)
    ch_gegs['C2']['plot y'] = np.linspace(ch_gegs['C2']['L'].sum()+100,100,101)
    ch_gegs['C2']['plot color'] = 'b'

    ch_gegs['C3']['plot x'] = np.linspace(100,ch_gegs['C3']['L'].sum()+100,101)
    ch_gegs['C3']['plot y'] = -np.linspace(100,ch_gegs['C3']['L'].sum()+100,101)
    ch_gegs['C3']['plot color'] = 'black'

    #ch_gegs['C3']['plot x'] = np.flip(ch_gegs['C3']['plot x'])
    #ch_gegs['C3']['plot y'] = np.flip(ch_gegs['C3']['plot y'])

    return ch_gegs
