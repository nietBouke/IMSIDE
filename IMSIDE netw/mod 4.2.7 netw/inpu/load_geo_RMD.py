# =============================================================================
#  properties of the Rhine-Meuse delta
# this file contains dictionaires where the data of the channels in a network 
# are specified, e.g. widht, depth, length, and where they are connected to
# also their coordinates, for plotting purposes, are specified. 
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

def geo_RMD1():
    # =============================================================================
    #     Input: geometry
    # =============================================================================
    ch_gegs = {}
    ch_gegs['Hollandse IJssel'] =  { 'Name' : 'Hollandse IJssel' , #1
               'H' : 4. ,
               'L' : np.array([19700], dtype=float),
               'b' : np.array([45,150], dtype=float), #one longer than L
               'dx' : np.array([197], dtype=float), #same length as L
               'loc x=0' : 'j2' ,
               'loc x=-L': 'w1'
               }

    ch_gegs['Lek'] =  { 'Name' : 'Lek' , #2
               'H' : 5.3 ,
               'L' : np.array([32000,10000], dtype=float),
               'b' : np.array([136,200,260], dtype=float), #one longer than L
               'dx' : np.array([1000,250], dtype=float), #same length as L
               'loc x=0' : 'j1' ,
               'loc x=-L': 'w2'
               }

    ch_gegs['Nieuwe Maas 2 old'] = { 'Name' : 'Nieuwe Maas 2' , #3
               'H' : 8.1 ,
               'L' : np.array([4900], dtype=float),
               'b' : np.array([250,400], dtype=float), #one longer than L
               'dx' : np.array([350], dtype=float), #same length as L
               'loc x=0' : 'j2' ,
               'loc x=-L': 'j1'
               }
    
    ch_gegs['Nieuwe Maas 1 old'] = { 'Name' : 'Nieuwe Maas 1' , #4
               'H' : 11 , #wel echt gemiddeld dit
               'L' : np.array([18750], dtype=float),
               'b' : np.array([400,500], dtype=float), #one longer than L
               'dx' : np.array([375], dtype=float), #same length as L
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j2'
               }    

    ch_gegs['Nieuwe Waterweg v2'] =  { 'Name' : 'Nieuwe Waterweg' , #5
               'H' : 16 ,
               'L' : np.array([16800], dtype=float),
               'b' : np.array([600,600], dtype=float), #one longer than L
               'dx' : np.array([200], dtype=float), #same length as L
               'loc x=0' : 'j12' ,
               'loc x=-L': 'j3'
               }

    ch_gegs['Noord'] =  { 'Name' : 'Noord' , #6
               'H' : 7 , #moeilijk vast te stellen
               'L' : np.array([8600], dtype=float),
               'b' : np.array([250,220], dtype=float), #one longer than L
               'dx' : np.array([430], dtype=float), #same length as L
               'loc x=0' : 'j6' ,
               'loc x=-L': 'j1'
               }

    ch_gegs['Oude Maas 1'] =  { 'Name' : 'Oude Maas 1' , #7
               'H' : 13 ,
               'L' : np.array([3100], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([310], dtype=float), #same length as L
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j4'
               }

    ch_gegs['Hartelkanaal v2'] =  { 'Name' : 'Hartelkanaal' ,#8
               'H' : 7.6,
               'L' : np.array([21200,1500,3000], dtype=float),
               'b' : np.array([310,310,600,1500], dtype=float), #one longer than L
               'dx' : np.array([200,100,300], dtype=float), #same length as L
               'loc x=0' : 'j12' ,
               'loc x=-L': 'j4'
               }

    ch_gegs['Breeddiep'] = { 'Name' : 'Breeddiep' ,#8
               'H' : 16 ,
               'L' : np.array([2500], dtype=float),
               'b' : np.array([1200,1200], dtype=float), #one longer than L
               'dx' : np.array([250], dtype=float), #same length as L
               'loc x=0' : 's1' ,
               'loc x=-L': 'j12'
               }

    ch_gegs['Oude Maas 2'] =  { 'Name' : 'Oude Maas 2' , #9
               'H' : 14 ,
               'L' : np.array([8250], dtype=float),
               'b' : np.array([317,317], dtype=float), #one longer than L
               'dx' : np.array([250], dtype=float), #same length as L
               'loc x=0' : 'j4' ,
               'loc x=-L': 'j8'
               }

    ch_gegs['Waal'] =  { 'Name' : 'Waal' , #10, Boven Merwede in bestand
               'H' : 4 ,
               'L' : np.array([45200], dtype=float),
               'b' : np.array([214,500], dtype=float), #one longer than L
               'dx' : np.array([1130], dtype=float), #same length as L
               'loc x=0' : 'j5' ,
               'loc x=-L': 'r1'
               }

    ch_gegs['Beneden Merwede'] =  { 'Name' : 'Beneden Merwede' , #11
               'H' : 6 ,
               'L' : np.array([15300], dtype=float),
               'b' : np.array([200/2,300], dtype=float), #one longer than L #groins not taken into account!
               'dx' : np.array([510], dtype=float), #same length as L
               'loc x=0' : 'j6' ,
               'loc x=-L': 'j5'
               }

    ch_gegs['Oude Maas 4'] =  { 'Name' : 'Oude Maas 4' , #12
               'H' : 7 ,
               'L' : np.array([4260], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([426], dtype=float), #same length as L
               'loc x=0' : 'j7' ,
               'loc x=-L': 'j6'
               }

    ch_gegs['Oude Maas 3'] =  { 'Name' : 'Oude Maas 3' , #13
               'H' : 10.2 ,
               'L' : np.array([15000], dtype=float),
               'b' : np.array([240,350], dtype=float), #one longer than L
               'dx' : np.array([500], dtype=float), #same length as L
               'loc x=0' : 'j8' ,
               'loc x=-L': 'j7'
               }

    ch_gegs['Nieuwe Merwede'] =  { 'Name' : 'Nieuwe Merwede' , #14
               'H' : 5 ,
               'L' : np.array([19600], dtype=float),
               'b' : np.array([400,730], dtype=float), #one longer than L
               'dx' : np.array([1960], dtype=float), #same length as L
               'loc x=0' : 'j9' ,
               'loc x=-L': 'j5'
               }

    ch_gegs['Dordtsche Kil'] =  { 'Name' : 'Dordtsche Kil' , #15
               'H' : 10.7 ,
               'L' : np.array([9400], dtype=float),
               'b' : np.array([300,300], dtype=float), #one longer than L
               'dx' : np.array([470], dtype=float), #same length as L
               'loc x=0' : 'j7' ,
               'loc x=-L': 'j10'
               }

    ch_gegs['Spui'] =  { 'Name' : 'Spui' , #16
               'H' :  6.4,
               'L' : np.array([17400,7400], dtype=float),
               'b' : np.array([250,250,250], dtype=float), #one longer than L
               'dx' : np.array([855,370], dtype=float), #same length as L
               'loc x=0' : 'j8' ,
               'loc x=-L': 'j11'
               }

    ch_gegs['Maas'] =  { 'Name' : 'Maas' , #17
               'H' : 5.3 ,
               'L' : np.array([62300], dtype=float),
               'b' : np.array([97,416], dtype=float), #one longer than L
               'dx' : np.array([6230], dtype=float), #same length as L
               'loc x=0' : 'j9' ,
               'loc x=-L': 'r2'
               }

    ch_gegs['Hollands Diep 2'] =  { 'Name' : 'Hollands Diep 2' , #18
               'H' : 6.2 ,
               'L' : np.array([3810], dtype=float),
               'b' : np.array([1600,1100], dtype=float), #one longer than L
               'dx' : np.array([381], dtype=float), #same length as L
               'loc x=0' : 'j10' ,
               'loc x=-L': 'j9'
               }

    ch_gegs['Hollands Diep 1'] =  { 'Name' : 'Hollands Diep 1' , #19
               'H' : 7.6 ,
               'L' : np.array([32000], dtype=float),
               'b' : np.array([1630,2000], dtype=float), #one longer than L
               'dx' : np.array([3200], dtype=float), #same length as L
               'loc x=0' : 'j11' ,
               'loc x=-L': 'j10'
               }

    ch_gegs['Haringvliet'] =  { 'Name' : 'Haringvliet' , #20
               'H' : 8.7 ,
               'L' : np.array([11700], dtype=float),
               'b' : np.array([2420,2420], dtype=float), #one longer than L
               'dx' : np.array([900], dtype=float), #same length as L
               'loc x=0' : 'h1' ,
               'loc x=-L': 'j11'
               }

    #optional: fix the tides
    for key in list(ch_gegs.keys()): ch_gegs[key]['Ut']= 1

    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================

    path_RM = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Geo_RijnMaas/'
    list_channels = [f for f in os.listdir(path_RM) if f.endswith('.kml')]
    RM_coords = {}
    for i in range(len(list_channels)):
        RM_coords[list_channels[i][:-4]] = np.array(kml_subtract_latlon(path_RM+list_channels[i]),dtype=float)

    cs = ['r','olive','b','dimgray','c','orange','g','m','y','gold','indigo','tan','skyblue','maroon','peru',\
          'beige','plum','silver','forestgreen','darkkhaki','rosybrown','teal','snow','aliceblue','mintcream']
    count=0
    for key in list(ch_gegs.keys()):
        ch_gegs[key]['plot x'] = np.flip(RM_coords[key][:,0])
        ch_gegs[key]['plot y'] = np.flip(RM_coords[key][:,1])
        ch_gegs[key]['plot color'] = cs[count]
        count+=1

    return ch_gegs

def geo_RMD2():
    # =============================================================================
    #     Input: geometry
    # =============================================================================
    ch_gegs = {}
    ch_gegs['Hollandse IJssel'] =  { 'Name' : 'Hollandse IJssel' , #1
               'H' : 4. ,
               'L' : np.array([19700], dtype=float),
               'b' : np.array([45,150], dtype=float), #one longer than L
               'dx' : np.array([197], dtype=float), #same length as L
               'loc x=0' : 'j2' ,
               'loc x=-L': 'w1'
               }

    ch_gegs['Lek'] =  { 'Name' : 'Lek' , #2
               'H' : 5.3 ,
               'L' : np.array([32000,10000], dtype=float),
               'b' : np.array([136,200,260], dtype=float), #one longer than L
               'dx' : np.array([1000,250], dtype=float), #same length as L
               'loc x=0' : 'j1' ,
               'loc x=-L': 'w2'
               }

    ch_gegs['Nieuwe Maas 2 old'] = { 'Name' : 'Nieuwe Maas 2' , #3
               'H' : 8.1 ,
               'L' : np.array([4900], dtype=float),
               'b' : np.array([250,400], dtype=float), #one longer than L
               'dx' : np.array([100], dtype=float), #same length as L
               'loc x=0' : 'j2' ,
               'loc x=-L': 'j1'
               }
    
    ch_gegs['Nieuwe Maas 1 old'] = { 'Name' : 'Nieuwe Maas 1' , #4
               'H' : 11 , #wel echt gemiddeld dit
               'L' : np.array([18750], dtype=float),
               'b' : np.array([400,500], dtype=float), #one longer than L
               'dx' : np.array([250], dtype=float), #same length as L
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j2'
               }    

    ch_gegs['Nieuwe Waterweg v2'] =  { 'Name' : 'Nieuwe Waterweg' , #5
               'H' : 16 ,
               'L' : np.array([16800], dtype=float),
               'b' : np.array([600,600], dtype=float), #one longer than L
               'dx' : np.array([50], dtype=float), #same length as L
               'loc x=0' : 'j12' ,
               'loc x=-L': 'j3'
               }

    ch_gegs['Noord'] =  { 'Name' : 'Noord' , #6
               'H' : 7 , #moeilijk vast te stellen
               'L' : np.array([8600], dtype=float),
               'b' : np.array([250,220], dtype=float), #one longer than L
               'dx' : np.array([430], dtype=float), #same length as L
               'loc x=0' : 'j6' ,
               'loc x=-L': 'j1'
               }

    ch_gegs['Oude Maas 1'] =  { 'Name' : 'Oude Maas 1' , #7
               'H' : 13 ,
               'L' : np.array([3100], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([100], dtype=float), #same length as L
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j4'
               }


    ch_gegs['Hartelkanaal v2'] =  { 'Name' : 'Hartelkanaal' ,#8
               'H' : 7.6,
               'L' : np.array([22200,500,3000], dtype=float),
               'b' : np.array([310,310,600,1500], dtype=float), #one longer than L
               'dx' : np.array([200,100,100], dtype=float), #same length as L
               'loc x=0' : 'j12' ,
               'loc x=-L': 'j4'
               }

    ch_gegs['Breeddiep'] = { 'Name' : 'Breeddiep' ,#8
               'H' : 16 ,
               'L' : np.array([2500], dtype=float),
               'b' : np.array([1200,1200], dtype=float), #one longer than L
               'dx' : np.array([50], dtype=float), #same length as L
               'loc x=0' : 's1' ,
               'loc x=-L': 'j12'
               }

    ch_gegs['Oude Maas 2'] =  { 'Name' : 'Oude Maas 2' , #9
               'H' : 14 ,
               'L' : np.array([8250], dtype=float),
               'b' : np.array([317,317], dtype=float), #one longer than L
               'dx' : np.array([125], dtype=float), #same length as L
               'loc x=0' : 'j4' ,
               'loc x=-L': 'j8'
               }

    ch_gegs['Waal'] =  { 'Name' : 'Waal' , #10, Boven Merwede in bestand
               'H' : 4 ,
               'L' : np.array([45200], dtype=float),
               'b' : np.array([214,500], dtype=float), #one longer than L
               'dx' : np.array([1130], dtype=float), #same length as L
               'loc x=0' : 'j5' ,
               'loc x=-L': 'r1'
               }

    ch_gegs['Beneden Merwede'] =  { 'Name' : 'Beneden Merwede' , #11
               'H' : 6 ,
               'L' : np.array([15300], dtype=float),
               'b' : np.array([200/2,300], dtype=float), #one longer than L #groins not taken into account!
               'dx' : np.array([510], dtype=float), #same length as L
               'loc x=0' : 'j6' ,
               'loc x=-L': 'j5'
               }

    ch_gegs['Oude Maas 4'] =  { 'Name' : 'Oude Maas 4' , #12
               'H' : 7 ,
               'L' : np.array([4260], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([426], dtype=float), #same length as L
               'loc x=0' : 'j7' ,
               'loc x=-L': 'j6'
               }

    ch_gegs['Oude Maas 3'] =  { 'Name' : 'Oude Maas 3' , #13
               'H' : 10.2 ,
               'L' : np.array([15000], dtype=float),
               'b' : np.array([240,350], dtype=float), #one longer than L
               'dx' : np.array([500], dtype=float), #same length as L
               'loc x=0' : 'j8' ,
               'loc x=-L': 'j7'
               }

    ch_gegs['Nieuwe Merwede'] =  { 'Name' : 'Nieuwe Merwede' , #14
               'H' : 5 ,
               'L' : np.array([19600], dtype=float),
               'b' : np.array([400,730], dtype=float), #one longer than L
               'dx' : np.array([400], dtype=float), #same length as L
               'loc x=0' : 'j9' ,
               'loc x=-L': 'j5'
               }

    ch_gegs['Dordtsche Kil'] =  { 'Name' : 'Dordtsche Kil' , #15
               'H' : 10.7 ,
               'L' : np.array([9400], dtype=float),
               'b' : np.array([300,300], dtype=float), #one longer than L
               'dx' : np.array([470], dtype=float), #same length as L
               'loc x=0' : 'j7' ,
               'loc x=-L': 'j10'
               }

    ch_gegs['Spui'] =  { 'Name' : 'Spui' , #16
               'H' :  6.4,
               'L' : np.array([17400,7400], dtype=float),
               'b' : np.array([250,250,250], dtype=float), #one longer than L
               'dx' : np.array([1000,185], dtype=float), #same length as L
               'loc x=0' : 'j8' ,
               'loc x=-L': 'j11'
               }

    ch_gegs['Maas'] =  { 'Name' : 'Maas' , #17
               'H' : 5.3 ,
               'L' : np.array([62300], dtype=float),
               'b' : np.array([97,416], dtype=float), #one longer than L
               'dx' : np.array([623], dtype=float), #same length as L
               'loc x=0' : 'j9' ,
               'loc x=-L': 'r2'
               }

    ch_gegs['Hollands Diep 2'] =  { 'Name' : 'Hollands Diep 2' , #18
               'H' : 6.2 ,
               'L' : np.array([3810], dtype=float),
               'b' : np.array([1600,1100], dtype=float), #one longer than L
               'dx' : np.array([381], dtype=float), #same length as L
               'loc x=0' : 'j10' ,
               'loc x=-L': 'j9'
               }

    ch_gegs['Hollands Diep 1'] =  { 'Name' : 'Hollands Diep 1' , #19
               'H' : 7.6 ,
               'L' : np.array([32000], dtype=float),
               'b' : np.array([1630,2000], dtype=float), #one longer than L
               'dx' : np.array([1000], dtype=float), #same length as L
               'loc x=0' : 'j11' ,
               'loc x=-L': 'j10'
               }

    ch_gegs['Haringvliet'] =  { 'Name' : 'Haringvliet' , #20
               'H' : 8.7 ,
               'L' : np.array([11700], dtype=float),
               'b' : np.array([2420,2420], dtype=float), #one longer than L
               'dx' : np.array([900], dtype=float), #same length as L
               'loc x=0' : 'h1' ,
               'loc x=-L': 'j11'
               }

    #optional: fix the tides
    for key in list(ch_gegs.keys()): 
        #ch_gegs[key]['H']= 12
        ch_gegs[key]['Ut']= 1

    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================

    path_RM = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Geo_RijnMaas/'
    list_channels = [f for f in os.listdir(path_RM) if f.endswith('.kml')]
    RM_coords = {}
    for i in range(len(list_channels)):
        RM_coords[list_channels[i][:-4]] = np.array(kml_subtract_latlon(path_RM+list_channels[i]),dtype=float)

    cs = ['r','olive','b','dimgray','c','orange','g','m','y','gold','indigo','tan','skyblue','maroon','peru',\
          'beige','plum','silver','forestgreen','darkkhaki','rosybrown','teal','snow','aliceblue','mintcream']
    count=0
    for key in list(ch_gegs.keys()):
        ch_gegs[key]['plot x'] = np.flip(RM_coords[key][:,0])
        ch_gegs[key]['plot y'] = np.flip(RM_coords[key][:,1])
        ch_gegs[key]['plot color'] = cs[count]
        count+=1

    return ch_gegs

def geo_RMD3():
    # =============================================================================
    #     Input: geometry
    # =============================================================================
    ch_gegs = {}
    ch_gegs['Hollandse IJssel'] =  { 'Name' : 'Hollandse IJssel' , #1
               'H' : 4. ,
               'L' : np.array([16000,3700], dtype=float),
               'b' : np.array([45,127,150], dtype=float), #one longer than L
               'dx' : np.array([250,100], dtype=float), #same length as L
               'loc x=0' : 'j2' ,
               'loc x=-L': 'w1'
               }

    ch_gegs['Lek'] =  { 'Name' : 'Lek' , #2
               'H' : 5.3 ,
               'L' : np.array([32000,10000], dtype=float),
               'b' : np.array([136,200,260], dtype=float), #one longer than L
               'dx' : np.array([1000,250], dtype=float), #same length as L
               'loc x=0' : 'j1' ,
               'loc x=-L': 'w2'
               }

    ch_gegs['Nieuwe Maas 2 old'] = { 'Name' : 'Nieuwe Maas 2' , #3
               'H' : 8.1 ,
               'L' : np.array([4900], dtype=float),
               'b' : np.array([250,400], dtype=float), #one longer than L
               'dx' : np.array([100], dtype=float), #same length as L
               'loc x=0' : 'j2' ,
               'loc x=-L': 'j1'
               }
    
    ch_gegs['Nieuwe Maas 1 old'] = { 'Name' : 'Nieuwe Maas 1' , #4
               'H' : 11 , #wel echt gemiddeld dit
               'L' : np.array([18750], dtype=float),
               'b' : np.array([400,500], dtype=float), #one longer than L
               'dx' : np.array([250], dtype=float), #same length as L
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j2'
               }    

    ch_gegs['Nieuwe Waterweg v2'] =  { 'Name' : 'Nieuwe Waterweg' , #5
               'H' : 16 ,
               'L' : np.array([16800], dtype=float),
               'b' : np.array([600,600], dtype=float), #one longer than L
               'dx' : np.array([50], dtype=float), #same length as L
               'loc x=0' : 'j12' ,
               'loc x=-L': 'j3'
               }

    ch_gegs['Noord'] =  { 'Name' : 'Noord' , #6
               'H' : 7 , #moeilijk vast te stellen
               'L' : np.array([8600], dtype=float),
               'b' : np.array([250,220], dtype=float), #one longer than L
               'dx' : np.array([430], dtype=float), #same length as L
               'loc x=0' : 'j6' ,
               'loc x=-L': 'j1'
               }

    ch_gegs['Oude Maas 1'] =  { 'Name' : 'Oude Maas 1' , #7
               'H' : 13 ,
               'L' : np.array([3100], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([100], dtype=float), #same length as L
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j4'
               }


    ch_gegs['Hartelkanaal v2'] =  { 'Name' : 'Hartelkanaal' ,#8
               'H' : 7.6,
               'L' : np.array([20000,2700,3000], dtype=float),
               'b' : np.array([310,310,600,1500], dtype=float), #one longer than L
               'dx' : np.array([200,100,100], dtype=float), #same length as L
               'loc x=0' : 'j12' ,
               'loc x=-L': 'j4'
               }

    ch_gegs['Breeddiep'] = { 'Name' : 'Breeddiep' ,#8
               'H' : 16 ,
               'L' : np.array([2500], dtype=float),
               'b' : np.array([1200,1200], dtype=float), #one longer than L
               'dx' : np.array([50], dtype=float), #same length as L
               'loc x=0' : 's1' ,
               'loc x=-L': 'j12'
               }

    ch_gegs['Oude Maas 2'] =  { 'Name' : 'Oude Maas 2' , #9
               'H' : 14 ,
               'L' : np.array([8250], dtype=float),
               'b' : np.array([317,317], dtype=float), #one longer than L
               'dx' : np.array([125], dtype=float), #same length as L
               'loc x=0' : 'j4' ,
               'loc x=-L': 'j8'
               }

    ch_gegs['Waal'] =  { 'Name' : 'Waal' , #10, Boven Merwede in bestand
               'H' : 4 ,
               'L' : np.array([45200], dtype=float),
               'b' : np.array([214,500], dtype=float), #one longer than L
               'dx' : np.array([1130], dtype=float), #same length as L
               'loc x=0' : 'j5' ,
               'loc x=-L': 'r1'
               }

    ch_gegs['Beneden Merwede'] =  { 'Name' : 'Beneden Merwede' , #11
               'H' : 6 ,
               'L' : np.array([15300], dtype=float),
               'b' : np.array([200/2,300], dtype=float), #one longer than L #groins not taken into account!
               'dx' : np.array([510], dtype=float), #same length as L
               'loc x=0' : 'j6' ,
               'loc x=-L': 'j5'
               }

    ch_gegs['Oude Maas 4'] =  { 'Name' : 'Oude Maas 4' , #12
               'H' : 7 ,
               'L' : np.array([4260], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([426], dtype=float), #same length as L
               'loc x=0' : 'j7' ,
               'loc x=-L': 'j6'
               }

    ch_gegs['Oude Maas 3'] =  { 'Name' : 'Oude Maas 3' , #13
               'H' : 10.2 ,
               'L' : np.array([10000,5000], dtype=float),
               'b' : np.array([240,309,350], dtype=float), #one longer than L
               'dx' : np.array([500,100], dtype=float), #same length as L
               'loc x=0' : 'j8' ,
               'loc x=-L': 'j7'
               }

    ch_gegs['Nieuwe Merwede'] =  { 'Name' : 'Nieuwe Merwede' , #14
               'H' : 5 ,
               'L' : np.array([19600], dtype=float),
               'b' : np.array([400,730], dtype=float), #one longer than L
               'dx' : np.array([400], dtype=float), #same length as L
               'loc x=0' : 'j9' ,
               'loc x=-L': 'j5'
               }

    ch_gegs['Dordtsche Kil'] =  { 'Name' : 'Dordtsche Kil' , #15
               'H' : 10.7 ,
               'L' : np.array([9400], dtype=float),
               'b' : np.array([300,300], dtype=float), #one longer than L
               'dx' : np.array([470], dtype=float), #same length as L
               'loc x=0' : 'j7' ,
               'loc x=-L': 'j10'
               }

    ch_gegs['Spui'] =  { 'Name' : 'Spui' , #16
               'H' :  6.4,
               'L' : np.array([12000,10000,2800], dtype=float),
               'b' : np.array([250,250,250,250], dtype=float), #one longer than L
               'dx' : np.array([1000,500,50], dtype=float), #same length as L
               'loc x=0' : 'j8' ,
               'loc x=-L': 'j11'
               }

    ch_gegs['Maas'] =  { 'Name' : 'Maas' , #17
               'H' : 5.3 ,
               'L' : np.array([62300], dtype=float),
               'b' : np.array([97,416], dtype=float), #one longer than L
               'dx' : np.array([623], dtype=float), #same length as L
               'loc x=0' : 'j9' ,
               'loc x=-L': 'r2'
               }

    ch_gegs['Hollands Diep 2'] =  { 'Name' : 'Hollands Diep 2' , #18
               'H' : 6.2 ,
               'L' : np.array([3810], dtype=float),
               'b' : np.array([1600,1100], dtype=float), #one longer than L
               'dx' : np.array([381], dtype=float), #same length as L
               'loc x=0' : 'j10' ,
               'loc x=-L': 'j9'
               }

    ch_gegs['Hollands Diep 1'] =  { 'Name' : 'Hollands Diep 1' , #19
               'H' : 7.6 ,
               'L' : np.array([32000], dtype=float),
               'b' : np.array([1630,2000], dtype=float), #one longer than L
               'dx' : np.array([1000], dtype=float), #same length as L
               'loc x=0' : 'j11' ,
               'loc x=-L': 'j10'
               }

    ch_gegs['Haringvliet'] =  { 'Name' : 'Haringvliet' , #20
               'H' : 8.7 ,
               'L' : np.array([11700], dtype=float),
               'b' : np.array([2420,2420], dtype=float), #one longer than L
               'dx' : np.array([900], dtype=float), #same length as L
               'loc x=0' : 'h1' ,
               'loc x=-L': 'j11'
               }

    #optional: fix the tides
    for key in list(ch_gegs.keys()): ch_gegs[key]['Ut']= 1

    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================

    path_RM = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Geo_RijnMaas/'
    list_channels = [f for f in os.listdir(path_RM) if f.endswith('.kml')]
    RM_coords = {}
    for i in range(len(list_channels)):
        RM_coords[list_channels[i][:-4]] = np.array(kml_subtract_latlon(path_RM+list_channels[i]),dtype=float)

    cs = ['r','olive','b','dimgray','c','orange','g','m','y','gold','indigo','tan','skyblue','maroon','peru',\
          'beige','plum','silver','forestgreen','darkkhaki','rosybrown','teal','snow','aliceblue','mintcream']
    count=0
    for key in list(ch_gegs.keys()):
        ch_gegs[key]['plot x'] = np.flip(RM_coords[key][:,0])
        ch_gegs[key]['plot y'] = np.flip(RM_coords[key][:,1])
        ch_gegs[key]['plot color'] = cs[count]
        count+=1

    return ch_gegs

def geo_RMD4():
    # =============================================================================
    #     Input: geometry
    # =============================================================================
    ch_gegs = {}
    ch_gegs['Hollandse IJssel'] =  { 'Name' : 'Hollandse IJssel' , #1
               'H' : 3.5 ,
               'L' : np.array([19700], dtype=float),
               'b' : np.array([45,150], dtype=float), #one longer than L
               'dx' : np.array([197], dtype=float), #same length as L
               'loc x=0' : 'j2' ,
               'loc x=-L': 'w1'
               }

    ch_gegs['Lek'] =  { 'Name' : 'Lek' , #2
               'H' : 4.7 ,
               'L' : np.array([42000], dtype=float),
               'b' : np.array([136,260], dtype=float), #one longer than L
               'dx' : np.array([600], dtype=float), #same length as L
               'loc x=0' : 'j1' ,
               'loc x=-L': 'w2'
               }

    ch_gegs['Nieuwe Maas 4'] = { 'Name' : 'Nieuwe Maas 4' , #3
               'H' : 8.1 ,
               'L' : np.array([4900], dtype=float),
               'b' : np.array([250,370], dtype=float), #one longer than L
               'dx' : np.array([490], dtype=float), #same length as L
               'loc x=0' : 'j2' ,
               'loc x=-L': 'j1'
               }

    ch_gegs['Nieuwe Maas 3'] = { 'Name' : 'Nieuwe Maas 3' , #3
               'H' : 10 ,
               'L' : np.array([10800], dtype=float),
               'b' : np.array([400,400], dtype=float), #one longer than L
               'dx' : np.array([400], dtype=float), #same length as L
               'Ut' : 1.76 ,
               'loc x=0' : 'j13' ,
               'loc x=-L': 'j2'
               }

    ch_gegs['Nieuwe Maas 2'] = { 'Name' : 'Nieuwe Maas 2' , #3
               'H' : 11 ,
               'L' : np.array([2750], dtype=float),
               'b' : np.array([400,450], dtype=float), #one longer than L
               'dx' : np.array([250], dtype=float), #same length as L
               'loc x=0' : 'j14' ,
               'loc x=-L': 'j13'
               }

    ch_gegs['Nieuwe Maas 1'] = { 'Name' : 'Nieuwe Maas 1' , #3
               'H' : 14 ,
               'L' : np.array([5100], dtype=float),
               'b' : np.array([450,500], dtype=float), #one longer than L
               'dx' : np.array([510], dtype=float), #same length as L
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j14'
               }

    ch_gegs['Waalhaven'] = { 'Name' : 'Waalhaven' , #3
               'H' : 13 ,
               'L' : np.array([2500], dtype=float),
               'b' : np.array([1250,350], dtype=float), #one longer than L
               'dx' : np.array([500], dtype=float), #same length as L
               'Ut' : 1.76 ,
               'loc x=0' : 'j13' ,
               'loc x=-L': 'w3'
               }

    ch_gegs['Eemhaven'] = { 'Name' : 'Eemhaven' , #3
               'H' : 13,
               'L' : np.array([2000], dtype=float),
               'b' : np.array([800,300], dtype=float), #one longer than L
               'dx' : np.array([400], dtype=float), #same length as L
               'loc x=0' : 'j14' ,
               'loc x=-L': 'w4'
               }

    ch_gegs['Nieuwe Waterweg v2'] =  { 'Name' : 'Nieuwe Waterweg' , #5
               'H' : 16 ,
               'L' : np.array([16800], dtype=float),
               'b' : np.array([600,600], dtype=float), #one longer than L
               'dx' : np.array([400], dtype=float), #same length as L
               'loc x=0' : 'j12' ,
               'loc x=-L': 'j3'
               }

    ch_gegs['Noord'] =  { 'Name' : 'Noord' , #6
               'H' : 5.9+2 ,
               'L' : np.array([8620], dtype=float),
               'b' : np.array([250,300], dtype=float), #one longer than L
               'dx' : np.array([431], dtype=float), #same length as L
               'loc x=0' : 'j6' ,
               'loc x=-L': 'j1'
               }

    ch_gegs['Oude Maas 1'] =  { 'Name' : 'Oude Maas 1' , #7
               'H' : 13 ,
               'L' : np.array([3100], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([310], dtype=float), #same length as L
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j4'
               }


    ch_gegs['Hartelkanaal v2'] =  { 'Name' : 'Hartelkanaal' ,#8
               'H' : 7.6 +1,
               'L' : np.array([22200,500,3000], dtype=float),
               'b' : np.array([310,310,600,1500], dtype=float), #one longer than L
               'dx' : np.array([600,200,300], dtype=float), #same length as L
               'loc x=0' : 'j12' ,
               'loc x=-L': 'j4'
               }

    ch_gegs['Breeddiep'] = { 'Name' : 'Breeddiep' ,#8
               'H' : 16 ,
               'L' : np.array([2500], dtype=float),
               'b' : np.array([1200,1200], dtype=float), #one longer than L
               'dx' : np.array([250], dtype=float), #same length as L
               'loc x=0' : 's1' ,
               'loc x=-L': 'j12'
               }

    ch_gegs['Oude Maas 2'] =  { 'Name' : 'Oude Maas 2' , #9
               'H' : 14 ,
               'L' : np.array([8260], dtype=float),
               'b' : np.array([317,317], dtype=float), #one longer than L
               'dx' : np.array([413], dtype=float), #same length as L
               'loc x=0' : 'j4' ,
               'loc x=-L': 'j8'
               }

    ch_gegs['Waal'] =  { 'Name' : 'Waal' , #10, Boven Merwede in bestand
               'H' : 4 ,
               'L' : np.array([45200], dtype=float),
               'b' : np.array([214,500], dtype=float), #one longer than L
               'dx' : np.array([1130], dtype=float), #same length as L
               'loc x=0' : 'j5' ,
               'loc x=-L': 'r1'
               }

    ch_gegs['Beneden Merwede'] =  { 'Name' : 'Beneden Merwede' , #11
               'H' : 6 ,
               'L' : np.array([15300], dtype=float),
               'b' : np.array([200,320], dtype=float), #one longer than L
               'dx' : np.array([510], dtype=float), #same length as L
               'loc x=0' : 'j6' ,
               'loc x=-L': 'j5'
               }

    ch_gegs['Oude Maas 4'] =  { 'Name' : 'Oude Maas 4' , #12
               'H' : 5 ,
               'L' : np.array([4260], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([426], dtype=float), #same length as L
               'loc x=0' : 'j7' ,
               'loc x=-L': 'j6'
               }

    ch_gegs['Oude Maas 3'] =  { 'Name' : 'Oude Maas 3' , #13
               'H' : 9.2 ,
               'L' : np.array([15000], dtype=float),
               'b' : np.array([240,350], dtype=float), #one longer than L
               'dx' : np.array([500], dtype=float), #same length as L
               'loc x=0' : 'j8' ,
               'loc x=-L': 'j7'
               }

    ch_gegs['Nieuwe Merwede'] =  { 'Name' : 'Nieuwe Merwede' , #14
               'H' : 4.3+2 ,
               'L' : np.array([19600], dtype=float),
               'b' : np.array([400,730], dtype=float), #one longer than L
               'dx' : np.array([400], dtype=float), #same length as L
               'loc x=0' : 'j9' ,
               'loc x=-L': 'j5'
               }

    ch_gegs['Dordtsche Kil'] =  { 'Name' : 'Dordtsche Kil' , #15
               'H' : 9.2 ,
               'L' : np.array([9400], dtype=float),
               'b' : np.array([350,350], dtype=float), #one longer than L
               'dx' : np.array([470], dtype=float), #same length as L
               'loc x=0' : 'j7' ,
               'loc x=-L': 'j10'
               }

    ch_gegs['Spui'] =  { 'Name' : 'Spui' , #16
               'H' :  6.4,
               'L' : np.array([17400], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([600], dtype=float), #same length as L
               'loc x=0' : 'j8' ,
               'loc x=-L': 'j11'
               }

    ch_gegs['Maas'] =  { 'Name' : 'Maas' , #17
               'H' : 5.3 ,
               'L' : np.array([62300], dtype=float),
               'b' : np.array([97,416], dtype=float), #one longer than L
               'dx' : np.array([1246], dtype=float), #same length as L
               'loc x=0' : 'j9' ,
               'loc x=-L': 'r2'
               }

    ch_gegs['Hollands Diep 2'] =  { 'Name' : 'Hollands Diep 2' , #18
               'H' : 6.2 ,
               'L' : np.array([3810], dtype=float),
               'b' : np.array([1600,1100], dtype=float), #one longer than L
               'dx' : np.array([381], dtype=float), #same length as L
               'loc x=0' : 'j10' ,
               'loc x=-L': 'j9'
               }

    ch_gegs['Hollands Diep 1'] =  { 'Name' : 'Hollands Diep 1' , #19
               'H' : 7.6 ,
               'L' : np.array([32000], dtype=float),
               'b' : np.array([1630,2000], dtype=float), #one longer than L
               'dx' : np.array([1000], dtype=float), #same length as L
               'loc x=0' : 'j11' ,
               'loc x=-L': 'j10'
               }

    ch_gegs['Haringvliet'] =  { 'Name' : 'Haringvliet' , #20
               'H' : 8.7 ,
               'L' : np.array([11700], dtype=float),
               'b' : np.array([2420,2420], dtype=float), #one longer than L
               'dx' : np.array([1170], dtype=float), #same length as L
               'loc x=0' : 'h1' ,
               'loc x=-L': 'j11'
               }

    #optional: fix the tides
    for key in list(ch_gegs.keys()): ch_gegs[key]['Ut']= 1

    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================

    path_RM = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Geo_RijnMaas/'
    list_channels = [f for f in os.listdir(path_RM) if f.endswith('.kml')]
    RM_coords = {}
    for i in range(len(list_channels)):
        RM_coords[list_channels[i][:-4]] = np.array(kml_subtract_latlon(path_RM+list_channels[i]),dtype=float)

    cs = ['r','olive','b','dimgray','c','orange','g','m','y','gold','indigo','tan','skyblue','maroon','peru',\
          'beige','plum','silver','forestgreen','darkkhaki','rosybrown','teal','snow','aliceblue','mintcream']
    count=0
    for key in list(ch_gegs.keys()):
        ch_gegs[key]['plot x'] = np.flip(RM_coords[key][:,0])
        ch_gegs[key]['plot y'] = np.flip(RM_coords[key][:,1])
        ch_gegs[key]['plot color'] = cs[count]
        count+=1

    return ch_gegs

def geo_RMD5():
    # =============================================================================
    #     Input: geometry
    # =============================================================================
    ch_gegs = {}
    ch_gegs['Hollandse IJssel'] =  { 'Name' : 'Hollandse IJssel' , #1
               'H' : 3.5 ,
               'L' : np.array([19700], dtype=float),
               'b' : np.array([45,150], dtype=float), #one longer than L
               'dx' : np.array([197], dtype=float), #same length as L
               'loc x=0' : 'j2' ,
               'loc x=-L': 'w1'
               }

    ch_gegs['Lek'] =  { 'Name' : 'Lek' , #2
               'H' : 4.7 ,
               'L' : np.array([42000], dtype=float),
               'b' : np.array([136,260], dtype=float), #one longer than L
               'dx' : np.array([600], dtype=float), #same length as L
               'loc x=0' : 'j1' ,
               'loc x=-L': 'w2'
               }

    ch_gegs['Nieuwe Maas 4'] = { 'Name' : 'Nieuwe Maas 4' , #3
               'H' : 8.1 ,
               'L' : np.array([4900], dtype=float),
               'b' : np.array([250,370], dtype=float), #one longer than L
               'dx' : np.array([490], dtype=float), #same length as L
               'loc x=0' : 'j2' ,
               'loc x=-L': 'j1'
               }

    ch_gegs['Nieuwe Maas 3'] = { 'Name' : 'Nieuwe Maas 3' , #3
               'H' : 10 ,
               'L' : np.array([10800], dtype=float),
               'b' : np.array([400,400], dtype=float), #one longer than L
               'dx' : np.array([400], dtype=float), #same length as L
               'loc x=0' : 'j13' ,
               'loc x=-L': 'j2'
               }

    ch_gegs['Nieuwe Maas 2'] = { 'Name' : 'Nieuwe Maas 2' , #3
               'H' : 11 ,
               'L' : np.array([2750], dtype=float),
               'b' : np.array([400,450], dtype=float), #one longer than L
               'dx' : np.array([25], dtype=float), #same length as L
               'loc x=0' : 'j14' ,
               'loc x=-L': 'j13'
               }

    ch_gegs['Nieuwe Maas 1'] = { 'Name' : 'Nieuwe Maas 1' , #3
               'H' : 14 ,
               'L' : np.array([5100], dtype=float),
               'b' : np.array([450,500], dtype=float), #one longer than L
               'dx' : np.array([25], dtype=float), #same length as L
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j14'
               }

    ch_gegs['Waalhaven'] = { 'Name' : 'Waalhaven' , #3
               'H' : 13 ,
               'L' : np.array([2500], dtype=float),
               'b' : np.array([1250,350], dtype=float), #one longer than L
               'dx' : np.array([500], dtype=float), #same length as L
               'loc x=0' : 'j13' ,
               'loc x=-L': 'w3'
               }

    ch_gegs['Eemhaven'] = { 'Name' : 'Eemhaven' , #3
               'H' : 13,
               'L' : np.array([2000], dtype=float),
               'b' : np.array([800,300], dtype=float), #one longer than L
               'dx' : np.array([400], dtype=float), #same length as L
               'loc x=0' : 'j14' ,
               'loc x=-L': 'w4'
               }

    ch_gegs['Nieuwe Waterweg v2'] =  { 'Name' : 'Nieuwe Waterweg' , #5
               'H' : 16 ,
               'L' : np.array([16800], dtype=float),
               'b' : np.array([600,600], dtype=float), #one longer than L
               'dx' : np.array([20], dtype=float), #same length as L
               'loc x=0' : 'j12' ,
               'loc x=-L': 'j3'
               }

    ch_gegs['Noord'] =  { 'Name' : 'Noord' , #6
               'H' : 5.9+2 ,
               'L' : np.array([8620], dtype=float),
               'b' : np.array([250,300], dtype=float), #one longer than L
               'dx' : np.array([431], dtype=float), #same length as L
               'loc x=0' : 'j6' ,
               'loc x=-L': 'j1'
               }

    ch_gegs['Oude Maas 1'] =  { 'Name' : 'Oude Maas 1' , #7
               'H' : 13 ,
               'L' : np.array([3100], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([20], dtype=float), #same length as L
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j4'
               }


    ch_gegs['Hartelkanaal v2'] =  { 'Name' : 'Hartelkanaal' ,#8
               'H' : 7.6 +1,
               'L' : np.array([22200,500,3000], dtype=float),
               'b' : np.array([310,310,600,1500], dtype=float), #one longer than L
               'dx' : np.array([20,20,20], dtype=float), #same length as L
               'loc x=0' : 'j12' ,
               'loc x=-L': 'j4'
               }

    ch_gegs['Breeddiep'] = { 'Name' : 'Breeddiep' ,#8
               'H' : 16 ,
               'L' : np.array([2500], dtype=float),
               'b' : np.array([1200,1200], dtype=float), #one longer than L
               'dx' : np.array([20], dtype=float), #same length as L
               'loc x=0' : 's1' ,
               'loc x=-L': 'j12'
               }

    ch_gegs['Oude Maas 2'] =  { 'Name' : 'Oude Maas 2' , #9
               'H' : 14 ,
               'L' : np.array([8260], dtype=float),
               'b' : np.array([317,317], dtype=float), #one longer than L
               'dx' : np.array([20], dtype=float), #same length as L
               'loc x=0' : 'j4' ,
               'loc x=-L': 'j8'
               }

    ch_gegs['Waal'] =  { 'Name' : 'Waal' , #10, Boven Merwede in bestand
               'H' : 4 ,
               'L' : np.array([45200], dtype=float),
               'b' : np.array([214,500], dtype=float), #one longer than L
               'dx' : np.array([1130], dtype=float), #same length as L
               'loc x=0' : 'j5' ,
               'loc x=-L': 'r1'
               }

    ch_gegs['Beneden Merwede'] =  { 'Name' : 'Beneden Merwede' , #11
               'H' : 6 ,
               'L' : np.array([15300], dtype=float),
               'b' : np.array([200,320], dtype=float), #one longer than L
               'dx' : np.array([510], dtype=float), #same length as L
               'loc x=0' : 'j6' ,
               'loc x=-L': 'j5'
               }

    ch_gegs['Oude Maas 4'] =  { 'Name' : 'Oude Maas 4' , #12
               'H' : 5 ,
               'L' : np.array([4260], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([426], dtype=float), #same length as L
               'loc x=0' : 'j7' ,
               'loc x=-L': 'j6'
               }

    ch_gegs['Oude Maas 3'] =  { 'Name' : 'Oude Maas 3' , #13
               'H' : 9.2 ,
               'L' : np.array([15000], dtype=float),
               'b' : np.array([240,350], dtype=float), #one longer than L
               'dx' : np.array([500], dtype=float), #same length as L
               'loc x=0' : 'j8' ,
               'loc x=-L': 'j7'
               }

    ch_gegs['Nieuwe Merwede'] =  { 'Name' : 'Nieuwe Merwede' , #14
               'H' : 4.3+2 ,
               'L' : np.array([19600], dtype=float),
               'b' : np.array([400,730], dtype=float), #one longer than L
               'dx' : np.array([400], dtype=float), #same length as L
               'loc x=0' : 'j9' ,
               'loc x=-L': 'j5'
               }

    ch_gegs['Dordtsche Kil'] =  { 'Name' : 'Dordtsche Kil' , #15
               'H' : 9.2 ,
               'L' : np.array([9400], dtype=float),
               'b' : np.array([350,350], dtype=float), #one longer than L
               'dx' : np.array([470], dtype=float), #same length as L
               'loc x=0' : 'j7' ,
               'loc x=-L': 'j10'
               }

    ch_gegs['Spui'] =  { 'Name' : 'Spui' , #16
               'H' :  6.4,
               'L' : np.array([17400], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([600], dtype=float), #same length as L
               'loc x=0' : 'j8' ,
               'loc x=-L': 'j11'
               }

    ch_gegs['Maas'] =  { 'Name' : 'Maas' , #17
               'H' : 5.3 ,
               'L' : np.array([62300], dtype=float),
               'b' : np.array([97,416], dtype=float), #one longer than L
               'dx' : np.array([1246], dtype=float), #same length as L
               'loc x=0' : 'j9' ,
               'loc x=-L': 'r2'
               }

    ch_gegs['Hollands Diep 2'] =  { 'Name' : 'Hollands Diep 2' , #18
               'H' : 6.2 ,
               'L' : np.array([3810], dtype=float),
               'b' : np.array([1600,1100], dtype=float), #one longer than L
               'dx' : np.array([381], dtype=float), #same length as L
               'loc x=0' : 'j10' ,
               'loc x=-L': 'j9'
               }

    ch_gegs['Hollands Diep 1'] =  { 'Name' : 'Hollands Diep 1' , #19
               'H' : 7.6 ,
               'L' : np.array([32000], dtype=float),
               'b' : np.array([1630,2000], dtype=float), #one longer than L
               'dx' : np.array([600], dtype=float), #same length as L
               'loc x=0' : 'j11' ,
               'loc x=-L': 'j10'
               }

    ch_gegs['Haringvliet'] =  { 'Name' : 'Haringvliet' , #20
               'H' : 8.7 ,
               'L' : np.array([11700], dtype=float),
               'b' : np.array([2420,2420], dtype=float), #one longer than L
               'dx' : np.array([450], dtype=float), #same length as L
               'loc x=0' : 'h1' ,
               'loc x=-L': 'j11'
               }

    #optional: fix the tides
    for key in list(ch_gegs.keys()): ch_gegs[key]['Ut']= 1

    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================

    path_RM = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Geo_RijnMaas/'
    list_channels = [f for f in os.listdir(path_RM) if f.endswith('.kml')]
    RM_coords = {}
    for i in range(len(list_channels)):
        RM_coords[list_channels[i][:-4]] = np.array(kml_subtract_latlon(path_RM+list_channels[i]),dtype=float)

    cs = ['r','olive','b','dimgray','c','orange','g','m','y','gold','indigo','tan','skyblue','maroon','peru',\
          'beige','plum','silver','forestgreen','darkkhaki','rosybrown','teal','snow','aliceblue','mintcream']
    count=0
    for key in list(ch_gegs.keys()):
        ch_gegs[key]['plot x'] = np.flip(RM_coords[key][:,0])
        ch_gegs[key]['plot y'] = np.flip(RM_coords[key][:,1])
        ch_gegs[key]['plot color'] = cs[count]
        count+=1

    return ch_gegs


def geo_RMD6():
    # =============================================================================
    #     Input: geometry
    # =============================================================================
    ch_gegs = {}
    ch_gegs['Hollandse IJssel'] =  { 'Name' : 'Hollandse IJssel' , #1
               'H' : 3.5 ,
               'L' : np.array([19700], dtype=float),
               'b' : np.array([45,150], dtype=float), #one longer than L
               'dx' : np.array([197], dtype=float), #same length as L
               'loc x=0' : 'j2' ,
               'loc x=-L': 'w1'
               }

    ch_gegs['Lek'] =  { 'Name' : 'Lek' , #2
               'H' : 4.7 ,
               'L' : np.array([42000], dtype=float),
               'b' : np.array([136,260], dtype=float), #one longer than L
               'dx' : np.array([600], dtype=float), #same length as L
               'loc x=0' : 'j1' ,
               'loc x=-L': 'w2'
               }

    ch_gegs['Nieuwe Maas 4'] = { 'Name' : 'Nieuwe Maas 4' , #3
               'H' : 11 ,
               'L' : np.array([4900], dtype=float),
               'b' : np.array([250,370], dtype=float), #one longer than L
               'dx' : np.array([490], dtype=float), #same length as L
               'loc x=0' : 'j2' ,
               'loc x=-L': 'j1'
               }

    ch_gegs['Nieuwe Maas 3'] = { 'Name' : 'Nieuwe Maas 3' , #3
               'H' : 11 ,
               'L' : np.array([10800], dtype=float),
               'b' : np.array([400,400], dtype=float), #one longer than L
               'dx' : np.array([400], dtype=float), #same length as L
               'loc x=0' : 'j13' ,
               'loc x=-L': 'j2'
               }

    ch_gegs['Nieuwe Maas 2'] = { 'Name' : 'Nieuwe Maas 2' , #3
               'H' : 11 ,
               'L' : np.array([2750], dtype=float),
               'b' : np.array([400,450], dtype=float), #one longer than L
               'dx' : np.array([250], dtype=float), #same length as L
               'loc x=0' : 'j14' ,
               'loc x=-L': 'j13'
               }

    ch_gegs['Nieuwe Maas 1'] = { 'Name' : 'Nieuwe Maas 1' , #3
               'H' : 11 ,
               'L' : np.array([5100], dtype=float),
               'b' : np.array([450,500], dtype=float), #one longer than L
               'dx' : np.array([510], dtype=float), #same length as L
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j14'
               }

    ch_gegs['Waalhaven'] = { 'Name' : 'Waalhaven' , #3
               'H' : 13 ,
               'L' : np.array([2500], dtype=float),
               'b' : np.array([1250,350], dtype=float), #one longer than L
               'dx' : np.array([500], dtype=float), #same length as L
               'loc x=0' : 'j13' ,
               'loc x=-L': 'w3'
               }

    ch_gegs['Eemhaven'] = { 'Name' : 'Eemhaven' , #3
               'H' : 13,
               'L' : np.array([2000], dtype=float),
               'b' : np.array([800,300], dtype=float), #one longer than L
               'dx' : np.array([400], dtype=float), #same length as L
               'loc x=0' : 'j14' ,
               'loc x=-L': 'w4'
               }

    ch_gegs['Nieuwe Waterweg v2'] =  { 'Name' : 'Nieuwe Waterweg' , #5
               'H' : 16 ,
               'L' : np.array([16800], dtype=float),
               'b' : np.array([600,600], dtype=float), #one longer than L
               'dx' : np.array([400], dtype=float), #same length as L
               'loc x=0' : 'j12' ,
               'loc x=-L': 'j3'
               }

    ch_gegs['Noord'] =  { 'Name' : 'Noord' , #6
               'H' : 5.9 ,
               'L' : np.array([8620], dtype=float),
               'b' : np.array([250,300], dtype=float), #one longer than L
               'dx' : np.array([431], dtype=float), #same length as L
               'loc x=0' : 'j6' ,
               'loc x=-L': 'j1'
               }

    ch_gegs['Oude Maas 1'] =  { 'Name' : 'Oude Maas 1' , #7
               'H' : 13 ,
               'L' : np.array([3100], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([310], dtype=float), #same length as L
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j4'
               }


    ch_gegs['Hartelkanaal v2'] =  { 'Name' : 'Hartelkanaal' ,#8
               'H' : 7.6 ,
               'L' : np.array([22200,500,3000], dtype=float),
               'b' : np.array([310,310,600,1500], dtype=float), #one longer than L
               'dx' : np.array([600,200,300], dtype=float), #same length as L
               'loc x=0' : 'j12' ,
               'loc x=-L': 'j4'
               }

    ch_gegs['Breeddiep'] = { 'Name' : 'Breeddiep' ,#8
               'H' : 18 ,
               'L' : np.array([2500], dtype=float),
               'b' : np.array([1200,1200], dtype=float), #one longer than L
               'dx' : np.array([250], dtype=float), #same length as L
               'loc x=0' : 's1' ,
               'loc x=-L': 'j12'
               }

    ch_gegs['Oude Maas 2'] =  { 'Name' : 'Oude Maas 2' , #9
               'H' : 14 ,
               'L' : np.array([8260], dtype=float),
               'b' : np.array([317,317], dtype=float), #one longer than L
               'dx' : np.array([413], dtype=float), #same length as L
               'loc x=0' : 'j4' ,
               'loc x=-L': 'j8'
               }

    ch_gegs['Waal'] =  { 'Name' : 'Waal' , #10, Boven Merwede in bestand
               'H' : 4 ,
               'L' : np.array([45200], dtype=float),
               'b' : np.array([214,500], dtype=float), #one longer than L
               'dx' : np.array([1130], dtype=float), #same length as L
               'loc x=0' : 'j5' ,
               'loc x=-L': 'r1'
               }

    ch_gegs['Beneden Merwede'] =  { 'Name' : 'Beneden Merwede' , #11
               'H' : 6 ,
               'L' : np.array([15300], dtype=float),
               'b' : np.array([200,320], dtype=float), #one longer than L
               'dx' : np.array([510], dtype=float), #same length as L
               'loc x=0' : 'j6' ,
               'loc x=-L': 'j5'
               }

    ch_gegs['Oude Maas 4'] =  { 'Name' : 'Oude Maas 4' , #12
               'H' : 5 ,
               'L' : np.array([4260], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([426], dtype=float), #same length as L
               'loc x=0' : 'j7' ,
               'loc x=-L': 'j6'
               }

    ch_gegs['Oude Maas 3'] =  { 'Name' : 'Oude Maas 3' , #13
               'H' : 9.2 ,
               'L' : np.array([15000], dtype=float),
               'b' : np.array([240,350], dtype=float), #one longer than L
               'dx' : np.array([500], dtype=float), #same length as L
               'loc x=0' : 'j8' ,
               'loc x=-L': 'j7'
               }

    ch_gegs['Nieuwe Merwede'] =  { 'Name' : 'Nieuwe Merwede' , #14
               'H' : 4.3 ,
               'L' : np.array([19600], dtype=float),
               'b' : np.array([400,730], dtype=float), #one longer than L
               'dx' : np.array([400], dtype=float), #same length as L
               'loc x=0' : 'j9' ,
               'loc x=-L': 'j5'
               }

    ch_gegs['Dordtsche Kil'] =  { 'Name' : 'Dordtsche Kil' , #15
               'H' : 9.2 ,
               'L' : np.array([9400], dtype=float),
               'b' : np.array([350,350], dtype=float), #one longer than L
               'dx' : np.array([470], dtype=float), #same length as L
               'loc x=0' : 'j7' ,
               'loc x=-L': 'j10'
               }

    ch_gegs['Spui'] =  { 'Name' : 'Spui' , #16
               'H' :  6.4,
               'L' : np.array([17400], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([600], dtype=float), #same length as L
               'loc x=0' : 'j8' ,
               'loc x=-L': 'j11'
               }

    ch_gegs['Maas'] =  { 'Name' : 'Maas' , #17
               'H' : 5.3 ,
               'L' : np.array([62300], dtype=float),
               'b' : np.array([97,416], dtype=float), #one longer than L
               'dx' : np.array([1246], dtype=float), #same length as L
               'loc x=0' : 'j9' ,
               'loc x=-L': 'r2'
               }

    ch_gegs['Hollands Diep 2'] =  { 'Name' : 'Hollands Diep 2' , #18
               'H' : 6.2 ,
               'L' : np.array([3810], dtype=float),
               'b' : np.array([1600,1100], dtype=float), #one longer than L
               'dx' : np.array([381], dtype=float), #same length as L
               'loc x=0' : 'j10' ,
               'loc x=-L': 'j9'
               }

    ch_gegs['Hollands Diep 1'] =  { 'Name' : 'Hollands Diep 1' , #19
               'H' : 7.6 ,
               'L' : np.array([32000], dtype=float),
               'b' : np.array([1630,2000], dtype=float), #one longer than L
               'dx' : np.array([1000], dtype=float), #same length as L
               'loc x=0' : 'j11' ,
               'loc x=-L': 'j10'
               }

    ch_gegs['Haringvliet'] =  { 'Name' : 'Haringvliet' , #20
               'H' : 8.7 ,
               'L' : np.array([11700], dtype=float),
               'b' : np.array([2420,2420], dtype=float), #one longer than L
               'dx' : np.array([1170], dtype=float), #same length as L
               'loc x=0' : 'h1' ,
               'loc x=-L': 'j11'
               }

    #optional: fix the tides
    for key in list(ch_gegs.keys()): ch_gegs[key]['Ut']= 1

    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================

    path_RM = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Geo_RijnMaas/'
    list_channels = [f for f in os.listdir(path_RM) if f.endswith('.kml')]
    RM_coords = {}
    for i in range(len(list_channels)):
        RM_coords[list_channels[i][:-4]] = np.array(kml_subtract_latlon(path_RM+list_channels[i]),dtype=float)

    cs = ['r','olive','b','dimgray','c','orange','g','m','y','gold','indigo','tan','skyblue','maroon','peru',\
          'beige','plum','silver','forestgreen','darkkhaki','rosybrown','teal','snow','aliceblue','mintcream']
    count=0
    for key in list(ch_gegs.keys()):
        ch_gegs[key]['plot x'] = np.flip(RM_coords[key][:,0])
        ch_gegs[key]['plot y'] = np.flip(RM_coords[key][:,1])
        ch_gegs[key]['plot color'] = cs[count]
        count+=1

    return ch_gegs

def geo_RMD7():
    # =============================================================================
    #     Input: geometry
    # =============================================================================
    ch_gegs = {}
    ch_gegs['Hollandse IJssel'] =  { 'Name' : 'Hollandse IJssel' , #1
               'H' : 4. ,
               'L' : np.array([19700], dtype=float),
               'b' : np.array([45,150], dtype=float), #one longer than L
               'dx' : np.array([197], dtype=float), #same length as L
               'loc x=0' : 'j2' ,
               'loc x=-L': 'w1'
               }

    ch_gegs['Lek'] =  { 'Name' : 'Lek' , #2
               'H' : 5.3 ,
               'L' : np.array([32000,10000], dtype=float),
               'b' : np.array([136,200,260], dtype=float), #one longer than L
               'dx' : np.array([1000,250], dtype=float), #same length as L
               'loc x=0' : 'j1' ,
               'loc x=-L': 'w2'
               }

    ch_gegs['Nieuwe Maas 2 old'] = { 'Name' : 'Nieuwe Maas 2' , #3
               'H' : 8.1 ,
               'L' : np.array([4900], dtype=float),
               'b' : np.array([250,400], dtype=float), #one longer than L
               'dx' : np.array([350], dtype=float), #same length as L
               'loc x=0' : 'j2' ,
               'loc x=-L': 'j1'
               }
    
    ch_gegs['Nieuwe Maas 1 old'] = { 'Name' : 'Nieuwe Maas 1' , #4
               'H' : 11 , #wel echt gemiddeld dit
               'L' : np.array([18750], dtype=float),
               'b' : np.array([400,500], dtype=float), #one longer than L
               'dx' : np.array([375], dtype=float), #same length as L
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j2'
               }    

    ch_gegs['Nieuwe Waterweg v2'] =  { 'Name' : 'Nieuwe Waterweg' , #5
               'H' : 16 ,
               'L' : np.array([16800], dtype=float),
               'b' : np.array([600,600], dtype=float), #one longer than L
               'dx' : np.array([200], dtype=float), #same length as L
               'loc x=0' : 'j12' ,
               'loc x=-L': 'j3'
               }

    ch_gegs['Noord'] =  { 'Name' : 'Noord' , #6
               'H' : 7 , #moeilijk vast te stellen
               'L' : np.array([8600], dtype=float),
               'b' : np.array([250,220], dtype=float), #one longer than L
               'dx' : np.array([430], dtype=float), #same length as L
               'loc x=0' : 'j6' ,
               'loc x=-L': 'j1'
               }

    ch_gegs['Oude Maas 1'] =  { 'Name' : 'Oude Maas 1' , #7
               'H' : 13 ,
               'L' : np.array([3100], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([310], dtype=float), #same length as L
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j4'
               }

    ch_gegs['Hartelkanaal v2'] =  { 'Name' : 'Hartelkanaal' ,#8
               'H' : 7.6,
               'L' : np.array([25800], dtype=float),
               'b' : np.array([310,1500], dtype=float), #one longer than L
               'dx' : np.array([300], dtype=float), #same length as L
               'loc x=0' : 'j12' ,
               'loc x=-L': 'j4'
               }

    ch_gegs['Breeddiep'] = { 'Name' : 'Breeddiep' ,#8
               'H' : 16 ,
               'L' : np.array([2500], dtype=float),
               'b' : np.array([1200,1200], dtype=float), #one longer than L
               'dx' : np.array([250], dtype=float), #same length as L
               'loc x=0' : 's1' ,
               'loc x=-L': 'j12'
               }

    ch_gegs['Oude Maas 2'] =  { 'Name' : 'Oude Maas 2' , #9
               'H' : 14 ,
               'L' : np.array([8250], dtype=float),
               'b' : np.array([317,317], dtype=float), #one longer than L
               'dx' : np.array([250], dtype=float), #same length as L
               'loc x=0' : 'j4' ,
               'loc x=-L': 'j8'
               }

    ch_gegs['Waal'] =  { 'Name' : 'Waal' , #10, Boven Merwede in bestand
               'H' : 4 ,
               'L' : np.array([45200], dtype=float),
               'b' : np.array([214,500], dtype=float), #one longer than L
               'dx' : np.array([1130], dtype=float), #same length as L
               'loc x=0' : 'j5' ,
               'loc x=-L': 'r1'
               }

    ch_gegs['Beneden Merwede'] =  { 'Name' : 'Beneden Merwede' , #11
               'H' : 6 ,
               'L' : np.array([15300], dtype=float),
               'b' : np.array([200/2,300], dtype=float), #one longer than L #groins not taken into account!
               'dx' : np.array([510], dtype=float), #same length as L
               'loc x=0' : 'j6' ,
               'loc x=-L': 'j5'
               }

    ch_gegs['Oude Maas 4'] =  { 'Name' : 'Oude Maas 4' , #12
               'H' : 7 ,
               'L' : np.array([4260], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([426], dtype=float), #same length as L
               'loc x=0' : 'j7' ,
               'loc x=-L': 'j6'
               }

    ch_gegs['Oude Maas 3'] =  { 'Name' : 'Oude Maas 3' , #13
               'H' : 10.2 ,
               'L' : np.array([15000], dtype=float),
               'b' : np.array([240,350], dtype=float), #one longer than L
               'dx' : np.array([500], dtype=float), #same length as L
               'loc x=0' : 'j8' ,
               'loc x=-L': 'j7'
               }

    ch_gegs['Nieuwe Merwede'] =  { 'Name' : 'Nieuwe Merwede' , #14
               'H' : 5 ,
               'L' : np.array([19600], dtype=float),
               'b' : np.array([400,730], dtype=float), #one longer than L
               'dx' : np.array([400], dtype=float), #same length as L
               'loc x=0' : 'j9' ,
               'loc x=-L': 'j5'
               }

    ch_gegs['Dordtsche Kil'] =  { 'Name' : 'Dordtsche Kil' , #15
               'H' : 10.7 ,
               'L' : np.array([9400], dtype=float),
               'b' : np.array([300,300], dtype=float), #one longer than L
               'dx' : np.array([470], dtype=float), #same length as L
               'loc x=0' : 'j7' ,
               'loc x=-L': 'j10'
               }

    ch_gegs['Spui'] =  { 'Name' : 'Spui' , #16
               'H' :  6.4,
               'L' : np.array([17400,7400], dtype=float),
               'b' : np.array([250,250,250], dtype=float), #one longer than L
               'dx' : np.array([1000,370], dtype=float), #same length as L
               'loc x=0' : 'j8' ,
               'loc x=-L': 'j11'
               }

    ch_gegs['Maas'] =  { 'Name' : 'Maas' , #17
               'H' : 5.3 ,
               'L' : np.array([62300], dtype=float),
               'b' : np.array([97,416], dtype=float), #one longer than L
               'dx' : np.array([623], dtype=float), #same length as L
               'loc x=0' : 'j9' ,
               'loc x=-L': 'r2'
               }

    ch_gegs['Hollands Diep 2'] =  { 'Name' : 'Hollands Diep 2' , #18
               'H' : 6.2 ,
               'L' : np.array([3810], dtype=float),
               'b' : np.array([1600,1100], dtype=float), #one longer than L
               'dx' : np.array([381], dtype=float), #same length as L
               'loc x=0' : 'j10' ,
               'loc x=-L': 'j9'
               }

    ch_gegs['Hollands Diep 1'] =  { 'Name' : 'Hollands Diep 1' , #19
               'H' : 7.6 ,
               'L' : np.array([32000], dtype=float),
               'b' : np.array([1630,2000], dtype=float), #one longer than L
               'dx' : np.array([1000], dtype=float), #same length as L
               'loc x=0' : 'j11' ,
               'loc x=-L': 'j10'
               }

    ch_gegs['Haringvliet'] =  { 'Name' : 'Haringvliet' , #20
               'H' : 8.7 ,
               'L' : np.array([11700], dtype=float),
               'b' : np.array([2420,2420], dtype=float), #one longer than L
               'dx' : np.array([900], dtype=float), #same length as L
               'loc x=0' : 'h1' ,
               'loc x=-L': 'j11'
               }

    #optional: fix the tides
    for key in list(ch_gegs.keys()): ch_gegs[key]['Ut']= 1

    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================

    path_RM = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Geo_RijnMaas/'
    list_channels = [f for f in os.listdir(path_RM) if f.endswith('.kml')]
    RM_coords = {}
    for i in range(len(list_channels)):
        RM_coords[list_channels[i][:-4]] = np.array(kml_subtract_latlon(path_RM+list_channels[i]),dtype=float)

    cs = ['r','olive','b','dimgray','c','orange','g','m','y','gold','indigo','tan','skyblue','maroon','peru',\
          'beige','plum','silver','forestgreen','darkkhaki','rosybrown','teal','snow','aliceblue','mintcream']
    count=0
    for key in list(ch_gegs.keys()):
        ch_gegs[key]['plot x'] = np.flip(RM_coords[key][:,0])
        ch_gegs[key]['plot y'] = np.flip(RM_coords[key][:,1])
        ch_gegs[key]['plot color'] = cs[count]
        count+=1

    return ch_gegs

def geo_RMD8():
    # =============================================================================
    #     Input: geometry
    # =============================================================================
    ch_gegs = {}
    ch_gegs['Hollandse IJssel'] =  { 'Name' : 'Hollandse IJssel' , #1
               'H' : 4. ,
               'L' : np.array([16000,3700], dtype=float),
               'b' : np.array([45,127,150], dtype=float), #one longer than L
               'dx' : np.array([250,100], dtype=float), #same length as L
               'loc x=0' : 'j2' ,
               'loc x=-L': 'w1'
               }

    ch_gegs['Lek'] =  { 'Name' : 'Lek' , #2
               'H' : 5.3 ,
               'L' : np.array([32000,10000], dtype=float),
               'b' : np.array([136,200,260], dtype=float), #one longer than L
               'dx' : np.array([1000,250], dtype=float), #same length as L
               'loc x=0' : 'j1' ,
               'loc x=-L': 'w2'
               }

    ch_gegs['Nieuwe Maas 2 old'] = { 'Name' : 'Nieuwe Maas 2' , #3
               'H' : 8.1 ,
               'L' : np.array([4900], dtype=float),
               'b' : np.array([250,400], dtype=float), #one longer than L
               'dx' : np.array([100], dtype=float), #same length as L
               'loc x=0' : 'j2' ,
               'loc x=-L': 'j1'
               }
    
    ch_gegs['Nieuwe Maas 1 old'] = { 'Name' : 'Nieuwe Maas 1' , #4
               'H' : 11 , #wel echt gemiddeld dit
               'L' : np.array([18750], dtype=float),
               'b' : np.array([400,500], dtype=float), #one longer than L
               'dx' : np.array([250], dtype=float), #same length as L
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j2'
               }    

    ch_gegs['Nieuwe Waterweg v2'] =  { 'Name' : 'Nieuwe Waterweg' , #5
               'H' : 16 ,
               'L' : np.array([16800], dtype=float),
               'b' : np.array([600,600], dtype=float), #one longer than L
               'dx' : np.array([100], dtype=float), #same length as L
               'loc x=0' : 'j12' ,
               'loc x=-L': 'j3'
               }

    ch_gegs['Noord'] =  { 'Name' : 'Noord' , #6
               'H' : 7 , #moeilijk vast te stellen
               'L' : np.array([8600], dtype=float),
               'b' : np.array([250,220], dtype=float), #one longer than L
               'dx' : np.array([430], dtype=float), #same length as L
               'loc x=0' : 'j6' ,
               'loc x=-L': 'j1'
               }

    ch_gegs['Oude Maas 1'] =  { 'Name' : 'Oude Maas 1' , #7
               'H' : 13 ,
               'L' : np.array([3100], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([100], dtype=float), #same length as L
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j4'
               }


    ch_gegs['Hartelkanaal v2'] =  { 'Name' : 'Hartelkanaal' ,#8
               'H' : 7.6,
               'L' : np.array([25800], dtype=float),
               'b' : np.array([310,1500], dtype=float), #one longer than L
               'dx' : np.array([100], dtype=float), #same length as L
               'loc x=0' : 'j12' ,
               'loc x=-L': 'j4'
               }

    ch_gegs['Breeddiep'] = { 'Name' : 'Breeddiep' ,#8
               'H' : 16 ,
               'L' : np.array([2500], dtype=float),
               'b' : np.array([1200,1200], dtype=float), #one longer than L
               'dx' : np.array([100], dtype=float), #same length as L
               'loc x=0' : 's1' ,
               'loc x=-L': 'j12'
               }

    ch_gegs['Oude Maas 2'] =  { 'Name' : 'Oude Maas 2' , #9
               'H' : 14 ,
               'L' : np.array([8250], dtype=float),
               'b' : np.array([317,317], dtype=float), #one longer than L
               'dx' : np.array([125], dtype=float), #same length as L
               'loc x=0' : 'j4' ,
               'loc x=-L': 'j8'
               }

    ch_gegs['Waal'] =  { 'Name' : 'Waal' , #10, Boven Merwede in bestand
               'H' : 4 ,
               'L' : np.array([45200], dtype=float),
               'b' : np.array([214,500], dtype=float), #one longer than L
               'dx' : np.array([1130], dtype=float), #same length as L
               'loc x=0' : 'j5' ,
               'loc x=-L': 'r1'
               }

    ch_gegs['Beneden Merwede'] =  { 'Name' : 'Beneden Merwede' , #11
               'H' : 6 ,
               'L' : np.array([15300], dtype=float),
               'b' : np.array([200/2,300], dtype=float), #one longer than L #groins not taken into account!
               'dx' : np.array([510], dtype=float), #same length as L
               'loc x=0' : 'j6' ,
               'loc x=-L': 'j5'
               }

    ch_gegs['Oude Maas 4'] =  { 'Name' : 'Oude Maas 4' , #12
               'H' : 7 ,
               'L' : np.array([4260], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([426], dtype=float), #same length as L
               'loc x=0' : 'j7' ,
               'loc x=-L': 'j6'
               }

    ch_gegs['Oude Maas 3'] =  { 'Name' : 'Oude Maas 3' , #13
               'H' : 10.2 ,
               'L' : np.array([10000,5000], dtype=float),
               'b' : np.array([240,309,350], dtype=float), #one longer than L
               'dx' : np.array([500,100], dtype=float), #same length as L
               'loc x=0' : 'j8' ,
               'loc x=-L': 'j7'
               }

    ch_gegs['Nieuwe Merwede'] =  { 'Name' : 'Nieuwe Merwede' , #14
               'H' : 5 ,
               'L' : np.array([19600], dtype=float),
               'b' : np.array([400,730], dtype=float), #one longer than L
               'dx' : np.array([400], dtype=float), #same length as L
               'loc x=0' : 'j9' ,
               'loc x=-L': 'j5'
               }

    ch_gegs['Dordtsche Kil'] =  { 'Name' : 'Dordtsche Kil' , #15
               'H' : 10.7 ,
               'L' : np.array([9400], dtype=float),
               'b' : np.array([300,300], dtype=float), #one longer than L
               'dx' : np.array([470], dtype=float), #same length as L
               'loc x=0' : 'j7' ,
               'loc x=-L': 'j10'
               }

    ch_gegs['Spui'] =  { 'Name' : 'Spui' , #16
               'H' :  6.4,
               'L' : np.array([12000,10000,2800], dtype=float),
               'b' : np.array([250,250,250,250], dtype=float), #one longer than L
               'dx' : np.array([1000,500,100], dtype=float), #same length as L
               'loc x=0' : 'j8' ,
               'loc x=-L': 'j11'
               }

    ch_gegs['Maas'] =  { 'Name' : 'Maas' , #17
               'H' : 5.3 ,
               'L' : np.array([62300], dtype=float),
               'b' : np.array([97,416], dtype=float), #one longer than L
               'dx' : np.array([623], dtype=float), #same length as L
               'loc x=0' : 'j9' ,
               'loc x=-L': 'r2'
               }

    ch_gegs['Hollands Diep 2'] =  { 'Name' : 'Hollands Diep 2' , #18
               'H' : 6.2 ,
               'L' : np.array([3810], dtype=float),
               'b' : np.array([1600,1100], dtype=float), #one longer than L
               'dx' : np.array([381], dtype=float), #same length as L
               'loc x=0' : 'j10' ,
               'loc x=-L': 'j9'
               }

    ch_gegs['Hollands Diep 1'] =  { 'Name' : 'Hollands Diep 1' , #19
               'H' : 7.6 ,
               'L' : np.array([32000], dtype=float),
               'b' : np.array([1630,2000], dtype=float), #one longer than L
               'dx' : np.array([1000], dtype=float), #same length as L
               'loc x=0' : 'j11' ,
               'loc x=-L': 'j10'
               }

    ch_gegs['Haringvliet'] =  { 'Name' : 'Haringvliet' , #20
               'H' : 8.7 ,
               'L' : np.array([11700], dtype=float),
               'b' : np.array([2420,2420], dtype=float), #one longer than L
               'dx' : np.array([900], dtype=float), #same length as L
               'loc x=0' : 'h1' ,
               'loc x=-L': 'j11'
               }

    #optional: fix the tides
    for key in list(ch_gegs.keys()): ch_gegs[key]['Ut']= 1

    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================

    path_RM = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Geo_RijnMaas/'
    list_channels = [f for f in os.listdir(path_RM) if f.endswith('.kml')]
    RM_coords = {}
    for i in range(len(list_channels)):
        RM_coords[list_channels[i][:-4]] = np.array(kml_subtract_latlon(path_RM+list_channels[i]),dtype=float)

    cs = ['r','olive','b','dimgray','c','orange','g','m','y','gold','indigo','tan','skyblue','maroon','peru',\
          'beige','plum','silver','forestgreen','darkkhaki','rosybrown','teal','snow','aliceblue','mintcream']
    count=0
    for key in list(ch_gegs.keys()):
        ch_gegs[key]['plot x'] = np.flip(RM_coords[key][:,0])
        ch_gegs[key]['plot y'] = np.flip(RM_coords[key][:,1])
        ch_gegs[key]['plot color'] = cs[count]
        count+=1

    return ch_gegs

def geo_RMD9():
    # =============================================================================
    #     Input: geometry
    # =============================================================================
    ch_gegs = {}
    ch_gegs['Hollandse IJssel'] =  { 'Name' : 'Hollandse IJssel' , #1
               'H' : 4. ,
               'L' : np.array([19700], dtype=float),
               'b' : np.array([45,150], dtype=float), #one longer than L
               'dx' : np.array([197], dtype=float), #same length as L
               'loc x=0' : 'j2' ,
               'loc x=-L': 'w1'
               }

    ch_gegs['Lek'] =  { 'Name' : 'Lek' , #2
               'H' : 5.3 ,
               'L' : np.array([32000,10000], dtype=float),
               'b' : np.array([136,200,260], dtype=float), #one longer than L
               'dx' : np.array([1000,250], dtype=float), #same length as L
               'loc x=0' : 'j1' ,
               'loc x=-L': 'w2'
               }

    ch_gegs['Nieuwe Maas 2 old'] = { 'Name' : 'Nieuwe Maas 2' , #3
               'H' : 8.1 ,
               'L' : np.array([4900], dtype=float),
               'b' : np.array([250,400], dtype=float), #one longer than L
               'dx' : np.array([350], dtype=float), #same length as L
               'loc x=0' : 'j2' ,
               'loc x=-L': 'j1'
               }
    
    ch_gegs['Nieuwe Maas 1 old'] = { 'Name' : 'Nieuwe Maas 1' , #4
               'H' : 11 , #wel echt gemiddeld dit
               'L' : np.array([18750], dtype=float),
               'b' : np.array([400,500], dtype=float), #one longer than L
               'dx' : np.array([375], dtype=float), #same length as L
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j2'
               }    

    ch_gegs['Nieuwe Waterweg v2'] =  { 'Name' : 'Nieuwe Waterweg' , #5
               'H' : 16 ,
               'L' : np.array([16800], dtype=float),
               'b' : np.array([600,600], dtype=float), #one longer than L
               'dx' : np.array([200], dtype=float), #same length as L
               'loc x=0' : 'j12' ,
               'loc x=-L': 'j3'
               }

    ch_gegs['Noord'] =  { 'Name' : 'Noord' , #6
               'H' : 7 , #moeilijk vast te stellen
               'L' : np.array([8600], dtype=float),
               'b' : np.array([250,220], dtype=float), #one longer than L
               'dx' : np.array([430], dtype=float), #same length as L
               'loc x=0' : 'j6' ,
               'loc x=-L': 'j1'
               }

    ch_gegs['Oude Maas 1'] =  { 'Name' : 'Oude Maas 1' , #7
               'H' : 13 ,
               'L' : np.array([3100], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([310], dtype=float), #same length as L
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j4'
               }

    ch_gegs['Hartelkanaal v2'] =  { 'Name' : 'Hartelkanaal' ,#8
               'H' : 7.6,
               'L' : np.array([25800], dtype=float),
               'b' : np.array([310,1500], dtype=float), #one longer than L
               'dx' : np.array([300], dtype=float), #same length as L
               'loc x=0' : 'j12' ,
               'loc x=-L': 'j4'
               }

    ch_gegs['Breeddiep'] = { 'Name' : 'Breeddiep' ,#8
               'H' : 16 ,
               'L' : np.array([2500], dtype=float),
               'b' : np.array([1200,1200], dtype=float), #one longer than L
               'dx' : np.array([250], dtype=float), #same length as L
               'loc x=0' : 's1' ,
               'loc x=-L': 'j12'
               }

    ch_gegs['Oude Maas 2'] =  { 'Name' : 'Oude Maas 2' , #9
               'H' : 14 ,
               'L' : np.array([8250], dtype=float),
               'b' : np.array([317,317], dtype=float), #one longer than L
               'dx' : np.array([250], dtype=float), #same length as L
               'loc x=0' : 'j4' ,
               'loc x=-L': 'j8'
               }

    ch_gegs['Waal'] =  { 'Name' : 'Waal' , #10, Boven Merwede in bestand
               'H' : 4 ,
               'L' : np.array([45200], dtype=float),
               'b' : np.array([214,500], dtype=float), #one longer than L
               'dx' : np.array([1130], dtype=float), #same length as L
               'loc x=0' : 'j5' ,
               'loc x=-L': 'r1'
               }

    ch_gegs['Beneden Merwede'] =  { 'Name' : 'Beneden Merwede' , #11
               'H' : 6 ,
               'L' : np.array([15300], dtype=float),
               'b' : np.array([200/2,300], dtype=float), #one longer than L #groins not taken into account!
               'dx' : np.array([510], dtype=float), #same length as L
               'loc x=0' : 'j6' ,
               'loc x=-L': 'j5'
               }

    ch_gegs['Oude Maas 4'] =  { 'Name' : 'Oude Maas 4' , #12
               'H' : 7 ,
               'L' : np.array([4260], dtype=float),
               'b' : np.array([250,250], dtype=float), #one longer than L
               'dx' : np.array([426], dtype=float), #same length as L
               'loc x=0' : 'j7' ,
               'loc x=-L': 'j6'
               }

    ch_gegs['Oude Maas 3'] =  { 'Name' : 'Oude Maas 3' , #13
               'H' : 10.2 ,
               'L' : np.array([15000], dtype=float),
               'b' : np.array([240,350], dtype=float), #one longer than L
               'dx' : np.array([500], dtype=float), #same length as L
               'loc x=0' : 'j8' ,
               'loc x=-L': 'j7'
               }

    ch_gegs['Nieuwe Merwede'] =  { 'Name' : 'Nieuwe Merwede' , #14
               'H' : 5 ,
               'L' : np.array([19600], dtype=float),
               'b' : np.array([400,730], dtype=float), #one longer than L
               'dx' : np.array([400], dtype=float), #same length as L
               'loc x=0' : 'j9' ,
               'loc x=-L': 'j5'
               }

    ch_gegs['Dordtsche Kil'] =  { 'Name' : 'Dordtsche Kil' , #15
               'H' : 10.7 ,
               'L' : np.array([9400], dtype=float),
               'b' : np.array([300,300], dtype=float), #one longer than L
               'dx' : np.array([470], dtype=float), #same length as L
               'loc x=0' : 'j7' ,
               'loc x=-L': 'j10'
               }

    ch_gegs['Spui'] =  { 'Name' : 'Spui' , #16
               'H' :  6.4,
               'L' : np.array([17400,7400], dtype=float),
               'b' : np.array([250,250,250], dtype=float), #one longer than L
               'dx' : np.array([1000,370], dtype=float), #same length as L
               'loc x=0' : 'j8' ,
               'loc x=-L': 'j11'
               }

    ch_gegs['Maas'] =  { 'Name' : 'Maas' , #17
               'H' : 5.3 ,
               'L' : np.array([62300], dtype=float),
               'b' : np.array([97,416], dtype=float), #one longer than L
               'dx' : np.array([623], dtype=float), #same length as L
               'loc x=0' : 'j9' ,
               'loc x=-L': 'r2'
               }

    ch_gegs['Hollands Diep 2'] =  { 'Name' : 'Hollands Diep 2' , #18
               'H' : 6.2 ,
               'L' : np.array([3810], dtype=float),
               'b' : np.array([1600,1100], dtype=float), #one longer than L
               'dx' : np.array([381], dtype=float), #same length as L
               'loc x=0' : 'j10' ,
               'loc x=-L': 'j9'
               }

    ch_gegs['Hollands Diep 1'] =  { 'Name' : 'Hollands Diep 1' , #19
               'H' : 7.6 ,
               'L' : np.array([32000], dtype=float),
               'b' : np.array([1630,2000], dtype=float), #one longer than L
               'dx' : np.array([1000], dtype=float), #same length as L
               'loc x=0' : 'j11' ,
               'loc x=-L': 'j10'
               }

    ch_gegs['Haringvliet'] =  { 'Name' : 'Haringvliet' , #20
               'H' : 8.7 ,
               'L' : np.array([11700], dtype=float),
               'b' : np.array([2420,2420], dtype=float), #one longer than L
               'dx' : np.array([900], dtype=float), #same length as L
               'loc x=0' : 'h1' ,
               'loc x=-L': 'j11'
               }

    #optional: fix the tides
    for key in list(ch_gegs.keys()): ch_gegs[key]['Ut']= 1

    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================

    path_RM = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Geo_RijnMaas/'
    list_channels = [f for f in os.listdir(path_RM) if f.endswith('.kml')]
    RM_coords = {}
    for i in range(len(list_channels)):
        RM_coords[list_channels[i][:-4]] = np.array(kml_subtract_latlon(path_RM+list_channels[i]),dtype=float)

    cs = ['r','olive','b','dimgray','c','orange','g','m','y','gold','indigo','tan','skyblue','maroon','peru',\
          'beige','plum','silver','forestgreen','darkkhaki','rosybrown','teal','snow','aliceblue','mintcream']
    count=0
    for key in list(ch_gegs.keys()):
        ch_gegs[key]['plot x'] = np.flip(RM_coords[key][:,0])
        ch_gegs[key]['plot y'] = np.flip(RM_coords[key][:,1])
        ch_gegs[key]['plot color'] = cs[count]
        count+=1

    return ch_gegs


