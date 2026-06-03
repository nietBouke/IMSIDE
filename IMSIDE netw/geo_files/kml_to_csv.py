#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:03:13 2022

@author: biemo004
"""
import numpy as np
from bs4 import BeautifulSoup
import os

list_channels = [f for f in os.listdir() if f.endswith('.kml')]

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

RM_coords = {}
for i in range(len(list_channels)):
    RM_coords[list_channels[i][:-4]] = np.array(kml_subtract_latlon(list_channels[i]),dtype=float)
