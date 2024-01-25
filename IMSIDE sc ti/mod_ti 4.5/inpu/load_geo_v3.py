# =============================================================================
# In this file the geometry is loaded 
# 
# =============================================================================
import numpy as np



def geo_try3():
    
    Ln = np.array([50000,100000,5000,25000])
    bsn = np.array([1000,1000,1000,1000,1000*np.exp(10)])
    dxn= np.array([10000,10000,1000,5000])
    dxn= np.array([1000,500,100,100])
    #Hn = np.array([10,13,15,15])
    Hn = np.array([5,10,10,10])

    
    return Hn, Ln, bsn, dxn

def geo_GUA1():

    Ln = np.array([25000,10000,72000,3000,25000])
    bsn = np.array([150,250,450,500,3000,3000*np.exp(10)])
    dxn= np.array([1000,500,100,100,100])
    #dxn= np.array([100,100,100,100,100])
    
    Hn= np.array([7.1]*5)
    #Hn= np.array([7.1,7.1,7.1,7.1,10])
    return Hn, Ln, bsn, dxn

def geo_DLW1():
    
    Hn= np.array([7.4]*4)
    
    Ln = np.array([60000,142000,13000,25000])
    bsn = np.array([200,500,41500,18000,18000*np.exp(10)])
    dxn= np.array([2000,1000,250,200])
    
    return Hn, Ln, bsn, dxn    



def geo_LOI1():

    
    Ln = np.array([86600,5500,39300,14100,25000])
    bsn = np.array([110,500,250,1300,4000,4000*np.exp(10)])
    dxn= np.array([866,500,300,300,250])
    #dxn= np.array([866,500,50,50,100])
    #Hn = np.array([3.6,10,12,12,12])
    Hn = np.array([3.6,9,9,12,12])
    Hn = np.array([4.,9,9,12,12])
    
    
    return Hn, Ln, bsn, dxn