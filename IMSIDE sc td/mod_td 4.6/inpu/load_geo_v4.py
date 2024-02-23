# =============================================================================
# In this file the geometry is loaded 
# 
# =============================================================================
import numpy as np

def geo_GUA2():
    
    
    Ln = np.array([25000,10000,72000,3000,25000])
    bsn = np.array([150,250,450,500,3000,3000*np.exp(10)])
    dxn= np.array([1000,500,500,250,250])
    #dxn= np.array([1000,2000,2000,500,1000])
    Hn= np.array([7.1]*5)
    #Hn= np.array([7.1,7.1,6.8,7.5,7.1])

    return Hn, Ln, bsn, dxn


def geo_DLW2():
    
    Hn= np.array([7.4]*4)

    
    Ln = np.array([60000,142000,13000,25000])
    bsn = np.array([200,500,41500,18000,18000*np.exp(10)])
    dxn= np.array([3000,1000,500,500])
    
    return Hn, Ln, bsn, dxn




def geo_LOI2():
    #bit more rounded numbers so we can work with cruder grid

    Ln = np.array([87000,5500,39600,14000,25000])
    bsn = np.array([110,500,250,1300,4000,4000*np.exp(10)])
    dxn= np.array([8700,500,300,250,250])
    #dxn= np.array([866,500,100,100,100])
    #Hn = np.array([3.6,10,12,12,12])
    Hn = np.array([2.5,9,9,12,12])
    
    
    return Hn, Ln, bsn, dxn