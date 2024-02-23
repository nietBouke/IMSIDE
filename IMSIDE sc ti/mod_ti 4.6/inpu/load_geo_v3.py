# =============================================================================
# In this file the geometry is loaded 
# 
# =============================================================================
import numpy as np



def geo_try1():
    
    Ln = np.array([90000,10000,10000,25000])
    bsn = np.array([1000,1000,1000,1000,1000*np.exp(10)])
    #dxn= np.array([1000,100,50,100])
    dxn= np.array([1000,500,500,250])
    #Hn = np.array([10,13,15,15])
    Hn = np.array([10,7,10,10])

    
    return Hn, Ln, bsn, dxn

def geo_try2():
    
    Ln = np.array([90000,10000,10000,25000])
    bsn = np.array([1000,1000,1000,1000,1000*np.exp(10)])
    #dxn= np.array([1000,100,50,100])
    dxn= np.array([1000,500,500,250])*5
    #Hn = np.array([10,13,15,15])
    Hn = np.array([10,9,10,10])

    
    return Hn, Ln, bsn, dxn


def geo_GUA1():

    Ln = np.array([25000,10000,72000,3000,25000])
    bsn = np.array([150,250,450,500,3000,3000*np.exp(10)])
    #dxn= np.array([1000,500,200,100,100])
    dxn= np.array([1000,500,360,200,200])
    #dxn= np.array([5000,500,500,250,250])
    
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
    #Hn = np.array([3.6,9,9,12,12])
    Hn = np.array([2.5,9,9,12,12])
    
    
    return Hn, Ln, bsn, dxn

def geo_GUA2():
    
    Ln = np.array([25000,10000,72000,3000,25000])
    bsn = np.array([150,250,500,450,3000,3000*np.exp(10)])
    dxn= np.array([1000,500,720,300,250])
    
    Ln = np.array([25000,10000,12000,10000,10000,10000,10000,10000,10000,3000,25000])
    bsn = np.array([150,250,450,520,400,470,520,450,440,500,3000,3000*np.exp(10)])
    dxn= np.array([1000,500,1000,1000,1000,1000,1000,1000,1000,300,250])
    
    
    Hn= np.array([7.1]*11)
    return Hn, Ln, bsn, dxn