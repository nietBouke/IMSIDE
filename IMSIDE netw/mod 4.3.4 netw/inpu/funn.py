# =============================================================================
#  properties of the imaginary delta 
# would be fun if it existed
# but the humour of morphodynamics is more complicated than mine
# =============================================================================

import numpy as np


def geo_fun():
    nn = 11
    
    # =============================================================================
    #     Input: geometry
    # =============================================================================
    ch_gegs = {}
    '''
    ch_gegs['C1'] =  { 'Name' : 'C1' , #1
               'H' : 10. ,
               'L' : np.array([10000], dtype=float),
               'b' : np.array([1000,1000], dtype=float), #one longer than L
               'dx' : np.array([1000], dtype=float), #same length as L
               'Ut' : None ,
               'loc x=0' : 'j2' ,
               'loc x=-L': 'w1'
               }
    '''
    nchan = 32
    for ch in range(1,nchan+1):
        ch_gegs['C'+str(ch)] =  { 'Name' : 'C'+str(ch) , 
                   'H' : 10. ,
                   'L' : np.array([10000], dtype=float),
                   'b' : np.array([1000,1000], dtype=float), #one longer than L
                   'dx' : np.array([1000], dtype=float), #same length as L
                   'Ut' : None ,
                   'plot color': 'black'
                   }
    
    
    #connections - elements of the adjecency matrix
    ch_gegs['C1']['loc x=0'] = 'j1'
    ch_gegs['C1']['loc x=-L'] = 'r1'
    ch_gegs['C1']['plot x'] = np.linspace(0,1,nn)
    ch_gegs['C1']['plot y'] = np.linspace(6,5,nn)
    
    ch_gegs['C2']['loc x=0'] = 'j1'
    ch_gegs['C2']['loc x=-L'] = 'j2'
    ch_gegs['C2']['plot x'] = np.linspace(2,1,nn)
    ch_gegs['C2']['plot y'] = np.linspace(5,5,nn)
    
    ch_gegs['C3']['loc x=0'] = 'j2'
    ch_gegs['C3']['loc x=-L'] = 'j3'
    ch_gegs['C3']['plot x'] = np.linspace(3,2,nn)
    ch_gegs['C3']['plot y'] = np.linspace(5,5,nn)
    
    ch_gegs['C4']['loc x=0'] = 'j3'
    ch_gegs['C4']['loc x=-L'] = 'j4'
    ch_gegs['C4']['plot x'] = np.linspace(4,3,nn)
    ch_gegs['C4']['plot y'] = np.linspace(5,5,nn)
    
    ch_gegs['C5']['loc x=0'] = 'j4'
    ch_gegs['C5']['loc x=-L'] = 'j5'
    ch_gegs['C5']['plot x'] = np.linspace(5,4,nn)
    ch_gegs['C5']['plot y'] = np.linspace(5,5,nn)
    
    ch_gegs['C6']['loc x=0'] = 's1'
    ch_gegs['C6']['loc x=-L'] = 'j5'
    ch_gegs['C6']['plot x'] = np.linspace(5,6,nn)
    ch_gegs['C6']['plot y'] = np.linspace(5,6,nn)
    
    ch_gegs['C7']['loc x=0'] = 'j1'
    ch_gegs['C7']['loc x=-L'] = 'j6'
    ch_gegs['C7']['plot x'] = np.linspace(1,1,nn)
    ch_gegs['C7']['plot y'] = np.linspace(4,5,nn)
    
    ch_gegs['C8']['loc x=0'] = 'j2'
    ch_gegs['C8']['loc x=-L'] = 'j6'
    ch_gegs['C8']['plot x'] = np.linspace(1,2,nn)
    ch_gegs['C8']['plot y'] = np.linspace(4,5,nn)
    
    ch_gegs['C9']['loc x=0'] = 'j3'
    ch_gegs['C9']['loc x=-L'] = 'j7'
    ch_gegs['C9']['plot x'] = np.linspace(3,3,nn)
    ch_gegs['C9']['plot y'] = np.linspace(4,5,nn)
    
    ch_gegs['C10']['loc x=0'] = 'j4'
    ch_gegs['C10']['loc x=-L'] = 'j8'
    ch_gegs['C10']['plot x'] = np.linspace(5,4,nn)
    ch_gegs['C10']['plot y'] = np.linspace(4,5,nn)
    
    ch_gegs['C11']['loc x=0'] = 'j5'
    ch_gegs['C11']['loc x=-L'] = 'j8'
    ch_gegs['C11']['plot x'] = np.linspace(5,5,nn)
    ch_gegs['C11']['plot y'] = np.linspace(4,5,nn)
    
    ch_gegs['C12']['loc x=0'] = 'j6'
    ch_gegs['C12']['loc x=-L'] = 'j9'
    ch_gegs['C12']['plot x'] = np.linspace(1,1,nn)
    ch_gegs['C12']['plot y'] = np.linspace(3,4,nn)
    
    ch_gegs['C13']['loc x=0'] = 'j7'
    ch_gegs['C13']['loc x=-L'] = 'j10'
    ch_gegs['C13']['plot x'] = np.linspace(2,3,nn)
    ch_gegs['C13']['plot y'] = np.linspace(3,4,nn)
    
    ch_gegs['C14']['loc x=0'] = 'j7'
    ch_gegs['C14']['loc x=-L'] = 'j11'
    ch_gegs['C14']['plot x'] = np.linspace(4,3,nn)
    ch_gegs['C14']['plot y'] = np.linspace(3,4,nn)
    
    ch_gegs['C15']['loc x=0'] = 'j8'
    ch_gegs['C15']['loc x=-L'] = 'j12'
    ch_gegs['C15']['plot x'] = np.linspace(5,5,nn)
    ch_gegs['C15']['plot y'] = np.linspace(3,4,nn)
    
    ch_gegs['C16']['loc x=0'] = 'j9'
    ch_gegs['C16']['loc x=-L'] = 'j10'
    ch_gegs['C16']['plot x'] = np.linspace(2,1,nn)
    ch_gegs['C16']['plot y'] = np.linspace(3,3,nn)
    
    ch_gegs['C17']['loc x=0'] = 'j11'
    ch_gegs['C17']['loc x=-L'] = 'j12'
    ch_gegs['C17']['plot x'] = np.linspace(5,4,nn)
    ch_gegs['C17']['plot y'] = np.linspace(3,3,nn)
    
    ch_gegs['C18']['loc x=0'] = 'j9'
    ch_gegs['C18']['loc x=-L'] = 'j14'
    ch_gegs['C18']['plot x'] = np.linspace(1,1,nn)
    ch_gegs['C18']['plot y'] = np.linspace(2,3,nn)
    
    ch_gegs['C19']['loc x=0'] = 'j10'
    ch_gegs['C19']['loc x=-L'] = 'j13'
    ch_gegs['C19']['plot x'] = np.linspace(3,2,nn)
    ch_gegs['C19']['plot y'] = np.linspace(2,3,nn)
    
    ch_gegs['C20']['loc x=0'] = 'j11'
    ch_gegs['C20']['loc x=-L'] = 'j13'
    ch_gegs['C20']['plot x'] = np.linspace(3,4,nn)
    ch_gegs['C20']['plot y'] = np.linspace(2,3,nn)
    
    ch_gegs['C21']['loc x=0'] = 'j12'
    ch_gegs['C21']['loc x=-L'] = 'j15'
    ch_gegs['C21']['plot x'] = np.linspace(5,5,nn)
    ch_gegs['C21']['plot y'] = np.linspace(2,3,nn)
    
    ch_gegs['C22']['loc x=0'] = 'j14'
    ch_gegs['C22']['loc x=-L'] = 'j16'
    ch_gegs['C22']['plot x'] = np.linspace(1,1,nn)
    ch_gegs['C22']['plot y'] = np.linspace(1,2,nn)
    
    ch_gegs['C23']['loc x=0'] = 'j14'
    ch_gegs['C23']['loc x=-L'] = 'j17'
    ch_gegs['C23']['plot x'] = np.linspace(2,1,nn)
    ch_gegs['C23']['plot y'] = np.linspace(1,2,nn)
    
    ch_gegs['C24']['loc x=0'] = 'j13'
    ch_gegs['C24']['loc x=-L'] = 'j18'
    ch_gegs['C24']['plot x'] = np.linspace(3,3,nn)
    ch_gegs['C24']['plot y'] = np.linspace(1,2,nn)
    
    ch_gegs['C25']['loc x=0'] = 'j15'
    ch_gegs['C25']['loc x=-L'] = 'j19'
    ch_gegs['C25']['plot x'] = np.linspace(4,5,nn)
    ch_gegs['C25']['plot y'] = np.linspace(1,2,nn)
    
    ch_gegs['C26']['loc x=0'] = 'j15'
    ch_gegs['C26']['loc x=-L'] = 'j20'
    ch_gegs['C26']['plot x'] = np.linspace(5,5,nn)
    ch_gegs['C26']['plot y'] = np.linspace(1,2,nn)
    
    ch_gegs['C27']['loc x=0'] = 's2'
    ch_gegs['C27']['loc x=-L'] = 'j16'
    ch_gegs['C27']['plot x'] = np.linspace(1,0,nn)
    ch_gegs['C27']['plot y'] = np.linspace(1,0,nn)
    
    ch_gegs['C28']['loc x=0'] = 'j17'
    ch_gegs['C28']['loc x=-L'] = 'j16'
    ch_gegs['C28']['plot x'] = np.linspace(1,2,nn)
    ch_gegs['C28']['plot y'] = np.linspace(1,1,nn)
    
    ch_gegs['C29']['loc x=0'] = 'j18'
    ch_gegs['C29']['loc x=-L'] = 'j17'
    ch_gegs['C29']['plot x'] = np.linspace(2,3,nn)
    ch_gegs['C29']['plot y'] = np.linspace(1,1,nn)
    
    ch_gegs['C30']['loc x=0'] = 'j19'
    ch_gegs['C30']['loc x=-L'] = 'j18'
    ch_gegs['C30']['plot x'] = np.linspace(3,4,nn)
    ch_gegs['C30']['plot y'] = np.linspace(1,1,nn)
    
    ch_gegs['C31']['loc x=0'] = 'j20'
    ch_gegs['C31']['loc x=-L'] = 'j19'
    ch_gegs['C31']['plot x'] = np.linspace(4,5,nn)
    ch_gegs['C31']['plot y'] = np.linspace(1,1,nn)
    
    ch_gegs['C32']['loc x=0'] = 'j20'
    ch_gegs['C32']['loc x=-L'] = 'r2'   
    ch_gegs['C32']['plot x'] = np.linspace(6,5,nn)
    ch_gegs['C32']['plot y'] = np.linspace(0,1,nn)
    
    
    '''
    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================
    ch_gegs['C1']['plot x'] = np.linspace(0,-ch_gegs['C1']['L'].sum(),101)
    ch_gegs['C1']['plot y'] = np.zeros(101)
    ch_gegs['C1']['plot color'] = 'r'

    ch_gegs['C2']['plot x'] = np.linspace(ch_gegs['C2']['L'].sum(),0,101)
    ch_gegs['C2']['plot y'] = np.linspace(ch_gegs['C2']['L'].sum(),0,101)
    ch_gegs['C2']['plot color'] = 'b'

    ch_gegs['C3']['plot x'] = np.linspace(ch_gegs['C3']['L'].sum(),0,101)
    ch_gegs['C3']['plot y'] = -np.linspace(ch_gegs['C3']['L'].sum(),0,101)
    ch_gegs['C3']['plot color'] = 'black'
    '''

    
    
    
    
    
    return ch_gegs
    
    #print(ch_gegs)
    
#inp_fundelta1_geo()