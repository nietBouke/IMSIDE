import numpy as np
#import matplotlib.pyplot as plt         
# =============================================================================
# Freshwater distribution in a random network
# Equations are generated by the computer
# From here it should go automatic...
# =============================================================================

def Qdist_calc(self, Qall, prt = False):
    #load Q
    Qriv, Qweir, Qhar, n_sea = Qall
    
    
    #some functions
    def calc_vol(b,L): 
        # =============================================================================
        # this function calculates the 'volume' of a channel, which is relevant for the water distribution
        # see eq. 12 in fork paper
        # =============================================================================
        vol_seg = np.zeros(len(L))
        for i in range(len(L)):
            if b[i]==b[i+1]: vol_seg[i] = L[i]/b[i+1]
            else: 
                Lb = L[i]/np.log(b[i]/b[i+1])
                vol_seg[i] = Lb/b[i+1]*(1-np.exp(-L[i]/Lb)) 
        return vol_seg
        
    def calc_facQ(Av,vol,H):
        facQtot = 0
        for dom in range(self.ch_pars[key]['n_seg']):
            facQtot += (6/5*Av*vol[dom])/(self.g*H[dom]**3)  
            
        #this function calculates what happens in eq. 11 in fork paper
        return facQtot
   
    def solvec_to_solch(solvec):
       # =============================================================================
       # this function converts the solution vector of river discharge a more easy form
       # form solvec is Q,r,s,w,h,j
       # form sol_ch is n_ch,3. For every channel: first the Q, then the eta at x=-L, then the eta at x=0
       # =============================================================================
       
       sol_ch = np.zeros((self.n_ch,3))
       #add discharge
       sol_ch[:,0] = solvec[:self.n_ch]
       
       for j in range(self.n_ch): 
           #x=-L parts
           i_here = int(self.ch_gegs[self.ch_keys[j]]['loc x=-L'][1:])-1
           if self.ch_gegs[self.ch_keys[j]]['loc x=-L'][0] == 'r': sol_ch[j,1] = solvec[self.n_ch+i_here] 
           elif self.ch_gegs[self.ch_keys[j]]['loc x=-L'][0] == 'w': sol_ch[j,1] = solvec[self.n_ch+self.n_r+i_here] 
           elif self.ch_gegs[self.ch_keys[j]]['loc x=-L'][0] == 'h': sol_ch[j,1] = solvec[self.n_ch+self.n_r+self.n_w+i_here] 
           elif self.ch_gegs[self.ch_keys[j]]['loc x=-L'][0] == 's': sol_ch[j,1] = solvec[self.n_ch+self.n_r+self.n_w+self.n_h+i_here] 
           elif self.ch_gegs[self.ch_keys[j]]['loc x=-L'][0] == 'j': sol_ch[j,1] = solvec[self.n_ch+self.n_r+self.n_w+self.n_h+self.n_s+i_here] 
           else: print('ERROR')
           
           #x=0 parts
           i_here = int(self.ch_gegs[self.ch_keys[j]]['loc x=0'][1:])-1
           if self.ch_gegs[self.ch_keys[j]]['loc x=0'][0] == 'r': sol_ch[j,2] = solvec[self.n_ch+i_here] 
           elif self.ch_gegs[self.ch_keys[j]]['loc x=0'][0] == 'w': sol_ch[j,2] = solvec[self.n_ch+self.n_r+i_here] 
           elif self.ch_gegs[self.ch_keys[j]]['loc x=0'][0] == 'h': sol_ch[j,2] = solvec[self.n_ch+self.n_r+self.n_w+i_here] 
           elif self.ch_gegs[self.ch_keys[j]]['loc x=0'][0] == 's': sol_ch[j,2] = solvec[self.n_ch+self.n_r+self.n_w+self.n_h+i_here] 
           elif self.ch_gegs[self.ch_keys[j]]['loc x=0'][0] == 'j': sol_ch[j,2] = solvec[self.n_ch+self.n_r+self.n_w+self.n_h+self.n_s+i_here] 
           else: print('ERROR')
           
       return sol_ch 
   
    #add properties for river discharge 
    for key in self.ch_keys:
        self.ch_pars[key]['vol']  = calc_vol(self.ch_gegs[key]['b'],self.ch_gegs[key]['L'])
        self.ch_pars[key]['facQ'] = calc_facQ(self.ch_pars[key]['Av'],self.ch_pars[key]['vol'],self.ch_gegs[key]['Hn'])
  
    # =============================================================================
    # We solve the system of equations for the discharge distrubtion using Newton-Raphson
    # It could also be solved directly, since it is a linear system, which is a bit easier. 
    # But I only realised that later, and this also works perfectly
    # =============================================================================
    #build solution vector
    def sol_RD(ans):
        #create empty solutin vector
        sol = np.zeros(self.n_unk)
        #convert unknown to nice format
        ans_ch = solvec_to_solch(ans)

        #add prescribed discharge
        for i in range(self.n_r): #rivers
            j=0
            while j < self.n_ch:
                if self.ch_gegs[self.ch_keys[j]]['loc x=0'] == 'r'+str(i+1) or self.ch_gegs[self.ch_keys[j]]['loc x=-L'] == 'r'+str(i+1) :
                    sol[i] = ans_ch[j,0] - Qriv[i]
                j=j+1  
        for i in range(self.n_w): #weirs
            j=0
            while j < self.n_ch:
                if self.ch_gegs[self.ch_keys[j]]['loc x=0'] == 'w'+str(i+1) or self.ch_gegs[self.ch_keys[j]]['loc x=-L'] == 'w'+str(i+1) :
                    sol[self.n_r+i] = ans_ch[j,0] - Qweir[i]
                j=j+1  
        for i in range(self.n_h): #hars
            j=0
            while j < self.n_ch:
                if self.ch_gegs[self.ch_keys[j]]['loc x=0'] == 'h'+str(i+1) or self.ch_gegs[self.ch_keys[j]]['loc x=-L'] == 'h'+str(i+1) :
                    sol[self.n_r+self.n_w+i] = ans_ch[j,0] - Qhar[i]
                j=j+1  
                
        #add prescribed sea level 
        for i in range(self.n_s):
            j=0
            while j < self.n_ch:
                if self.ch_gegs[self.ch_keys[j]]['loc x=-L'] == 's'+str(i+1) :
                    sol[self.n_r+self.n_w+self.n_h+i] = ans_ch[j,1] - n_sea[i]
                elif self.ch_gegs[self.ch_keys[j]]['loc x=0'] == 's'+str(i+1) :
                    sol[self.n_r+self.n_w+self.n_h+i] = ans_ch[j,2] - n_sea[i]
                j=j+1  
    
        
        #add condition at junctions
        for i in range(self.n_j):
            while j < self.n_ch:
                if self.ch_gegs[self.ch_keys[j]]['loc x=0'] == 'j'+str(i+1) :
                    sol[self.n_r+self.n_w+self.n_h+self.n_s+i] = sol[self.n_r+self.n_w+self.n_h+self.n_s+i] + ans_ch[j,0]
                elif self.ch_gegs[self.ch_keys[j]]['loc x=-L'] == 'j'+str(i+1) :
                    sol[self.n_r+self.n_w+self.n_h+self.n_s+i] = sol[self.n_r+self.n_w+self.n_h+self.n_s+i] - ans_ch[j,0]
                j=j+1  
                
                
        #calculate water level in channels 
        for i in range(self.n_ch):
            sol[self.n_r+self.n_w+self.n_h+self.n_s+self.n_j+i] = ans_ch[i,1]-ans_ch[i,2]-self.ch_pars[self.ch_keys[i]]['facQ'] * ans_ch[i,0]

        return sol
    
    #build Jacobian
    def jac_RD(ans):
        jac = np.zeros((self.n_unk,self.n_unk))
    
        #add prescribed discharge
        for i in range(self.n_r): #river channels
            j=0
            while j < self.n_ch:
                if self.ch_gegs[self.ch_keys[j]]['loc x=0'] == 'r'+str(i+1) or self.ch_gegs[self.ch_keys[j]]['loc x=-L'] == 'r'+str(i+1) :
                    jac[i,j] = 1
                j=j+1  
        for i in range(self.n_w): #weir channels
            j=0
            while j < self.n_ch:
                if self.ch_gegs[self.ch_keys[j]]['loc x=0'] == 'w'+str(i+1) or self.ch_gegs[self.ch_keys[j]]['loc x=-L'] == 'w'+str(i+1) :
                    jac[self.n_r+i,j] = 1
                j=j+1  
        for i in range(self.n_h): #har channels
            j=0
            while j < self.n_ch:
                if self.ch_gegs[self.ch_keys[j]]['loc x=0'] == 'h'+str(i+1) or self.ch_gegs[self.ch_keys[j]]['loc x=-L'] == 'h'+str(i+1) :
                    jac[self.n_r+self.n_w+i,j] = 1
                j=j+1  

        #add prescribed sea level 
        for i in range(self.n_s):
            j=0
            while j < self.n_ch:
                if self.ch_gegs[self.ch_keys[j]]['loc x=-L'] == 's'+str(i+1) :
                    ji_here = self.n_ch+self.n_r+self.n_w+self.n_h+int(self.ch_gegs[self.ch_keys[j]]['loc x=-L'][1:])-1
                    jac[self.n_r+self.n_w+self.n_h+i,ji_here] = 1
                    
                elif self.ch_gegs[self.ch_keys[j]]['loc x=0'] == 's'+str(i+1) :
                    ji_here = self.n_ch+self.n_r+self.n_w+self.n_h+int(self.ch_gegs[self.ch_keys[j]]['loc x=0'][1:])-1
                    jac[self.n_r+self.n_w+self.n_h+i,ji_here] = 1
                j=j+1  
    
        #add conditions at junctions
        for i in range(self.n_j):
            j=0
            while j < self.n_ch:
                if self.ch_gegs[self.ch_keys[j]]['loc x=0'] == 'j'+str(i+1) :
                    jac[self.n_r+self.n_w+self.n_h+self.n_s+i,j] = 1
                elif self.ch_gegs[self.ch_keys[j]]['loc x=-L'] == 'j'+str(i+1) :
                    jac[self.n_r+self.n_w+self.n_h+self.n_s+i,j] = -1
                j=j+1  
                
        #convert to water level in channels 
        for i in range(self.n_ch):
            #x=-L
            i_here = int(self.ch_gegs[self.ch_keys[i]]['loc x=-L'][1:])-1
            if self.ch_gegs[self.ch_keys[i]]['loc x=-L'][0] == 'r': ji_here = self.n_ch+i_here
            elif self.ch_gegs[self.ch_keys[i]]['loc x=-L'][0] == 'w': ji_here = self.n_ch+self.n_r+i_here
            elif self.ch_gegs[self.ch_keys[i]]['loc x=-L'][0] == 'h': ji_here = self.n_ch+self.n_r+self.n_w+i_here
            elif self.ch_gegs[self.ch_keys[i]]['loc x=-L'][0] == 's': ji_here = self.n_ch+self.n_r+self.n_w+self.n_h+i_here
            elif self.ch_gegs[self.ch_keys[i]]['loc x=-L'][0] == 'j': ji_here = self.n_ch+self.n_r+self.n_w+self.n_h+self.n_s+i_here
            else: print('ERROR')
            jac[self.n_r+self.n_w+self.n_h+self.n_s+self.n_j+i,ji_here] = 1
            #x=0
            i_here = int(self.ch_gegs[self.ch_keys[i]]['loc x=0'][1:])-1
            if self.ch_gegs[self.ch_keys[i]]['loc x=0'][0] == 'r': ji_here = self.n_ch + i_here
            elif self.ch_gegs[self.ch_keys[i]]['loc x=0'][0] == 'w': ji_here = self.n_ch+self.n_r+i_here
            elif self.ch_gegs[self.ch_keys[i]]['loc x=0'][0] == 'h': ji_here = self.n_ch+self.n_r+self.n_w+i_here
            elif self.ch_gegs[self.ch_keys[i]]['loc x=0'][0] == 's': ji_here = self.n_ch+self.n_r+self.n_w+self.n_h+i_here
            elif self.ch_gegs[self.ch_keys[i]]['loc x=0'][0] == 'j': ji_here = self.n_ch+self.n_r+self.n_w+self.n_h+self.n_s+i_here
            else: print('ERROR')
            jac[self.n_r+self.n_w+self.n_h+self.n_s+self.n_j+i,ji_here] = -1
            #discharge 
            jac[self.n_r+self.n_w+self.n_h+self.n_s+self.n_j+i,i] = -self.ch_pars[self.ch_keys[i]]['facQ']

        return jac
    
    
    #implement Newton-Raphson algoritm
    init = np.zeros(self.n_unk)
    #do the first iteration
    jaco, solu = jac_RD(init) , sol_RD(init)
    rd_n = init - np.linalg.solve(jaco,solu)  # this is faster then np.linalg.solve (at least for a sufficiently large matrix)
    #print(rd_n)
    t=1
    #print('That was iteration step ', t)
    rd=init
    
    # do the rest of the iterations
    while np.max(np.abs(rd-rd_n))>10e-6: #check whether the algoritm has converged
        rd = rd_n.copy() #update
        jaco = jac_RD(rd) 
        solu = sol_RD(rd)
        rd_n = rd - np.linalg.solve(jaco,solu)  
          
        #if t==1: 
            #print(jaco)
            #print(solu)
            #print(rd_n)
        
        t=1+t
        #print('That was iteration step ', t)
        if t>=10: break
    
    if t<10:
        if prt == True: print('The algoritm for river discharge has converged ')  
    else:
        print('ERROR: no convergence')
    
    if prt == True: print('The river discharge over the channels is, respectively:',rd[:self.n_ch], '\n')
    
    #add the calculated river discharge to the object so it can be used for the salinity calculations
    
    Qsave = {}
    
    count = 0
    for key in self.ch_keys:
        Qsave[key] = rd[count]
        count+=1
    
    
    return Qsave

   


