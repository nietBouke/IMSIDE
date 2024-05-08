# =============================================================================
# here we define the contirbution to the jacobian and solution vector by the boundaries 
# also the equations for the boundary layer correction are here
# =============================================================================
import numpy as np

def solu_bnd_subtidal(self, ans, pars_st, indi, version):

    so = np.zeros(self.di3[-1]*self.M)
    
    #river boundary - everything is zero
    #so[2*self.M] = ans[2*self.M] - self.sri / self.soc
    #so[2*self.M+1:3*self.M] = ans[2*self.M+1:3*self.M]

    #river boundary 2 - total transport is Q*sriv
    so[self.M*2] = pars_st['C14a'] * (ans[self.M*2]- self.sri/self.soc) + pars_st['C14d'] * (-3*ans[2*self.M] + 4*ans[3*self.M] - ans[4*self.M]) /(2*self.dl[[0]]) 
    
    if version in ['A','C','D']:
        so[self.M*2] += np.sum([pars_st['C14b'][n-1] * ans[2*self.M+n] for n in range(1,self.M) ]) \
        + np.sum([pars_st['C14c'][n-1] * ans[2*self.M+n] for n in range(1,self.M) ]) * (-3*ans[2*self.M] + 4*ans[3*self.M] - ans[4*self.M]) /(2*self.dl[[0]])
        
    so[2*self.M+1:3*self.M] = (-3*ans[2*self.M+1:3*self.M] + 4*ans[3*self.M+1:4*self.M] - ans[4*self.M+1:5*self.M]) /(2*self.dl[[0]])
    
    #sea boundary
    so[self.M*(self.di3[-1]-3)] = 1 - ans[self.M*(self.di3[-1]-3)]
    so[self.M*(self.di3[-1]-3)+1:self.M*(self.di3[-1]-2)] = ans[self.M*(self.di3[-1]-3)+1:self.M*(self.di3[-1]-2)]
    
    # =============================================================================
    # inner boundary - salinities equal 
    # =============================================================================
    so[indi['bnd_lft']] = ans[indi['bnd_rgt']]  - ans[indi['bnd_lft']]

    '''
    for ib in range(self.ndom-1):
        #depth-averaged salinity always equal 
        so[indi['i_s0'][ib]] = ans[indi['i_s0'][ib]] - ans[indi['i_s_1'][ib]]
        
        #on vertical levels 
        H1, H2 = self.Hn[ib],self.Hn[ib+1]
    
        sn1 = ans[indi['i_s0'][ib]+np.arange(1,self.M)]
        sn2 = ans[indi['i_s_1'][ib]+np.arange(1,self.M)]

        if H1>H2: 
            z_here = np.linspace(-H2,0,self.N)
            for m in range(self.N): so[indi['i_s0'][ib]+1+m] = np.sum(sn1 * np.cos(np.pi*np.arange(1,self.M)/H1*z_here[m])) - np.sum(sn2 * np.cos(np.pi*np.arange(1,self.M)/H2*z_here[m]))
        elif H2>H1:
            z_here = np.linspace(-H1,0,self.N)
            for m in range(self.N): so[indi['i_s0'][ib]+1+m] = np.sum(sn1 * np.cos(np.pi*np.arange(1,self.M)/H1*z_here[m])) - np.sum(sn2 * np.cos(np.pi*np.arange(1,self.M)/H2*z_here[m]))
        else: #depths are equal
            so[indi['i_s0'][ib]+np.arange(1,self.M)] = sn1 - sn2
    '''
    # =============================================================================
    # inner boundary - flux - subtidal contribution
    # =============================================================================
    #salinities at boundaries
    sb_1 = ans[indi['i_s_1']]
    sb_2 = ans[(self.di3[1:-1]-4) * self.M]
    sb_3 = ans[(self.di3[1:-1]-5) * self.M]
    sn_1 = ans[((self.di3[1:-1]-3)*self.M+np.arange(1,self.M)[:,np.newaxis]).T]
    sn_2 = ans[((self.di3[1:-1]-4)*self.M+np.arange(1,self.M)[:,np.newaxis]).T]
    sn_3 = ans[((self.di3[1:-1]-5)*self.M+np.arange(1,self.M)[:,np.newaxis]).T]
    
    sb0 = ans[(self.di3[1:-1]+2) * self.M]
    sb1 = ans[(self.di3[1:-1]+3) * self.M]
    sb2 = ans[(self.di3[1:-1]+4) * self.M]
    sn0 = ans[((self.di3[1:-1]+2)*self.M+np.arange(1,self.M)[:,np.newaxis]).T]
    sn1 = ans[((self.di3[1:-1]+3)*self.M+np.arange(1,self.M)[:,np.newaxis]).T]
    sn2 = ans[((self.di3[1:-1]+4)*self.M+np.arange(1,self.M)[:,np.newaxis]).T]

    #flux equal
    #depth-averaged flux
    so[indi['i_s_1']] += pars_st['C13a_1'] * sb_1 + pars_st['C13d_1'] * (3*sb_1 - 4*sb_2 + sb_3)/(2*self.dl[self.di[ :-2]]) - (pars_st['C13a0'] * sb0 + pars_st['C13d0'] * (-3*sb0 + 4*sb1 - sb2)/(2*self.dl[self.di[1:-1]]))
    
    
    if version in ['A','C','D']:
        so[indi['i_s_1']] += np.sum(pars_st['C13b_1'] * sn_1 + pars_st['C13c_1'] * sn_1 * (3*sb_1 - 4*sb_2 + sb_3)[:,np.newaxis]/(2*self.dl[self.di[ :-2],np.newaxis]) ,1)  - \
                            (np.sum(pars_st['C13b0'] * sn0 + pars_st['C13c0'] * sn0 * (-3*sb0 + 4*sb1 - sb2)[:,np.newaxis]/(2*self.dl[self.di[1:-1],np.newaxis]) ,1) )

    
    #print(pars_st['C13a_1'] * sb_1 * self.soc)
    #print( pars_st['C13a0'] * sb0 * self.soc)
    #print(pars_st['C13d_1'] * (3*sb_1 - 4*sb_2 + sb_3)/(2*self.dl[self.di[ :-2]]) * self.soc)
    #print(pars_st['C13d0'] * (-3*sb0 + 4*sb1 - sb2)/(2*self.dl[self.di[1:-1]]) * self.soc)

    #depth-perturbed flux    
    #if version in ['A','B','C']:
    #    dpf = pars_st['C15a_1'][:,np.newaxis] * (3*sn_1 - 4*sn_2 + sn_3)/(2*self.dl[self.di[ :-2],np.newaxis]) \
    #        - pars_st['C15a0'][:,np.newaxis] * (-3*sn0 + 4*sn1 - sn2)/(2*self.dl[self.di[1:-1],np.newaxis]) 
                
    if version in ['A','B','C','D']:
        dpf = pars_st['C15a_1'][:,np.newaxis] * (3*sn_1 - 4*sn_2 + sn_3)/(2*self.dl[self.di[ :-2],np.newaxis]) \
             + pars_st['C15b_1'] * sn_1 * (3*sn_1 - 4*sn_2 + sn_3)/(2*self.dl[self.di[ :-2],np.newaxis]) \
             + np.sum([pars_st['C15c_1'][:,n] * sn_1[:,n,np.newaxis] for n in range(self.N)] ,0) * ((3*sb_1 - 4*sb_2 + sb_3)/(2*self.dl[self.di[ :-2]]))[:,np.newaxis] \
             + pars_st['C15d_1']*sn_1 \
             + np.sum([pars_st['C15e_1'][:,:,n] * sn_1[:,n,np.newaxis] for n in range(self.N)] ,0) \
             - (pars_st['C15a0'][:,np.newaxis] * (-3*sn0 + 4*sn1 - sn2)/(2*self.dl[self.di[1:-1],np.newaxis]) \
                 + pars_st['C15b0'] * sn0 * (-3*sn0 + 4*sn1 - sn2)/(2*self.dl[self.di[1:-1],np.newaxis]) \
                 + np.sum([pars_st['C15c0'][:,n] * sn0[:,n,np.newaxis] for n in range(self.N)] ,0) * ((-3*sb0 + 4*sb1 - sb2)/(2*self.dl[self.di[1:-1]]))[:,np.newaxis] \
                 + pars_st['C15d0']*sn0 \
                 + np.sum([pars_st['C15e0'][:,:,n] * sn0[:,n,np.newaxis] for n in range(self.N)] ,0)           
                 )

    so[indi['bnd2_rgt']] += dpf.flatten()
        
    return so


def jaco_bnd_subtidal(self, ans, pars_st, indi, version):
    
    jac = np.zeros((self.di3[-1]*self.M,self.di3[-1]*self.M))
    #river boundary - salt is prescribed
    #jac[np.arange(self.M)+2*self.M,np.arange(self.M)+2*self.M] = jac[np.arange(self.M)+2*self.M,np.arange(self.M)+2*self.M] + 1
    #river boundary - flux is Q*sriv
    
    jac[2*self.M, 2*self.M] = pars_st['C14a'] - 3*pars_st['C14d'] /(2*self.dl[[0]])
    jac[2*self.M, 3*self.M] = 4*pars_st['C14d'] /(2*self.dl[[0]])
    jac[2*self.M, 4*self.M] =-1*pars_st['C14d'] /(2*self.dl[[0]])

    if version in ['A','C','D']:
        jac[2*self.M, 2*self.M] += - 3/(2*self.dl[[0]]) *np.sum([pars_st['C14c'][n-1] * ans[2*self.M+n] for n in range(1,self.M) ]) 
        jac[2*self.M, 3*self.M] +=   4/(2*self.dl[[0]])  * np.sum([pars_st['C14c'][n-1] * ans[2*self.M+n] for n in range(1,self.M) ]) 
        jac[2*self.M, 4*self.M] += - 1/(2*self.dl[[0]])  * np.sum([pars_st['C14c'][n-1] * ans[2*self.M+n] for n in range(1,self.M) ]) 
        jac[2*self.M, 2*self.M + np.arange(1,self.M)] = pars_st['C14b'] + pars_st['C14c'] * (-3*ans[2*self.M] + 4*ans[3*self.M] - ans[4*self.M]) /(2*self.dl[[0]]) 

    jac[2*self.M + np.arange(1,self.M), 2*self.M + np.arange(1,self.M)] = -3/(2*self.dl[[0]]) 
    jac[2*self.M + np.arange(1,self.M), 3*self.M + np.arange(1,self.M)] =  4/(2*self.dl[[0]]) 
    jac[2*self.M + np.arange(1,self.M), 4*self.M + np.arange(1,self.M)] = -1/(2*self.dl[[0]]) 
    

    # sea boundary 
    jac[self.M*(self.di3[-1]-3),self.M*(self.di3[-1]-3)] = jac[self.M*(self.di3[-1]-3),self.M*(self.di3[-1]-3)] - 1
    jac[np.arange(self.M*(self.di3[-1]-3)+1,self.M*(self.di3[-1]-2)),np.arange(self.M*(self.di3[-1]-3)+1,self.M*(self.di3[-1]-2))] =  1

    
    # =============================================================================
    # inner boundary - salinity equal
    # =============================================================================
    jac[indi['bnd_lft'], indi['bnd_lft']] = -1                
    jac[indi['bnd_lft'], indi['bnd_rgt']] = 1
    '''
    for ib in range(self.ndom-1):
        #depth-averaged salinity always equal 
        jac[indi['i_s0'][ib],indi['i_s0'][ib]] = 1
        jac[indi['i_s0'][ib],indi['i_s_1'][ib]] = -1
        
        #on vertical levels 
        H1, H2 = self.Hn[ib],self.Hn[ib+1]
    
        if H1>H2: 
            z_here = np.linspace(-H2,0,self.N)
            for m in range(self.N): 
                for k in range(self.N): 
                    jac[indi['i_s0'][ib]+1+m , indi['i_s0'][ib]+1+k]  += np.cos(np.pi*(k+1)/H1*z_here[m])
                    jac[indi['i_s0'][ib]+1+m , indi['i_s_1'][ib]+1+k] += -np.cos(np.pi*(k+1)/H2*z_here[m])
                
        elif H2>H1:
            z_here = np.linspace(-H1,0,self.N)
            for m in range(self.N): 
                for k in range(self.N): 
                    jac[indi['i_s0'][ib]+1+m , indi['i_s0'][ib]+1+k]  += np.cos(np.pi*(k+1)/H1*z_here[m])
                    jac[indi['i_s0'][ib]+1+m , indi['i_s_1'][ib]+1+k] += -np.cos(np.pi*(k+1)/H2*z_here[m])
    
        else: #depths are equal
            for m in range(self.N): 
                jac[indi['i_s0'][ib]+1+m,indi['i_s0'][ib]+m+1]  += 1
                jac[indi['i_s0'][ib]+1+m,indi['i_s_1'][ib]+m+1] += -1
    '''            
    # =============================================================================
    # inner boundary - flux - subtidal contribution
    # =============================================================================
    #salinities at boundaries
    sb_1 = ans[indi['i_s_1']]
    sb_2 = ans[(self.di3[1:-1]-4) * self.M]
    sb_3 = ans[(self.di3[1:-1]-5) * self.M]
    sn_1 = ans[((self.di3[1:-1]-3)*self.M+np.arange(1,self.M)[:,np.newaxis]).T]
    sn_2 = ans[((self.di3[1:-1]-4)*self.M+np.arange(1,self.M)[:,np.newaxis]).T]
    sn_3 = ans[((self.di3[1:-1]-5)*self.M+np.arange(1,self.M)[:,np.newaxis]).T]
    
    sb0 = ans[(self.di3[1:-1]+2) * self.M]
    sb1 = ans[(self.di3[1:-1]+3) * self.M]
    sb2 = ans[(self.di3[1:-1]+4) * self.M]
    sn0 = ans[((self.di3[1:-1]+2)*self.M+np.arange(1,self.M)[:,np.newaxis]).T]
    sn1 = ans[((self.di3[1:-1]+3)*self.M+np.arange(1,self.M)[:,np.newaxis]).T]
    sn2 = ans[((self.di3[1:-1]+4)*self.M+np.arange(1,self.M)[:,np.newaxis]).T]

    
    #flux
    #depth-averaged 
    jac[indi['i_s_1'] , indi['i_s_1']] += pars_st['C13a_1'] + pars_st['C13d_1'] * 3/(2*self.dl[self.di[ :-2]]) 
    jac[indi['i_s_1'] , indi['i_s_2']] += pars_st['C13d_1'] *-4/(2*self.dl[self.di[ :-2]]) 
    jac[indi['i_s_1'] , indi['i_s_3']] += pars_st['C13d_1'] * 1/(2*self.dl[self.di[ :-2]])
    
    jac[indi['i_s_1'] , indi['i_s0']] += - pars_st['C13a0'] + pars_st['C13d0'] * 3/(2*self.dl[self.di[1:-1]]) 
    jac[indi['i_s_1'] , indi['i_s1']] +=  pars_st['C13d0'] *-4/(2*self.dl[self.di[1:-1]]) 
    jac[indi['i_s_1'] , indi['i_s2']] +=  pars_st['C13d0'] * 1/(2*self.dl[self.di[1:-1]]) 
    
    
    if version in ['A','C','D']:
        jac[indi['i_s_1'] , indi['i_s_1']] += np.sum(pars_st['C13c_1'] * sn_1 * 3/(2*self.dl[self.di[ :-2,np.newaxis]]),1) #+ jtr_da[0]
        jac[indi['i_s_1'] , indi['i_s_2']] += np.sum(pars_st['C13c_1'] * sn_1 *-4/(2*self.dl[self.di[ :-2,np.newaxis]]),1) #+ jtr_da[1]
        jac[indi['i_s_1'] , indi['i_s_3']] += np.sum(pars_st['C13c_1'] * sn_1 * 1/(2*self.dl[self.di[ :-2,np.newaxis]]),1) #+ jtr_da[2]
        
        jac[indi['i_s_1'] , indi['i_s0']] += np.sum(pars_st['C13c0'] * sn0 * 3/(2*self.dl[self.di[1:-1,np.newaxis]]),1) #- jtl_da[0]
        jac[indi['i_s_1'] , indi['i_s1']] += np.sum(pars_st['C13c0'] * sn0 *-4/(2*self.dl[self.di[1:-1,np.newaxis]]),1) #- jtl_da[1]
        jac[indi['i_s_1'] , indi['i_s2']] += np.sum(pars_st['C13c0'] * sn0 * 1/(2*self.dl[self.di[1:-1,np.newaxis]]),1) #- jtl_da[2]
        
        for k in range(1,self.M):
            jac[indi['i_s_1'] , indi['i_s_1'] + k] += pars_st['C13b_1'][:,k-1] + pars_st['C13c_1'][:,k-1] * (3*sb_1 - 4*sb_2 + sb_3)/(2*self.dl[self.di[ :-2]]) #+ jtr_da[3][k-1]
            jac[indi['i_s_1'] , indi['i_s0']  + k] += -pars_st['C13b0'][:,k-1] - pars_st['C13c0'][:,k-1] * (-3*sb0 + 4*sb1 - sb2)/(2*self.dl[self.di[1:-1]]) #- jtl_da[3][k-1]
    
    
    #depth-varying   
    if version in ['A','B','C','D']:
        jac[indi['bnd2_rgt'], indi['bnd2_rgt']-2*self.M] += (np.repeat(pars_st['C15a_1'],self.N).reshape((len(pars_st['C15a_1']),self.N)) * 1/(2*self.dl[self.di[ :-2],np.newaxis]) ).flatten()
        jac[indi['bnd2_rgt'], indi['bnd2_rgt']-self.M]   += (np.repeat(pars_st['C15a_1'],self.N).reshape((len(pars_st['C15a_1']),self.N)) *-4/(2*self.dl[self.di[ :-2],np.newaxis]) ).flatten()
        jac[indi['bnd2_rgt'], indi['bnd2_rgt']]     += (np.repeat(pars_st['C15a_1'],self.N).reshape((len(pars_st['C15a_1']),self.N)) * 3/(2*self.dl[self.di[ :-2],np.newaxis])).flatten() #sn_1
    
        jac[indi['bnd2_rgt'], indi['bnd2_lft']]     += (np.repeat(pars_st['C15a0'],self.N).reshape((len(pars_st['C15a0']),self.N)) * 3/(2*self.dl[self.di[1:-1],np.newaxis]) ).flatten()#sn0
        jac[indi['bnd2_rgt'], indi['bnd2_lft']+self.M]   += (np.repeat(pars_st['C15a0'],self.N).reshape((len(pars_st['C15a0']),self.N)) *-4/(2*self.dl[self.di[1:-1],np.newaxis])  ).flatten() #sn1
        jac[indi['bnd2_rgt'], indi['bnd2_lft']+2*self.M] += (np.repeat(pars_st['C15a0'],self.N).reshape((len(pars_st['C15a0']),self.N)) * 1/(2*self.dl[self.di[1:-1],np.newaxis])  ).flatten() #sn2
         
    #if version in ['D']:
        
        jac[indi['bnd2_rgt'],np.repeat(indi['i_s_3'],self.N)] += (np.sum([pars_st['C15c_1'][:,n] * sn_1[:,n,np.newaxis] for n in range(self.N)] ,0) * 1/(2*self.dl[self.di[ :-2],np.newaxis]) ).flatten() #sb_3+ jtr_dv[2].T
        jac[indi['bnd2_rgt'],np.repeat(indi['i_s_2'],self.N)] += (np.sum([pars_st['C15c_1'][:,n] * sn_1[:,n,np.newaxis] for n in range(self.N)] ,0) *-4/(2*self.dl[self.di[ :-2],np.newaxis])).flatten() #sb_2 + jtr_dv[1].T
        jac[indi['bnd2_rgt'],np.repeat(indi['i_s_1'],self.N)] += (np.sum([pars_st['C15c_1'][:,n] * sn_1[:,n,np.newaxis] for n in range(self.N)] ,0) * 3/(2*self.dl[self.di[ :-2],np.newaxis])).flatten()  #sb_1 + jtr_dv[0].T
        
        jac[indi['bnd2_rgt'],np.repeat(indi['i_s0'],self.N)] += (np.sum([pars_st['C15c0'][:,n] * sn0[:,n,np.newaxis] for n in range(self.N)] ,0) * 3/(2*self.dl[self.di[1:-1],np.newaxis])).flatten()#sb0 - jtl_dv[0].T
        jac[indi['bnd2_rgt'],np.repeat(indi['i_s1'],self.N)] += (np.sum([pars_st['C15c0'][:,n] * sn0[:,n,np.newaxis] for n in range(self.N)] ,0) *-4/(2*self.dl[self.di[1:-1],np.newaxis])).flatten()#sb1 - jtl_dv[1].T
        jac[indi['bnd2_rgt'],np.repeat(indi['i_s2'],self.N)] += (np.sum([pars_st['C15c0'][:,n] * sn0[:,n,np.newaxis] for n in range(self.N)] ,0) * 1/(2*self.dl[self.di[1:-1],np.newaxis])).flatten()#sb2 - jtl_dv[2].T
        
        jac[indi['bnd2_rgt'], indi['bnd2_rgt']-2*self.M] += ( pars_st['C15b_1'] * sn_1 * 1/(2*self.dl[self.di[ :-2],np.newaxis])).flatten() #sn_3
        jac[indi['bnd2_rgt'], indi['bnd2_rgt']-self.M]   += ( pars_st['C15b_1'] * sn_1 *-4/(2*self.dl[self.di[ :-2],np.newaxis])).flatten() #sn_2
        jac[indi['bnd2_rgt'], indi['bnd2_rgt']]     += (pars_st['C15d_1'] + pars_st['C15b_1'] * sn_1 * 3/(2*self.dl[self.di[ :-2],np.newaxis]) + pars_st['C15b_1'] * (3*sn_1 - 4*sn_2 + sn_3)/(2*self.dl[self.di[ :-2],np.newaxis])).flatten() #sn_1
        
        jac[indi['bnd2_rgt'], indi['bnd2_lft']]     += (-pars_st['C15d0'] + pars_st['C15b0'] * sn0 * 3/(2*self.dl[self.di[1:-1],np.newaxis]) - pars_st['C15b0'] * (-3*sn0 + 4*sn1 - sn2)/(2*self.dl[self.di[1:-1],np.newaxis]) ).flatten()#sn0
        jac[indi['bnd2_rgt'], indi['bnd2_lft']+self.M]   += ( pars_st['C15b0'] * sn0 *-4/(2*self.dl[self.di[1:-1],np.newaxis]) ).flatten() #sn1
        jac[indi['bnd2_rgt'], indi['bnd2_lft']+2*self.M] += ( pars_st['C15b0'] * sn0 * 1/(2*self.dl[self.di[1:-1],np.newaxis]) ).flatten() #sn2
        
        for k in range(1,self.M):
            jac[indi['bnd2_rgt'] , np.repeat(indi['i_s_1'],self.N) + k] += (pars_st['C15e_1'][:,:,k-1] + pars_st['C15c_1'][:,k-1] * ((3*sb_1 - 4*sb_2 + sb_3)/(2*self.dl[self.di[ :-2]]))[:,np.newaxis]).flatten() #sn_1 in sum + (jtr_dv[3][k-1]).T 
            jac[indi['bnd2_rgt'] , np.repeat(indi['i_s0'] ,self.N) + k] += (-pars_st['C15e0'][:,:,k-1] -  pars_st['C15c0'][:,k-1] * ((-3*sb0 + 4*sb1 - sb2)/(2*self.dl[self.di[1:-1]]))[:,np.newaxis]).flatten() #sn0 in sum - (jtl_dv[3][k-1]).T 
        
            
    
    
    
    return jac







