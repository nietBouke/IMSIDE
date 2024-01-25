# =============================================================================
# analyze the boundary layer correction
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

#calculate the tidal salinity, plus the boundary layer correction

def func1(self,sss,indi):

    ss2 = np.delete(sss , np.concatenate([indi['bnl_rgt'].flatten(),indi['bnl_lft'].flatten(),indi['bnl_bnd']]))
    
    #calculate total salinity 
    s_b = np.transpose([np.reshape(ss2,(self.di[-1],self.M))[:,0]]*self.nz)
    sn = np.reshape(ss2,(self.di[-1],self.M))[:,1:]
    s_p = np.array([np.sum([sn[i,n-1]*np.cos(np.pi*n *self.z_nd) for n in range(1,self.M)],0) for i in range(self.di[-1])])
    s = (s_b+s_p)*self.soc
    s[np.where((s<0) & (s>-0.0001))]= 1e-10 #remove negative values due to numerical precision
    
    Lint = -self.px[np.where(s[:,0]>2)[0][0]]-self.Ln[-1]/1000
    
    #salinity in tidal cycle
    if len(self.tid_comp)>1: print('Not sure about the relative timing of the tidal components')
    t= np.linspace(0,44700,self.nt)
    sti = 0
    for i in range(len(self.tid_comp)):
        tid_set = self.tid_sets[self.tid_comp[i]]
        tid_geg = self.tid_gegs[self.tid_comp[i]]
        tid_inp = self.tidal_salinity(sss, tid_set, tid_geg)  
        
        sti += np.real(tid_inp['st'][:,:,np.newaxis]* np.exp(1j*tid_geg['omega']*t))
        
    stot = s[:,:,np.newaxis] + sti
        
    return stot

stot = func1(run,out4,run.ii_all)
print(stot.shape)
print(run.di)

#%%
ttide = np.linspace(0,44700,run.nt)/3600

plt.contourf(ttide, run.zlist[0,run.di[-3]], stot[run.di[-3]])
plt.colorbar()
plt.show()

plt.contourf(ttide, run.zlist[0,run.di[-3]],stot[run.di[-3]-1])
plt.colorbar()
plt.show()

#%%calculate now the boundary layer correction. 

def calc_cor(self, zout, indi): #for plotting purposes afterwards 
    
    #calculate salinity correction
    Bm_rgt = zout[((self.di3[1:-1]-2)*self.M+np.arange(self.M)[:,np.newaxis])] + 1j * zout[((self.di3[1:-1]-2)*self.M+np.arange(self.M,2*self.M)[:,np.newaxis])]
    Bm_lft = zout[( self.di3[1:-1]   *self.M+np.arange(self.M)[:,np.newaxis])] + 1j * zout[( self.di3[1:-1]   *self.M+np.arange(self.M,2*self.M)[:,np.newaxis])]
           
    stc = np.zeros((self.di[-1],self.nz,self.nt)) 

    
    for j in range(len(self.tid_comp)):
        tid_set = self.tid_sets[self.tid_comp[j]]
        tid_geg = self.tid_gegs[self.tid_comp[j]]

        #calculate x 
        x_temp = [np.arange(self.nxn[i])*self.dxn[i] for i in range(self.ndom)]
        x_lft = [x_temp[i][np.where(x_temp[i]<-np.sqrt(tid_geg['epsL'])*np.log(self.tol))[0]] for i in range(self.ndom)][1:]
        x_rgt = [x_temp[i][np.where(x_temp[i]>x_temp[i][-1]+np.sqrt(tid_geg['epsL'])*np.log(self.tol))[0]]-self.Ln[i] for i in range(self.ndom)][:-1]
               
        stc_lft   = [np.exp(-x_lft[i][:,np.newaxis]/np.sqrt(tid_geg['epsL'])) * np.sum(Bm_lft[:,i] * np.cos(np.arange(self.M)*np.pi*self.z_nd[:,np.newaxis]) ,1) * self.soc for i in range(self.ndom-1)]
        stc_rgt   = [np.exp(x_rgt[i][:,np.newaxis]/np.sqrt(tid_geg['epsL'])) * np.sum(Bm_rgt[:,i] * np.cos(np.arange(self.M)*np.pi*self.z_nd[:,np.newaxis]) ,1) * self.soc for i in range(self.ndom-1)]
                
        #salinity for plotting
        t= np.linspace(0,44700,self.nt)
        stc_i = np.zeros((self.di[-1],self.nz) , dtype = complex)
        for i in range(self.ndom-1):
            stc[self.di[1+i]+np.arange(len(x_lft[i]))]    += -np.real(stc_lft[i][:,:,np.newaxis] * np.exp(1j*tid_geg['omega']*t))
            stc[self.di[1+i]+np.arange(-len(x_rgt[i]),0)] += -np.real(stc_rgt[i][:,:,np.newaxis] * np.exp(1j*tid_geg['omega']*t))
            
            stc_i[self.di[1+i]+np.arange(len(x_lft[i]))]    = stc_lft[i]
            stc_i[self.di[1+i]+np.arange(-len(x_rgt[i]),0)] = stc_rgt[i]
        '''
        #terms for fluxes 
        flux = np.zeros(self.di[-1]) 
        for i in range(self.ndom-1):
            flux[self.di[1+i]+np.arange(len(x_lft[i]))]    = -1/4 * np.real(self.ut[self.di[1+i]+np.arange(len(x_lft[i]))] *np.conj(stc_lft[i]) + np.conj(self.ut[self.di[1+i]+np.arange(len(x_lft[i]))]) *stc_lft[i]).mean(1) 
            flux[self.di[1+i]+np.arange(-len(x_rgt[i]),0)] = -1/4 * np.real(self.ut[self.di[1+i]+np.arange(-len(x_rgt[i]),0)]*np.conj(stc_rgt[i]) + np.conj(self.ut[self.di[1+i]+np.arange(-len(x_rgt[i]),0)])*stc_rgt[i]).mean(1) 
        '''
        
        
        T_Tc = self.b*self.H*1/4 * np.real(tid_geg['ut']*np.conj(stc_i)   + np.conj(tid_geg['ut'])*stc_i).mean(1) 
        print(np.max(np.abs(T_Tc)))
    return stc


sti_cor = calc_cor(run, out4, run.ii_all)

stot = func1(run,out4,run.ii_all)*0 + calc_cor(run, out4, run.ii_all)

ttide = np.linspace(0,44700,run.nt)/3600

plt.contourf(ttide, run.zlist[0,run.di[-3]], stot[run.di[-3]])
plt.colorbar()
plt.show()

plt.contourf(ttide, run.zlist[0,run.di[-3]],stot[run.di[-3]-1])
plt.colorbar()
plt.show()
