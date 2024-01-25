# =============================================================================
# functions for tidal model
# analytical solutions to the tidal salt balance with w included. 
# =============================================================================
import numpy as np

def sti_coefs(z,c1,pars):
    # =============================================================================
    # prepare coefficients for tidal functions 
    # =============================================================================
    dA,H,B,n = pars
    
    #calculate some returning parts for better perfomance
    zn = z/H
    c1_wo   = np.sqrt(c1)
    
    sin_pinz= np.sin(np.pi*n*zn)
    sin_pin = np.sin(np.pi*n)
    sin_Hcw = np.sin(H/c1_wo)

    cos_pinz= np.cos(np.pi*n*zn)
    cos_pin = np.cos(np.pi*n)
    cos_zcw = np.cos(z/c1_wo)
        
    #coefficients
    c2_coef = -(-B*H**2*np.cosh(dA*zn) - B*H*c1_wo*dA*np.cos(z/c1_wo)*np.sinh(dA)*1/np.sin(H/c1_wo) + H**2 + c1*dA**2)/(H**2 + c1*dA**2)
    c3_coef = (H*(-H*sin_pinz  - np.pi*c1_wo*n*(-cos_pin + np.cos(H/c1_wo))*cos_zcw*1/sin_Hcw + np.pi*c1_wo*n*np.sin(z/c1_wo))/(H**2 - np.pi**2*c1*n**2))
    c4_coef = (H**2*(-B*H**7*sin_pinz *np.sinh(dA*zn) - np.pi*B*H**6*c1_wo*n*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw - B*H**5*c1*dA**2*sin_pinz *np.sinh(dA*zn) + 3*np.pi**2*B*H**5*c1*n**2*sin_pinz *np.sinh(dA*zn) \
         + np.pi*B*H**4*c1**(3/2)*dA**2*n*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw + 3*np.pi**3*B*H**4*c1**(3/2)*n**3*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw + 2*np.pi**2*B*H**3*c1**2*dA**2*n**2*sin_pinz *np.sinh(dA*zn) \
         - 3*np.pi**4*B*H**3*c1**2*n**4*sin_pinz *np.sinh(dA*zn) - 2*np.pi**3*B*H**2*c1**(5/2)*dA**2*n**3*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw - 3*np.pi**5*B*H**2*c1**(5/2)*n**5*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw \
         - np.pi**4*B*H*c1**3*dA**2*n**4*sin_pinz *np.sinh(dA*zn) + np.pi**6*B*H*c1**3*n**6*sin_pinz *np.sinh(dA*zn) + 2*np.pi*B*H*c1*dA*n*(H**2 - np.pi**2*c1*n**2)**2*cos_pinz*np.cosh(dA*zn) \
         + np.pi**5*B*c1**(7/2)*dA**2*n**5*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw + np.pi**7*B*c1**(7/2)*n**7*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw \
         - B*c1_wo*dA*(H**2 - np.pi**2*c1*n**2)**2*(H**2 + c1*dA**2 + np.pi**2*c1*n**2)*sin_pin*cos_zcw*np.cosh(dA)*1/sin_Hcw + np.pi*H**6*c1_wo*dA*n*cos_pin*cos_zcw*1/sin_Hcw \
         + H**6*c1_wo*dA*sin_pin*cos_zcw*1/sin_Hcw + H**6*dA*z*sin_pinz  - 2*np.pi*H**5*c1*dA*n*cos_pinz + 2*np.pi*H**4*c1**(3/2)*dA**3*n*cos_pin*cos_zcw*1/sin_Hcw \
         + 2*H**4*c1**(3/2)*dA**3*sin_pin*cos_zcw*1/sin_Hcw - 3*np.pi**3*H**4*c1**(3/2)*dA*n**3*cos_pin*cos_zcw*1/sin_Hcw - np.pi**2*H**4*c1**(3/2)*dA*n**2*sin_pin*cos_zcw*1/sin_Hcw \
         + 2*H**4*c1*dA**3*z*sin_pinz  - 3*np.pi**2*H**4*c1*dA*n**2*z*sin_pinz  - 4*np.pi*H**3*c1**2*dA**3*n*cos_pinz + 4*np.pi**3*H**3*c1**2*dA*n**3*cos_pinz + np.pi*H**2*c1**(5/2)*dA**5*n*cos_pin*cos_zcw*1/sin_Hcw \
         + H**2*c1**(5/2)*dA**5*sin_pin*cos_zcw*1/sin_Hcw + 4*np.pi**2*H**2*c1**(5/2)*dA**3*n**2*sin_pin*cos_zcw*1/sin_Hcw + 3*np.pi**5*H**2*c1**(5/2)*dA*n**5*cos_pin*cos_zcw*1/sin_Hcw\
         - np.pi**4*H**2*c1**(5/2)*dA*n**4*sin_pin*cos_zcw*1/sin_Hcw + H**2*c1**2*dA**5*z*sin_pinz  + 3*np.pi**4*H**2*c1**2*dA*n**4*z*sin_pinz  - 2*np.pi*H*c1**3*dA**5*n*cos_pinz - 4*np.pi**3*H*c1**3*dA**3*n**3*cos_pinz \
         - 2*np.pi**5*H*c1**3*dA*n**5*cos_pinz - np.pi**3*c1**(7/2)*dA**5*n**3*cos_pin*cos_zcw*1/sin_Hcw + np.pi**2*c1**(7/2)*dA**5*n**2*sin_pin*cos_zcw*1/sin_Hcw \
         - 2*np.pi**5*c1**(7/2)*dA**3*n**5*cos_pin*cos_zcw*1/sin_Hcw + 2*np.pi**4*c1**(7/2)*dA**3*n**4*sin_pin*cos_zcw*1/sin_Hcw - np.pi**7*c1**(7/2)*dA*n**7*cos_pin*cos_zcw*1/sin_Hcw \
         + np.pi**6*c1**(7/2)*dA*n**6*sin_pin*cos_zcw*1/sin_Hcw - np.pi**2*c1**3*dA**5*n**2*z*sin_pinz  - 2*np.pi**4*c1**3*dA**3*n**4*z*sin_pinz  \
         - np.pi**6*c1**3*dA*n**6*z*sin_pinz )/(dA*(H**2 - np.pi**2*c1*n**2)**2*(H**4 + 2*H**2*c1*(dA**2 - np.pi**2*n**2) + c1**2*(dA**2 + np.pi**2*n**2)**2)))

    #for depth-averages
    c2p_coef = c2_coef - c2_coef.mean(2)[:,:,np.newaxis]
    c3p_coef = c3_coef - c3_coef.mean(2)[:,:,np.newaxis]
    c4p_coef = c4_coef - c4_coef.mean(2)[:,:,np.newaxis]
        
    #for derivatives to z
    dc2dz_coef = B*H*dA*(-np.sin(z/c1_wo)*np.sinh(dA)*1/sin_Hcw + np.sinh(dA*zn))/(H**2 + c1*dA**2)
    dc3dz_coef = (np.pi*H*n*((cos_pin*1/sin_Hcw - 1/np.tan(H/c1_wo))*np.sin(z/c1_wo) - np.cos(z/c1_wo) + cos_pinz)/(-H**2 + np.pi**2*c1*n**2))
    dc4dz_coef = (-H*(H*c1**(3/2)*(np.pi*B*n*(3*np.pi**2*H**4*n**2 - 3*np.pi**4*H**2*c1*n**4 + np.pi**6*c1**2*n**6 + dA**2*(H**2 - np.pi**2*c1*n**2)**2)*cos_pin*np.sinh(dA) - np.pi*dA*n*(H**4*(-2*dA**2 + 3*np.pi**2*n**2) - H**2*c1*(dA**4 + 3*np.pi**4*n**4) \
                    + np.pi**2*c1**2*n**2*(dA**2 + np.pi**2*n**2)**2)*cos_pin + dA*(H**4*(2*dA**2 - np.pi**2*n**2) + H**2*c1*(dA**4 + 4*np.pi**2*dA**2*n**2 - np.pi**4*n**4) + np.pi**2*c1**2*n**2*(dA**2 + np.pi**2*n**2)**2)*sin_pin)*np.sin(z/c1_wo)*1/sin_Hcw \
                    + c1_wo*(-np.pi*B*H**7*n*np.sin(z/c1_wo)*cos_pin*np.sinh(dA)*1/sin_Hcw + np.pi*B*H**7*n*cos_pinz*np.sinh(dA*zn) - np.pi*B*H**5*c1*dA**2*n*cos_pinz*np.sinh(dA*zn) - 3*np.pi**3*B*H**5*c1*n**3*cos_pinz*np.sinh(dA*zn) \
                    + 2*np.pi**3*B*H**3*c1**2*dA**2*n**3*cos_pinz*np.sinh(dA*zn) + 3*np.pi**5*B*H**3*c1**2*n**5*cos_pinz*np.sinh(dA*zn) - np.pi**5*B*H*c1**3*dA**2*n**5*cos_pinz*np.sinh(dA*zn) \
                    - np.pi**7*B*H*c1**3*n**7*cos_pinz*np.sinh(dA*zn) + B*H*dA*(H**2 + c1*(dA**2 + np.pi**2*n**2))*(H**2 - np.pi**2*c1*n**2)**2*sin_pinz*np.cosh(dA*zn) - B*H*dA*(H**2 - np.pi**2*c1*n**2)**2*(H**2 + c1*dA**2 \
                    + np.pi**2*c1*n**2)*sin_pin*np.sin(z/c1_wo)*np.cosh(dA)*1/sin_Hcw + np.pi*H**7*dA*n*np.sin(z/c1_wo)*cos_pin*1/sin_Hcw + H**7*dA*sin_pin*np.sin(z/c1_wo)*1/sin_Hcw - H**7*dA*sin_pinz \
                    - np.pi*H**6*dA*n*z*cos_pinz - 2*H**5*c1*dA**3*sin_pinz + np.pi**2*H**5*c1*dA*n**2*sin_pinz - 2*np.pi*H**4*c1*dA**3*n*z*cos_pinz + 3*np.pi**3*H**4*c1*dA*n**3*z*cos_pinz - H**3*c1**2*dA**5*sin_pinz \
                    - 4*np.pi**2*H**3*c1**2*dA**3*n**2*sin_pinz + np.pi**4*H**3*c1**2*dA*n**4*sin_pinz - np.pi*H**2*c1**2*dA**5*n*z*cos_pinz - 3*np.pi**5*H**2*c1**2*dA*n**5*z*cos_pinz - np.pi**2*H*c1**3*dA**5*n**2*sin_pinz \
                    - 2*np.pi**4*H*c1**3*dA**3*n**4*sin_pinz - np.pi**6*H*c1**3*dA*n**6*sin_pinz + np.pi**3*c1**3*dA**5*n**3*z*cos_pinz + 2*np.pi**5*c1**3*dA**3*n**5*z*cos_pinz \
                    + np.pi**7*c1**3*dA*n**7*z*cos_pinz))/(c1_wo*dA*(H**2 - np.pi**2*c1*n**2)**2*(H**4 + 2*H**2*c1*(dA**2 - np.pi**2*n**2) + c1**2*(dA**2 + np.pi**2*n**2)**2)))

    
    return c2_coef, c3_coef, c4_coef , c2p_coef, c3p_coef, c4p_coef , dc2dz_coef, dc3dz_coef, dc4dz_coef


def sti(c,pars,w_on = 1): 
    # =============================================================================
    # calculate tidal salinity (complex notation, so \hat s_ti)
    # newer version which has a better performance but should have the same result as the old formulation
    # =============================================================================
    c2,c3,c4 = c
    c3,c4 = c3*w_on , c4*w_on
    c2_coef, c3_coef, c4_coef = pars
    
    sbar = (c2*c2_coef)[0]
    spri = c3 * c3_coef + c4 * c4_coef

    sall = sbar+spri.sum(0)

    return sall

'''
def stib(c,pars,w_on = 1): 
    # =============================================================================
    # calculate tidal salinity (complex notation, so \hat s_ti)
    # =============================================================================
    c1,c2,c3,c4 = c
    c3,c4 = c3*w_on , c4*w_on
    dA,H,B,n0 = pars    
    sbar = 0

    def spri(n):
        return -B*H*c4*np.sin(np.pi*n)*np.cosh(dA)/(dA**2 + np.pi**2*n**2) + B*(np.pi*H*c4*n*np.cos(np.pi*n) + c2*(dA**2 + np.pi**2*n**2))*np.sinh(dA)/(dA**3 + np.pi**2*dA*n**2) \
            + (H*c4*np.sin(np.pi*n) - np.pi*n*(H*c4 + c3)*np.cos(np.pi*n) + np.pi*n*(-np.pi*c2*n + c3))/(np.pi**2*n**2)

    out = sbar+spri(n0).sum(0)

    return out



def utisti(geg_all, geg_st, geg_ut, w_on = 1): 
    # =============================================================================
    # calculate tidal transport (complex notation, so \hat s_ti)
    # newer version which has a better performance but should have the same result as the old formulation
    # obsolete since the numerical computation is faster, with comparable accuracy
    # =============================================================================
    
    c1,c2,c3,c4, dA, B = geg_st
    c5,dAu,Bu = geg_ut
    H,n0 = geg_all
    
    c3,c4 = c3*w_on , c4*w_on
    
    #calculate some returning parts for better perfomance
    c1_wo = np.sqrt(c1) 
    sh_dA = np.sinh(dA)
    tan_Hc1 = np.tan(H/c1_wo)
    
    utstc2 = ( -c2*c5*(B*dAu*(Bu*H**2*dA**2*(H**2 + c1*dA**2)*np.cosh(dAu) - (dA**2 - dAu**2)*(-Bu*H*c1**(3/2)*dA**2*dAu*1/tan_Hc1*np.sinh(dAu) + H**4 + H**2*c1*(dA**2 + dAu**2) + c1**2*dA**2*dAu**2))*np.sinh(dA) + dA*(H**2 + c1*dAu**2)*(-2*Bu*(B*H**2*dAu**2*np.cosh(dA) \
             + (H**2 + c1*dA**2)*(dA**2 - dAu**2))*np.sinh(dAu) + 2*dAu*(H**2 + c1*dA**2)*(dA**2 - dAu**2))/2)/(dA*dAu*(H**2 + c1*dA**2)*(H**2 + c1*dAu**2)*(dA**2 - dAu**2)))[0]
    
    #utstc2 = (2*c2*c5*Bu*B*H**2) / ((H**2+c1*dA**2)*(H**2+c1*dAu**2)) #ChatGPT output, I do not trust this
        
    def utstc34(n):
        #sin_pinz = np.sin(np.pi*n*zn)
        cos_pin = np.cos(np.pi*n)
        sin_pin = np.sin(np.pi*n)
        #cos_zcw = np.cos(z/c1_wo)
        #sin_Hcw = np.sin(H/c1_wo)
        
        return c3*c5*(np.pi**2*Bu*H**2*n**2*(-H**2 + np.pi**2*c1*n**2)*cos_pin*np.cosh(dAu) + np.pi*Bu*H*dAu*n*(H*(H**2 + c1*dAu**2)*sin_pin + np.pi*c1**(3/2)*n*(dAu**2 + np.pi**2*n**2)*cos_pin*1/tan_Hc1 - np.pi*c1**(3/2)*n*(dAu**2 + np.pi**2*n**2)*1/np.sin(H/c1_wo))*np.sinh(dAu) \
                + (H**2 - np.pi**2*c1*n**2)*(-H**2*(dAu**2 - np.pi**2*n**2*(Bu - 1)) - c1*(dAu**4 + np.pi**2*dAu**2*n**2) + (H**2 + c1*dAu**2)*(dAu**2 + np.pi**2*n**2)*cos_pin))/(np.pi*n*(-H**2 + np.pi**2*c1*n**2)*(H**2 + c1*dAu**2)*(dAu**2 + np.pi**2*n**2)) \
                + c4*c5*H*(B*Bu*H**8*(dA*(2*np.pi*dAu*n*cos_pin*np.sinh(dAu) + (dA**2 - dAu**2 + np.pi**2*n**2)*sin_pin*np.cosh(dAu))*np.cosh(dA) + (dAu*(-dA**2 + dAu**2 + np.pi**2*n**2)*sin_pin*np.sinh(dAu) - np.pi*n*(dA**2 + dAu**2 + np.pi**2*n**2)*cos_pin*np.cosh(dAu))*np.sinh(dA))/((dA**2 - 2*dA*dAu + dAu**2 + np.pi**2*n**2)*(dA**2 + 2*dA*dAu + dAu**2 + np.pi**2*n**2)) + np.pi*B*Bu*H**7*c1*n*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*cos_pin*np.sinh(dA)/(H**2 + c1*dAu**2) + B*Bu*H**6*c1*dA**2*(dA*(2*np.pi*dAu*n*cos_pin*np.sinh(dAu) + (dA**2 - dAu**2 + np.pi**2*n**2)*sin_pin*np.cosh(dAu))*np.cosh(dA) + (dAu*(-dA**2 + dAu**2 + np.pi**2*n**2)*sin_pin*np.sinh(dAu) - np.pi*n*(dA**2 + dAu**2 + np.pi**2*n**2)*cos_pin*np.cosh(dAu))*np.sinh(dA))/((dA**2 - 2*dA*dAu + dAu**2 + np.pi**2*n**2)*(dA**2 + 2*dA*dAu + dAu**2 + np.pi**2*n**2)) - 3*np.pi**2*B*Bu*H**6*c1*n**2*(dA*(2*np.pi*dAu*n*cos_pin*np.sinh(dAu) + (dA**2 - dAu**2 + np.pi**2*n**2)*sin_pin*np.cosh(dAu))*np.cosh(dA) + (dAu*(-dA**2 + dAu**2 + np.pi**2*n**2)*sin_pin*np.sinh(dAu) - np.pi*n*(dA**2 + dAu**2 + np.pi**2*n**2)*cos_pin*np.cosh(dAu))*np.sinh(dA))/((dA**2 - 2*dA*dAu + dAu**2 + np.pi**2*n**2)*(dA**2 + 2*dA*dAu + dAu**2 + np.pi**2*n**2)) - np.pi*B*Bu*H**5*c1**2*dA**2*n*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*cos_pin*np.sinh(dA)/(H**2 + c1*dAu**2) - 3*np.pi**3*B*Bu*H**5*c1**2*n**3*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*cos_pin*np.sinh(dA)/(H**2 + c1*dAu**2) - 2*np.pi**2*B*Bu*H**4*c1**2*dA**2*n**2*(dA*(2*np.pi*dAu*n*cos_pin*np.sinh(dAu) + (dA**2 - dAu**2 + np.pi**2*n**2)*sin_pin*np.cosh(dAu))*np.cosh(dA) + (dAu*(-dA**2 + dAu**2 + np.pi**2*n**2)*sin_pin*np.sinh(dAu) - np.pi*n*(dA**2 + dAu**2 + np.pi**2*n**2)*cos_pin*np.cosh(dAu))*np.sinh(dA))/((dA**2 - 2*dA*dAu + dAu**2 + np.pi**2*n**2)*(dA**2 + 2*dA*dAu + dAu**2 + np.pi**2*n**2)) + 3*np.pi**4*B*Bu*H**4*c1**2*n**4*(dA*(2*np.pi*dAu*n*cos_pin*np.sinh(dAu) + (dA**2 - dAu**2 + np.pi**2*n**2)*sin_pin*np.cosh(dAu))*np.cosh(dA) + (dAu*(-dA**2 + dAu**2 + np.pi**2*n**2)*sin_pin*np.sinh(dAu) - np.pi*n*(dA**2 + dAu**2 + np.pi**2*n**2)*cos_pin*np.cosh(dAu))*np.sinh(dA))/((dA**2 - 2*dA*dAu + dAu**2 + np.pi**2*n**2)*(dA**2 + 2*dA*dAu + dAu**2 + np.pi**2*n**2)) + 2*np.pi**3*B*Bu*H**3*c1**3*dA**2*n**3*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*cos_pin*np.sinh(dA)/(H**2 + c1*dAu**2) + 3*np.pi**5*B*Bu*H**3*c1**3*n**5*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*cos_pin*np.sinh(dA)/(H**2 + c1*dAu**2) + np.pi**4*B*Bu*H**2*c1**3*dA**2*n**4*(dA*(2*np.pi*dAu*n*cos_pin*np.sinh(dAu) + (dA**2 - dAu**2 + np.pi**2*n**2)*sin_pin*np.cosh(dAu))*np.cosh(dA) + (dAu*(-dA**2 + dAu**2 + np.pi**2*n**2)*sin_pin*np.sinh(dAu) - np.pi*n*(dA**2 + dAu**2 + np.pi**2*n**2)*cos_pin*np.cosh(dAu))*np.sinh(dA))/((dA**2 - 2*dA*dAu + dAu**2 + np.pi**2*n**2)*(dA**2 + 2*dA*dAu + dAu**2 + np.pi**2*n**2)) - np.pi**6*B*Bu*H**2*c1**3*n**6*(dA*(2*np.pi*dAu*n*cos_pin*np.sinh(dAu) + (dA**2 - dAu**2 + np.pi**2*n**2)*sin_pin*np.cosh(dAu))*np.cosh(dA) + (dAu*(-dA**2 + dAu**2 + np.pi**2*n**2)*sin_pin*np.sinh(dAu) - np.pi*n*(dA**2 + dAu**2 + np.pi**2*n**2)*cos_pin*np.cosh(dAu))*np.sinh(dA))/((dA**2 - 2*dA*dAu + dAu**2 + np.pi**2*n**2)*(dA**2 + 2*dA*dAu + dAu**2 + np.pi**2*n**2)) - np.pi**5*B*Bu*H*c1**4*dA**2*n**5*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*cos_pin*np.sinh(dA)/(H**2 + c1*dAu**2) - np.pi**7*B*Bu*H*c1**4*n**7*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*cos_pin*np.sinh(dA)/(H**2 + c1*dAu**2) + B*Bu*H*c1*dA*(H**2 + c1*(dA**2 + np.pi**2*n**2))*(H**2 - np.pi**2*c1*n**2)**2*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*sin_pin*np.cosh(dA)/(H**2 + c1*dAu**2) - 2*np.pi*B*Bu*c1*dA*n*(H**3 - np.pi**2*H*c1*n**2)**2*(dA*(-2*np.pi*dAu*n*sin_pin*np.sinh(dAu) + (dA**2 - dAu**2 + np.pi**2*n**2)*cos_pin*np.cosh(dAu))*np.sinh(dA) + (dAu*(-dA**2 + dAu**2 + np.pi**2*n**2)*cos_pin*np.sinh(dAu) + np.pi*n*(dA**2 + dAu**2 + np.pi**2*n**2)*sin_pin*np.cosh(dAu))*np.cosh(dA))/((dA**2 - 2*dA*dAu + dAu**2 + np.pi**2*n**2)*(dA**2 + 2*dA*dAu + dAu**2 + np.pi**2*n**2)) - B*H**8*(dA*sin_pin*np.cosh(dA) - np.pi*n*cos_pin*np.sinh(dA))/(dA**2 + np.pi**2*n**2) - B*H**6*c1*dA**2*(dA*sin_pin*np.cosh(dA) - np.pi*n*cos_pin*np.sinh(dA))/(dA**2 + np.pi**2*n**2) + 3*np.pi**2*B*H**6*c1*n**2*(dA*sin_pin*np.cosh(dA) - np.pi*n*cos_pin*np.sinh(dA))/(dA**2 + np.pi**2*n**2) - np.pi*B*H**6*c1*n*cos_pin*np.sinh(dA) + 2*np.pi**2*B*H**4*c1**2*dA**2*n**2*(dA*sin_pin*np.cosh(dA) - np.pi*n*cos_pin*np.sinh(dA))/(dA**2 + np.pi**2*n**2) + np.pi*B*H**4*c1**2*dA**2*n*cos_pin*np.sinh(dA) + 3*np.pi**4*B*H**4*c1**2*n**4*(-dA*sin_pin*np.cosh(dA) + np.pi*n*cos_pin*np.sinh(dA))/(dA**2 + np.pi**2*n**2) + 3*np.pi**3*B*H**4*c1**2*n**3*cos_pin*np.sinh(dA) + np.pi**4*B*H**2*c1**3*dA**2*n**4*(-dA*sin_pin*np.cosh(dA) + np.pi*n*cos_pin*np.sinh(dA))/(dA**2 + np.pi**2*n**2) - 2*np.pi**3*B*H**2*c1**3*dA**2*n**3*cos_pin*np.sinh(dA) + np.pi**6*B*H**2*c1**3*n**6*(dA*sin_pin*np.cosh(dA) - np.pi*n*cos_pin*np.sinh(dA))/(dA**2 + np.pi**2*n**2) - 3*np.pi**5*B*H**2*c1**3*n**5*cos_pin*np.sinh(dA) + np.pi**5*B*c1**4*dA**2*n**5*cos_pin*np.sinh(dA) + np.pi**7*B*c1**4*n**7*cos_pin*np.sinh(dA) + 2*np.pi*B*c1*dA*n*(H**3 - np.pi**2*H*c1*n**2)**2*(dA*cos_pin*np.sinh(dA) + np.pi*n*sin_pin*np.cosh(dA))/(dA**2 + np.pi**2*n**2) - B*c1*dA*(H**2 + c1*(dA**2 + np.pi**2*n**2))*(H**2 - np.pi**2*c1*n**2)**2*sin_pin*np.cosh(dA) + Bu*H**8*dA*(-dAu*(2*np.pi*n*cos_pin + (dAu**2 + np.pi**2*n**2)*sin_pin)*np.sinh(dAu) + (np.pi*n*(dAu**2 + np.pi**2*n**2)*cos_pin + (dAu**2 - np.pi**2*n**2)*sin_pin)*np.cosh(dAu))/(dAu**2 + np.pi**2*n**2)**2 - np.pi*Bu*H**7*c1*dA*n*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*cos_pin/(H**2 + c1*dAu**2) - Bu*H**7*c1*dA*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*sin_pin/(H**2 + c1*dAu**2) + 2*Bu*H**6*c1*dA**3*(-dAu*(2*np.pi*n*cos_pin + (dAu**2 + np.pi**2*n**2)*sin_pin)*np.sinh(dAu) + (np.pi*n*(dAu**2 + np.pi**2*n**2)*cos_pin + (dAu**2 - np.pi**2*n**2)*sin_pin)*np.cosh(dAu))/(dAu**2 + np.pi**2*n**2)**2 + 3*np.pi**2*Bu*H**6*c1*dA*n**2*(dAu*(2*np.pi*n*cos_pin + (dAu**2 + np.pi**2*n**2)*sin_pin)*np.sinh(dAu) - (np.pi*n*(dAu**2 + np.pi**2*n**2)*cos_pin + (dAu**2 - np.pi**2*n**2)*sin_pin)*np.cosh(dAu))/(dAu**2 + np.pi**2*n**2)**2 + 2*np.pi*Bu*H**6*c1*dA*n*(dAu*cos_pin*np.sinh(dAu) + np.pi*n*sin_pin*np.cosh(dAu))/(dAu**2 + np.pi**2*n**2) - 2*np.pi*Bu*H**5*c1**2*dA**3*n*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*cos_pin/(H**2 + c1*dAu**2) - 2*Bu*H**5*c1**2*dA**3*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*sin_pin/(H**2 + c1*dAu**2) + 3*np.pi**3*Bu*H**5*c1**2*dA*n**3*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*cos_pin/(H**2 + c1*dAu**2) + np.pi**2*Bu*H**5*c1**2*dA*n**2*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*sin_pin/(H**2 + c1*dAu**2) + Bu*H**4*c1**2*dA**5*(-dAu*(2*np.pi*n*cos_pin + (dAu**2 + np.pi**2*n**2)*sin_pin)*np.sinh(dAu) + (np.pi*n*(dAu**2 + np.pi**2*n**2)*cos_pin + (dAu**2 - np.pi**2*n**2)*sin_pin)*np.cosh(dAu))/(dAu**2 + np.pi**2*n**2)**2 + 4*np.pi*Bu*H**4*c1**2*dA**3*n*(dAu*cos_pin*np.sinh(dAu) + np.pi*n*sin_pin*np.cosh(dAu))/(dAu**2 + np.pi**2*n**2) + 3*np.pi**4*Bu*H**4*c1**2*dA*n**4*(-dAu*(2*np.pi*n*cos_pin + (dAu**2 + np.pi**2*n**2)*sin_pin)*np.sinh(dAu) + (np.pi*n*(dAu**2 + np.pi**2*n**2)*cos_pin + (dAu**2 - np.pi**2*n**2)*sin_pin)*np.cosh(dAu))/(dAu**2 + np.pi**2*n**2)**2 - 4*np.pi**3*Bu*H**4*c1**2*dA*n**3*(dAu*cos_pin*np.sinh(dAu) + np.pi*n*sin_pin*np.cosh(dAu))/(dAu**2 + np.pi**2*n**2) - np.pi*Bu*H**3*c1**3*dA**5*n*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*cos_pin/(H**2 + c1*dAu**2) - Bu*H**3*c1**3*dA**5*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*sin_pin/(H**2 + c1*dAu**2) - 4*np.pi**2*Bu*H**3*c1**3*dA**3*n**2*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*sin_pin/(H**2 + c1*dAu**2) - 3*np.pi**5*Bu*H**3*c1**3*dA*n**5*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*cos_pin/(H**2 + c1*dAu**2) + np.pi**4*Bu*H**3*c1**3*dA*n**4*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*sin_pin/(H**2 + c1*dAu**2) + np.pi**2*Bu*H**2*c1**3*dA**5*n**2*(dAu*(2*np.pi*n*cos_pin + (dAu**2 + np.pi**2*n**2)*sin_pin)*np.sinh(dAu) - (np.pi*n*(dAu**2 + np.pi**2*n**2)*cos_pin + (dAu**2 - np.pi**2*n**2)*sin_pin)*np.cosh(dAu))/(dAu**2 + np.pi**2*n**2)**2 + 2*np.pi*Bu*H**2*c1**3*dA**5*n*(dAu*cos_pin*np.sinh(dAu) + np.pi*n*sin_pin*np.cosh(dAu))/(dAu**2 + np.pi**2*n**2) + 2*np.pi**4*Bu*H**2*c1**3*dA**3*n**4*(dAu*(2*np.pi*n*cos_pin + (dAu**2 + np.pi**2*n**2)*sin_pin)*np.sinh(dAu) - (np.pi*n*(dAu**2 + np.pi**2*n**2)*cos_pin + (dAu**2 - np.pi**2*n**2)*sin_pin)*np.cosh(dAu))/(dAu**2 + np.pi**2*n**2)**2 + 4*np.pi**3*Bu*H**2*c1**3*dA**3*n**3*(dAu*cos_pin*np.sinh(dAu) + np.pi*n*sin_pin*np.cosh(dAu))/(dAu**2 + np.pi**2*n**2) + np.pi**6*Bu*H**2*c1**3*dA*n**6*(dAu*(2*np.pi*n*cos_pin + (dAu**2 + np.pi**2*n**2)*sin_pin)*np.sinh(dAu) - (np.pi*n*(dAu**2 + np.pi**2*n**2)*cos_pin + (dAu**2 - np.pi**2*n**2)*sin_pin)*np.cosh(dAu))/(dAu**2 + np.pi**2*n**2)**2 + 2*np.pi**5*Bu*H**2*c1**3*dA*n**5*(dAu*cos_pin*np.sinh(dAu) + np.pi*n*sin_pin*np.cosh(dAu))/(dAu**2 + np.pi**2*n**2) + np.pi**3*Bu*H*c1**4*dA**5*n**3*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*cos_pin/(H**2 + c1*dAu**2) - np.pi**2*Bu*H*c1**4*dA**5*n**2*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*sin_pin/(H**2 + c1*dAu**2) + 2*np.pi**5*Bu*H*c1**4*dA**3*n**5*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*cos_pin/(H**2 + c1*dAu**2) - 2*np.pi**4*Bu*H*c1**4*dA**3*n**4*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*sin_pin/(H**2 + c1*dAu**2) + np.pi**7*Bu*H*c1**4*dA*n**7*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*cos_pin/(H**2 + c1*dAu**2) - np.pi**6*Bu*H*c1**4*dA*n**6*(H*np.cosh(dAu) + c1_wo*dAu*1/tan_Hc1*np.sinh(dAu))*sin_pin/(H**2 + c1*dAu**2) - H**8*dA*(np.pi*n*cos_pin - sin_pin)/(np.pi**2*n**2) - 2*H**6*c1*dA**3*(np.pi*n*cos_pin - sin_pin)/(np.pi**2*n**2) + np.pi*H**6*c1*dA*n*cos_pin + 3*H**6*c1*dA*(np.pi*n*cos_pin - sin_pin) - H**6*c1*dA*sin_pin - H**4*c1**2*dA**5*(np.pi*n*cos_pin - sin_pin)/(np.pi**2*n**2) + 2*np.pi*H**4*c1**2*dA**3*n*cos_pin - 2*H**4*c1**2*dA**3*sin_pin - 3*np.pi**3*H**4*c1**2*dA*n**3*cos_pin - 3*np.pi**2*H**4*c1**2*dA*n**2*(np.pi*n*cos_pin - sin_pin) + 3*np.pi**2*H**4*c1**2*dA*n**2*sin_pin + np.pi*H**2*c1**3*dA**5*n*cos_pin + H**2*c1**3*dA**5*(np.pi*n*cos_pin - sin_pin) - H**2*c1**3*dA**5*sin_pin + 2*np.pi**2*H**2*c1**3*dA**3*n**2*(np.pi*n*cos_pin - sin_pin) + 3*np.pi**5*H**2*c1**3*dA*n**5*cos_pin + np.pi**4*H**2*c1**3*dA*n**4*(np.pi*n*cos_pin - sin_pin) - 3*np.pi**4*H**2*c1**3*dA*n**4*sin_pin - np.pi**3*c1**4*dA**5*n**3*cos_pin + np.pi**2*c1**4*dA**5*n**2*sin_pin - 2*np.pi**5*c1**4*dA**3*n**5*cos_pin + 2*np.pi**4*c1**4*dA**3*n**4*sin_pin - np.pi**7*c1**4*dA*n**7*cos_pin + np.pi**6*c1**4*dA*n**6*sin_pin)/(dA*(H**2 - np.pi**2*c1*n**2)**2*(H**4 + 2*H**2*c1*(dA**2 - np.pi**2*n**2) + c1**2*(dA**2 + np.pi**2*n**2)**2))
                
                
    out = utstc2+utstc34(n0).sum(0)
    return out
'''

