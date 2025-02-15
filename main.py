# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:22:51 2025

@author: gfrpurba
"""
# =============================================================================
# from qiskit.circuit import Parameter
# from qiskit import QuantumCircuit
# from qiskit import QuantumRegister, ClassicalRegister
# from qiskit import transpile
# from qiskit.visualization import plot_histogram
# =============================================================================
import numpy as np
from math import pi
import scipy

# %% defining parameters
fc = 30e9               #carrier frequency [Hz]                
lambda_w = 3e8 / fc;    #wavelength [m]
d = lambda_w/2          #distance of elemen antennas [m]
Nt = 512                #elemen of antennas
N_user = 10             #number of users
r_circle_min = 4        #min diameter circle [m]
r_circle_max = 100      #max diameter circle [m]
L = 5                   #number of paths
kappa_list = np.arange(0.5,10,0.5);
kappa = kappa_list[0];  #Rician factor
sigma_aod = np.pi/180*5;    #angle of departure

# %% defining functions
def ula(phi, N_ant, f = 30e9):
    c = 3e8
    lambda_w = c / f
    d = lambda_w/2
    k_wave_num = (2*np.pi)/lambda_w
    phase = np.zeros(N_ant, dtype=complex)
    #s = np.zeros(bits_reshape.shape[0], dtype=complex)
    for n in range (1,N_ant):
        result = np.exp(1j*k_wave_num*(n*d)*np.sin(phi))
        phase[n] = result    
    return (1/np.sqrt(N_ant))*phase   

ula(0.1, 5)

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)  
    return(theta, rho)

# %% generate_user_multipath
def generate_user_in_circle_multipath(Nt, d, r_min, r_max, fc, N_user, sigma_aod, L, kappa):
    
    H_multi_user = np.zeros((N_user,Nt),dtype=complex)
    x_list = np.zeros((1,N_user)); y_list = np.zeros((1,N_user));
    cnt = 0
    trsh_theta = np.pi/3;
    c = 3e8;
    
    while(cnt <= N_user-1):
        x_1 = np.abs(np.random.random()*r_max)  #making random coordinate x_axis user 
        y_1 = (np.random.random()-0.5)*2*r_max  #making random coordinate y_axis user
        theta_1, rho_1 = cart2pol(x_1, y_1);    #carthesian to polar [angle, magnitude]
        r_1 = x_1**2+y_1**2;
        if (r_1 < r_max**2) and (r_1 > r_min**2) and (-trsh_theta < theta_1 < trsh_theta):
            x_list[:,cnt] = x_1;
            y_list[:,cnt] = y_1;
            cnt = cnt+1;
            
    [theta_list, r_list]  = cart2pol(x_list, y_list);
    nn = np.arange(-(Nt-1)/2,(Nt-1)/2,1);   #position of element from - 0 +
    theta_aod  = np.sqrt(sigma_aod)*np.random.random([N_user,L]);
    ssf = (np.random.random([N_user,L]) + 1j*np.random.random([N_user,L]))/np.sqrt(2);
    #allocate a factor to fade the NLoS channel
    alpha = 1;
    beta = np.sqrt(alpha/(L));
    ssf = ssf*beta;

    for i_user in range(N_user):
        for l in range(L+1):
            if l != L+1:
                r0 = r_list[i_user];
                theta0 = theta_list[i_user]+theta_aod[i_user,l];
                r = np.sqrt(r0**2 + (nn*d)**2 - 2*r0*nn*d*np.sin(theta0));
                at = np.exp(-1j*2*np.pi*fc*(r - r0)/c)/np.sqrt(Nt);
                H_multi_user[i_user, :] = H_multi_user[i_user, :] + ssf[i_user,l]*at*np.sqrt(1/(1+kappa));
            else:
                r0 = r_list[i_user];
                theta0 = theta_list[i_user];
                r = np.sqrt(r0**2 + (nn*d)**2 - 2*r0*nn*d*np.sin(theta0));
                at = np.exp(-1j*2*np.pi*fc*(r - r0)/c)/np.sqrt(Nt);
                H_multi_user[i_user, :] = H_multi_user[i_user, :] + at*np.sqrt(kappa/(1+kappa));
    
    if L ==0:
        H_multi_user = H_multi_user/np.sqrt(kappa/(1+kappa));
        
    return H_multi_user

generate_user_in_circle_multipath(Nt, d, r_circle_min,r_circle_max, fc, 10, sigma_aod, L, 1)    
        
        
        
        
        