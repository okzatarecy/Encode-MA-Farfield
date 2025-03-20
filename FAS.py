# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 15:49:38 2025

@author: orecy
"""

import numpy as np
import matplotlib.pyplot as plt # Import matplotlib.pyplot
c = 3e8
f = 3e9
Lambda = c / f
d_BS = Lambda/2; d_U = Lambda/2
N = 3 # Number of antennas in Base station
M = 3 # Number of antennas in User
m0 = 2 # number of ports are selected for signal reception >> bagaiamana cara menentukan jumlah port yang hidup?
r_m = np.array([1, 3]); #index of selected ports (example)
bits = np.random.randint(0, 2, 1000) # Generate 1000 random bits (0 or 1)
s = 2 * bits - 1  # Map bits to BPSK symbols (-1 or 1)
z = np.sqrt(1/2) * (np.random.randn(m0) + 1j * np.random.randn(m0))
k_nd_BS = (2 * (np.arange(1, N+1) - 1) - N + 1)/2 * d_BS; # The coordinate of antenna n
k_rmd_U = (2 * (r_m - 1) - M + 1)/2 * d_U; # Coordinate of activated port r_m
L_t = 2; # Number of transmit paths
L_r = 2; # Number of receive paths

# %%
bits = np.random.randint(0, 2, 1000)
bits_reshape = bits.reshape(-1, 2)

# Map bits to QPSK symbols using 3GPP equations
s = np.zeros(bits_reshape.shape[0], dtype=complex)
for i in range(bits_reshape.shape[0]):
    b0 = bits_reshape[i, 0]
    b1 = bits_reshape[i, 1]
    # 3GPP QPSK mapping equations:
    I = (1 - 2 * b0) / np.sqrt(2)
    Q = (1 - 2 * b1) / np.sqrt(2)
    s[i] = I + 1j * Q  # Create the complex QPSK symbol

s = s.reshape(N,-1)
Q = np.cov(s)   #matrix covariance using

print(np.sum(np.diag(Q))/N)
s.shape

# %%

eigenvalues = np.linalg.eigvals(Q)
is_psd = np.all(eigenvalues >= 0) # to check is Q a positive semidefinite matrix
print(is_psd)
print(Q.shape)

# %% Transmitter side

scatters_coordinate_t = np.array([[1, 1.2], [1, -1.2]]); # scatters coordinate form the origin O_t (x, y)
l_t = np.sqrt(scatters_coordinate_t[:, 0]**2 + scatters_coordinate_t[:,1]**2 \
              - (2 * scatters_coordinate_t[:,0] * scatters_coordinate_t[:,1] \
                 * np.cos(90*np.pi/180))); # distance from scatter to the origin O_t
theta_t = np.arccos(scatters_coordinate_t[:, 0] / l_t) # elevation from origin to scatters (elevation transmit path)
print(f'Distance O_t to scatter = {l_t}')
print(f'Elevation theta {theta_t} and elevation in degree {theta_t*(90/np.pi)}')


k_n = k_nd_BS / d_BS;
rho_scatter1 = -k_nd_BS * np.sin(theta_t[0]) - ((k_n**2 * d_BS**2 * np.sin(theta_t[0])**2) / 2*l_t[0])
rho_scatter2 = -k_nd_BS * np.sin(theta_t[1]) - ((k_n**2 * d_BS**2 * np.sin(theta_t[1])**2) / 2*l_t[1])
rho_t = np.array([rho_scatter1, rho_scatter2])
print(rho_t)


signal_phase_different_tx = (2*np.pi*rho_t/Lambda);
a = np.exp(1j*(2*np.pi/Lambda)*rho_t[:,0]) # transmit field response vector
A = np.exp(1j*(2*np.pi/Lambda)*rho_t)   # The field response vectors of all the N transmit antennas
print(signal_phase_different_tx)
print(np.angle(A))

# %% Receiver side

scatters_coordinate_r = np.array([[-1, 1.2], [-1, -1.2]]); # scatters coordinate form the origin O_r (x, y)
l_r = np.sqrt(scatters_coordinate_r[:, 0]**2 + scatters_coordinate_r[:,1]**2 \
              - (2 * scatters_coordinate_r[:,0] * scatters_coordinate_r[:,1] \
                 * np.cos(90*np.pi/180)))
theta_r = np.arccos(scatters_coordinate_r[:, 0] / l_r) # elevation from origin to scatters (elevation receiver path)
print(f'Distance O_r to scatter = {l_r}')
print(f'Elevation theta {theta_r} and elevation in degree {theta_r*(90/np.pi)}')


k_rm = k_rmd_U / d_U;
rho_scatter1_r = -k_rmd_U * np.sin(theta_r[0]) - ((k_rm**2 * d_U**2 * np.sin(theta_r[0])**2) / 2*l_r[0])
rho_scatter2_r = -k_rmd_U * np.sin(theta_r[1]) - ((k_rm**2 * d_U**2 * np.sin(theta_r[1])**2) / 2*l_r[1])
rho_r = np.array([rho_scatter1_r, rho_scatter2_r]);
print(rho_r)

b = np.exp(1j*(2*np.pi/Lambda)*rho_r[:,0]) # Receive field response vector
B = np.exp(1j*(2*np.pi/Lambda)*rho_r)   # The field response vectors of all the m_o receive antennas
B_Hermition = B.conj().T

O = np.random.normal(loc=0, scale=1, size=(2, 2)) + 1j*np.random.normal(loc=0, scale=1, size=(2, 2)) # (i.i.d.) Gaussian random variable with zero mean and variance α2
H = B_Hermition@O@A
print(H.shape)
print(O)

# %% Single point achiavable rate

sigma = np.sqrt(1/(20**(0/10)))
I = np.eye(m0)
R_calc = I + (H@Q@(H.conj().T)) / sigma**2 
R = np.log2(np.linalg.det(R_calc))
R = np.mean(R).real
print(R_calc)
print(R)

# %%
c = 3e8
f = 3e9
Lambda = c / f
d_BS = Lambda/2; d_U = Lambda/2
N = 3 # Number of antennas in Base station
M = 3 # Number of antennas in User
m0 = 2 # number of ports are selected for signal reception >> bagaiamana cara menentukan jumlah port yang hidup?
r_m = np.array([1, 3]); #index of selected ports (example)
bits = np.random.randint(0, 2, 1000) # Generate 1000 random bits (0 or 1)
s = 2 * bits - 1  # Map bits to BPSK symbols (-1 or 1)
z = np.sqrt(1/2) * (np.random.randn(m0) + 1j * np.random.randn(m0))
k_nd_BS = (2 * (np.arange(1, N+1) - 1) - N + 1)/2 * d_BS; # The coordinate of antenna n
k_rmd_U = (2 * (r_m - 1) - M + 1)/2 * d_U; # Coordinate of activated port r_m
L_t = 2; # Number of transmit paths
L_r = 2; # Number of receive paths

def channel_gen(k_nd_BS, d_BS, k_rmd_U, d_U):
    scatters_coordinate_t = np.array([[1, 1.2], [1, -1.2]]); # scatters coordinate form the origin O_t (x, y)
    l_t = np.sqrt(scatters_coordinate_t[:, 0]**2 + scatters_coordinate_t[:,1]**2 \
                  - (2 * scatters_coordinate_t[:,0] * scatters_coordinate_t[:,1] \
                     * np.cos(90*np.pi/180))); # distance from scatter to the origin O_t
    theta_t = np.arccos(scatters_coordinate_t[:, 0] / l_t) # elevation from origin to scatters (elevation transmit path)

    k_n = k_nd_BS / d_BS;
    rho_scatter1 = -k_nd_BS * np.sin(theta_t[0]) - ((k_n**2 * d_BS**2 * np.sin(theta_t[0])**2) / 2*l_t[0])
    rho_scatter2 = -k_nd_BS * np.sin(theta_t[1]) - ((k_n**2 * d_BS**2 * np.sin(theta_t[1])**2) / 2*l_t[1])
    rho_t = np.array([rho_scatter1, rho_scatter2])


    signal_phase_different_tx = (2*np.pi*rho_t/Lambda);
    a = np.exp(1j*(2*np.pi/Lambda)*rho_t[:,0]) # transmit field response vector
    A = np.exp(1j*(2*np.pi/Lambda)*rho_t)   # The field response vectors of all the N transmit antennas

    scatters_coordinate_r = np.array([[-1, 1.2], [-1, -1.2]]); # scatters coordinate form the origin O_r (x, y)
    l_r = np.sqrt(scatters_coordinate_r[:, 0]**2 + scatters_coordinate_r[:,1]**2 \
                  - (2 * scatters_coordinate_r[:,0] * scatters_coordinate_r[:,1] \
                     * np.cos(90*np.pi/180)))
    theta_r = np.arccos(scatters_coordinate_r[:, 0] / l_r) # elevation from origin to scatters (elevation receiver path)

    k_rm = k_rmd_U / d_U;
    rho_scatter1_r = -k_rmd_U * np.sin(theta_r[0]) - ((k_rm**2 * d_U**2 * np.sin(theta_r[0])**2) / 2*l_r[0])
    rho_scatter2_r = -k_rmd_U * np.sin(theta_r[1]) - ((k_rm**2 * d_U**2 * np.sin(theta_r[1])**2) / 2*l_r[1])
    rho_r = np.array([rho_scatter1_r, rho_scatter2_r]);

    b = np.exp(1j*(2*np.pi/Lambda)*rho_r[:,0]) # Receive field response vector
    B = np.exp(1j*(2*np.pi/Lambda)*rho_r)   # The field response vectors of all the m_o receive antennas
    B_Hermition = B.conj().T

    O = np.random.normal(loc=0, scale=1, size=(2, 2)) + 1j*np.random.normal(loc=0, scale=1, size=(2, 2)) # (i.i.d.) Gaussian random variable with zero mean and variance α2
    ch_gen = B_Hermition@O@A
    
    return ch_gen

chan = channel_gen(k_nd_BS, d_BS, k_rmd_U, d_U)
# %%
    

N_data = 2
H_sample_real = []
H_sample_imag = []
for i_channel in range(N_data):
    
    scatters_coordinate_t = np.array([[1, 1.2], [1, -1.2]]); # scatters coordinate form the origin O_t (x, y)
    l_t = np.sqrt(scatters_coordinate_t[:, 0]**2 + scatters_coordinate_t[:,1]**2 \
                  - (2 * scatters_coordinate_t[:,0] * scatters_coordinate_t[:,1] \
                     * np.cos(90*np.pi/180))); # distance from scatter to the origin O_t
    theta_t = np.arccos(scatters_coordinate_t[:, 0] / l_t) # elevation from origin to scatters (elevation transmit path)

    k_n = k_nd_BS / d_BS;
    rho_scatter1 = -k_nd_BS * np.sin(theta_t[0]) - ((k_n**2 * d_BS**2 * np.sin(theta_t[0])**2) / 2*l_t[0])
    rho_scatter2 = -k_nd_BS * np.sin(theta_t[1]) - ((k_n**2 * d_BS**2 * np.sin(theta_t[1])**2) / 2*l_t[1])
    rho_t = np.array([rho_scatter1, rho_scatter2])


    signal_phase_different_tx = (2*np.pi*rho_t/Lambda);
    a = np.exp(1j*(2*np.pi/Lambda)*rho_t[:,0]) # transmit field response vector
    A = np.exp(1j*(2*np.pi/Lambda)*rho_t)   # The field response vectors of all the N transmit antennas

    scatters_coordinate_r = np.array([[-1, 1.2], [-1, -1.2]]); # scatters coordinate form the origin O_r (x, y)
    l_r = np.sqrt(scatters_coordinate_r[:, 0]**2 + scatters_coordinate_r[:,1]**2 \
                  - (2 * scatters_coordinate_r[:,0] * scatters_coordinate_r[:,1] \
                     * np.cos(90*np.pi/180)))
    theta_r = np.arccos(scatters_coordinate_r[:, 0] / l_r) # elevation from origin to scatters (elevation receiver path)

    k_rm = k_rmd_U / d_U;
    rho_scatter1_r = -k_rmd_U * np.sin(theta_r[0]) - ((k_rm**2 * d_U**2 * np.sin(theta_r[0])**2) / 2*l_r[0])
    rho_scatter2_r = -k_rmd_U * np.sin(theta_r[1]) - ((k_rm**2 * d_U**2 * np.sin(theta_r[1])**2) / 2*l_r[1])
    rho_r = np.array([rho_scatter1_r, rho_scatter2_r]);

    b = np.exp(1j*(2*np.pi/Lambda)*rho_r[:,0]) # Receive field response vector
    B = np.exp(1j*(2*np.pi/Lambda)*rho_r)   # The field response vectors of all the m_o receive antennas
    B_Hermition = B.conj().T

    O = np.random.normal(loc=0, scale=1, size=(2, 2)) + 1j*np.random.normal(loc=0, scale=1, size=(2, 2)) # (i.i.d.) Gaussian random variable with zero mean and variance α2
    ch_gen = B_Hermition@O@A
    
    inputs_og = np.reshape(ch_gen,(-1,1))
    inputs = np.round(inputs_og, 5)
    H_real = np.real(inputs).flatten()
    H_imag = np.imag(inputs).flatten()    
    
    H_sample_real.append(H_real)
    H_sample_imag.append(H_imag)
        
    # H_samp = generate_user_in_circle_multipath(Nt, d, r_circle_min, r_circle_max, fc, N_user, sigma_aod, L, kappa)
    # inputs= np.reshape(H_samp,(-1,1))
    # inputs = np.round(input_og, 5)
    # H_real = np.real(inputs).flatten()
    # H_imag = np.imag(inputs).flatten()
    
    # H_sample_real.append(H_real)
    # H_sample_imag.append(H_imag)



