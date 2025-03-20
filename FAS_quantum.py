# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 19:40:33 2025

@author: orecy
"""

import numpy as np
import matplotlib.pyplot as plt # Import matplotlib.pyplot

import qiskit
from qiskit.circuit import Parameter
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import transpile
from qiskit.result import marginal_counts
from qiskit.visualization import plot_histogram
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import Pauli, SparsePauliOp
import numpy as np
from math import pi
import scipy
from scipy.special import jv

# %% initialization

c = 3e8
f = 3e9
Lambda = c / f
d_BS = Lambda/2; d_U = Lambda/2
N = 3 # Number of antennas in Base station
N_BS = N
M = 4 # Number of antennas in User
m0 = 2 # number of ports are selected for signal reception >> bagaiamana cara menentukan jumlah port yang hidup?
r_m = np.array([1, 4]); #index of selected ports (example)
bits = np.random.randint(0, 2, 1000) # Generate 1000 random bits (0 or 1)
s = 2 * bits - 1  # Map bits to BPSK symbols (-1 or 1)
z = np.sqrt(1/2) * (np.random.randn(m0) + 1j * np.random.randn(m0))
k_nd_BS = (2 * (np.arange(1, N+1) - 1) - N + 1)/2 * d_BS; # The coordinate of antenna n
k_rmd_U = (2 * (r_m - 1) - M + 1)/2 * d_U; # Coordinate of activated port r_m
L_t = 2; # Number of transmit paths
L_r = 2; # Number of receive paths

# %% generate channel

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

# %% Simplified channel model FAS
N_port = 6
N_sample = 1
W = 0.5
x = np.sqrt(1/2)*np.random.randn(N_port,N_sample)
y = np.sqrt(1/2)*np.random.randn(N_port, N_sample)
print(x,y)

h_ch = np.zeros((N_port,N_sample), dtype=complex)
mu = np.zeros(N_port)
h_ch[0,:] = x[0,:] + 1j*y[0,:]     #reference port
mu[0] = 1                       #reference port (mu is spatial correlation between port using bessel func)
for i_sample in range(N_sample):
    
    for i_port in range(1,N_port):
        mu[i_port] = jv(0, 2*np.pi * (abs((i_port+1)-1)/(N_port-1)) * W)
        h_ch[i_port,:] = (np.sqrt(1-mu[i_port]**2) * x[i_port,:] + mu[i_port] * x[0,:] + 
                        1j*(np.sqrt(1-mu[i_port]**2) * y[i_port,:] + mu[i_port] * y[0,:]))
        
    
inputs_og = np.reshape(h_ch,(-1,1))
inputs = np.round(inputs_og, 5)
H_real = np.real(inputs).flatten()
H_imag = np.imag(inputs).flatten()
# H_real_pick = H_real[0::2]
# H_imag_pick = H_imag[0::2]

# %%


xy = np.vstack((x, y))
Q = np.cov(mu)
eigenvalues = np.linalg.eigvals(Q)
is_psd = np.all(eigenvalues >= 0) # to check is Q a positive semidefinite matrix
print(is_psd)
print(Q.shape)

# %% learning

# N_data = 1
# H_sample_real = []
# H_sample_imag = []

# for i_channel in range(N_data):
#     ch_gen = channel_gen(k_nd_BS, d_BS, k_rmd_U, d_U)

#     inputs_og = np.reshape(ch_gen,(-1,1))
#     inputs = np.round(inputs_og, 5)
#     H_real = np.real(inputs).flatten()
#     H_imag = np.imag(inputs).flatten()    

#     H_sample_real.append(H_real)
#     H_sample_imag.append(H_imag)
    
# %%
w_1 = 0
w_2 = 0
w_3 = 0
w_4 = 0
w_5 = 0
w_6 = 0

def ave_meas(count):
     total = count.get('0', 0) + count.get('1', 0)
     return count.get('1', 0) / total if total > 0 else 0

def Q_sampler_est(H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6, shots):
        
    q = QuantumRegister(H_real.size, 'q')
    c = ClassicalRegister(H_real.size, 'c')
    qc1= QuantumCircuit(q,c)
        
    for k in range(H_real.size):
        qc1.h(q[k])
            
    qc1.barrier()
        
    for k in range(H_real.size):
        qc1.ry(H_real[k], q[k])
            
    for i in range(H_real.size):
        qc1.rz(H_imag[i], q[i])
        
    qc1.barrier()
    
    for k in range(H_real.size-2):    
        qc1.cx(q[k], q[k+1])

    qc1.cx(q[H_real.size-1], q[0])
        
    qc1.barrier()
        
    qc1.u(w_1, 0, 0, q[0])
    qc1.u(w_2, 0, 0, q[1])
    qc1.u(w_3, 0, 0, q[2])
    qc1.u(w_4, 0, 0, q[3])
    qc1.u(w_5, 0, 0, q[4])
    qc1.u(w_6, 0, 0, q[5])
    qc1.barrier()
        
    qc1.measure(q[0], c[0]) 
    qc1.measure(q[1], c[1]) 
    qc1.measure(q[2], c[2]) 
    qc1.measure(q[3], c[3])
    qc1.measure(q[4], c[4]) 
    qc1.measure(q[5], c[5])     
        
    sampler = StatevectorSampler()
        
    job_sam = sampler.run( [(qc1)], shots = shots)
    result_sam = job_sam.result()
    counts_sam = result_sam[0].data.c.get_counts()
        
    simp_counts_01 = marginal_counts(counts_sam, indices=[5])
    simp_counts_02 = marginal_counts(counts_sam, indices=[4])
    simp_counts_03 = marginal_counts(counts_sam, indices=[3])
    simp_counts_04 = marginal_counts(counts_sam, indices=[2])
    simp_counts_05 = marginal_counts(counts_sam, indices=[1])
    simp_counts_06 = marginal_counts(counts_sam, indices=[0])
        # counts_sam = result_sam[0].data.c.get_counts()
        
    out1 = ave_meas(simp_counts_01)
    out2 = ave_meas(simp_counts_02)
    out3 = ave_meas(simp_counts_03)
    out4 = ave_meas(simp_counts_04)
    out5 = ave_meas(simp_counts_05)
    out6 = ave_meas(simp_counts_06)
        
    out = [out1, out2, out3, out4, out5, out6]

    return qc1, counts_sam, out, out1, out2, out3, out4, out5, out6

#H_real = H_sample_real[0]
#H_imag = H_sample_imag[0]
qc1, counts_sam,out, out1, out2, out3, out4, out5, out6 = Q_sampler_est(H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6, shots=1024)
print("measurement_average_01 =",out[0])
print("measurement_average_02 =",out[1])
print("measurement_average_03 =",out[2])
print("measurement_average_04 =",out[3])
print("measurement_average_05 =",out[4])
print("measurement_average_06 =",out[5])
# %%

qc1.draw()
plot_histogram(counts_sam, sort='value_desc')

# %% loss function
ptx = 5
# def loss(N_BS, ch_gen, H_real, H_imag, w_1, w_2, w_3, w_4, shots):
#     qc1, counts_sam,out, out1, out2, out3, out4, out5, out6 = Q_sampler_est(H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6, shots)
#     ptx = 5
#     sigma_n = 1
    
#     v1 = np.exp(1j*(out1+out4))
#     v2 = np.exp(1j*(out2+out5))
#     v3 = np.exp(1j*(out3+out6))
    
#     V = np.array([v1, v2, v3])
#     cap_P1 = np.log2(1+(ptx*np.abs(ch_gen[0,:]@v1)**2/sigma_n))
#     cap_P2 = np.log2(1+(ptx*np.abs(ch_gen[1,:]@V)**2/sigma_n))
#     print(cap_P1)
#     # print(cap_P2)
#     # print(cap_P1+cap_P2)
#     return loss

# los = loss(h_ch, H_real_pick, H_imag_pick, w_1, w_2, w_3, w_4, shots=1024)
# %%
def loss(h_ch, H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6):
    
    qc1, counts_sam,out, out1, out2, out3, out4, out5, out6 = Q_sampler_est(H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6, shots=1024)
    ptx = 5
    sigma_n = 1
    
    Q = np.array([out1, out2, out3, out4, out5, out6])
    
    indices = np.argpartition(Q, -3)[-3:]  # Indeks from max 3
    indices = np.sort(indices)  # Urutkan indeks agar tetap dalam urutan asli A

    # Ambil nilai berdasarkan indeks yang sudah diurutkan
    P = Q[indices]

    # print(P)
    
    V = np.array([np.exp(1j*(np.sum(P)))])
    
    h_max = h_ch[indices]
    #h_recon = np.zeros_like(h_ch)
    #[indices] = h_max
    
    cap = ptx*np.abs(h_max @ V)**2/sigma_n
    rate= np.log2(1+cap)
    sum_rate = np.sum(rate)
    #cap_P2 = np.log2(1+(ptx*np.abs(ch_gen[1,:]@V)**2/sigma_n))
    # print(cap_P2)
    # print(cap_P1+cap_P2)
    
    loss = -(sum_rate)
    return loss

los = loss(h_ch, H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6)

# %%

def gradient(h_ch, H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6, w_index):
        
    shift = np.pi/2
        
    w = np.array([w_1, w_2, w_3, w_4, w_5, w_6])
        
    w_min = w
    w_plus = w
        
    w_min[w_index] = w_min[w_index] - shift
    loss_min = loss(h_ch, H_real, H_imag, w_min[0], w_min[1], w_min[2], w_min[3], w_min[4], w_min[5])
        
    w_plus[w_index] = w_plus[w_index] + shift
    loss_plus = loss(h_ch, H_real, H_imag, w_plus[0], w_plus[1], w_plus[2], w_plus[3], w_plus[4], w_plus[5])
        
    grad = (1/2*np.sin(shift)) * (loss_min-loss_plus)
        
    return grad, loss_min, loss_plus

w_1 = np.pi
w_2 = np.pi
w_3 = np.pi
w_4 = np.pi

grad_1, loss_min, loss_plus = gradient(h_ch, H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6, 1)
