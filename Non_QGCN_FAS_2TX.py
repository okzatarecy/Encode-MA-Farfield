#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 12:29:40 2025

@author: okzatarecy
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
from numpy import linalg as LA
# %% initialization

#### This code for non graph based quantum

# %%
# 
c = 3e8
f = 3e9
Lambda = c / f
d_BS = Lambda/2; d_U = Lambda/2
N = 2 # Number of antennas in Base station
N_BS = N
M = 3 # Number of antennas in User
m0 = 3 # number of ports are selected for signal reception >> bagaiamana cara menentukan jumlah port yang hidup?
r_m = np.array([1,2,3]); #index of selected ports (example)
bits = np.random.randint(0, 2, 1000) # Generate 1000 random bits (0 or 1)
s = 2 * bits - 1  # Map bits to BPSK symbols (-1 or 1)
z = np.sqrt(1/2) * (np.random.randn(m0) + 1j * np.random.randn(m0))
k_nd_BS = (2 * (np.arange(1, N+1) - 1) - N + 1)/2 * d_BS; # The coordinate of antenna n
k_rmd_U = (2 * (r_m - 1) - M + 1)/2 * d_U; # Coordinate of activated port r_m
L_t = 2; # Number of transmit paths
L_r = 2; # Number of receive paths
bandwidth = 1.5e6 #channel bandwidth

 # %%
def channel_gen(k_nd_BS, d_BS, k_rmd_U, d_U):
    angle1 = 80
    angle2 = 100
    scatters_coordinate_t = np.array([[1.7, 1.5], [0.3,-1.2]]); # scatters coordinate form the origin O_t (x, y)
    l_t = np.sqrt(scatters_coordinate_t[:, 0]**2 + scatters_coordinate_t[:,1]**2 \
                  - (2 * scatters_coordinate_t[:,0] * scatters_coordinate_t[:,1] \
                     * np.cos(angle1*np.pi/180))); # distance from scatter to the origin O_t
    theta_t = np.arccos(scatters_coordinate_t[:, 0] / l_t) # elevation from origin to scatters (elevation transmit path)

    k_n = k_nd_BS / d_BS;
    rho_scatter1 = -k_nd_BS * np.sin(theta_t[0]) - ((k_n**2 * d_BS**2 * np.sin(theta_t[0])**2) / 2*l_t[0])
    rho_scatter2 = -k_nd_BS * np.sin(theta_t[1]) - ((k_n**2 * d_BS**2 * np.sin(theta_t[1])**2) / 2*l_t[1])
    rho_t = np.array([rho_scatter1, rho_scatter2])


    signal_phase_different_tx = (2*np.pi*rho_t/Lambda);
    a = np.exp(1j*(2*np.pi/Lambda)*rho_t[:,0]) # transmit field response vector
    A = np.exp(1j*(2*np.pi/Lambda)*rho_t)   # The field response vectors of all the N transmit antennas

    scatters_coordinate_r = np.array([[-0.3, 1.5], [-1.7, -1.2]]); # scatters coordinate form the origin O_r (x, y)
    l_r = np.sqrt(scatters_coordinate_r[:, 0]**2 + scatters_coordinate_r[:,1]**2 \
                  - (2 * scatters_coordinate_r[:,0] * scatters_coordinate_r[:,1] \
                     * np.cos(angle2*np.pi/180)))
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


ch_gen = channel_gen(k_nd_BS, d_BS, k_rmd_U, d_U)
h_ch = np.reshape(ch_gen,(-1,1))
H_real = np.round(np.real(h_ch),5).flatten()
H_imag = np.round(np.imag(h_ch),5).flatten()

# N_port = 3
# N_BS = 2
# WL = 0.5

# def ch_simp(N_port, N_BS, WL):
#     # H_sample_real = []
#     # H_sample_imag = []

#     for i_BS in range(N_BS):
#         x = np.sqrt(1/2)*np.random.randn(N_port,N_BS)
#         y = np.sqrt(1/2)*np.random.randn(N_port,N_BS)
#         h_ch = np.zeros((N_port,N_BS), dtype=complex)
#         mu = np.zeros(N_port)
#         h_ch[0,:] = x[0,:] + 1j*y[0,:]     #reference port
#         mu[0] = 1                       #reference port (mu is spatial correlation between port using bessel func)
#         for i_port in range(1,N_port):
#             mu[i_port] = jv(0, 2*np.pi * (abs((i_port+1)-1)/(N_port-1)) * WL)
#             h_ch[i_port,:] = (np.sqrt(1-mu[i_port]**2) * x[i_port,:] + mu[i_port] * x[0,:] + 
#                               1j*(np.sqrt(1-mu[i_port]**2) * y[i_port,:] + mu[i_port] * y[0,:]))          
            
#             #inputs = np.round(h_ch[:,i_sample], 5)
#         H_real = np.round(np.real(h_ch),5)
#         H_imag = np.round(np.imag(h_ch),5)
#     return h_ch, H_real, H_imag

# ch_gen, H_real, H_imag = ch_simp(N_port, N_BS, WL)
# h_ch = np.reshape(ch_gen,(-1,1))
# H_real = H_real.flatten()
# H_imag = H_imag.flatten()


# %%

def Q_sampler_est(h_ch, H_real, H_imag, params_U):
        
    q = QuantumRegister(H_real.size, 'q')
    c = ClassicalRegister(6, 'c')
    qc1= QuantumCircuit(q,c)
    
    # h_abs1 = np.abs(h_ch)
    # h_abs = h_abs1.flatten()
    # h_max = np.max(h_abs)     
    # angle_amp =  (h_abs / h_max) * np.pi
    # angle_phase = np.angle(h_ch).flatten()
    
    # for k in range(H_real.size):    
    #     qc1.ry(angle_amp[k], q[k])
            
    # for i in range(H_real.size):
    #     qc1.rz(angle_phase[i], q[i])
       
    for k in range(H_real.size):
        qc1.ry(H_real[k], q[k])
                
    for i in range(H_real.size):
        qc1.rz(H_imag[i], q[i])
    qc1.barrier()
 
    
    qc1.cz(q[0], q[1])
    qc1.cz(q[1], q[2])
    qc1.cz(q[2], q[3])
    qc1.cz(q[3], q[4])
    qc1.cz(q[4], q[5])
    qc1.cz(q[5], q[0])

    qc1.barrier()
        
    qc1.ry(params_U[0,0], q[0])
    qc1.ry(params_U[1,0], q[1])
    qc1.ry(params_U[2,0], q[2])
    qc1.ry(params_U[3,0], q[3])
    qc1.ry(params_U[4,0], q[4])
    qc1.ry(params_U[5,0], q[5])
    
    qc1.barrier()
    
    # Superposition Combination
    # qc1.h(q[0])  # Superposition untuk out1, out3, out5
    qc1.cx(q[2], q[0])
    qc1.cx(q[4], q[0])
    
    # qc1.h(q[1])  # Superposition untuk out2, out4, out6
    qc1.cx(q[3], q[1])
    qc1.cx(q[5], q[1])
    
    # Gabungkan qubit q[0], q[2], q[4] menjadi output pertama
    # qc1.cx(q[0], q[2])  # Gabungkan q[0] dan q[2]
    # qc1.cx(q[2], q[4])  # Gabungkan q[2] dan q[4]

    # Gabungkan qubit q[1], q[3], q[5] menjadi output kedua
    # qc1.cx(q[1], q[3])  # Gabungkan q[1] dan q[3]
    # qc1.cx(q[3], q[5])  # Gabungkan q[3] dan q[5]
    # qc1.ccx(q[0], q[2], q[4])
    # qc1.ccx(q[1], q[3], q[5])
    qc1.barrier()
        
    qc1.measure(q[0], c[0]) 
    qc1.measure(q[1], c[1]) 
    return qc1

w_1 = np.pi
w_2 = np.pi
w_3 = np.pi
w_4 = np.pi
w_5 = np.pi
w_6 = np.pi
params_U = np.array([[w_1],
                     [w_2],
                     [w_3],
                     [w_4],
                     [w_5],
                     [w_6]
                     ])
qc = Q_sampler_est(ch_gen, H_real, H_imag, params_U)
# %%
#
def ave_meas(count):
     total = count.get('0', 0) + count.get('1', 0)
     return count.get('1', 0) / total if total > 0 else 0
 
def Q_decode(h_ch, H_real, H_imag, params_U, shots):    
       
    qc1 =  Q_sampler_est(ch_gen, H_real, H_imag, params_U)
    sampler = StatevectorSampler()
        
    job_sam = sampler.run( [(qc1)], shots = shots)
    result_sam = job_sam.result()
    counts_sam = result_sam[0].data.c.get_counts()
        
    simp_counts_01 = marginal_counts(counts_sam, indices=[0])
    simp_counts_02 = marginal_counts(counts_sam, indices=[1])

        # counts_sam = result_sam[0].data.c.get_counts()
        
    out1 = ave_meas(simp_counts_01)
    out2 = ave_meas(simp_counts_02)

        
    out = [out1, out2]

    return qc1, counts_sam, out, out1, out2

#H_real = H_sample_real[0]
#H_imag = H_sample_imag[0]
qc1, counts_sam,out, out1, out2 = Q_decode(ch_gen, H_real, H_imag, params_U, shots=1024)
print("measurement_average_01 =",out[0])
print("measurement_average_02 =",out[1])
# %% loss function

num_ports=3
H = ch_gen
ptx = 5
sigma_n = 1
    

def LossRate(ch_gen, H_real, H_imag, params_U, bandwidth):
    
    count_sam, qc, out, out1, out2 = Q_decode(ch_gen, H_real, H_imag, params_U, shots=1024)
    
    v1 = np.exp(1j*(out1))     # 1st BS
    v2 = np.exp(1j*(out2))     # 2nd BS
    V1 = v1/abs(v1)
    V2 = v2/abs(v2)
    Q = np.array([V1,V2])
    
    
    sinr1 = np.abs(ch_gen[0,:] @ Q)**2
    sinr2 = np.abs(ch_gen[1,:] @ Q)**2
    sinr3 = np.abs(ch_gen[2,:] @ Q)**2
    
    sinr_p1 = sinr1 / (sinr2+sinr3+sigma_n)
    sinr_p2 = sinr2 / (sinr1+sinr3+sigma_n)
    sinr_p3 = sinr3 / (sinr1+sinr2+sigma_n)
    
    sinr_all = np.array([sinr_p1,sinr_p2,sinr_p3])
    # best_port = np.argmax(sinr_all)
    # sum_rate = np.log2(1 + sinr_all[best_port])
    
    indices = np.argsort(sinr_all)[-2:]
    best_sinr = sinr_all[indices]
    
    sum_rate1 = np.log2(1 + best_sinr[0])
    sum_rate2 = np.log2(1 + best_sinr[1])
    sum_rate = sum_rate1+sum_rate2        

    loss = -1*(sum_rate)
    rate = bandwidth * sum_rate
    
    return loss, rate

loss, rate = LossRate(ch_gen, H_real, H_imag, params_U, bandwidth)

# %%
# ww = np.random.randn(3, 2) + 1j * np.random.randn(3, 2)
# num_ports=3
# H = ch_gen
# ptx = 5
# sigma_n = 1
    
# def loss(N_BS, ch_gen, H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6):
    
#     qc1, counts_sam,out, out1, out2, out3, out4, out5, out6 = Q_sampler_est(H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6, shots=2096)
    
#     v1 = np.exp(1j*(2*np.pi/Lambda)*(out1+out3+out5))     # 1st BS
#     v2 = np.exp(1j*(2*np.pi/Lambda)*(out2+out4+out6))     # 2nd BS
#     V1 = v1/abs(v1)
#     V2 = v2/abs(v2)
#     Q = np.array([V1,V2])
    
#     def compute_sinr(H, Q, num_ports, sigma_n):
#         sinr = np.zeros(num_ports)
#         for k in range(num_ports):
#             signal = np.abs(H[k,:].conj().T @ Q)**2  # Sinyal ke port p
#             interference = np.sum([np.abs(H[j, :].conj().T @ Q)**2 for j in range(num_ports) if j != k
#                                    ])
#             # interference = np.sum(np.abs(H @ Q)**2) - signal  # Interferensi dari port lain
#             sinr[k] = signal / (interference + sigma_n)  # Hitung SINR port p
        
#         best_port = np.argmax(sinr)
#         return sinr, best_port

#     sinr, best_port = compute_sinr(H, Q, num_ports, sigma_n)
#     # print(sinr)
    
#     sum_rate = np.log2(1 + sinr[best_port])

#     loss = -(sum_rate)
#     return loss

# los = loss(N_BS, ch_gen, H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6)
# %% gradient

def update_matrix_min(matrix_min, single_index, shift):
    m = len(matrix_min)     # Jumlah baris
    n = len(matrix_min[0])  # Jumlah kolom
    if 0 <= single_index < m * n:
        i = single_index // n  # Baris
        j = single_index % n   # Kolom
        value = matrix_min[i,j] - shift
        matrix_min[i,j] = value
        return matrix_min
    else:
        print("Error: Index out of range!")
        return matrix_min
    
def update_matrix_plus(matrix_plus, single_index, shift):
    m = len(matrix_plus)     # Jumlah baris
    n = len(matrix_plus[0])  # Jumlah kolom
    if 0 <= single_index < m * n:
        i = single_index // n  # Baris
        j = single_index % n   # Kolom
        value_plus = matrix_plus[i,j] + shift
        matrix_plus[i,j] = value_plus
        return matrix_plus
    else:
        print("Error: Index out of range!")
        return matrix_plus

def gradient(H, H_real, H_imag, params_U, w_index):  
    shift = np.pi/2
    
    params_U_copy_min = params_U.copy()
    params_U_copy_plus = params_U.copy()
    
    update_params_min = update_matrix_min(params_U_copy_min, w_index, shift)
    update_params_plus = update_matrix_plus(params_U_copy_plus, w_index, shift)
        
    loss_min,rate = LossRate(H, H_real, H_imag, update_params_min, bandwidth)
    
    loss_plus,rate = LossRate(H, H_real, H_imag, update_params_plus, bandwidth)
        
    # grad = (1/2*np.sin(shift)) * (loss_min-loss_plus)
    grad = (1/2*np.sin(shift)) * (loss_plus-loss_min)
    # grad = (loss_plus - loss_min) / 2
        
    return grad, loss_min, loss_plus

grad, loss_min, loss_plus = gradient(ch_gen, H_real, H_imag, params_U, 0) 

# %%

WL = 0.5
N_eps = 50
N_data = 2
learn_step = 2

w_1 = np.pi
w_2 = np.pi
w_3 = np.pi
w_4 = np.pi
w_5 = np.pi
w_6 = np.pi
params_U = np.array([[w_1],
                     [w_2],
                     [w_3],
                     [w_4],
                     [w_5],
                     [w_6]
                     ])

learn_step_init = learn_step
 
#Generate dataset channel
H_sample_real = []
H_sample_imag = []
h_ch = []   

loss_mean_array =[]
loss_min_array = []
loss_max_array = []
 
rate_mean_array = []
rate_min_array = []
rate_max_array = []
for i_eps in range(N_eps):
    
    for i_channel in range(N_data):
        ch_gen = channel_gen(k_nd_BS, d_BS, k_rmd_U, d_U)
        # ch_gen, H_real_s, H_imag_s = ch_simp(N_port, N_BS, WL) 

        inputs_og = np.reshape(ch_gen,(-1,1))
        # H_real = H_real_s.flatten()
        # H_imag = H_imag_s.flatten()    
        H_real = np.round(np.real(inputs_og),5).flatten()
        H_imag = np.round(np.imag(inputs_og),5).flatten()
        
        h_ch.append(ch_gen)
        H_sample_real.append(H_real)
        H_sample_imag.append(H_imag)
        
    loss_array =[]
    rate_array = []
    for i_data in range(N_data):
        
        for i_weight in range(params_U.size):
            
            row = i_weight // params_U.shape[1] 
            col = i_weight % params_U.shape[1]
            
            grad, loss_min, loss_plus = gradient(h_ch[i_data], H_sample_real[i_data], H_sample_imag[i_data], params_U, i_weight)
            # grad =0.5
            learn_step = learn_step_init / np.sqrt(i_eps+1)
            # learn_step = 0.5 * learn_step_init * (1+np.cos((np.pi*(i_eps+1))/N_eps))
            # learn_step = learn_step_init
            
            params_U[row, col] = params_U[row, col] - ((learn_step)*grad)
            # w = np.array(w[i_weight])
        
        loss_cal, rate_cal = LossRate(h_ch[i_data], H_sample_real[i_data], H_sample_imag[i_data], params_U, bandwidth)
        
        loss_array.append(loss_cal)
        rate_array.append(rate_cal)
    
    # w = w
    loss_mean_array.append(np.mean(loss_array))
    loss_min_array.append(np.min(loss_array))
    loss_max_array.append(np.max(loss_array))
    
    rate_mean_array.append(np.mean(rate_array))
    rate_min_array.append(np.min(rate_array))
    rate_max_array.append(np.max(rate_array))
    
    print("i_episode =",i_eps)
    print('optimized weight : ', np.array([params_U])) 
    print('gradient: ', grad)
    

print('Result - weight final: ', np.array([params_U]))  


# %% Plot trainig loss

# plt.plot(loss_mean_array, label="QNN $N_{data}$ ="+ str(N_data))
plt.plot(loss_mean_array, color='blue', label='Training Loss')
plt.fill_between(np.arange(N_eps), loss_max_array, loss_min_array, color='#ccccff')

# naming the x axis 
plt.xlabel('Training episode') 
# naming the y axis 
plt.ylabel('Loss') 


plt.grid(True)
plt.rc('grid', linestyle="dotted", color='grey')
plt.legend(loc='best')

# plt.savefig('training_loss_Non_plot3.svg', format='svg', dpi=1200, bbox_inches="tight")
plt.show()
# %%

plt.plot(rate_mean_array, color='blue', label='Rate')
plt.fill_between(np.arange(N_eps), rate_max_array, rate_min_array, color='blue', alpha=0.2)

# naming the x axis 
plt.xlabel('Episode') 
# naming the y axis 
plt.ylabel('Rate (bps/Hz)') 


plt.grid(True)
plt.rc('grid', linestyle="dotted", color='grey')
plt.legend(loc='lower right')

# plt.savefig('rate_Non_plot3.svg', format='svg', dpi=1200, bbox_inches="tight")
plt.show()
# %%
#
# Result - weight final Ndata=1 learnsteinit (3): [[1.52021903 4.64165236 2.97436532 2.9345749  3.82982091 2.70906777]]
# Result - weight final Ndata=10 learnsteinit (2): [[1.54010869 4.84064471 3.12875271 3.30091615 3.12799423 3.1104534 ]]
# Result - weight final  Ndata=20 learnsteinit (2):  [[1.6614066  4.70851437 2.96296494 3.09708533 3.16486484 3.25067301]]
# Result - weight final Ndata=40 learnsteinit (1): [[1.59559124 4.63071261 3.24797102 3.09708461 3.05688972 3.07325515]]  