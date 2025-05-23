# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 13:54:54 2025

@author: orecy
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit, ParameterVector
from qiskit import transpile
from qiskit.primitives import StatevectorSampler, StatevectorEstimator, Sampler
from qiskit.result import marginal_counts
from qiskit.visualization import plot_histogram
from qiskit.circuit import Gate
from math import pi
import scipy
from scipy.special import jv
from numpy import linalg as LA
import matplotlib.pyplot as plt

# %%

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
# # %%

# N_port = 3
# N_BS = 2
# WL = 0.5
# N_data = 100

# #Generate dataset channel
# H_sample_real = []
# H_sample_imag = []
# h_ch = []

# for i_channel in range(N_data):
#     # ch_gen = channel_gen(k_nd_BS, d_BS, k_rmd_U, d_U)
#     ch_gen, H_real_s, H_imag_s = ch_simp(N_port, N_BS, WL) 


#     inputs_og = np.reshape(ch_gen,(-1,1))
#     inputs = np.round(inputs_og, 5)
#     H_real = H_real_s.flatten()
#     H_imag = H_imag_s.flatten()    
    
#     h_ch.append(ch_gen)
#     H_sample_real.append(H_real)
#     H_sample_imag.append(H_imag)
    
# # Hitung channel gain untuk setiap sampel (||H_i||^2)
# channel_gains = np.array([
#     [np.linalg.norm(H[i])**2 for i in range(H.shape[0])]
#     for H in h_ch
# ])  # Hasil: (3 sampel × 3 ports)

# adjacency_matrix = np.corrcoef(channel_gains.T)

# # Pastikan diagonal bernilai 0 (tidak ada self-loop)
# a = np.fill_diagonal(adjacency_matrix, 0)


# %%

# ==============================
# 1. Data Preparation
# ==============================
# Channel matrix H (3 ports x 2 antennas)
# H = np.array([
#     [0.8+0.1j, 0.2-0.3j],  # Port 1
#     [0.3+0.4j, 0.7-0.2j],  # Port 2
#     [0.1+0.5j, 0.4-0.1j]   # Port 3
# ])
H = np.array([
    [0.8+0.1j],  # Port 1
    [0.3+0.4j],  # Port 2
    [0.1+0.5j]   # Port 3
])
H_real = np.real(H).flatten()
H_imag = np.imag(H).flatten()
# Adjacency matrix (nodes 0-1: antennas, 2-4: ports)
A = np.array([
    [0, 0, 1, 1, 1],  # Antenna 0
    [0, 0, 1, 1, 1],  # Antenna 1
    [1, 1, 0, 0, 0],  # Port 1
    [1, 1, 0, 0, 0],  # Port 2
    [1, 1, 0, 0, 0]   # Port 3
])


# %% quantum circuit

def Q_encode(H, H_real, H_imag, w_1, w_2, w_3, V0, V1):
    
    q1 = QuantumRegister(H.size, 'q1')
    q2 = QuantumRegister(4, 'q2')
    c2 = ClassicalRegister(1, 'c2')
    qc = QuantumCircuit(q1, q2, c2)  # Deklarasi semua register sekaligus
    
    
    # quantum state preparation
    for i_re in range(H.size): 
        qc.ry(H_real[i_re], q1[i_re])
        
    for i_im in range(H.size):
        qc.rz(H_imag[i_im], q1[i_im])
        
    qc.barrier()
    
    def create_u_gate(label,w_i):
        u = QuantumCircuit(2, name=f'U{label}')
        # u.h(0)
        u.cx(1,0)
        u.rz(np.pi/4, 0)
        u.ry(w_i, 1)
        u.cx(0, 1)
        u.rz(np.pi/4, 1)
        u.cx(1,0)
        return u.to_gate()
    
    u1 = create_u_gate(1, w_1).control(1)  #control(1) menambah 1 control di Ugate,
    qc.append(u1, [q1[0], q2[0], q2[1]])
    qc.swap(q2[0], q2[1])
    
    u2 = create_u_gate(2, w_2).control(1)
    qc.append(u2, [q1[1], q2[1], q2[2]])
    qc.swap(q2[1], q2[2])
    
    u3 = create_u_gate(3, w_3).control(1)
    qc.append(u3, [q1[2], q2[2], q2[3]])
    
    qc.barrier()
    
    def apply_controlled_V(qc, control, target, V):
        # qc.h(control)
        qc.crz(V[0], control, target)
        qc.cry(V[1], control, target)
        qc.crz(V[2], control, target)

    def quantum_pooling_layer(qc, control_qubit, target_qubit, V0, V1):
        # Apply V1 if control == 1
        qc.h(control_qubit)
        apply_controlled_V(qc, control_qubit, target_qubit, V1)

        # Apply V0 if control == 0
        qc.x(control_qubit)
        apply_controlled_V(qc, control_qubit, target_qubit, V0)
        qc.x(control_qubit)  # Reset
        
        qc.barrier()


    # qc.swap(q2[2], q2[1])
    # quantum_pooling_layer(qc, q2[0], q2[1],  V0, V1)
    # qc.swap(q2[2], q2[1])
    # quantum_pooling_layer(qc, q2[1], q2[2],  V0, V1)
    # qc.swap(q2[3], q2[2])
    # quantum_pooling_layer(qc, q2[2], q2[3],  V0, V1)
    
    qc.swap(q2[2], q2[1])
    qc.h(q2[0])
    # qc.barrier()
    qc.swap(q2[2], q2[1])
    qc.cx(q2[1], q2[0])
    # qc.barrier()
    qc.swap(q2[3], q2[2])
    qc.cx(q2[2], q2[0])
    

    # qc.h(q2[0])  # Superposition untuk out1, out3, out5
    # qc.cx(q2[1], q2[0])
    # qc.cx(q2[2], q2[0])
    
    qc.barrier()
    qc.measure(q2[0], c2[0])    

    
    return qc
w_1, w_2, w_3 = np.pi/2, np.pi/2, np.pi/2
V0 = (w_1, w_2, w_3)
V1 = (w_1, w_2, w_3)
qc2 = Q_encode(H, H_real, H_imag, w_1, w_2, w_3, V0, V1)

# %%
def ave_meas(count):
     total = count.get('0', 0) + count.get('1', 0)
     return count.get('1', 0) / total if total > 0 else 0
 
def Q_decode(H, H_real, H_imag, w_1, w_2, w_3, shots):
    
    qc = Q_encode(H, H_real, H_imag, w_1, w_2, w_3, V0, V1)
    sampler = StatevectorSampler()
    
    job = sampler.run( [(qc)], shots=shots)
    result = job.result()
    counts_sam = result[0].data.c2.get_counts()
    
    simp_counts_01 = marginal_counts(counts_sam, indices=[0])
    # simp_counts_02 = marginal_counts(counts_sam, indices=[1])
    # simp_counts_03 = marginal_counts(counts_sam, indices=[2])
       # counts_sam = result_sam[0].data.c.get_counts()
       
    out1 = ave_meas(simp_counts_01)
    # out2 = ave_meas(simp_counts_02)
    # out3 = ave_meas(simp_counts_03)
       
    out = [out1]
    return counts_sam, qc, out, out1

w_1, w_2, w_3 = np.pi, np.pi, np.pi
V0 = (w_1, w_2, w_3)
V1 = (w_1, w_2, w_3)
count_sam, qc, out, out1 = Q_decode(H, H_real, H_imag, w_1, w_2, w_3, shots=1024)

print("measurement_average_01 =",out[0])
# print("measurement_average_02 =",out[1])
# print("measurement_average_03 =",out[2])

aaaaa = plot_histogram(count_sam, sort='value_desc')
# %%
#
num_ports=3
ptx = 5
sigma_n = 1
    
def loss(H, H_real, H_imag, w_1, w_2, w_3):
    
    count_sam, qc, out, out1 = Q_decode(H, H_real, H_imag, w_1, w_2, w_3, shots=1024)
    
    v1 = np.exp(1j*(out1))     # 1st BS
    V1 = v1/abs(v1)

    Q = np.array([V1])
    
    
    sinr1 = np.abs(H[0,:] @ Q)**2
    sinr2 = np.abs(H[1,:] @ Q)**2
    sinr3 = np.abs(H[2,:] @ Q)**2
    
    sinr_p1 = sinr1 / (sinr2+sinr3+sigma_n)
    sinr_p2 = sinr2 / (sinr1+sinr3+sigma_n)
    sinr_p3 = sinr3 / (sinr1+sinr2+sigma_n)
    
    sinr_all = np.array([sinr_p1,sinr_p2,sinr_p3])
    best_port = np.argmax(sinr_all)
    sum_rate = np.log2(1 + sinr_all[best_port])
    
    # indices = np.argsort(sinr_all)[-2:]
    # best_sinr = sinr_all[indices]
    
    # sum_rate1 = np.log2(1 + best_sinr[0])
    # sum_rate = sum_rate     

    loss = -1*(sum_rate)
    return loss

los = loss(H, H_real, H_imag, w_1, w_2, w_3)
# %%

def gradient(H, H_real, H_imag, w_1, w_2, w_3, w_index):
        
    shift = np.pi/2
        
    w = np.array([w_1, w_2, w_3])
        
    w_min = w
    w_plus = w
    
    w_min[w_index] = w_min[w_index] - shift
    loss_min = loss(H, H_real, H_imag, w_min[0], w_min[1], w_min[2])
        
    w_plus[w_index] = w_plus[w_index] + shift
    loss_plus = loss(H, H_real, H_imag, w_plus[0], w_plus[1], w_plus[2])
        
    # grad = (1/2*np.sin(shift)) * (loss_min-loss_plus)
    # grad = (1/2*np.sin(shift)) * (loss_plus-loss_min)
    grad = (loss_plus - loss_min) / 2
        
    return grad, loss_min, loss_plus

w_1 = np.pi
w_2 = np.pi
w_3 = np.pi

grad_1, loss_min, loss_plus = gradient(H, H_real, H_imag, w_1, w_2, w_3, 1)  

# %%

N_port = 3
N_BS = 1
WL = 0.5

def ch_simp(N_port, N_BS, WL):
    # H_sample_real = []
    # H_sample_imag = []

    for i_BS in range(N_BS):
        x = np.sqrt(1/2)*np.random.randn(N_port,N_BS)
        y = np.sqrt(1/2)*np.random.randn(N_port,N_BS)
        h_ch = np.zeros((N_port,N_BS), dtype=complex)
        mu = np.zeros(N_port)
        h_ch[0,:] = x[0,:] + 1j*y[0,:]     #reference port
        mu[0] = 1                       #reference port (mu is spatial correlation between port using bessel func)
        for i_port in range(1,N_port):
            mu[i_port] = jv(0, 2*np.pi * (abs((i_port+1)-1)/(N_port-1)) * WL)
            h_ch[i_port,:] = (np.sqrt(1-mu[i_port]**2) * x[i_port,:] + mu[i_port] * x[0,:] + 
                              1j*(np.sqrt(1-mu[i_port]**2) * y[i_port,:] + mu[i_port] * y[0,:]))          
            
            #inputs = np.round(h_ch[:,i_sample], 5)
        H_real = np.round(np.real(h_ch),5)
        H_imag = np.round(np.imag(h_ch),5)
    return h_ch, H_real, H_imag

ch_gen, H_real, H_imag = ch_simp(N_port, N_BS, WL)
h_ch = np.reshape(ch_gen,(-1,1))
H_real = H_real.flatten()
H_imag = H_imag.flatten()
# %%
#
WL = 0.5
N_eps = 50
N_data = 1
learn_step = 0.1
w_1 = np.pi
w_2 = np.pi
w_3 = np.pi

w = np.array([w_1, w_2, w_3])

learn_step_init = learn_step

#Generate dataset channel
H_sample_real = []
H_sample_imag = []
h_ch = []

for i_channel in range(N_data):
    # ch_gen = channel_gen(k_nd_BS, d_BS, k_rmd_U, d_U)
    ch_gen, H_real_s, H_imag_s = ch_simp(N_port, N_BS, WL) 


    inputs_og = np.reshape(ch_gen,(-1,1))
    H_real = H_real_s.flatten()
    H_imag = H_imag_s.flatten()    
    
    h_ch.append(ch_gen)
    H_sample_real.append(H_real)
    H_sample_imag.append(H_imag)
    

loss_mean_array =[]
loss_min_array = []
loss_max_array = [] 
for i_eps in range(N_eps):
    loss_array =[]
    for i_data in range(N_data):
        
        for i_weight in range(len(w)):
            
            grad, loss_min, loss_plus = gradient(h_ch[i_data], H_sample_real[i_data], H_sample_imag[i_data], w[0], w[1], w[2], i_weight)
            
            learn_step = learn_step_init / np.sqrt(i_eps+1)
            # learn_step = 0.5 * learn_step_init * (1+np.cos((np.pi*(i_eps+1))/N_eps))
            # learn_step = learn_step_init
            
            w[i_weight] = w[i_weight] - ((learn_step)*grad)
            # w = np.array(w[i_weight])
        
        loss_cal = loss(h_ch[i_data], H_sample_real[i_data], H_sample_imag[i_data], w[0], w[1], w[2])
        
        loss_array.append(loss_cal)
    
    # w = w
    loss_mean_array.append(np.mean(loss_array))
    loss_min_array.append(np.min(loss_array))
    loss_max_array.append(np.max(loss_array))
    
    print("i_episode =",i_eps)
    print('optimized weight : ', np.array([w])) 
    print('gradient: ', grad)
    

print('Result - weight final: ', np.array([w]))  



plt.plot(loss_mean_array, label="QNN $N_{data}$ ="+ str(N_data))
# plt.fill_between(np.arange(N_eps), loss_max_array, loss_min_array, color='grey', alpha=0.5)

# naming the x axis 
plt.xlabel('Training episode') 
# naming the y axis 
plt.ylabel('Training loss') 


plt.grid(True)
plt.rc('grid', linestyle="dotted", color='grey')
plt.legend(loc='best')
plt.show()
