# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 13:54:54 2025

@author: orecy
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit, ParameterVector
from qiskit import transpile
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.result import marginal_counts
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

def Q_encode(H, H_real, H_imag, w_1, w_2, w_3):
    
    q1 = QuantumRegister(H.size, 'q1')
    qc1 = QuantumCircuit(q1)
    
    q2 = QuantumRegister(4, 'q2')
    c2 = ClassicalRegister(1, 'c2')
    qc2 = QuantumCircuit(q2,c2)
    
    final_qc = QuantumCircuit(q1, q2, c2)  # Deklarasi semua register sekaligus
    final_qc = final_qc.compose(qc1)       # Tambahkan qc1
    final_qc = final_qc.compose(qc2)       # Tambahkan qc2
    
    # quantum state preparation
    for i_re in range(H.size): 
        final_qc.ry(H_real[i_re], q1[i_re])
        
    for i_im in range(H.size):
        final_qc.rz(H_imag[i_im], q1[i_im])
        
    final_qc.barrier()
    
    def create_u_gate(label,w_i):
        u = QuantumCircuit(2, name=f'U{label}')
        u.cx(1,0)
        u.rz(np.pi/4, 0)
        u.ry(w_i, 1)
        u.cx(0, 1)
        u.rz(np.pi/4, 1)
        u.cx(1,0)
        return u.to_gate()
    
    u1 = create_u_gate(1, w_1).control(1)  #control(1) menambah 1 control di Ugate,
    final_qc.append(u1, [q1[0], q2[0], q2[1]])
    final_qc.swap(q2[0], q2[1])
    
    u2 = create_u_gate(2, w_2).control(1)
    final_qc.append(u2, [q1[1], q2[1], q2[2]])
    final_qc.swap(q2[1], q2[2])
    
    u3 = create_u_gate(3, w_3).control(1)
    final_qc.append(u3, [q1[2], q2[2], q2[3]])
    
    final_qc.barrier()
    
    final_qc.cx(q2[2],q2[3])
    final_qc.measure(q2[3], c2[0])
    
    return final_qc
w_1 = 0
w_2 = 0
w_3 = 0
qc2 = Q_encode(H, H_real, H_imag, w_1, w_2, w_3)


# def build_qgcn(H, A, num_layers=1):
#     num_nodes = len(A)
#     q = QuantumRegister(num_nodes, 'q')
#     c = ClassicalRegister(3, 'c')
#     qc = QuantumCircuit(q,c)
    
#     # Normalize amplitudes
#     h_abs = np.abs(H)
#     h_max = np.max(h_abs)
    
#     # Encode channel information
#     # for port in range(3):
#     #     for ant in range(2):
#     #         # Amplitude encoding (safe normalization)
#     #         angle_amp = (h_abs[port, ant] / h_max) * np.pi
#     #         qc.ry(angle_amp, ant)
            

#     #         # Phase encoding
#     #         angle_phase = np.angle(H[port, ant])
#     #         qc.rz(angle_phase, ant)
#     #         qc.barrier()
#     # qc.barrier()
    
    
#     for port in range(3):
#        # Gabungkan informasi kedua antenna
#        combined_angle = np.arctan(np.sum(h_abs[port,:])/h_max * np.pi/2)
#        qc.ry(combined_angle, port+2)  # Qubit 2-4 adalah port
       
#        # Encoding fase
#        mean_phase = np.mean(np.angle(H[port,:]))
#        qc.rz(mean_phase, port+2)
#        qc.barrier()
    
#     # Parameterized layers
#     theta = ParameterVector('θ', length=num_layers*(2*num_nodes + 1))
    
#     for layer in range(num_layers):
#         # Node feature transformation
#         for i in range(num_nodes):
#             qc.ry(theta[layer*num_nodes + i], i)
#             qc.rz(theta[layer*num_nodes + num_nodes + i], i)
#         qc.barrier()
#         # Graph convolution
#         for i in range(num_nodes):
#             for j in range(i+1, num_nodes):
#                 if A[i,j] == 1:
#                     qc.cz(i, j)
#                     qc.cry(theta[-1], i, j)  # Shared parameter
#                     qc.barrier()
                    
#     qc.measure(q[2], c[0])
#     qc.measure(q[3], c[1])
#     qc.measure(q[4], c[2])
#     return qc,theta

# circut,theta = build_qgcn(H, A, num_layers=1)

# %%
def ave_meas(count):
     total = count.get('0', 0) + count.get('1', 0)
     return count.get('1', 0) / total if total > 0 else 0
 
def Q_decode(H, H_real, H_imag, w_1, w_2, w_3, shots):
    
    qc = Q_encode(H, H_real, H_imag, w_1, w_2, w_3)
    sampler = StatevectorSampler() 
    
    job = sampler.run( [(qc)], shots=shots)
    result = job.result()
    counts_sam = result[0].data.c2.get_counts()
    
    simp_counts_01 = marginal_counts(counts_sam, indices=[0])

    
    out1 = ave_meas(simp_counts_01)
    
    out = [out1]
    return qc, out, out1

w_1 = np.pi
w_2 = np.pi
w_3 = np.pi
qc, out, out1 = Q_decode(H, H_real, H_imag, w_1, w_2, w_3, shots=1024)

print("measurement_average_01 =",out[0])


# %%
#

def train_qgcn(H, A, shots, epochs=100, lr=0.01):
    qc, theta = build_qgcn(H, A, num_layers=1)
    sampler = StatevectorSampler()
    param_values = np.random.rand(len(theta))
    qc_assigned = qc.assign_parameters({theta: param_values})
    
    job_sam = sampler.run( [(qc_assigned)], shots = shots)
    result_sam = job_sam.result()
    counts_sam = result_sam[0].data.c.get_counts()
    
    param_values = np.random.rand(len(theta)) * 2*np.pi
    history = []
    
    for epoch in range(epochs):
        # Gradient calculation
        gradients = np.zeros(len(theta))
        for i in range(len(theta)):
            # Plus shift
            params_plus = param_values.copy()
            params_plus[i] += np.pi/2
            qc_plus = qc.bind_parameters({theta: params_plus})
            state_plus = execute(qc_plus, backend).result().get_statevector()
            V_plus = state_plus[:2]
            rate_plus = np.sum([np.abs(H[k] @ V_plus)**2 for k in range(3)])
            
            # Minus shift
            params_minus = param_values.copy()
            params_minus[i] -= np.pi/2
            qc_minus = qc.bind_parameters({theta: params_minus})
            state_minus = execute(qc_minus, backend).result().get_statevector()
            V_minus = state_minus[:2]
            rate_minus = np.sum([np.abs(H[k] @ V_minus)**2 for k in range(3)])
            
            gradients[i] = (rate_plus - rate_minus) / 2
        
        # Update parameters
        param_values -= lr * gradients
        
        # Track progress
        current_state = execute(qc.bind_parameters({theta: param_values}), backend).result().get_statevector()
        sum_rate = np.sum([np.abs(H[k] @ current_state[:2])**2 for k in range(3)])
        history.append(sum_rate)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Sum Rate = {sum_rate:.4f}")
    
    return param_values, history

# # ==============================
# # 4. Run Training and Visualization
# # ==============================
optimal_params, history = train_qgcn(H, A)

# # Plot training progress
# plt.plot(history)
# plt.xlabel('Epoch')
# plt.ylabel('Sum Rate')
# plt.title('QGCN Training Progress')
# plt.show()

# # ==============================
# # 5. Result Analysis
# # ==============================
# qc_final = build_qgcn(H, A)[0].bind_parameters({theta: optimal_params})
# state_final = execute(qc_final, Aer.get_backend('statevector_simulator')).result().get_statevector()
# V_optimal = state_final[:2]
# rates = [np.abs(H[k] @ V_optimal)**2 for k in range(3)]
# best_port = np.argmax(rates)

# print("\n=== Final Results ===")
# print(f"Optimal Precoding Vector: {V_optimal}")
# print(f"Port Rates: {np.round(rates, 4)}")
# print(f"Best Port: Port {best_port + 1} (Rate: {rates[best_port]:.4f})")