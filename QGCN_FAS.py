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

def Q_encode(H, H_real, H_imag, w_1, w_2, w_3, V0, V1):
    
    q1 = QuantumRegister(H.size, 'q1')
    qc1 = QuantumCircuit(q1)
    
    q2 = QuantumRegister(4, 'q2')
    c2 = ClassicalRegister(3, 'c2')
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
    
    
    # def quantum_pooling_layer(final_qc, control_qubit, target_qubits, classical_bits, V0_params, V1_params):
    #     """
    #     Menerapkan pooling layer ke quantum circuit.

    #     Args:
    #         - qc             : QuantumCircuit object
    #         - control_qubit  : Qubit yang akan diukur (dan collapse)
    #         - target_qubit   : Qubit yang menerima hasil pooling
    #         - classical_bit  : ClassicalRegister index untuk menyimpan hasil pengukuran
    #         - V0_params      : Tuple (rz1, ry, rz2) untuk unitary V0 jika hasil ukur 0
    #         - V1_params      : Tuple (rz1, ry, rz2) untuk unitary V1 jika hasil ukur 1
    #         """
    #     final_qc.measure(control_qubit, classical_bits)
        
    #     with final_qc.switch(classical_bits) as case:
    #         with case(0):  # hasil ukur 0 → V0
    #             final_qc.rz(V0[0], target_qubits)
    #             final_qc.ry(V0[1], target_qubits)
    #             final_qc.rz(V0[2], target_qubits)

    #         with case(1):  # hasil ukur 1 → V1
    #             final_qc.rz(V1[0], target_qubits)
    #             final_qc.ry(V1[1], target_qubits)
    #             final_qc.rz(V1[2], target_qubits)

        # final_qc.rz(V0_params[0], target_qubits).c_if((classical_bits, 0))    
        # final_qc.ry(V0_params[1], target_qubits).if_test((classical_bits, 0))
        # final_qc.rz(V0_params[2], target_qubits).if_test((classical_bits, 0))
        
        # final_qc.rz(V1_params[0], target_qubits).if_test((classical_bits, 1))
        # final_qc.ry(V1_params[1], target_qubits).if_test((classical_bits, 1))
        # final_qc.rz(V1_params[2], target_qubits).if_test((classical_bits, 1))
        
    final_qc.swap(q2[2], q2[1])
    # quantum_pooling_layer(final_qc, q2[0], q2[1], c2[0], V0_params=V0, V1_params=V1)     
    final_qc.measure(q2[0], c2[0])
    final_qc.swap(q2[2], q2[1])
    final_qc.measure(q2[1], c2[1])
    final_qc.swap(q2[3], q2[2])
    final_qc.measure(q2[2], c2[2])
    
    return final_qc
w_1, w_2, w_3 = np.pi/6, np.pi/5, np.pi/4
V0 = (np.pi/4, np.pi/3, np.pi/2)
V1 = (np.pi/6, np.pi/4, np.pi/3)
qc2 = Q_encode(H, H_real, H_imag, w_1, w_2, w_3, V0, V1)


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
    simp_counts_02 = marginal_counts(counts_sam, indices=[1])
    simp_counts_03 = marginal_counts(counts_sam, indices=[2])
       # counts_sam = result_sam[0].data.c.get_counts()
       
    out1 = ave_meas(simp_counts_01)
    out2 = ave_meas(simp_counts_02)
    out3 = ave_meas(simp_counts_03)
       
    out = [out1, out2, out3]
    return qc, out, out1, out2, out3

w_1, w_2, w_3 = np.pi/6, np.pi/5, np.pi/4
V0 = (np.pi/4, np.pi/3, np.pi/2)
V1 = (np.pi/6, np.pi/4, np.pi/3)
qc, out, out1, out2, out3 = Q_decode(H, H_real, H_imag, w_1, w_2, w_3, shots=1024)

print("measurement_average_01 =",out[0])
print("measurement_average_02 =",out[1])
print("measurement_average_03 =",out[2])


# %%
#
num_ports=3
ptx = 5
sigma_n = 1
    
def loss(N_BS, ch_gen, H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6):
    
    qc, out, out1 = Q_decode(H, H_real, H_imag, w_1, w_2, w_3, shots=1024)
    
    # v1 = np.exp(1j*(2*np.pi/Lambda)*(out1))     # 1st BS
    # v2 = np.exp(1j*(2*np.pi/Lambda)*(out2))     # 2nd BS
    # v1 = np.exp(1j*2*np.pi*(out1))     # 1st BS
    # v2 = np.exp(1j*2*np.pi*(out2))     # 2nd BS
    v1 = np.exp(1j*(out1+(0)))     # 1st BS
    V1 = v1/abs(v1)
    V2 = v2/abs(v2)
    Q = np.array([V1,V2])
    
    
    sinr1 = np.abs(H[0,:] @ Q)**2
    sinr2 = np.abs(H[1,:] @ Q)**2
    sinr3 = np.abs(H[2,:] @ Q)**2
    
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
    return loss

los = loss(N_BS, ch_gen, H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6)