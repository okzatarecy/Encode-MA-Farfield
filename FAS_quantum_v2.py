# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 18:25:56 2025

@author: orecy
"""

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

# %% generate channel

def channel_gen(k_nd_BS, d_BS, k_rmd_U, d_U):
    angle1 = 80
    angle2 =180-angle1
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
        
    qc1.barrier()
    
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

    qc1.barrier()
        
    qc1.ry(w_1, q[0])
    qc1.ry(w_2, q[1])
    qc1.ry(w_3, q[2])
    qc1.ry(w_4, q[3])
    qc1.ry(w_5, q[4])
    qc1.ry(w_6, q[5])
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
        
    simp_counts_01 = marginal_counts(counts_sam, indices=[0])
    simp_counts_02 = marginal_counts(counts_sam, indices=[1])
    simp_counts_03 = marginal_counts(counts_sam, indices=[2])
    simp_counts_04 = marginal_counts(counts_sam, indices=[3])
    simp_counts_05 = marginal_counts(counts_sam, indices=[4])
    simp_counts_06 = marginal_counts(counts_sam, indices=[5])
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
# plot_histogram(counts_sam, sort='value_desc')

# %% loss function

# def loss(N_BS, ch_gen, H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6):
#     qc1, counts_sam,out, out1, out2, out3, out4, out5, out6 = Q_sampler_est(H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6, shots=1024)
#     ptx = 5
#     sigma_n = 1
    
#     v1 = np.exp(1j*(out1+out2))     # 1st port
#     v2 = np.exp(1j*(out3+out4))     # 2nd port
#     v3 = np.exp(1j*(out5+out6))     # 3rd port
    
#     Q = np.array([v1,v2,v3])
#     V_max = Q[np.argmax(np.abs(Q))]  #looking for biggest magnitude of complex number
#     V_norm = V_max/abs(V_max)
    
#     max_index = np.argmax(np.abs(Q))
#     snr = ptx*np.abs(ch_gen[max_index] * V_norm)**2/sigma_n
#     rate_BS1 = np.log2(1+snr[0])
#     rate_BS2 = np.log2(1+snr[1])
#     sum_rate = rate_BS1+rate_BS2
#     # print(cap_P2)
#     # print(cap_P1+cap_P2)
#     loss = -(sum_rate)
#     return loss

# los = loss(N_BS, ch_gen, H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6)

# %%
# ww = np.random.randn(3, 2) + 1j * np.random.randn(3, 2)
num_ports=3
H = ch_gen
ptx = 5
sigma_n = 1
    
def loss(N_BS, ch_gen, H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6):
    
    qc1, counts_sam,out, out1, out2, out3, out4, out5, out6 = Q_sampler_est(H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6, shots=1024)
    
    v1 = np.exp(1j*(out1+out3+out5))     # 1st port
    v2 = np.exp(1j*(out2+out4+out6))     # 2nd port
    V1 = v1/abs(v1)
    V2 = v2/abs(v2)
    Q = np.array([V1,V2])
    
    def compute_sinr(H, Q, sigma_n):
        signal = np.abs(H.conj().T * Q)**2  # Sinyal ke port p
        interference = np.sum(np.abs(H * Q)**2) - signal  # Interferensi dari port lain
        sinr = signal / (interference + sigma_n)  # Hitung SINR port p
        return sinr

    sinr = compute_sinr(H, Q, sigma_n)

    def compute_sum_rate(sinr, selected_port):
        return np.log2(1 + sinr[selected_port])
    

    Q = np.array([v1,v2,v3])
    V_max = Q[np.argmax(np.abs(Q))]  #looking for biggest magnitude of complex number
    V_norm = V_max/abs(V_max)
    
    max_index = np.argmax(np.abs(Q))
    snr = ptx*np.abs(ch_gen[max_index] * V_norm)**2/sigma_n
    rate_BS1 = np.log2(1+snr[0])
    rate_BS2 = np.log2(1+snr[1])
    sum_rate = rate_BS1+rate_BS2
    # print(cap_P2)
    # print(cap_P1+cap_P2)
    loss = -(sum_rate)
    return loss

los = loss(N_BS, ch_gen, H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6)
# %%

def gradient(N_BS, ch_gen, H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6, w_index):
        
    shift = np.pi/2
        
    w = np.array([w_1, w_2, w_3, w_4, w_5, w_6])
        
    w_min = w
    w_plus = w
    
    w_min[w_index] = w_min[w_index] - shift
    loss_min = loss(N_BS, ch_gen, H_real, H_imag, w_min[0], w_min[1], w_min[2], w_min[3], w_min[4], w_min[5])
        
    w_plus[w_index] = w_plus[w_index] + shift
    loss_plus = loss(N_BS, ch_gen, H_real, H_imag, w_plus[0], w_plus[1], w_plus[2], w_plus[3], w_plus[4], w_plus[5])
        
    grad = (1/2*np.sin(shift)) * (loss_min-loss_plus)
        
    return grad, loss_min, loss_plus

w_1 = np.pi
w_2 = np.pi
w_3 = np.pi
w_4 = np.pi
w_5 = np.pi
w_6 = np.pi

grad_1, loss_min, loss_plus = gradient(N_BS, ch_gen, H_real, H_imag, w_1, w_2, w_3, w_4, w_5, w_6, 1)  

# %%


    
    # ch_gen = channel_gen(k_nd_BS, d_BS, k_rmd_U, d_U)
    # h_ch = np.reshape(ch_gen,(-1,1))
    # H_real = np.round(np.real(h_ch),5).flatten()
    # H_imag = np.round(np.imag(h_ch),5).flatten()

# %%

WL = 0.5
N_eps = 20
N_data = 50
learn_step = 0.1
w_1 = np.pi
w_2 = np.pi
w_3 = np.pi
w_4 = np.pi
w_5 = np.pi
w_6 = np.pi

w = np.array([w_1, w_2, w_3, w_4, w_5, w_6])

learn_step_init = learn_step

#Generate dataset channel
H_sample_real = []
H_sample_imag = []
h_ch = []

for i_channel in range(N_data):
    ch_gen = channel_gen(k_nd_BS, d_BS, k_rmd_U, d_U)

    inputs_og = np.reshape(ch_gen,(-1,1))
    inputs = np.round(inputs_og, 5)
    H_real = np.real(inputs).flatten()
    H_imag = np.imag(inputs).flatten()    
    
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
            
            grad, loss_min, loss_plus = gradient(N_BS, h_ch[i_data], H_sample_real[i_data], H_sample_imag[i_data], w[0], w[1], w[2], w[3], w[4], w[5], i_weight)
            
            learn_step = learn_step_init / np.sqrt(i_eps+1)
            # learn_step = 0.5 * learn_step_init * (1+np.cos((np.pi*(i_eps+1))/N_eps))
            
            w[i_weight] = w[i_weight] - ((learn_step)*grad)
            # w = np.array(w[i_weight])
        
        loss_cal = loss(N_BS, h_ch[i_data], H_sample_real[i_data], H_sample_imag[i_data], w[0], w[1], w[2], w[3], w[4], w[5])
        
        loss_array.append(loss_cal)
    
    # w = w
    loss_mean_array.append(np.mean(loss_array))
    loss_min_array.append(np.min(loss_array))
    loss_max_array.append(np.max(loss_array))
    
    print("i_episode =",i_eps)
    print('optimized weight : ', np.array([w])) 
    print('gradient: ', grad)
    

print('Result - weight final: ', np.array([w]))  

    
# plt.plot(loss_mean_array, label='QNN Loss, $N_{data}=50$')
# plt.grid(True)
# plt.title('QNN Loss')
# plt.xlabel('Episode')
# plt.ylabel('Loss')

# plt.legend(loc='best')
# plt.show()

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
    
   
