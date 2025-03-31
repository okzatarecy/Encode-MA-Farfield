# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 19:04:15 2025

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
from numpy import linalg as LA
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

# def channel_gen(k_nd_BS, d_BS, k_rmd_U, d_U):
#     angle1 = 80
#     angle2 = 80
#     scatters_coordinate_t = np.array([[1.7, 1.5], [0.3,-1.2]]); # scatters coordinate form the origin O_t (x, y)
#     l_t = np.sqrt(scatters_coordinate_t[:, 0]**2 + scatters_coordinate_t[:,1]**2 \
#                   - (2 * scatters_coordinate_t[:,0] * scatters_coordinate_t[:,1] \
#                      * np.cos(angle1*np.pi/180))); # distance from scatter to the origin O_t
#     theta_t = np.arccos(scatters_coordinate_t[:, 0] / l_t) # elevation from origin to scatters (elevation transmit path)

#     k_n = k_nd_BS / d_BS;
#     rho_scatter1 = -k_nd_BS * np.sin(theta_t[0]) - ((k_n**2 * d_BS**2 * np.sin(theta_t[0])**2) / 2*l_t[0])
#     rho_scatter2 = -k_nd_BS * np.sin(theta_t[1]) - ((k_n**2 * d_BS**2 * np.sin(theta_t[1])**2) / 2*l_t[1])
#     rho_t = np.array([rho_scatter1, rho_scatter2])


#     signal_phase_different_tx = (2*np.pi*rho_t/Lambda);
#     a = np.exp(1j*(2*np.pi/Lambda)*rho_t[:,0]) # transmit field response vector
#     A = np.exp(1j*(2*np.pi/Lambda)*rho_t)   # The field response vectors of all the N transmit antennas

#     scatters_coordinate_r = np.array([[-0.3, 1.5], [-1.7, -1.2]]); # scatters coordinate form the origin O_r (x, y)
#     l_r = np.sqrt(scatters_coordinate_r[:, 0]**2 + scatters_coordinate_r[:,1]**2 \
#                   - (2 * scatters_coordinate_r[:,0] * scatters_coordinate_r[:,1] \
#                      * np.cos(angle2*np.pi/180)))
#     theta_r = np.arccos(scatters_coordinate_r[:, 0] / l_r) # elevation from origin to scatters (elevation receiver path)

#     k_rm = k_rmd_U / d_U;
#     rho_scatter1_r = -k_rmd_U * np.sin(theta_r[0]) - ((k_rm**2 * d_U**2 * np.sin(theta_r[0])**2) / 2*l_r[0])
#     rho_scatter2_r = -k_rmd_U * np.sin(theta_r[1]) - ((k_rm**2 * d_U**2 * np.sin(theta_r[1])**2) / 2*l_r[1])
#     rho_r = np.array([rho_scatter1_r, rho_scatter2_r]);

#     b = np.exp(1j*(2*np.pi/Lambda)*rho_r[:,0]) # Receive field response vector
#     B = np.exp(1j*(2*np.pi/Lambda)*rho_r)   # The field response vectors of all the m_o receive antennas
#     B_Hermition = B.conj().T

#     O = np.random.normal(loc=0, scale=1, size=(2, 2)) + 1j*np.random.normal(loc=0, scale=1, size=(2, 2)) # (i.i.d.) Gaussian random variable with zero mean and variance α2
#     ch_gen = B_Hermition@O@A
    
#     return ch_gen

# ch_gen = channel_gen(k_nd_BS, d_BS, k_rmd_U, d_U)
# h_ch = np.reshape(ch_gen,(-1,1))
# H_real = np.round(np.real(h_ch),5).flatten()
# H_imag = np.round(np.imag(h_ch),5).flatten()

# %%
N_port = 3
N_BS = 2
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
w_1 = 0
w_2 = 0
w_3 = 0
w_4 = 0
w_5 = 0
w_6 = 0

def ave_meas(count):
     total = count.get('0', 0) + count.get('1', 0)
     return count.get('1', 0) / total if total > 0 else 0

def Q_sampler_est(ch_gen, H_real, H_imag, w_1, w_2, w_3, shots):
        
    q = QuantumRegister(ch_gen.shape[0], 'q')
    c = ClassicalRegister(ch_gen.shape[0], 'c')
    qc1= QuantumCircuit(q,c)
        
    qc1.barrier()
    
    for j in range(ch_gen.shape[0]):
        for k in range(ch_gen.shape[1]):
            qc1.ry(LA.norm(ch_gen[j,k]), q[j])
        

    qc1.barrier()
    
    qc1.cx(q[0], q[1])
    qc1.cx(q[1], q[2])
    qc1.cx(q[2], q[0])

    qc1.barrier()
        
    qc1.ry(w_1, q[0])
    qc1.ry(w_2, q[1])
    qc1.ry(w_3, q[2])
    
    qc1.barrier()
    
    # qc1.cz(q[0], q[1])
    # qc1.cz(q[1], q[2])
    # qc1.cz(q[2], q[3])
    # qc1.cz(q[3], q[4])
    # qc1.cz(q[4], q[5])
    # qc1.cz(q[5], q[0])
    
    # qc1.barrier()
        
    qc1.measure(q[0], c[0]) 
    qc1.measure(q[1], c[1]) 
    qc1.measure(q[2], c[2]) 
   
        
    sampler = StatevectorSampler()
        
    job_sam = sampler.run( [(qc1)], shots = shots)
    result_sam = job_sam.result()
    counts_sam = result_sam[0].data.c.get_counts()
        
    simp_counts_01 = marginal_counts(counts_sam, indices=[0])
    simp_counts_02 = marginal_counts(counts_sam, indices=[1])
    simp_counts_03 = marginal_counts(counts_sam, indices=[2])

        # counts_sam = result_sam[0].data.c.get_counts()
        
    out1 = ave_meas(simp_counts_01)
    out2 = ave_meas(simp_counts_02)
    out3 = ave_meas(simp_counts_03)

        
    out = [out1, out2, out3]

    return qc1, counts_sam, out, out1, out2, out3

#H_real = H_sample_real[0]
#H_imag = H_sample_imag[0]
qc1, counts_sam,out, out1, out2, out3 = Q_sampler_est(ch_gen, H_real, H_imag, w_1, w_2, w_3, shots=1024)
print("measurement_average_01 =",out[0])
print("measurement_average_02 =",out[1])
print("measurement_average_03 =",out[2])

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
    
def loss(N_BS, ch_gen, H_real, H_imag, w_1, w_2, w_3):
    
    qc1, counts_sam,out, out1, out2, out3 = Q_sampler_est(ch_gen, H_real, H_imag, w_1, w_2, w_3, shots=1024)
    
    v1 = np.exp(1j*(2*np.pi/Lambda)*(out1))     # 1st port
    v2 = np.exp(1j*(2*np.pi/Lambda)*(out2))     # 2nd port
    v3 = np.exp(1j*(2*np.pi/Lambda)*(out3))     # 3rd port
    V1 = v1/abs(v1)
    V2 = v2/abs(v2)
    V3 = v3/abs(v3)
    Q = np.array([V1,V2,V3])
    
    def compute_sinr(H, Q, num_ports, sigma_n):
        sinr_port = np.zeros(num_ports)
        for k in range(num_ports):
            signal = np.abs(H[k,:].conj().T * Q[k])**2  # Sinyal ke port p
            interference = np.sum([np.abs(H[j, :].conj().T * Q[j])**2 for j in range(num_ports) if j != k
                                   ])
            # interference = np.sum(np.abs(H @ Q)**2) - signal  # Interferensi dari port lain
            sinr = signal / (interference + sigma_n)  # Hitung SINR port p
            sinr_port[k] = np.sum(sinr)
            
        best_port = np.argmax(sinr_port)
        return sinr_port, best_port

    sinr, best_port = compute_sinr(H, Q, num_ports, sigma_n)
    # print(sinr)
    
    sum_rate = (np.log2(1 + sinr[best_port])) 

    loss = -(sum_rate)
    return loss

los = loss(N_BS, ch_gen, H_real, H_imag, w_1, w_2, w_3)
# %%

def gradient(N_BS, ch_gen, H_real, H_imag, w_1, w_2, w_3, w_index):
        
    shift = np.pi/2
        
    w = np.array([w_1, w_2, w_3])
        
    w_min = w
    w_plus = w
    
    w_min[w_index] = w_min[w_index] - shift
    loss_min = loss(N_BS, ch_gen, H_real, H_imag, w_min[0], w_min[1], w_min[2])
        
    w_plus[w_index] = w_plus[w_index] + shift
    loss_plus = loss(N_BS, ch_gen, H_real, H_imag, w_plus[0], w_plus[1], w_plus[2])
        
    grad = (1/2*np.sin(shift)) * (loss_min-loss_plus)
        
    return grad, loss_min, loss_plus

w_1 = np.pi
w_2 = np.pi
w_3 = np.pi

grad_1, loss_min, loss_plus = gradient(N_BS, ch_gen, H_real, H_imag, w_1, w_2, w_3, 1)  

# %%

WL = 0.5
N_eps = 10
N_data = 100
learn_step = 1e-4
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
    ch_gen, H_real, H_imag = ch_simp(N_port, N_BS, WL) 


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
            
            grad, loss_min, loss_plus = gradient(N_BS, h_ch[i_data], H_sample_real[i_data], H_sample_imag[i_data], w[0], w[1], w[2], i_weight)
            
            learn_step = learn_step_init / np.sqrt(i_eps+1)
            # learn_step = 0.5 * learn_step_init * (1+np.cos((np.pi*(i_eps+1))/N_eps))
            
            w[i_weight] = w[i_weight] - ((learn_step)*grad)
            # w = np.array(w[i_weight])
        
        loss_cal = loss(N_BS, h_ch[i_data], H_sample_real[i_data], H_sample_imag[i_data], w[0], w[1], w[2])
        
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
    
   
