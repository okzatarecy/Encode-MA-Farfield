# -*- coding: utf-8 -*-
"""
Created on Fri May  9 15:31:32 2025

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
bandwidth = 6.5e6 #channel bandwidth

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

# %% quantum circuit

def Q_encode(H, H_real, H_imag, params_U):
    
    q1 = QuantumRegister(6, 'q1')
    q2 = QuantumRegister(5, 'q2')
    c2 = ClassicalRegister(2, 'c2')
    qc = QuantumCircuit(q1, q2, c2)  # Deklarasi semua register sekaligus
    
    # hada = [h for h in range(6)]
    # qc.h(hada)
    # quantum state preparation
    for i_re in range(6): 
        qc.ry(H_real[i_re], q1[i_re])
        
    for i_im in range(6):
        qc.rz(H_imag[i_im], q1[i_im])
    
    edges = [e for e in range(6)]
    qc.x(edges)
    
    
    for i_re in range(5): 
        qc.ry(0.5, q2[i_re])
        
    for i_im in range(5):
        qc.rz(0.5, q2[i_im])
        
    # edges = [e+6 for e in range(5)]
    # qc.h(edges)
    
    qc.barrier()
    
    def create_u_gate(label, theta1):
        gateU = QuantumCircuit(2, name=f'U{label}')  
        gateU.h(1)
        gateU.cz(0, 1)
        gateU.ry(theta1 , 0)
        gateU.cz(0,1)
        
        return gateU.to_gate()
    
    # qc.x(q1[0])
    u1 = create_u_gate(1, params_U[0,0]).control(1)
    qc.append(u1, [q1[0], q2[0], q2[1]])
    qc.swap(q2[0], q2[1])
    qc.barrier()
    
    # qc.x(q1[1])
    u2 = create_u_gate(2, params_U[1,0]).control(1)
    qc.append(u2, [q1[1], q2[1], q2[2]])
    qc.swap(q2[1], q2[2])
    qc.barrier()
    
    # qc.x(q1[2])
    u3= create_u_gate(3, params_U[2,0]).control(1)
    qc.append(u3, [q1[2], q2[2], q2[3]])
    qc.swap(q2[3], q2[4])
    qc.barrier()
    
    # qc.x(q1[3])
    u4= create_u_gate(4, params_U[3,0]).control(1)
    qc.append(u4, [q1[3], q2[3], q2[4]])
    qc.swap(q2[3], q2[1])
    qc.swap(q2[3], q2[2])
    qc.barrier()
    
    # qc.x(q1[4])
    u5= create_u_gate(5,params_U[4,0]).control(1)
    qc.append(u5, [q1[4], q2[1], q2[2]])
    qc.swap(q2[1], q2[0])
    qc.barrier()
    
    # qc.x(q1[5])
    u6= create_u_gate(6,params_U[5,0]).control(1)
    qc.append(u6, [q1[5], q2[1], q2[0]])
    # qc.cz(q2[5], q2[6])
    # qc.swap(q2[4], q2[5])
    
    qc.barrier()
    qc.measure(q2[0], c2[0])    
    qc.measure(q2[1], c2[1])
    
    return qc
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
qc2 = Q_encode(ch_gen, H_real, H_imag, params_U)

# %%
def ave_meas(count):
     total = count.get('0', 0) + count.get('1', 0)
     return count.get('1', 0) / total if total > 0 else 0
 
def Q_decode(h_ch, H_real, H_imag, params_U, shots):
    
    qc = Q_encode(h_ch, H_real, H_imag, params_U)
    sampler = StatevectorSampler()
    
    job = sampler.run( [(qc)], shots=shots)
    result = job.result()
    counts_sam = result[0].data.c2.get_counts()
    
    simp_counts_01 = marginal_counts(counts_sam, indices=[0])
    simp_counts_02 = marginal_counts(counts_sam, indices=[1])
    
    out1 = ave_meas(simp_counts_01)
    out2 = ave_meas(simp_counts_02)
    
    out = [out1, out2]

    return counts_sam, qc, out, out1, out2

count_sam, qc, out, out1, out2 = Q_decode(ch_gen, H_real, H_imag, params_U, shots=1024)

print("measurement_average_01 =",out[0])
print("measurement_average_02 =",out[1])
# print("measurement_average_03 =",out[2])
# %%
#
num_ports=3
ptx = 5
sigma_n = 1
    
# def loss(ch_gen, H_real, H_imag, params_U):
    
#     count_sam, qc, out, out1, out2 = Q_decode(ch_gen, H_real, H_imag, params_U, shots=1024)
    
#     v1 = np.exp(1j*(out1))     # 1st BS
#     v2 = np.exp(1j*(out2))     # 2nd BS
#     V1 = v1/abs(v1)
#     V2 = v2/abs(v2)
#     Q = np.array([V1,V2])
    
    
#     sinr1 = np.abs(ch_gen[0,:] @ Q)**2
#     sinr2 = np.abs(ch_gen[1,:] @ Q)**2
#     sinr3 = np.abs(ch_gen[2,:] @ Q)**2
    
#     sinr_p1 = sinr1 / (sinr2+sinr3+sigma_n)
#     sinr_p2 = sinr2 / (sinr1+sinr3+sigma_n)
#     sinr_p3 = sinr3 / (sinr1+sinr2+sigma_n)
    
#     sinr_all = np.array([sinr_p1,sinr_p2,sinr_p3])
#     # best_port = np.argmax(sinr_all)
#     # sum_rate = np.log2(1 + sinr_all[best_port])
    
#     indices = np.argsort(sinr_all)[-2:]
#     best_sinr = sinr_all[indices]
    
#     sum_rate1 = np.log2(1 + best_sinr[0])
#     sum_rate2 = np.log2(1 + best_sinr[1])
#     sum_rate = sum_rate1+sum_rate2        

#     loss = -1*(sum_rate)
#     return loss

# los = loss(ch_gen, H_real, H_imag, params_U)

# %% sum-rate func

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
learn_step = 3
shift = np.pi/2

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
# h_ch = np.array([[-3.38808382+1.61964646j,  2.41701005+0.34117205j],
#        [-0.40897523-3.47990012j, -0.98520206+3.2845099j ],
#        [ 3.44889666+1.20110836j, -2.55814964-3.85044008j]])        
    
# H_sample_real = np.array([-3.38808,  2.41701, -0.40898, -0.9852 ,  3.4489 , -2.55815])
# H_sample_imag = np.array([ 1.61965,  0.34117, -3.4799 ,  3.28451,  1.20111, -3.85044])   

loss_mean_array =[]
loss_min_array = []
loss_max_array = []
 
rate_mean_array = []
rate_min_array = []
rate_max_array = []

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
    

for i_eps in range(N_eps):
    loss_array =[]
    rate_array = []
    params_U_min = params_U.copy()
    params_U_plus = params_U.copy()
    
    # iteration each weight for N_data
    for i_weight in range(params_U.shape[0]):
        
        for j_weight in range(params_U.shape[1]):
            loss_idata = np.zeros(N_data)
            loss_min_idata = np.zeros(N_data)
            loss_plus_idata = np.zeros(N_data)
            
            params_U_min[i_weight, j_weight] = params_U_min[i_weight, j_weight] - shift
            params_U_plus[i_weight, j_weight] = params_U_plus[i_weight, j_weight] + shift
           
            for i_data in range(N_data):
                H_idata = h_ch[i_data]
                H_real_idata = H_sample_real[i_data]
                H_imag_idata = H_sample_imag[i_data]
                loss_min_idata[i_data], rate_min = LossRate(H_idata, H_real_idata, H_imag_idata, params_U_min, bandwidth)   
                loss_plus_idata[i_data], rate_plus = LossRate(H_idata, H_real_idata, H_imag_idata, params_U_plus, bandwidth)
                    
            Loss_min = np.mean(loss_min_idata)
            Loss_plus = np.mean(loss_plus_idata)
            grad = (1/2*np.sin(shift)) * (Loss_plus-Loss_min)
            learn_step = learn_step_init / np.sqrt(i_eps+1)
            params_U[i_weight, j_weight] = params_U[i_weight, j_weight] - ((learn_step)*grad)

    
    for i_data in range(N_data):
        H_idata = h_ch[i_data]
        H_real_idata = H_sample_real[i_data]
        H_imag_idata = H_sample_imag[i_data]
        loss_cal, rate_cal = LossRate(H_idata, H_real_idata,  H_imag_idata, params_U, bandwidth)
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

# plt.savefig('training_loss_revise_plot.svg', format='svg', dpi=1200, bbox_inches="tight")
plt.show()

  # %% plot sum_rate

plt.plot(rate_mean_array, color='blue', label='Rate')
plt.fill_between(np.arange(N_eps), rate_max_array, rate_min_array, color='blue', alpha=0.2)

# naming the x axis 
plt.xlabel('Episode') 
# naming the y axis 
plt.ylabel('Rate (bps/Hz)') 


plt.grid(True)
plt.rc('grid', linestyle="dotted", color='grey')
plt.legend(loc='lower right')

# plt.savefig('rate_revise_plot.svg', format='svg', dpi=1200, bbox_inches="tight")
plt.show()


# # %% Save figure using pickle (training loss)

# import pickle
# # === Your Plotting Code ===
# fig, ax = plt.subplots()  # Create figure and axes object

# plt.plot(loss_mean_array, color='blue', label='Training Loss')
# plt.fill_between(np.arange(N_eps), loss_max_array, loss_min_array, color='blue', alpha=0.2)

# # naming the x axis 
# plt.xlabel('Training episode') 
# # naming the y axis 
# plt.ylabel('Loss') 


# plt.grid(True)
# plt.rc('grid', linestyle="dotted", color='grey')
# plt.legend(loc='best')

# # === Save the figure object to pickle ===
# with open('QGNN_training_enc_VE_revise.pkl', 'wb') as f:
#     pickle.dump(fig, f)

# plt.show()

# # %% Save figure using pickle (Rate)

# # === Your Plotting Code ===
# fig, ax = plt.subplots()  # Create figure and axes object

# plt.plot(rate_mean_array, color='blue', label='Rate')
# plt.fill_between(np.arange(N_eps), rate_max_array, rate_min_array, color='blue', alpha=0.2)

# # naming the x axis 
# plt.xlabel('Episode') 
# # naming the y axis 
# plt.ylabel('Rate (bps/Hz)') 


# plt.grid(True)
# plt.rc('grid', linestyle="dotted", color='grey')
# plt.legend(loc='lower right')

# # === Save the figure object to pickle ===
# with open('QGNN_rate_enc_VE_revise.pkl', 'wb') as f:
#     pickle.dump(fig, f)

# plt.show()
# # %% To Load the Saved Figure Later

# import matplotlib.pyplot as plt
# import pickle

# # Load the figure from the Pickle file
# with open('QGNN_training.pkl', 'rb') as f:
#     fig = pickle.load(f)

# # === Access the axes and the lines ===
# ax = fig.axes[0]  # Assuming only one subplot (axes)
# ax.set_title('Updated Title: Quantum Neural Network')
# # ax.set_title('New Title Here', fontsize=14, fontweight='bold', loc='center')
# # Access all lines in the plot (returns a list of Line2D objects)
# lines = ax.get_lines()
# # Example: Change the color of the first line
# lines[0].set_color('red')  # Change to any color you want, e.g., 'blue', '#FF5733'

# # Optional: Change line style or linewidth as well
# # lines[0].set_linestyle('--')
# # lines[0].set_linewidth(2)
# # lines[0].set_color('blue')
# # lines[0].set_marker('o')
# # lines[0].set_markersize(5)
# # lines[0].set_label('Updated Training Loss') 


# ax.legend()
# # === Show the updated figure ===
# plt.show()
# # %%
# # Result - weight final Ndata= 1 learnsteinit (2): Result - weight final:  
#  # [[[3.13964884]
#   # [3.1550537 ]
#   # [3.12002723]
#   # [3.39378661]
#   # [1.70773878]
#   # [4.67869578]]]

# # Result - weight final Ndata= 1 learnsteinit (2): Result - weight final:   
# # [[[3.15754561]
# #   [3.14888535]
# #   [3.11466766]
# #   [4.6987544 ]
# #   [4.70513408]
# #   [1.57652649]]]

# # Result - weight final Ndata= 2 learnsteinit (2): Result - weight final: 
#     # [[[3.24421359]
#     #   [3.19915444]
#     #   [3.03174556]
#     #   [4.67610933]
#     #   [4.68151942]
#     #   [1.56348701]]]