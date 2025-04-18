# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:23:12 2025

@author: orecy
"""
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

# %% defining parameters
fc = 30e9               #carrier frequency [Hz]                
lambda_w = 3e8 / fc;    #wavelength [m]
d = lambda_w/2          #distance of elemen antennas [m]
Nt = 2               #elemen of antennas
N_user = 2              #number of users
r_circle_min = 4        #min diameter circle [m]
r_circle_max = 200      #max diameter circle [m]
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
            
    theta_list, r_list  = cart2pol(x_list, y_list);
    nn = np.arange(-(Nt-1)/2,(Nt)/2,1);   #position of element from - 0 +
    theta_aod  = np.sqrt(sigma_aod)*np.random.random([N_user,L]);
    ssf = (np.random.random([N_user,L]) + 1j*np.random.random([N_user,L]))/np.sqrt(2);
    #allocate a factor to fade the NLoS channel
    alpha = 1;
    beta = np.sqrt(alpha/(L));
    ssf = ssf*beta;

    for i_user in range(N_user):
        for l in range(L+1):
            if l != L:
                r0 = r_list[0,i_user];
                theta0 = theta_list[0,i_user]+theta_aod[i_user,l];
                r = np.sqrt(r0**2 + (nn*d)**2 - 2*r0*nn*d*np.sin(theta0));
                at = np.exp(-1j*2*np.pi*fc/c*(r - r0)*np.sin(theta0))/np.sqrt(Nt);
                H_multi_user[i_user, :] = H_multi_user[i_user, :] + ssf[i_user,l]*at*np.sqrt(1/(1+kappa));
            else:
                r0 = r_list[0,i_user];
                theta0 = theta_list[0,i_user];
                r = np.sqrt(r0**2 + (nn*d)**2 - 2*r0*nn*d*np.sin(theta0));
                at = np.exp(-1j*2*np.pi*fc/c*(r - r0)*np.sin(theta0))/np.sqrt(Nt);
                H_multi_user[i_user, :] = H_multi_user[i_user, :] + at*np.sqrt(kappa/(1+kappa));
    
    if L ==0:
        H_multi_user = H_multi_user/np.sqrt(kappa/(1+kappa));
        
    return H_multi_user

H = generate_user_in_circle_multipath(Nt, d, r_circle_min, r_circle_max, fc, N_user, sigma_aod, L, kappa)    

# %% quantum circuit preparation

# def Q_encode(N_user, H_real, H_imag, w_1, w_2, w_3, w_4):
    
#     q = QuantumRegister(4, 'q')
#     c = ClassicalRegister(4, 'c')
#     qc1= QuantumCircuit(q,c)
    
#     for k in range(N_user*2):
#         qc1.h(q[k])
        
#     qc1.barrier()
    
#     for k in range(N_user*2):
#         qc1.ry(H_real[k], q[k])
        
#     for i in range(N_user*2):
#         qc1.rz(H_imag[i], q[i])
        
#     qc1.measure(q[0], c[0]) 
#     qc1.measure(q[1], c[1]) 
#     qc1.measure(q[2], c[2]) 
#     qc1.measure(q[3], c[3])    
#     # qc1.measure_all()
        
        
#     return qc1
    
# %% QC decode sampler

# def Q_decode_sampler(N_user, H_real, H_imag, w_1, w_2, w_3, w_4, shots):
    
#     qc1 = Q_encode(N_user, H_real, H_imag, w_1, w_2, w_3, w_4)
#     sampler = StatevectorSampler()
    
#     params = {f"real_{i}": H_real[i] for i in range(len(H_real))}
#     params.update( {f"imag_{i}": H_imag[i] for i in range(len(H_imag))})
    
#     job = sampler.run( [(qc1)], shots = shots)
#     result = job.result()
#     counts = result[0].data.c.get_counts()
    
#     return counts

# %%

# def Q_decode_est(N_user, H_real, H_imag, w_1, w_2, w_3, w_4, shots):
    
#     q = QuantumRegister(4, 'q')
#     c = ClassicalRegister(4, 'c')
#     qc1= QuantumCircuit(q)
    
#     for k in range(N_user*2):
#         qc1.h(q[k])
        
#     qc1.barrier()
    
#     for k in range(N_user*2):
#         qc1.ry(H_real[k], q[k])
        
#     for i in range(N_user*2):
#         qc1.rz(H_imag[i], q[i])
    
#     observables = [[Pauli("ZIII")],
#                    [Pauli("IZII")],
#                    [Pauli("IIZI")],
#                    [Pauli("IIII")]
#                    #[SparsePauliOp(["IIII", "XXYY"], [0.5, 0.5])]
#                    ]
    
#     estimator =StatevectorEstimator()
    
#     pub = (qc1, observables)
#     job = estimator.run([pub])
#     result = job.result()[0]
#     return result
    # %% try encode
    # w_1 = 0
    # w_2 = 0
    # w_3 = 0
    # w_4 = 0
    # shots = 50
    # input_og = np.reshape(H,(-1,1))
    # inputs = np.round(input_og, 5)
    # H_real = np.real(inputs).flatten()
    # H_imag = np.imag(inputs).flatten()
    
    # qc1 = Q_encode(N_user, H_real, H_imag, w_1, w_2, w_3, w_4)
    
# %% try decode sampler
    # inputs_real = np.array([np.real(inputs[:,0])])
    # inputs_imag = np.array([np.imag(inputs[:,0])])
    # input_r = inputs_real[0].tolist()
    # input_i = inputs_imag[0].tolist()
    # counts_sampler = Q_decode_sampler(N_user, H_real, H_imag, w_1, w_2, w_3, w_4, shots)
    # qc1.draw()

    # plot_histogram(counts_sampler, sort='value_desc')
# %% try decode estimator
    # inputs_real = np.array([np.real(inputs[:,0])])
    # inputs_imag = np.array([np.imag(inputs[:,0])])
    # input_r = inputs_real[0].tolist()
    # input_i = inputs_imag[0].tolist()
    # counts = Q_decode_est(N_user, H_real, H_imag, w_1, w_2, w_3, w_4, shots)
    # qc1.draw()
    # counts.data.evs

    # plot_histogram(counts, sort='value_desc')

# %%
w_1 = 0
w_2 = 0
w_3 = 0
w_4 = 0
    
input_og = np.reshape(H,(-1,1))
inputs = np.round(input_og, 5)
H_real = np.real(inputs).flatten()
H_imag = np.imag(inputs).flatten()

def Q_sampler_est(N_user, H_real, H_imag, w_1, w_2, w_3, w_4, shots):
        
    q = QuantumRegister(4, 'q')
    c = ClassicalRegister(4, 'c')
    qc1= QuantumCircuit(q)
        
    for k in range(N_user*2):
        qc1.h(q[k])
            
    qc1.barrier()
        
    for k in range(N_user*2):
        qc1.ry(H_real[k], q[k])
            
    for i in range(N_user*2):
        qc1.rz(H_imag[i], q[i])
        
    observables = [[Pauli("ZIII")],
                   [Pauli("IZII")],
                   [Pauli("IIZI")],
                   [Pauli("IIII")]
                       #[SparsePauliOp(["IIII", "XXYY"], [0.5, 0.5])]
                       ]
        
    estimator =StatevectorEstimator()
        
    pub_est = (qc1, observables)
    job_est = estimator.run([pub_est])
    result_est = job_est.result()[0]
        
    result_est = result_est.data.evs
         
    qc1.measure_all()
            
    sampler = StatevectorSampler()
        
    params = {f"real_{i}": H_real[i] for i in range(len(H_real))}
    params.update( {f"imag_{i}": H_imag[i] for i in range(len(H_imag))})
        
    job_sam = sampler.run( [(qc1)], shots = shots)
    result_sam = job_sam.result()
    counts_sam = result_sam[0].data.meas.get_counts()
        
    return result_est, counts_sam

expected, sampler = Q_sampler_est(N_user, H_real, H_imag, w_1, w_2, w_3, w_4, 4096)
plot_histogram(sampler, sort='value_desc')

# %%

    
def ave_meas(count):
     total = count.get('0', 0) + count.get('1', 0)
     return count.get('1', 0) / total if total > 0 else 0

def Q_sampler_est(N_user, H_real, H_imag, w_1, w_2, w_3, w_4, shots):
        
    q = QuantumRegister(4, 'q')
    c = ClassicalRegister(4, 'c')
    qc1= QuantumCircuit(q,c)
        
    # for k in range(N_user*2):
    #     qc1.h(q[k])
            
    # qc1.barrier()
        
    for k in range(N_user*2):
        qc1.ry(H_real[k], q[k])
            
    for i in range(N_user*2):
        qc1.rz(H_imag[i], q[i])
        
    qc1.barrier()
        
    qc1.cz(q[0], q[1])
    qc1.cz(q[1], q[2])
    qc1.cz(q[2], q[3])
    # qc1.cz(q[3], q[4])       
        
    qc1.barrier()
        
    qc1.rz(w_1, q[0])
    qc1.rz(w_2, q[1])
    qc1.rz(w_3, q[2])
    qc1.rz(w_4, q[3])
    qc1.barrier()
        
    qc1.measure(q[0], c[0]) 
    qc1.measure(q[1], c[1]) 
    qc1.measure(q[2], c[2]) 
    qc1.measure(q[3], c[3])    
        
    sampler = StatevectorSampler()
        
    job_sam = sampler.run( [(qc1)], shots = shots)
    result_sam = job_sam.result()
    counts_sam = result_sam[0].data.c.get_counts()
        
    simp_counts_01 = marginal_counts(counts_sam, indices=[0])
    simp_counts_02 = marginal_counts(counts_sam, indices=[1])
    simp_counts_03 = marginal_counts(counts_sam, indices=[2])
    simp_counts_04 = marginal_counts(counts_sam, indices=[3])
        # counts_sam = result_sam[0].data.c.get_counts()
        
    out1 = ave_meas(simp_counts_01)
    out2 = ave_meas(simp_counts_02)
    out3 = ave_meas(simp_counts_03)
    out4 = ave_meas(simp_counts_04)
        
    out = [simp_counts_01, simp_counts_02, simp_counts_03, simp_counts_04]

    return qc1, counts_sam, out, out1, out2, out3, out4
# %%
w_1 = 0 
w_2 = 0
w_3 = 0
w_4 = 0
shots = 1024
    
input_og = np.reshape(H,(-1,1))
inputs = np.round(input_og, 5)
H_real = np.real(inputs).flatten()
H_imag = np.imag(inputs).flatten()   
qc1, counts_sam,out, out1, out2, out3, out4 = Q_sampler_est(N_user, H_real, H_imag, w_1, w_2, w_3, w_4, shots)
qc1.draw()
plot_histogram(counts_sam, sort='value_desc')
# %%

print("measurement_average_01 =",out1)
print("measurement_average_02 =",out2)
print("measurement_average_03 =",out3)
print("measurement_average_04 =",out4)
# expected, sampler = Q_sampler_est(N_user, H_real, H_imag, w_1, w_2, w_3, w_4, 4096)

# %% function loss

def loss(N_user, N_ant, H_real, H_imag, w_1, w_2, w_3, w_4):
        
    qc1, counts_sam,out, out1, out2, out3, out4 = Q_sampler_est(N_user, H_real, H_imag, w_1, w_2, w_3, w_4, shots)
    ptx = 5;
    sigma_n = 1;
    v1 = np.array([np.exp(1j*out1), np.exp(1j*out3)])       #precoding s.t opt out 2 antenna for user 1
    v2 = np.array([np.exp(1j*out2), np.exp(1j*out4)])       #precoding s.t opt out 2 antenna for user 1
    
    V1_norm = v1/abs(v1)
    V2_norm = v2/abs(v2)    
    SNR_1 = ptx*np.abs(H[0,:]@V1_norm)**2/sigma_n
    SNR_2 = ptx*np.abs(H[1,:]@V2_norm)**2/sigma_n
    R_k1 = np.log2(1+SNR_1)
    R_k2 = np.log2(1+SNR_2)
        # print(R_k1)
        # print(R_k2)
        
    Rsum = R_k1+R_k2
        
    loss = -Rsum
    return loss
# %%
    
loss_try = loss(N_user, Nt, H_real, H_imag, w_1, w_2, w_3, w_4)
    
# %% gradient calculation

def gradient(N_user, N_ant, H_real, H_imag, w_1, w_2, w_3, w_4, w_index):
        
    shift = np.pi/2
        
    w = np.array([w_1, w_2, w_3, w_4])
        
    w_min = w
    w_plus = w
        
    w_min[w_index] = w_min[w_index] - shift
    loss_min = loss(N_user, N_ant, H_real, H_imag, w_min[0], w_min[1], w_min[2], w_min[3])
        
    w_plus[w_index] = w_plus[w_index] + shift
    loss_plus = loss(N_user, N_ant, H_real, H_imag, w_plus[0], w_plus[1], w_plus[2], w_plus[3])
        
    grad = (1/2*np.sin(shift)) * (loss_min-loss_plus)
        
    return grad, loss_min, loss_plus
       
# %%
    w_1 = np.pi
    w_2 = np.pi
    w_3 = np.pi
    w_4 = np.pi
    
    grad_1, loss_min, loss_plus = gradient(N_user, Nt, H_real, H_imag, w_1, w_2, w_3, w_4, 1)
# %%
from matplotlib.pylab import plt
N_eps = 10

N_data = 100
learn_step = 0.1

w_1 = np.pi
w_2 = np.pi
w_3 = np.pi
w_4 = np.pi

w = np.array([ w_1, w_2, w_3, w_4 ])

learn_step_init = learn_step


H_sample_real =[]
H_sample_imag = []
for kk in np.arange(N_data):
    
    H_samp = generate_user_in_circle_multipath(Nt, d, r_circle_min, r_circle_max, fc, N_user, sigma_aod, L, kappa)
    inputs_og = np.reshape(H_samp,(-1,1))
    inputs = np.round(inputs_og, 5)
    H_real = np.real(inputs).flatten()
    H_imag = np.imag(inputs).flatten()
    
    H_sample_real.append(H_real)
    H_sample_imag.append(H_imag)
    # print(H_sample_real)
    # print(H_sample_imag)
    
loss_mean_array = []
for i_eps in range(N_eps):
    
    loss_array = []
    
    for i_data in range(N_data):
        
        for i_weight in range(len(w)):
            
            grad, loss_min, loss_plus = gradient(N_user, Nt, H_sample_real[i_data], H_sample_imag[i_data], w[0], w[1], w[2], w[3], i_weight)
            
            learn_step = learn_step_init / np.sqrt(i_eps+1)
            # learn_step = 0.5 * learn_step_init * (1+np.cos((np.pi*(i_eps+1))/N_eps))
            
            w[i_weight] = w[i_weight] - ((learn_step)*grad)
            
            
        loss_cal = loss(N_user, Nt, H_sample_real[i_data], H_sample_imag[i_data], w[0], w[1], w[2], w[3])
        
        loss_array.append(loss_cal)
        
    loss_mean_array.append(np.mean(loss_array))
    
    print("i_episode =",i_eps)
    print('optimized weight : ', np.array([w])) 
    print('gradient: ', grad)
    

print('Result - weight cloud : ', np.array([w]))  


plt.plot(loss_mean_array, label='QNN Loss, $N_{data}=50$')
plt.grid(True)
plt.title('QNN Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')

plt.legend(loc='best')
plt.show()