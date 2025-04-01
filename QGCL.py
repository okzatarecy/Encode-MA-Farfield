from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
import numpy as np

# 1. Inisialisasi register kuantum
edges = QuantumRegister(6, 'edges')  # 6 edge qubits (e0-e5)  
nodes = QuantumRegister(4, 'nodes')  # 4 node qubits (v0-v3) feature
qc = QuantumCircuit(edges, nodes)

# 2. Inisialisasi contoh
for i in range(6): qc.x(edges[i])  # Aktifkan semua edges
for i in range(4): qc.ry(np.pi/4, nodes[i])  # Inisialisasi fitur node

# 3. Definisikan 2-qubit unitary gate U
def create_u_gate(label):
    u = QuantumCircuit(2, name=f'U{label}')
    u.cx(0,1)
    u.rz(np.pi/4, 0)
    u.ry(np.pi/4, 1)
    u.cx(0,1)
    return u.to_gate()

# 4. Implementasi sesuai urutan gambar
# U1: nodes 0-1
u1 = create_u_gate(1).control(1)
qc.append(u1, [edges[0], nodes[0], nodes[1]])
qc.swap(nodes[0], nodes[1])

# U2: nodes 1-2
u2 = create_u_gate(2).control(1)
qc.append(u2, [edges[1], nodes[1], nodes[2]])
qc.swap(nodes[1], nodes[2])

# U3: nodes 2-3
u3 = create_u_gate(3).control(1)
qc.append(u3, [edges[2], nodes[2], nodes[3]])

# U4: nodes 0-1
u4 = create_u_gate(4).control(1)
qc.append(u4, [edges[3], nodes[0], nodes[1]])
qc.swap(nodes[0], nodes[1])
qc.swap(nodes[1], nodes[2])

# U5: nodes 2-3
u5 = create_u_gate(5).control(1)
qc.append(u5, [edges[4], nodes[2], nodes[3]])
qc.swap(nodes[0], nodes[1])
qc.swap(nodes[1], nodes[2])

# U6: nodes 2-3
u6 = create_u_gate(6).control(1)
qc.append(u6, [edges[5], nodes[2], nodes[3]])

# 5. Visualisasi circuit
print("Quantum Graph Convolutional Layer:")
print(qc.draw(fold=-1, output='text'))