# -*- coding: utf-8 -*-
"""
Created on Wed May 14 17:52:24 2025

@author: orecy
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# Contoh data: dua grup, dengan jumlah port berbeda
data = [
    [2, 11],       # group 1: a = 3 ports
    [2, 14],  # group 2: b = 4 ports
    [2, 17],             # group 3: 2 ports
    [2, 20]
]

data = np.array(data)
n_groups = data.shape[0]
bar_width = 0.35
spacing = 1.0

# Hitung posisi x untuk setiap grup
indices = np.arange(n_groups) * (0.5 * bar_width + spacing)

# Plot
plt.figure(figsize=(8, 4))
plt.bar(indices, data[:, 0], width=bar_width, color='tomato', edgecolor='black', label=r'2 qubit gate $\mathsf{U}(\theta_i)$')
plt.bar(indices + bar_width, data[:, 1], width=bar_width, color='skyblue',  edgecolor='black',  label='Whole circuit of QGNN')

# X-axis label
# X-axis label
port_numbers = [3, 4, 5, 6]  # Label sesuai jumlah grup
xtick_labels = [fr'$N_\mathrm{{ports}} = {p}$' for p in port_numbers]
plt.xticks(indices + bar_width / 2, xtick_labels)

plt.ylabel('Required qubits')
plt.legend(loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.gca().yaxis.set_major_locator(MultipleLocator(2))
plt.tight_layout()

plt.savefig('Required qubits.svg', format='svg', dpi=1200, bbox_inches="tight")
plt.show()