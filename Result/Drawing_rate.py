# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:10:16 2025

@author: orecy
"""

## This file is to generate the figure

import pickle
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch


# # === Access the axes and the lines ===

with open('D:/code/Encode-MA-Farfield/Result/12 May 2025/QGNN wo H/QGNN_rate_enc_VE_revise_100data.pkl', 'rb') as f:
    fig1 = pickle.load(f)
    
with open('D:/code/Encode-MA-Farfield/Result/12 May 2025/Non_QGNN/Non_QGNN_rate_100data_fix.pkl', 'rb') as f:
    fig2 = pickle.load(f)
    
# Get axes from each figure
ax1 = fig1.axes[0]
ax2 = fig2.axes[0]

# Get Line2D data
line1 = ax1.get_lines()[0]
x1, y1 = line1.get_xdata(), line1.get_ydata()

line2 = ax2.get_lines()[0]
x2, y2 = line2.get_xdata(), line2.get_ydata()

#Get shaded region
min_fig1 = y1-np.std(y1)
max_fig1 = y1+np.std(y1)

min_fig2 = y2-np.std(y2)
max_fig2 = y2+np.std(y2)

# Create a new figure
fig, ax = plt.subplots(figsize=(5,2))

# Plot the training loss lines
# ax.plot(x1, y1, label='QGNN', color='red', linewidth=1, marker='o', markevery=5)
# ax.fill_between(x1, max_fig1, min_fig1, color='red', alpha=0.2)
ax.plot(x2, y2, label='Non-QGNN', color='blue', linestyle='--', linewidth=1, marker='s', markevery=5)
ax.fill_between(x2, max_fig2, min_fig2, color='blue', alpha=0.2)
ax.legend(loc='lower right')  # Legend akan muncul sekarang


# Beautify the plot
# ax.set_title('Training Loss Comparison with Confidence Interval')
ax.set_xlabel('Episodes')
ax.set_ylabel('Rate (bps/Hz)')
ax.grid(True)

plt.savefig('Rate_compare_100_nonQGNN.svg', format='svg', dpi=1200, bbox_inches="tight")
plt.show()