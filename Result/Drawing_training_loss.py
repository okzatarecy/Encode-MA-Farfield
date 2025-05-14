# -*- coding: utf-8 -*-
"""
Created on Wed May 14 13:31:10 2025

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

with open('D:/code/Encode-MA-Farfield/Result/12 May 2025/QGNN wo H/QGNN_training_enc_VE_revise_100data.pkl', 'rb') as f:
    fig1 = pickle.load(f)
    
with open('D:/code/Encode-MA-Farfield/Result/12 May 2025/Non_QGNN/Non_QGNN_training_100data_fix.pkl', 'rb') as f:
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
fig, ax = plt.subplots(figsize=(5, 4))

# Plot the training loss lines
ax.plot(x1, y1, label='QGNN', color='red', linewidth=1, marker='o', markevery=5)
ax.fill_between(x1, max_fig1, min_fig1, color='red', alpha=0.2)
ax.plot(x2, y2, label='Non-QGNN', color='blue', linestyle='--', linewidth=1, marker='s', markevery=5)
ax.fill_between(x2, max_fig2, min_fig2, color='blue', alpha=0.2)
ax.legend(loc='upper right')  # Legend akan muncul sekarang

# ==== Buat inset di kiri bawah (area kosong) ====
ax_inset = inset_axes(
    ax,
    width="100%", height="100%",
    bbox_to_anchor=(0.23, 0.25, 0.4, 0.4),  # kiri-bawah posisi inset
    bbox_transform=ax.transAxes,
    loc='center'
)

# Plot ulang data zoom (episode 0–10)
mask_zoom = (x1 >= 0) & (x1 <= 10)
x_zoom = x1[mask_zoom]
y_zoom = y1[mask_zoom]
min_zoom = y_zoom - np.std(y1)
max_zoom = y_zoom + np.std(y1)

ax_inset.plot(x_zoom, y_zoom, color='red', marker='o', linestyle='-', linewidth=1, markersize=3)
ax_inset.fill_between(x_zoom, max_zoom, min_zoom, color='red', alpha=0.2)
ax_inset.set_xlim(0, 10)
ax_inset.set_ylim(min(min_zoom)-0.005, max(max_zoom)+0.005)
ax_inset.tick_params(labelsize=8)
ax_inset.grid(True)
# ax_inset.xaxis.set_major_locator(MultipleLocator(1))

# ==== Tambahkan panah otomatis ke titik tengah zoom ====
x_source = x_zoom[1]
y_source = y_zoom[1]

con = ConnectionPatch(
    xyA=(0, 0), coordsA=ax_inset.transAxes,        # pojok kiri bawah inset
    xyB=(x_source, y_source), coordsB=ax.transData,  # titik pada kurva utama
    arrowstyle="->", color="black", linewidth=1
)
fig.add_artist(con)

ax.set_ylim(-1.6, -1.45)
# ax.yaxis.set_major_locator(MultipleLocator(0.03))
# Beautify the plot
# ax.set_title('Training Loss Comparison with Confidence Interval')
ax.set_xlabel('Episodes')
ax.set_ylabel('Training Loss')
ax.grid(True)

plt.savefig('Training_loss_compare_100.svg', format='svg', dpi=1200, bbox_inches="tight")
plt.show()