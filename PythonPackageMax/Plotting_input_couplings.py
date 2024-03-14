import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# load in data
input1 = sp.io.loadmat('./Experimentation/full_align_L5_q4.mat')
input2 = sp.io.loadmat('./Experimentation/full_align_L6_q6.mat')
input3 = sp.io.loadmat('./Experimentation/full_align_L10_q10.mat')
L1 = input1['L']
L2 = input2['L']
L3 = input3['L']
q1 = input1['q']
q2 = input2['q']
q3 = input3['q']

fronorm_inp1 = input1['Jfro']
fronorm_inp2 = input2['Jfro']
fronorm_inp3 = input3['Jfro']

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12, 6))

im1 = ax1.imshow(fronorm_inp1, cmap='viridis')
ax1.set_title("Experiment 1\n L = %d, q= %d" % (L1,q1))
ax1.set_xticks(np.arange(L1))
ax1.set_yticks(np.arange(L1))
ax1.set_xlabel("Position")
ax1.set_ylabel("Position")

im2 = ax2.imshow(fronorm_inp2, cmap='viridis')
ax2.set_title("Experiment 2\n L = %d, q= %d" % (L2,q2))
ax2.set_xticks(np.arange(L2))
ax2.set_yticks(np.arange(L2))
ax2.set_xlabel("Position")
ax2.set_ylabel("Position")

im3 = ax3.imshow(fronorm_inp3, cmap='viridis')
ax3.set_title("Experiment 3\n L = %d, q= %d" % (L3,q3))
ax3.set_xticks(np.arange(L3))
ax3.set_yticks(np.arange(L3))
ax3.set_xlabel("Position")
ax3.set_ylabel("Position")


plt.tight_layout()
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im2, cax=cbar_ax)
plt.show()
