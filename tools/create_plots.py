"""
Script to create plots of Fig. 2 in the main paper
This runs best in jupyter notebook
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import json
import cv2
import glob

"""
This data consists of the success rate of the 11 models in Fig. 2 in the main paper
Each row contains the success rate for 1 model over 4 environmnetal conditions for multiple iterations
[[i_0, i_1, i_2, i_3] for training, [i_0, i_1, i_2, i_3] for NW, [i_0, i_1, i_2, i_3] for NT, [i_0, i_1, i_2, i_3] for NTW]
NW = New Weather, NT = New Town, NTW = New Town & Weather
"""
x = ['Iter 0', 'Iter 1', 'Iter 2', 'Iter 3']
models = [
    [[71, 71, 71, 71], [72, 72, 72, 72], [41, 41, 41, 41], [43, 43, 43, 43]],
    [[45, 54, 63, 66], [39, 47, 51, 57], [23, 25, 34, 36], [26, 32, 31, 36]],
    [[45, 47, 63, 63], [39, 31, 48, 60], [23, 20, 22, 35], [26, 20, 27, 25]],
    [[26, 47, 60, 58], [23, 26, 47, 54], [17, 24, 24, 28], [16, 27, 27, 26]],
    [[26, 50, 51, 48], [23, 35, 39, 35], [17, 16, 23, 26], [16, 22, 23, 21]],
    [[45, 52, 57, 53], [39, 39, 40, 41], [23, 27, 30, 29], [26, 32, 32, 31]],
    [[26, 45, 45, 45], [23, 40, 40, 41], [17, 21, 24, 21], [16, 23, 24, 23]],
    [[45, 50, 48, 53], [39, 39, 44, 40], [23, 23, 26, 25], [26, 23, 26, 21]],
    [[26, 46, 46, 36], [23, 41, 39, 38], [17, 19, 22, 19], [16, 22, 24, 23]],
    [[45, 45, 45, 45], [39, 39, 39, 39], [23, 23, 23, 23], [26, 26, 26, 26]],
    [[26, 26, 26, 26], [23, 23, 23, 23], [17, 17, 17, 17], [16, 16, 16, 16]],
]

# legend = ['cilrs', 'expert', 'darbpe', 'darbp', 'darb', 'dart', 'smilep', 'smile' 'daggerp', 'dagger', 'cilrsp']
# legend = ['CILRS', 'CILRS+', 'DAgger', 'DAgger+', 'SMILe', 'SMILe+', 'DART', 'DA-RB', 'DA-RB+', 'DA-RB+(E)', 'Expert']
# new_legend = []
# for i in range(1,12):
#     new_legend.append(legend[-i])
# new_legend


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#Gridspec demo
fig = plt.figure()
fig.set_size_inches(18,12)
fig.set_dpi(640)

rows   = 17 #the larger the number here, the smaller the spacing around the legend
start1 = 1
end1   = int((rows-1)/2)
start2 = end1
end2   = int(rows-1)

gspec = gridspec.GridSpec(ncols=4, nrows=rows)

axes = []
axes.append(fig.add_subplot(gspec[start1:end1,0:2]))
axes.append(fig.add_subplot(gspec[start1:end1,2:4]))
axes.append(fig.add_subplot(gspec[start2:end2,0:2]))
axes.append(fig.add_subplot(gspec[start2:end2,2:4]))
axes.append(fig.add_subplot(gspec[0,0:4])) # end2 here
 

# line style & labels
lines = []
for i in range(4):
    lines = []
    for j in range(len(legend)):
        key = legend[j]
        if key == 'Expert':
            linestyle = '-'
            line, = axes[i].plot(x, models[j][i], linestyle=linestyle, color='black')
        elif key == 'CILRS' or key == 'CILRS+':
            linestyle = '-'
            line, = axes[i].plot(x, models[j][i], linestyle=linestyle)
        else:
            linestyle = '--'
            line, = axes[i].plot(x, models[j][i], linestyle=linestyle, marker='^')
        # axes[i].set_xlabel('Iteration')
        axes[i].set_ylabel('Success Rate', size='x-large')
        
        lines.append(line)

axes[0].set_title('Training Conditions', fontsize='xx-large')
axes[1].set_title('New Weather', fontsize='xx-large')
axes[2].set_title('New Town', fontsize='xx-large')
axes[3].set_title('New Town & Weather', fontsize='xx-large')

for i in range(4):
    axes[i].tick_params(axis='x', labelsize=12)
    axes[i].tick_params(axis='y', labelsize=12)

'''
lines = []
for i in range(len(legend)):
    key = legend[i]
    if key == 'CILRS' or key == 'CILRS+' or key == 'Expert':
        linestyle = '-'
    else:
        linestyle = '--'
    line,  = axes[-1].plot(x, new_model[i], linestyle=linestyle)
    # print (line)
    lines.append(line)
'''
# line, _ = axes[-1].plot(x, new_model)
# handle, labels = ax[0,0].get_legend_handles_labels()

axes[-1].legend(lines, legend, loc='center', ncol=11, mode='expand', fontsize='x-large') # create legend on bottommost axis
axes[-1].set_axis_off() # don't show bottommost axis

fig.tight_layout()

plt.savefig('/is/sg2/aprakash/Dataset/plots.pdf')

# plt.show()


'''
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import OrderedDict

f, ax = plt.subplots(2,2)

curr_id = 0
for i in range(2):
    for j in range(2):
        for i in range(len(legend)):
            key = legend[i]
            if key == 'CILRS' or key == 'CILRS+' or key == 'Expert':
                linestyle = '-'
            else:
                linestyle = '--'
            ax[i,j].plot(x, models[key][curr_id], linestyle=linestyle)
        curr_id += 1

ax[0,0].set_title('Training Conditions')
ax[0,1].set_title('New Weather')
ax[1,0].set_title('New Town')
ax[1,1].set_title('New Town & Weather')

for i in range(2):
    for j in range(2):
        ax[i,j].set_xlabel('Iteration')
        ax[i,j].set_ylabel('Success Rate')     

#legend = ['expert', 'darbpe', 'darbp', 'darb', 'dart', 'smilep', 'smile' 'daggerp', 'dagger', 'cilrsp', 'cilrs']
plt.legend(models.keys(), loc=8)
plt.savefig('/is/sg2/aprakash/Dataset/plot.pdf')
plt.show()
'''