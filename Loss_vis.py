#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import pylab
from pylab import figure, show, legend
from mpl_toolkits.axes_grid1 import host_subplot

# read the log file
fp = open('/Users/zireael19andre/Desktop/Log/log.txt', 'r')

train_iterations = []
avg_loss = []
classfier_loss=[]
bbox_loss=[]
mask_loss=[]
RPN_bbox_loss=[]

for line in fp:
    # get train_iterations and train_loss
    if 'iter:' in line and 'loss:' in line:
        st_iter=line.find('iter:')
        st_loss=line.find(' loss:')
        #arr = re.findall(r'er: \b\d+\b,', line)
        train_iterations.append(int(line[st_iter+5:st_iter+11]))
        avg_loss.append(float(line[st_loss+6:st_loss+11]))

    if 'fier:' in line:
        st_classfi_loss=line.find('fier:')
        classfier_loss.append(float(line[st_classfi_loss+6:st_classfi_loss+11]))

    if 'reg:' in line:
        st_bbox_loss=line.find('reg:')
        bbox_loss.append(float(line[st_bbox_loss+5:st_bbox_loss+10]))

    if 'mask:' in line:
        st_mask_loss=line.find('mask:')
        mask_loss.append(float(line[st_mask_loss+6:st_mask_loss+11]))

fp.close()

host = host_subplot(111)
plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
par1 = host.twin()
par2 = host.twin()
par3 = host.twin()
# set labels
host.set_xlabel("iterations")
host.set_ylabel("loss")


# plot curves
p1, = host.plot(train_iterations, avg_loss, label="Avg_loss")
p2, = par1.plot(train_iterations, classfier_loss, label="Classfier_loss")
p3, = par2.plot(train_iterations, bbox_loss, label="BBOX_loss")
p4, = par3.plot(train_iterations, mask_loss, label="Mask_loss")
# set location of the legend,
# 1->rightup corner, 2->leftup corner, 3->leftdown corner
# 4->rightdown corner, 5->rightmid ...
host.legend(loc=1)

# set label color
host.axis["left"].label.set_color(p1.get_color())

# set the range of x axis of host and y axis of par1
host.set_xlim([0, 100000])
host.set_ylim([0, 2])
plt.draw()
plt.show()