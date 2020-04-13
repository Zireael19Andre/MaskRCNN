import matplotlib.pyplot as plt
import re
from mpl_toolkits.axes_grid1 import host_subplot

# read the log file/读取文件
fp = open('/Users/zireael19andre/Desktop/Log/kumamoto_log.txt', 'r')

train_iterations = []
avg_loss = []
classfier_loss=[]
bbox_loss=[]
mask_loss=[]
RPN_bbox_loss=[]


for line in fp:
    # get train_iterations and train_loss/获取损失和循环数
    if 'iter:' in line and 'loss:' in line:
        arr = re.findall(r'iter: (\d+)', line)
        arr2 = re.findall(r'loss: (\d\.\d+)',line)
        train_iterations.append(int(arr[0]))
        avg_loss.append(float(arr2[0]))


    if 'fier:' in line:
        arr3 = re.findall(r'_classifier: (\d\.\d+)',line)
        classfier_loss.append(float(arr3[0]))


    if 'reg:' in line:
        arr4 = re.findall(r'_reg: (\d\.\d+)',line)
        bbox_loss.append(float(arr4[0]))


    if 'mask:' in line:
        arr5 = re.findall(r'_mask: (\d\.\d+)',line)
        mask_loss.append(float(arr5[0]))


fp.close()

host = host_subplot(111)
plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window/调整边界
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
# set location of the legend,/设置图标位置
# 1->rightup corner, 2->leftup corner, 3->leftdown corner
# 4->rightdown corner, 5->rightmid ...
host.legend(loc=1)

# set label color/选色
host.axis["left"].label.set_color(p1.get_color())

# set the range of x axis of host and y axis of par1/坐标轴范围
host.set_xlim([0, 100000])
host.set_ylim([0, 2])
plt.draw()
plt.show()
