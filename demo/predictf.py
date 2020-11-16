#!--*-- coding:utf-8 --*--
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import random
import os

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 50, 20

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

output = os.listdir('/home/andre/maskrcnn-benchmark/predict/')

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")

config_file = "/home/andre/maskrcnn-benchmark/configs/e2e_mask_rcnn_R_101_FPN_1x.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
cfg.MODEL.WEIGHT = '/home/andre/maskrcnn-benchmark/output/model_final.pth'

coco_demo = COCODemo(
    cfg,
    min_image_size=500,
    confidence_threshold=0.7,
    masks_per_dim = 50,
)
url = ['https://s1.ax1x.com/2020/09/22/wOLyDK.jpg',
'https://s1.ax1x.com/2020/09/22/wOLsu6.jpg',
'https://s1.ax1x.com/2020/09/22/wOLDjx.jpg',
'https://s1.ax1x.com/2020/09/22/wOLBg1.jpg',
'https://s1.ax1x.com/2020/09/22/wOL03R.jpg',
'https://s1.ax1x.com/2020/09/22/wOLwC9.jpg',
'https://s1.ax1x.com/2020/09/22/wOLa4J.jpg',
'https://s1.ax1x.com/2020/09/22/wOLUN4.jpg',
'https://s1.ax1x.com/2020/09/22/wOLNEF.jpg',
'https://s1.ax1x.com/2020/09/22/wOLYHU.jpg',
'https://s1.ax1x.com/2020/09/22/wOLGuV.jpg',
'https://s1.ax1x.com/2020/09/22/wOLl3n.jpg',
'https://s1.ax1x.com/2020/09/22/wOZbBF.jpg',
'https://s1.ax1x.com/2020/09/22/wOZHnU.jpg',
'https://s1.ax1x.com/2020/09/22/wOZTXT.jpg',
'https://s1.ax1x.com/2020/09/22/wOZocV.jpg',
'https://s1.ax1x.com/2020/09/22/wOZI10.jpg',
'https://s1.ax1x.com/2020/09/22/wOZ5pq.jpg',
'https://s1.ax1x.com/2020/09/22/wOZhhn.jpg',
'https://s1.ax1x.com/2020/09/22/wOZfts.jpg',
'https://s1.ax1x.com/2020/09/22/wOZWkj.jpg',
'https://s1.ax1x.com/2020/09/22/wOZ27Q.jpg',
'https://s1.ax1x.com/2020/09/22/wOZg0g.jpg',
'https://s1.ax1x.com/2020/09/22/wOZcnS.jpg',
'https://s1.ax1x.com/2020/09/22/wOZyX8.jpg',
'https://s1.ax1x.com/2020/09/22/wOZs6f.jpg',
'https://s1.ax1x.com/2020/09/22/wOZr1P.jpg',
'https://s1.ax1x.com/2020/09/22/wOZDpt.jpg',
'https://s1.ax1x.com/2020/09/22/wOZ0fI.jpg',
'https://s1.ax1x.com/2020/09/22/wOZdkd.jpg',
'https://s1.ax1x.com/2020/09/22/wOZUTH.jpg',
'https://s1.ax1x.com/2020/09/22/wOZN0e.jpg',
'https://s1.ax1x.com/2020/09/22/wOZtmD.jpg',
'https://s1.ax1x.com/2020/09/22/wOZJOO.jpg',
'https://s1.ax1x.com/2020/09/22/wOZG6K.jpg',
'https://s1.ax1x.com/2020/09/22/wOZ8l6.jpg',
'https://s1.ax1x.com/2020/09/22/wOZ3Sx.jpg',
'https://s1.ax1x.com/2020/09/22/wOZlf1.jpg',
'https://s1.ax1x.com/2020/09/22/wOZQYR.jpg',
'https://s1.ax1x.com/2020/09/22/wOZMk9.jpg',
'https://s1.ax1x.com/2020/09/22/wOZuTJ.jpg',
'https://s1.ax1x.com/2020/09/22/wOZnw4.jpg',
'https://s1.ax1x.com/2020/09/22/wOZmmF.jpg',
'https://s1.ax1x.com/2020/09/22/wOZZOU.jpg'
]
for i in url:
    url_select = i
    image = load(url_select)

    predictions = coco_demo.compute_prediction(image)
    top_predictions = coco_demo.select_top_predictions(predictions)

    result = image.copy()
    plt.subplot(1,3,1)
    imshow(image)

    result_mask = coco_demo.overlay_mask(result,top_predictions)
    result = cv2.addWeighted(image,0.6,result_mask,0.4,0)
    plt.subplot(1,3,2)
    imshow(result)
    
    result = cv2.addWeighted(image,1.0,result_mask,0,0)
    result = coco_demo.overlay_class_names(result, top_predictions)
    result_box= coco_demo.overlay_boxes(result,top_predictions)
    plt.subplot(1,3,3)
    imshow(result_box)


    plt.savefig('/home/andre/maskrcnn-benchmark/predict/'+ url_select.split('/')[-1],dpi = 300)
    plt.clf()

