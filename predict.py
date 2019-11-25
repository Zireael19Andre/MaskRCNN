#!--*-- coding:utf-8 --*--

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from PIL import Image
import numpy as np
import cv2

pylab.rcParams['figure.figsize'] = 20, 20

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo


# configs file
config_file = "/home/andre/maskrcnn-benchmark/configs/e2e_mask_rcnn_R_101_FPN_1x.yaml"

cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
cfg.MODEL.WEIGHT = '/home/andre/maskrcnn-benchmark/output/model_final.pth'

coco_demo = COCODemo(
cfg,
min_image_size=500,
confidence_threshold=0.7,
masks_per_dim=20)

imgfile = '/home/andre/maskrcnn-benchmark/predict/Prediction.jpg'
pil_image = Image.open(imgfile).convert("RGB")

image = np.array(pil_image)[:, :, [2, 1, 0]]

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")


# forward predict
predictions = coco_demo.compute_prediction(image)
top_predictions = coco_demo.select_top_predictions(predictions)

result = image.copy()
plt.subplot(2,2,1)
imshow(image)

result_mask = coco_demo.overlay_mask(result,top_predictions)
result_mask = coco_demo.overlay_class_names(result_mask, top_predictions)
plt.subplot(2,2,2)
imshow(result_mask)

result_background = coco_demo.overlay_boxes(result,top_predictions)
plt.subplot(2,2,3)
imshow(result_background)

result = cv2.addWeighted(image,0.6,result_mask,0.4,0)
result = coco_demo.overlay_class_names(result, top_predictions)
plt.subplot(2,2,4)
imshow(result)
plt.show()
