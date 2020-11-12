# MaskRCNN
Work for research
---
BackGround
---
from FAIR(Facebook AI Research) Team ALG,I modified the network to fix missdection issue using aerial images.</br>
Paper submitted to Asian Conference on Remote Sensing - 2020.</br>
</hr>
- iteration_loss(exsample)
  - use the result log to draw iteration and all-type loss curve
  ![image](https://github.com/Zireael19Andre/MaskRCNN/blob/master/image/loss_vis.jpg)
- prediction(exsample)
  - MaskRCNN official solution (Pytorch ver) don't provide visulization for image prediction(Both BBox and Segm).
  I write a simple scripts.
  ![image](https://github.com/Zireael19Andre/MaskRCNN/blob/master/image/BBox.png)
  ![image](https://github.com/Zireael19Andre/MaskRCNN/blob/master/image/Segm.png)
