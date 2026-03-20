# Scene-text-detection
This project focuses on detecting text in natural scene images using a deep learning approach (Fcenet and radial encoder detection head)

The dataset and trained model can be downloaded via this google drive
https://drive.google.com/drive/folders/1Autl0i8yW4EUPrMsEN4zuAIlskh8wlnG?usp=sharing

All of the ground truth annotations from CoCoTextv2 and TotalText dataset are normalized into 14 vertices polygon format.

The Cocotextv2, totaltext and ctw1500 dataset images can be downloaded through official sites.

In order to run inference script/ fine tune on the model, load the model fcenet_radial_head_finetuned.pth 

the following dependencies are needed
mmocr 1.0.1
mmdet 3.1.0
mmcv 2.0.1
mmengine 0.10.7
conda 26.1.1


Backbone : ResNet50 (DCNv2) FPN
Det Head : Radial Regressor (20 Rays)
Input Shape : 512x512 pixels
Total Time : ~20.801 seconds (for 500 images)
FPS Rate : 24.03 Frames per Second
Latency : ~41.60 ms per Image