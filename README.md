# EnBCDet
Multiscale Self-Supervised Contrastive Learning for Enhanced Detection of Malignant Cells from Urine Cytology Samples with Explainability

Dependencies: 
pytorch-lightning = 0.9.1
PyTorch = 2.6.0

Dataset:
Custom Dataset = The dataset comprises 3238 images, with 1811 classified as benign and 1427 as malignant. Approximately 1500 cells from 150
JPEG images sized 1600x1400 were labeled.

Self-Supervised Pretraining:
To perform self-supervised pretraining, follow the steps provided in the ‘EnBCDet_Self_Supervised_Pretraining_Github_Copy.ipynb’ file in ‘Self Sup’ folder.

Object Detection:
To perform training and malignant cell detection from urine cytology samples, follow the code provided in the ‘main.py’ file in ‘ObjDet’ folder.

Explainability of Sel-Supervised Pretraining:
To get the CAM explanations of the self-supervised encoder, follow the ‘EnBCDet_encoder_heatmap_try.ipynb’ file in ‘XAI’ folder.

Citation:
Title =
Author = Arathy Menon N P and Ram S Iyer and Pournami P N and Jayaraj P B and Pranab Dey
Year = 2025
Journal = The Visual Computer 
url = ‘(url)’

