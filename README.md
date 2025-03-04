# EnBCDet
Explainable Self-Supervised Contrastive Learning for Accurate Detection of Malignant Cells from Urine Cytology Images

The primary objective of the study is to introduce a novel object detection model, EnBCDet, that leverages contrastive based self-supervised learning to detect malignant cells in urine cytology samples. We introduce a new and parameter efficient CSC Cascade block to enhance the performance of the YOLO framework. Further, explainability methods are incorporated to enhance the reliability of the model, along with the innovation of an entropy-based class activation mapping method to analyse the performance of self-supervised pretraining. The object detection backbone pretrained on unlabelled data is transferred and fine tuned with the labelled data for optimal performance. The method outperforms many previous object detection frameworks on the task in hand, in both performance, latency and parameter efficiency.

# Dependencies: 
pytorch-lightning = 0.9.1
PyTorch = 2.6.0

# Dataset:
Custom Dataset = The dataset comprises 3238 images, with 1811 classified as benign and 1427 as malignant. Approximately 1500 cells from 150
JPEG images sized 1600x1400 were labeled.

# Self-Supervised Pretraining:
To perform self-supervised pretraining, follow the steps provided in the ‘EnBCDet_Self_Supervised_Pretraining_Github_Copy.ipynb’ file in ‘Self Sup’ folder.

# Object Detection:
To perform training and malignant cell detection from urine cytology samples, follow the code provided in the ‘main.py’ file in ‘ObjDet’ folder.

# Explainability of Sel-Supervised Pretraining:
To get the CAM explanations of the self-supervised encoder, follow the ‘EnBCDet_encoder_heatmap_try.ipynb’ file in ‘XAI’ folder.

# Installing Dependencies
!pip install -r requirements.txt
Guarantees that all required libraries and their specific versions are installed, so that the code runs correctly without missing dependencies.
!!!!!!!!!!! Ensure you have Python 3.8+ and torch installed. If using CUDA, install torch with GPU support:!!!!!!!!!!

# Inference on Images/Videos 
python detect.py --source your_image.jpg --weights yolov5s.pt --conf 0.25

For webcam inference:
python detect.py --source 0 --weights yolov5s.pt --conf 0.25

# Training a Custom Model
To train YOLOv5 on a custom dataset:

Prepare the dataset in the YOLO format (images, labels, train.txt, val.txt).
Modify the dataset YAML file (data/custom.yaml):

yaml file:
train: path/to/train/images
val: path/to/val/images
nc: 2  # Number of classes
names: ['class1', 'class2']

# Train the model
python train.py --img 640 --batch 16 --epochs 50 --data data/custom.yaml --weights yolov5s.pt --device 0

# Exporting the Model
Converting to ther formats
Formats supported: ONNX, TensorRT, CoreML, OpenVINO, TF SavedModel, TFLite
python export.py --weights yolov5s.pt --include onnx

# Customization
Adjust Confidence and IoU Thresholds
python detect.py --weights yolov5s.pt --source data/images/ --conf 0.5 --iou-thres 0.45

# Resources
Ultralytics Documentation

# Troubleshooting
CUDA not detected? Run python -c "import torch; print(torch.cuda.is_available())"
No detections? Increase --conf or check dataset labels.


# Citation:
@article{Menon2025,
  title = {Explainable Self-Supervised Contrastive Learning for Accurate Detection of Malignant Cells from Urine Cytology Images},
  author = {Arathy Menon N P and Ram S Iyer and Pournami P N and Jayaraj P B and Pranab Dey},
  year = {2025},
  journal = {The Visual Computer},
  url = {<insert your URL here>}
}

