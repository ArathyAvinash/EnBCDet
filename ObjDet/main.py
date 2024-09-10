

import torch
import os
torch.backends.cudnn.enabled = True
os.environ["CUDA_VISIBLE_DEVICES"]="3"

torch.cuda.is_available()

#!python -m pip install --upgrade pip

#!pip install tensorflow==2.3.1

!pip install tensorboard==2.4.1
!pip install torch  #YOLOv5 runs on top of PyTorch, so we need to import it to the notebook

import torch # YOLOv5 implemented using pytorch

from IPython.display import Image #this is to render predictions

pwd

!pip install -r requirements.txt

!python train.py --img 416 --batch 8 --epochs 100 --data dataset.yaml --cfg ./models/custom_yolov5_senew2.yaml --weights ''

"""With the new se_new2architecture"""

!python train.py --img 416 --batch 8 --epochs 300 --data dataset.yaml --cfg ./models/custom_yolov5_senew2.yaml --weights ''

!python train.py --img 416 --batch 8 --epochs 300 --data dataset.yaml --cfg ./models/custom_yolov5_senew2.yaml --weights ''

python train.py --img 640 --batch 8 --epochs 300 --data dataset.yaml --cfg ./models/custom_yolov5_senew2.yaml --weights ''

"""#With pretrained weights after self supervision"""

!python train.py --img 416 --batch 16 --epochs 300 --data dataset.yaml --cfg ./models/custom_yolov5_senew2.yaml --weights runs/train/exp/weights/encoder.pt --cache

!python train.py --img 640 --batch 16 --epochs 300 --data dataset.yaml --cfg ./models/custom_yolov5_senew2.yaml --weights runs/train/exp/weights/encoder1.pt --cache

"""Third Architecture"""

!python detect.py --source /content/drive/MyDrive/YOLO_new/yolov5/High_Grade/P9142597.JPG --weights bestsenew2.pt --conf 0.4

Image(filename='/content/drive/MyDrive/YOLO_new/yolov5/runs/detect/exp32/P9142597.JPG', width=416)

!python detect.py --source /content/drive/MyDrive/YOLO_new/yolov5/High_Grade/P9153401.JPG --weights bestsenew2.pt  --conf 0.4

Image(filename='/content/drive/MyDrive/YOLO_new/yolov5/runs/detect/exp93/P9153401.JPG', width=416)

!python detect.py --source /content/drive/MyDrive/YOLO_new/yolov5/High_Grade/P9153407.JPG --weights bestsenew2.pt  --conf 0.4

Image(filename='/content/drive/MyDrive/YOLO_new/yolov5/runs/detect/exp97/P9153407.JPG', width=416)

!python detect.py --source /content/drive/MyDrive/YOLO_new/yolov5/High_Grade/HGimages/P9142581.JPG --weights bestsenew2.pt  --conf 0.1

"""With Conf 0.4"""

Image(filename='/content/drive/MyDrive/YOLO_new/yolov5/runs/detect/exp210/P9142581.JPG', width=416)

"""With Conf 0.1"""

Image(filename='/content/drive/MyDrive/YOLO_new/yolov5/runs/detect/exp223/P9142581.JPG', width=416)

!python val.py --weights bestsenew2.pt --data dataset-test.yaml --img 640
