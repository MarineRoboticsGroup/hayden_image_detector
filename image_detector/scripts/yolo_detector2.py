import ultralytics
from ultralytics import YOLO
from IPython.display import Image, display

# Check the installed version of ultralytics
ultralytics.checks()

# Mount Google Drive (if running in Google Colab)
from google.colab import drive
drive.mount('/content/drive')

# Change the directory to your YOLOv8 directory
import os
os.chdir('/content/drive/MyDrive/YOLOv8')

# List files in the current directory
print(os.listdir())

# Train the YOLOv8 model
os.system('yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=25 imgsz=320 plots=True')

# List files in the 'runs/detect/train/' directory
print(os.listdir('runs/detect/train/'))

# Display training results
display(Image(filename='runs/detect/train/confusion_matrix.png', width=600))
display(Image(filename='runs/detect/train/results.png', width=600))
display(Image(filename='runs/detect/train/val_batch0_pred.jpg', width=600))

# Validate the model
os.system('yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml')

# Predict using the trained model
os.system('yolo task=detect mode=predict model=runs/detect/train/weights/best.pt conf=0.25 source=/content/drive/MyDrive/YOLOv8/test/images')

# Display prediction results
import glob
for image_path in glob.glob('runs/detect/predict/*.jpg')[:3]:
    display(Image(filename=image_path, width=600))
    print("\n")
