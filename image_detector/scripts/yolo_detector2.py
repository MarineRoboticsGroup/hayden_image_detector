import ultralytics
from ultralytics import YOLO
from IPython.display import Image, display
import os
import glob

# Check the installed version of ultralytics
ultralytics.checks()

# Change to YOLOv8 directory
os.chdir('/home/haydentedrake/sonar_data')

# List files in the current directory
print(os.listdir())

# Train the YOLOv8 model
os.system('yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=25 imgsz=320 plots=True')

# List files in the 'runs/detect/train/' directory
print(os.listdir('runs/detect/train/'))

# Display training results
try:
    display(Image(filename='runs/detect/train/confusion_matrix.png', width=600))
    display(Image(filename='runs/detect/train/results.png', width=600))
    display(Image(filename='runs/detect/train/val_batch0_pred.jpg', width=600))
except Exception as e:
    print(f"Display error: {e}")
#
#  Validate the model
os.system('yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml')

# Predict using the trained model
os.system('yolo task=detect mode=predict model=runs/detect/train3/weights/best.pt conf=0.25 source=/content/drive/MyDrive/YOLOv8/test/images')

# Display prediction results
try:
    for image_path in glob.glob('runs/detect/predict/*.jpg')[:3]:
        display(Image(filename=image_path, width=600))
        print("\n")
except Exception as e:
    print(f"Display error: {e}")