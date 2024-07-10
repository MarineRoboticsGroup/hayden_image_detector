import rosbag
import cv2
from cv_bridge import CvBridge
import os
import shutil

# Path to the ROS bag file
#rosbag_file = '/home/haydentedrake/data/2024-02-16-13-43-49.bag'
input_dir = '/home/haydentedrake/catkin_ws/src/image_detector/scripts/yolo_sonar_dataset'
# Output directory
output_dir = 'yolo_sonar_split_dataset'
# Topic name of the sonar images
#image_topic = '/sonar_vertical/oculus_viewer/image'
class_name = 'sonar'

#bag = rosbag.Bag(rosbag_file, 'r')
#bridge = CvBridge()

#if not os.path.exists(output_dir):
    #os.makedirs(output_dir)

#count = 0
#for topic, msg, t in bag.read_messages(topics=[image_topic]):
    #try:
        #cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        #cv2.imwrite(os.path.join(output_dir, f'image_{count:06d}.png'), cv_img)
        #count += 1
    #except Exception as e:
        #print(f"Failed to convert image: {e}")

#bag.close()
#print(f"Extracted {count} images to {output_dir}")

class_dir = os.path.join(input_dir, class_name)
os.makedirs(class_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        shutil.move(os.path.join(input_dir, filename), os.path.join(class_dir, filename))

import splitfolders
splitfolders.ratio(input_dir, output=output_dir, seed=42,ratio=(0.7, 0.15, 0.15))

print("Dataset split into training validation, and test sets.")