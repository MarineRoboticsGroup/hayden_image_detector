#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os
import sys

# Hack for now until we set the PYTHONPATH properly
sys.path.append('/home/haydentedrake/Desktop/dino_vit_features')
#sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../src')

from cosegmentation import find_cosegmentation, draw_cosegmentation_binary_masks, draw_cosegmentation

import matplotlib.pyplot as plt

class ImageDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.callback)
        self.detection_pub = rospy.Publisher("/detection/image_raw", CompressedImage, queue_size=10)
       
    def callback(self, data: CompressedImage) -> None:
        try:
            np_arr = np.frombuffer(data.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # Insert your detector processing here
            detections = self.detect_objects(cv_image)
            compressed_image = CompressedImage()
            compressed_image.header = data.header
            compressed_image.format = "jpeg"
            compressed_image.data = np.array(cv2.imencode('.jpg', detections)[1]).tobytes()
            self.detection_pub.publish(compressed_image)
        except CvBridgeError as e:
            rospy.logerr("CvBridgeError: %s", e)
       
    def detect_objects(self, image: np.ndarray) -> np.ndarray:
        segmentation_masks, image_pil_list = find_cosegmentation(image, load_size=120, low_res_saliency_maps=False)
        result_image = draw_cosegmentation(segmentation_masks, image_pil_list)
        return result_image

def main():
    rospy.init_node('image_detector', anonymous=True)
    detector = ImageDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")

if __name__ == '__main__':
    main()
