#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import sys
sys.path.append('/home/haydentedrake/Desktop/dino-vit-features')
from dino_vit_features.cosegmentation import hi

class ImageDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.callback)
        self.detection_pub = rospy.Publisher("/detection/image_raw", CompressedImage, queue_size=10)
       
    def callback(self, data):
        try:
            np_arr = np.frombuffer(data.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # Insert your detector processing here
            detections = self.detect_objects(cv_image)
            compressed_image = CompressedImage()
            compressed_image.header = data.header
            compressed_image.format = "jpeg"
            compressed_image.data = np.array(cv2.imencode('.jpg', detections)[1]).tostring()
            self.detection_pub.publish(compressed_image)
        except CvBridgeError as e:
            rospy.logerr(e)
       
    def detect_objects(self, image):
        hi()
        print("Got image")
        return image

def main():
    rospy.init_node('image_detector', anonymous=True)
    detector = ImageDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")

if __name__ == '__main__':
    main()
