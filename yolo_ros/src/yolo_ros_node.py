#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
from ultralytics import YOLO

class YOLORosNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('yolo_ros_node', anonymous=True)
        
        # Load the YOLOv8 model
        self.model = YOLO('/home/haydentedrake/sonar_data/runs/detect/train3/weights/best.pt') 
        
        # Initialize the CvBridge class
        self.bridge = CvBridge()
        
        # Subscribe to the sonar image topic
        self.image_sub = rospy.Subscriber('/sonar_vertical/oculus_viewer/image', Image, self.image_callback)
        
        # Publisher for detection results
        self.detection_pub = rospy.Publisher('/yolo2_sonar_detections', Image, queue_size=10)
    
    def image_callback(self, data):
        # Convert ROS Image message to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridgeError: {e}")
            return
        
        # Run YOLO detection
        results = self.model(cv_image)
        
        # Draw the results on the image
        annotated_image = results[0].plot()
        
        # Convert OpenCV format image back to ROS Image message
        try:
            detection_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridgeError: {e}")
            return
        
        # Publish the detection results
        self.detection_pub.publish(detection_msg)
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        yolo_node = YOLORosNode()
        yolo_node.run()
    except rospy.ROSInterruptException:
        pass
