import rospy
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import cv2
import numpy as np

def main():
    rospy.init_node('simple_publisher')
    img_pub = rospy.Publisher('/sonar_detections', RosImage, queue_size=10)
    bridge = CvBridge()

    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        # Create a simple black image with a red dot
        img = np.zeros((480, 640, 3), np.uint8)
        cv2.circle(img, (320, 240), 50, (0, 0, 255), -1)

        img_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
        img_msg.header.stamp = rospy.Time.now()
        img_pub.publish(img_msg)
        
        rospy.loginfo("Published test image")
        rate.sleep()

if __name__ == '__main__':
    main()
