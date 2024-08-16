#!/usr/bin/env python
import numpy as np
import cv2
from scipy.interpolate import interp1d
import rospy
import cv_bridge
from sonar_oculus.msg import OculusPing
from sensor_msgs.msg import Image
from dynamic_reconfigure.server import Server
from sonar_oculus.cfg import ViewerConfig
import time

bridge = cv_bridge.CvBridge()
global res, height, rows, width, cols
res, height, rows, width, cols = None, None, None, None, None
cm = 1
to_rad = lambda bearing: bearing * np.pi / 18000

def generate_map_xy(msg):
    print("generate_map_xy is being triggered")

    _res = msg.range_resolution
    _height = msg.num_ranges * _res
    print(f"Resolution: {_res}, height: {_height}")
    if _res == 0:
        raise ValueError("Resolution (_res) must not be zero")
    _rows = int(np.ceil(_height / _res))
    _width = np.sin(to_rad(msg.bearings[-1] - msg.bearings[0]) / 2) * _height * 2 #msg.num_ranges * _res * np.sin(to_rad(msg.bearings[-1] - msg.bearings[0]) / 2) * 2
    _cols = int(np.ceil(_width / _res))

    global res, height, rows, width, cols
    res, height, rows, width, cols = _res, _height, _rows, _width, _cols
    print(f"Map dimensions set. rows: {rows}, cols: {cols}")

    bearings = to_rad(np.asarray(msg.bearings, dtype=np.float32))
    
    bearing_to_colindex = interp1d(bearings, range(len(bearings)), kind='linear', fill_value='extrapolate')
    print(f"bearing_to_colindex: {bearing_to_colindex} expected range: 0 to {cols-1}")

    XX, YY = np.meshgrid(range(cols), range(rows))
    x = res * (rows - YY)
    y = res * (-cols / 2.0 + XX + 0.5)
    b = np.arctan2(y, x) * -1
    r = np.sqrt(np.square(x) + np.square(y))
    map_y = np.asarray(r / res, dtype=np.float32)
    map_x = np.asarray(bearing_to_colindex(b), dtype=np.float32)
    return bearing_to_colindex, map_x, map_y

def add_detection(img, depth_bbox, label):
    top_corner = [depth_bbox[0] + depth_bbox[2], depth_bbox[1] + depth_bbox[3]]
    cv2.rectangle(img, pt1=[depth_bbox[0], depth_bbox[1]], pt2=top_corner, color=(255, 255, 255), thickness=2)
    cv2.putText(img, label, org=[depth_bbox[0], top_corner[1]+2], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=-1.0, color=(255, 255, 255))
    return img


def publish_detections(img, img_pub, map_x, map_y):
    print("ping callback triggered")
    
    remapped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        
    # Resize the remapped image to the expected output size if necessary
    if remapped_img.shape != (rows, cols):
        remapped_img = cv2.resize(remapped_img, (cols, rows))
        print(f"Resized remapped image shape: {remapped_img.shape}")

    # Normalize the image to enhance contrast
    remapped_img = cv2.normalize(remapped_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # Apply the color map if necessary
    remapped_img = cv2.applyColorMap(remapped_img, cm)

    # Convert the annotated or processed image back to a ROS message
    img_msg = bridge.cv2_to_imgmsg(remapped_img, encoding="bgr8")
    img_msg.header.stamp = rospy.Time.now()

    # Publish the image message
    img_pub.publish(img_msg)
    print("Published remapped sonar image")

#def config_callback(config, level):
    #global cm, raw
    #cm = config['Colormap']
    #raw = config['Polar']
    #return config

def initialize_viewer():
    rospy.init_node('oculus_viewer')
    global img_pub
    img_pub = rospy.Publisher('/sonar_vertical/oculus_viewer/detections', Image, queue_size=10)
    rospy.Subscriber('/sonar_vertical/oculus_node/ping', OculusPing, ping_callback, None, 10)
    rospy.spin()

if __name__ == '__main__':
    initialize_viewer()
    rospy.spin()
