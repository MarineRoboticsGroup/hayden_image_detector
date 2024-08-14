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
global res, height, rows, width, cols, map_x, map_y, f_bearings
res, height, rows, width, cols = None, None, None, None, None
f_bearings = None
map_x, map_y = None, None
cm = 1
to_rad = lambda bearing: bearing * np.pi / 18000
f_bearings = None

def generate_map_xy(msg):
    _res = msg.range_resolution
    _height = msg.num_ranges * _res
    if _res == 0:
        raise ValueError("Resolution (_res) must not be zero")
    _rows = int(np.ceil(_height / _res))
    _width = np.sin(to_rad(msg.bearings[-1] - msg.bearings[0]) / 2) * _height * 2 #msg.num_ranges * _res * np.sin(to_rad(msg.bearings[-1] - msg.bearings[0]) / 2) * 2
    _cols = int(np.ceil(_width / _res))

    global res, height, rows, width, cols, map_x, map_y, f_bearings
    if res == _res and height == _height and rows == _rows and width == _width and cols == _cols:
        return
    res, height, rows, width, cols = _res, _height, _rows, _width, _cols

    bearings = to_rad(np.asarray(msg.bearings, dtype=np.float32))
    f_bearings = interp1d(bearings, range(len(bearings)), kind='linear', fill_value='extrapolate')

    XX, YY = np.meshgrid(range(cols), range(rows))
    x = res * (rows - YY)
    y = res * (-cols / 2.0 + XX + 0.5)
    b = np.arctan2(y, x) * -1
    r = np.sqrt(np.square(x) + np.square(y))
    map_y = np.asarray(r / res, dtype=np.float32)
    map_x = np.asarray(f_bearings(b), dtype=np.float32)

def ping_callback(msg):
    generate_map_xy(msg)

    img = bridge.imgmsg_to_cv2(msg.ping, desired_encoding='passthrough')
    img = np.array(img, dtype=img.dtype, order='F')
    
    img.resize(rows, cols)
    img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    
    time.sleep(0.5)

    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img = cv2.applyColorMap(img, cm)

    img_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
    img_msg.header.stamp = rospy.Time.now()
    img_pub.publish(img_msg) 
    print("Published remapped sonar image")

def initialize_remapper():
    global img_pub
    rospy.init_node('sonar_remapper')
    img_pub = rospy.Publisher('/sonar_remapped_image', Image, queue_size=10)
    rospy.Subscriber('/sonar_vertical/oculus_node/ping', OculusPing, ping_callback)
    rospy.spin()

if __name__ == '__main__':
    initialize_remapper()
    rospy.spin()