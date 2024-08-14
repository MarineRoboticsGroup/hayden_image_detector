#!/usr/bin/env python
import numpy as np
import cv2
from scipy.interpolate import interp1d
import rospy
import cv_bridge
from sonar_oculus.msg import OculusPing
from sensor_msgs.msg import Image

bridge = cv_bridge.CvBridge()
map_x, map_y = None, None
res, height, rows, width, cols = None, None, None, None, None
f_bearings = None
cm = 1
to_rad = lambda bearing: bearing * np.pi / 18000

def generate_map_xy(msg):
    global res, height, rows, width, cols, map_x, map_y
    
    _res = msg.range_resolution
    _height = msg.num_ranges * _res
    if _res == 0:
        raise ValueError("Resolution (_res) must not be zero")
    _rows = int(np.ceil(_height / _res))
    _width = msg.num_ranges * _res * np.sin(to_rad(msg.bearings[-1] - msg.bearings[0]) / 2) * 2
    _cols = int(np.ceil(_width / _res))

    if res == _res and height == _height and rows == _rows and width == _width and cols == _cols:
        return
    res, height, rows, width, cols = _res, _height, _rows, _width, _cols

    bearings = to_rad(np.asarray(msg.bearings, dtype=np.float32))
    f_bearings = interp1d(bearings, range(len(bearings)), kind='linear', fill_value='extrapolate')

    map_x, map_y = np.zeros((rows, cols), dtype=np.float32), np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            r = i * res
            angle = msg.bearings[0] + (j * (msg.bearings[-1] - msg.bearings[0]) / cols)
            x = r * np.cos(angle)
            y = r * np.sin(angle)

            if _height <= 0:
                raise ValueError("_height must be positive and non-zero.")

            map_x[i, j] = 1 + (cols - 1) * (x/width + 0.5)
            map_y[i, j] = r / res

def ping_callback(msg):
    generate_map_xy(msg)

    img = bridge.imgmsg_to_cv2(msg.ping, desired_encoding='passthrough')
    img = np.array(img, dtype=img.dtype, order='F')
    
    remapped_img = img #cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    
    if remapped_img.shape != (rows, cols):
        remapped_img = cv2.resize(remapped_img, (cols, rows))

    remapped_img_bgr = cv2.cvtColor(remapped_img, cv2.COLOR_GRAY2BGR) if remapped_img.ndim == 2 else remapped_img
    remapped_img_bgr = cv2.normalize(remapped_img_bgr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    remapped_img_bgr = cv2.applyColorMap(remapped_img_bgr, cm)
        

    img_msg = bridge.cv2_to_imgmsg(remapped_img_bgr, encoding="bgr8")
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