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

REVERSE_Z = -1
global res, height, rows, width, cols, map_x, map_y, f_bearings
res, height, rows, width, cols = None, None, None, None, None
map_x, map_y = None, None
f_bearings = None

global cm, raw
cm = 1
raw = False
bridge = cv_bridge.CvBridge()

to_rad = lambda bearing: bearing * np.pi / 18000


def generate_map_xy(msg):
    _res = msg.range_resolution
    _height = msg.num_ranges * _res
    print(f"_height: {_height}, _res: {_res}")
    if _res == 0:
        raise ValueError("Resolution (_res) must not be zero")
    _rows = int(np.ceil(_height / _res))
    _width = np.sin(to_rad(msg.bearings[-1] - msg.bearings[0]) / 2) * _height * 2
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
    b = np.arctan2(y, x) * REVERSE_Z
    r = np.sqrt(np.square(x) + np.square(y))
    map_y = np.asarray(r / res, dtype=np.float32)
    map_x = np.asarray(f_bearings(b), dtype=np.float32)


def ping_callback(msg, img_pub, detected_coords=None):
    print("Ping callback triggered")
    
    if raw:
        img = bridge.imgmsg_to_cv2(msg.ping, desired_encoding='passthrough')
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img = cv2.applyColorMap(img, cm)
        img_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
        img_msg.header.stamp = rospy.Time.now()
        img_pub.publish(img_msg)
        print("Published raw sonar image") 
    else:
        generate_map_xy(msg)

        img = bridge.imgmsg_to_cv2(msg.ping, desired_encoding='passthrough')
        img = np.array(img, dtype=img.dtype, order='F')

        img.resize(rows, cols)
        img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

        time.sleep(0.5)

        print("Attempting to retrieve detected_coords from parameter server...")
        #detected_coords = rospy.get_param('/detected_coords', None)
        print(f"Retrieved detected_coords: {detected_coords}")
        
        if detected_coords:
            center_x, center_y = detected_coords
            print(f"Detected coordinates retrived: ({center_x}, {center_y})")
            print(f"Image dimensions: width={img.shape[1]}, height={img.shape[0]}")

            if 0 <= center_x < img.shape[1] and 0 <= center_y < img.shape[0]:
                cv2.circle(img, (center_x, center_y), 10, (255, 0, 0), -1)
                print(f"Drawing dot at {detected_coords}")
            else:
                print(f"Detected coordinates {detected_coords} are out of image bounds.")
        else:
            print("No detected coordinates found.")

        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img = cv2.applyColorMap(img, cm)
        
        img_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
        img_msg.header.stamp = rospy.Time.now()
        img_pub.publish(img_msg) 
        print("Published processed sonar image with detections")

def config_callback(config, level):
    global cm, raw
    cm = config['Colormap']
    raw = config['Polar']
    return config

def initialize_viewer():
    print("Initializing node...")
    rospy.init_node('oculus_viewer')
    print("Node initialized. Creating publisher...")
    global img_pub
    img_pub = rospy.Publisher('/sonar_vertical/oculus_viewer/image', Image, queue_size=10)
    print("Publisher created. Subscribing to topic...")
    ping_sub = rospy.Subscriber('/sonar_vertical/oculus_node/ping', OculusPing, ping_callback, None, 10)
    print("Dynamic reconfigure server started. Spinning...")
    server = Server(ViewerConfig, config_callback)

if __name__ == '__main__':
    initialize_viewer()
    rospy.spin()
