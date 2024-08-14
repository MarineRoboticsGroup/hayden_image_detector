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

def generate_map_xy(msg):
    global f_bearings
    print("generate_map_xy is being triggered")

    _res = msg.range_resolution
    _height = msg.num_ranges * _res
    print(f"Resolution: {_res}, height: {_height}")
    if _res == 0:
        raise ValueError("Resolution (_res) must not be zero")
    _rows = int(np.ceil(_height / _res))
    _width = np.sin(to_rad(msg.bearings[-1] - msg.bearings[0]) / 2) * _height * 2 #msg.num_ranges * _res * np.sin(to_rad(msg.bearings[-1] - msg.bearings[0]) / 2) * 2
    _cols = int(np.ceil(_width / _res))

    global res, height, rows, width, cols, map_x, map_y
    res, height, rows, width, cols = _res, _height, _rows, _width, _cols
    print(f"Map dimensions set. rows: {rows}, cols: {cols}")

    bearings = to_rad(np.asarray(msg.bearings, dtype=np.float32))
    print(f"Bearings (radians): {bearings}")
    
    f_bearings = interp1d(bearings, range(len(bearings)), kind='linear', fill_value='extrapolate')

    try:
        f_bearings = interp1d(bearings, range(len(bearings)), kind='linear', fill_value='extrapolate')
        print("f_bearings initialized successfully.")
    except Exception as e:
        print(f"Error initializing f_bearings: {e}")
        f_bearings = None

    XX, YY = np.meshgrid(range(cols), range(rows))
    x = res * (rows - YY)
    y = res * (-cols / 2.0 + XX + 0.5)
    b = np.arctan2(y, x) * -1
    r = np.sqrt(np.square(x) + np.square(y))
    map_y = np.asarray(r / res, dtype=np.float32)
    map_x = np.asarray(f_bearings(b), dtype=np.float32)

def ping_callback(msg, img_pub, detected_coords=None):
    print("ping callback triggered")
    
    img = bridge.imgmsg_to_cv2(msg.ping, desired_encoding='passthrough')
    img = np.array(img, dtype=img.dtype, order='F')

    remapped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        
    # Resize the remapped image to the expected output size if necessary
    if remapped_img.shape != (rows, cols):
        remapped_img = cv2.resize(remapped_img, (cols, rows))
        print(f"Resized remapped image shape: {remapped_img.shape}")

    # If detected coordinates are provided, annotate the image
    if detected_coords:
        center_x, center_y = detected_coords
        print(f"Detected coordinates retrieved: ({center_x}, {center_y})")

        if 0 <= center_x < remapped_img.shape[1] and 0 <= center_y < remapped_img.shape[0]:
            remapped_img_bgr = cv2.cvtColor(remapped_img, cv2.COLOR_GRAY2BGR)
            cv2.circle(remapped_img_bgr, (center_x, center_y), 10, (255, 0, 0), -1)
            print(f"Drawing dot at {detected_coords}")
        else:
            print(f"Detected coordinates {detected_coords} are out of image bounds.")
    else:
        print("No detected coordinates found.")
        # Convert the remapped image to BGR for consistency
        remapped_img_bgr = cv2.cvtColor(remapped_img, cv2.COLOR_GRAY2BGR)

    # Normalize the image to enhance contrast
    remapped_img_bgr = cv2.normalize(remapped_img_bgr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # Apply the color map if necessary
    remapped_img_bgr = cv2.applyColorMap(remapped_img_bgr, cm)

    # Convert the annotated or processed image back to a ROS message
    img_msg = bridge.cv2_to_imgmsg(remapped_img_bgr, encoding="bgr8")
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
