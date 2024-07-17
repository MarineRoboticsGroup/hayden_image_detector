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
res, height, rows, width, cols = None, None, None, None, None
map_x, map_y = None, None
f_bearings = None
cm = 1
raw = False

bridge = cv_bridge.CvBridge()
to_rad = lambda bearing: bearing * np.pi / 18000


def generate_map_xy(msg):
    global res, height, rows, width, cols, map_x, map_y, f_bearings
    
    _res = msg.range_resolution
    _height = msg.num_ranges * _res
    print(f"_height: {_height}, _res: {_res}")
    if _res == 0:
        raise ValueError("Resolution (_res) must not be zero")
    _rows = int(np.ceil(_height / _res))
    _width = np.sin(to_rad(msg.bearings[-1] - msg.bearings[0]) / 2) * _height * 2
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
            angle = msg.bearings[0] + j * (msg.bearings[-1] - msg.bearings[0]) / cols
            x = r * np.cos(to_rad(angle))
            y = r * np.sin(to_rad(angle))
            map_x[i, j] = f_bearings(np.arctan2(y, x) * REVERSE_Z)
            map_y[i, j] = r / res

    print(f"Generated map_x shape: {map_x.shape}, map_y shape: {map_y.shape}")

def ping_callback(msg, img_pub, detected_coords=None):
    #global cm, raw

    print("Ping callback triggered")
    
    if raw:
        img = bridge.imgmsg_to_cv2(msg.ping, desired_encoding='passthrough')
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img = cv2.applyColorMap(img, cm)
        print(f"Raw image shape: {img.shape}")
        img_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
        img_msg.header.stamp = rospy.Time.now()
        img_pub.publish(img_msg)
        print("Published raw sonar image") 
    else:
        print("Generating map coordinates")
        generate_map_xy(msg)

        img = bridge.imgmsg_to_cv2(msg.ping, desired_encoding='passthrough')
        print(f"Original ping image shape: {img.shape}")
        img = np.array(img, dtype=img.dtype, order='F')

        remapped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        print(f"Remapped image shape: {remapped_img.shape}")
        
        if remapped_img.shape != (rows, cols):
            remapped_img = cv2.resize(remapped_img, (cols, rows))
            print(f"Resized remapped image shape: {remapped_img.shape}")
        
        #time.sleep(0.5)

        if detected_coords:
            center_x, center_y = detected_coords
            print(f"Detected coordinates retrived: ({center_x}, {center_y})")
            print(f"Image dimensions: width={remapped_img.shape[1]}, height={remapped_img.shape[0]}")

            if 0 <= center_x < remapped_img.shape[1] and 0 <= center_y < remapped_img.shape[0]:
                remapped_img_bgr = cv2.cvtColor(remapped_img, cv2.COLOR_GRAY2BGR)
                cv2.circle(remapped_img_bgr, (center_x, center_y), 10, (255, 0, 0), -1)
                print(f"Drawing dot at {detected_coords}")
            else:
                print(f"Detected coordinates {detected_coords} are out of image bounds.")
        else:
            print("No detected coordinates found.")

        print(f"Image dimensions after drawing: {img.shape}")
        
        remapped_img_bgr = cv2.cvtColor(remapped_img, cv2.COLOR_GRAY2BGR) if remapped_img.ndim == 2 else remapped_img
        remapped_img_bgr = cv2.normalize(remapped_img_bgr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        remapped_img_bgr = cv2.applyColorMap(remapped_img_bgr, cm)
        
        print(f"Image dimensions before publishing: {remapped_img.shape}")
        
        #cv2.imwrite('/tmp/processed_sonar_image.png', remapped_img)
        #print("Saved processed sonar image to /tmp/processed_sonar_image.png")

        img_msg = bridge.cv2_to_imgmsg(remapped_img_bgr, encoding="bgr8")
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