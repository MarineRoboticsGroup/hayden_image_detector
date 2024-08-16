import pathlib
import message_filters
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import CameraInfo, Image
from ultralytics import YOLO
from PIL import Image as PILImage
from sensor_msgs.msg import Image as RosImage
import sys
import os
from collections import deque
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import rospy
import cv_bridge
from sonar_oculus.msg import OculusPing
from sensor_msgs.msg import Image
from dynamic_reconfigure.server import Server
from sonar_oculus.cfg import ViewerConfig
import time

oculus_viewer_path = os.path.expanduser('~/catkin_ws/src/sonar_oculus/scripts')
if oculus_viewer_path not in sys.path:
    sys.path.append(oculus_viewer_path)

from oculus_viewer import generate_map_xy, publish_detections, add_detection

from sonar_oculus.msg import OculusPing

cuda_available = torch.cuda.is_available()
cuda_device_count = torch.cuda.device_count()
print("CUDA Available:", cuda_available)
print("CUDA Device Count:", cuda_device_count)
if cuda_available and cuda_device_count > 0:
    print("Current CUDA Device:", torch.cuda.current_device())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))

bridge = cv_bridge.CvBridge()
global res, height, rows, width, cols
res, height, rows, width, cols = None, None, None, None, None
cm = 1
to_rad = lambda bearing: bearing * np.pi / 180
INTENSITY_THRESHOLD = 70

object_names = ['80-20', 'fish-cage', 'hand-net', 'ladder', 'lobster-cages', 'paddle', 'pipe', 'recycling-bin', 'seaweed', 'stairs', 'tire', 'towfish', 'trash-bin']

bearing_queue = deque(maxlen=10)
coord_history = {}
range_history = deque(maxlen=10)

def ping_to_range(msg: OculusPing, angle: float) -> float:
    img = bridge.imgmsg_to_cv2(msg.ping, desired_encoding="passthrough")
    print(f"Ping image shape: {img.shape}")
    angle_rad = to_rad(angle) # convert to radians
    r = np.linspace(0, msg.fire_msg.range, num=msg.num_ranges)
    az = to_rad(np.asarray(msg.bearings, dtype=np.float32))
    print(f"Angle (radians): {angle_rad}, Azimuth range: {az[0]} to {az[-1]}")
    
    if angle_rad < az[0] or angle_rad > az[-1]:
        print(f"Angle {angle_rad} out of azimuth range")
        return None

    closest_beam = np.argmin(np.abs(az - angle_rad))
    # filter range returns along closest beam
    filtered_ranges = savgol_filter(img[:, closest_beam], 21, 3)

    # image is num_ranges x num_beams
    min_range = 0
    max_range = msg.fire_msg.range
    found_min = False
    for idx, range in enumerate(filtered_ranges):
        if not found_min and range > INTENSITY_THRESHOLD:
            min_range = idx * msg.range_resolution
            found_min = True
        if found_min and range < INTENSITY_THRESHOLD:
            max_range = idx * msg.range_resolution
            print(f"Return detected: beam={closest_beam}, range=[{min_range}, {max_range}]") #, intensity={img[idx, closest_beam]}")
            return min_range,max_range
    print(f"No return detected for angle: {angle_rad}")
    return None, None

class ClosedSetDetector:
    def __init__(self) -> None:
        model_file = "/home/haydentedrake/Downloads/last.pt"
        self.model = YOLO(model_file)
        rospy.loginfo("Model loaded")
        self.img_pub = rospy.Publisher("/camera/yolo_img", RosImage, queue_size=10)
        self.sonar_img_pub = rospy.Publisher("/sonar_vertical/oculus_viewer/detections", RosImage, queue_size=10)
        self.br = CvBridge()
        rgb_topic = rospy.get_param("rgb_topic", "/usb_cam/image_raw_repub")
        depth_topic = rospy.get_param(
            "depth_topic", "/sonar_vertical/oculus_node/ping"
        )
        sonar_image_topic = rospy.get_param(
            "depth_topic", "/sonar_vertical/oculus_viewer/image"
        )
        self.rgb_img_sub = message_filters.Subscriber(rgb_topic, Image, queue_size=1)
        self.depth_img_sub = message_filters.Subscriber(
            depth_topic, OculusPing, queue_size=1
        )
        self.sonar_image_img_sub = message_filters.Subscriber(
            sonar_image_topic, Image, queue_size=1
        )
        # Synchronizer for RGB and depth images
        self.sync = message_filters.ApproximateTimeSynchronizer(
            (self.rgb_img_sub, self.depth_img_sub, self.sonar_image_img_sub), 100, 1
        )
        self.sync.registerCallback(self.forward_pass)
        print("Subscribers and synchronizer set up.")

    def forward_pass(self, rgb, depth, sonar_image):
        print("IN CALLBACK")
        global res, height, rows, width, cols

        # Pass rgb image to yolo, and get bbox detections.
        print("Processing images...")

        try:
            image_cv = self.br.imgmsg_to_cv2(rgb, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
            return
        
        CONF_THRESH=0.5
        results = self.model(image_cv, verbose=False, conf=CONF_THRESH)[0]
        
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = PILImage.fromarray(im_array[..., ::-1])  # RGB PIL image
            msg_yolo_detections = RosImage()
            msg_yolo_detections.header.stamp = rgb.header.stamp
            msg_yolo_detections.height = im.height
            msg_yolo_detections.width = im.width
            msg_yolo_detections.encoding = "rgb8"
            msg_yolo_detections.is_bigendian = False
            msg_yolo_detections.step = 3 * im.width
            msg_yolo_detections.data = np.array(im).tobytes()
            self.img_pub.publish(msg_yolo_detections)
            
        class_ids = results.boxes.cls.data.cpu().numpy()
        bboxes = results.boxes.xywh.data.cpu().numpy()
        confs = results.boxes.conf.data.cpu().numpy()

        CAM_FOV = 80 #degrees

        # Turn depth ping into cv2 img, and annotate with detections from rgb.
        # This image is rectangular, not triangular.

        img = bridge.imgmsg_to_cv2(depth.ping, desired_encoding='passthrough')
        img = np.array(img, dtype=img.dtype, order='F')
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        bearing_to_colindex, map_x, map_y = generate_map_xy(msg=depth)

        has_detections = False
        for class_id, bbox, conf in zip(class_ids, bboxes, confs):
            if class_id >= len(object_names):
                print(f"Class ID {class_id} out of range")
                continue
            
            class_id = int(class_id)
            obj_name = object_names[class_id]
            # bbox = [x, y, width, height] in rgb image coordinates
            def x_img_to_depth(x_img):
                x_deg = x_img/rgb.width * CAM_FOV - (CAM_FOV/2)
                x_rad = to_rad(x_deg) * -1.0
                x_depth = int(bearing_to_colindex(x_rad))
                return x_depth
            
            def range_to_y_depth(range):
                return int(range / depth.range_resolution)

            # Compute min/max range by looking only along center of detection
            center_x_img = bbox[0] + bbox[2]/2.0
            center_x_depth = x_img_to_depth(center_x_img)
            print(f"center_x_depth = {center_x_depth}")
            min_range, max_range = ping_to_range(depth, center_x_depth)

            if min_range is not None and min_range > 0:
                min_x_depth = x_img_to_depth(bbox[0])
                max_x_depth = x_img_to_depth(bbox[0] + bbox[3])
                min_y_depth = range_to_y_depth(min_range)
                min_box_height = 5 # in pixels
                max_y_depth = max(range_to_y_depth(max_range), min_y_depth + min_box_height) #

                # x, y, width, height
                depth_bbox = [min_x_depth, min_y_depth, max_x_depth - min_x_depth, max_y_depth - min_y_depth]                

                rospy.set_param('/detected_coords', depth_bbox)
                print(f"Detected coordinates set to parameter: {depth_bbox}")
                img_bgr = add_detection(img=img_bgr, depth_bbox=depth_bbox, label=obj_name)
                has_detections = True

        # Convert rectangular depth image into triangular and publish back to ROS (w/ detections).

        publish_detections(img_bgr, self.sonar_img_pub, map_x, map_y)
#        if has_detections:
#            exit()

if __name__ == "__main__":
    rospy.init_node("closed_set_detector")
    detector = ClosedSetDetector()
    bridge = CvBridge()
    rospy.spin()
