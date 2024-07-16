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
from sonar_oculus.msg import OculusPing
import yaml
from collections import deque
from scipy.interpolate import interp1d
import os

oculus_viewer_path = os.path.expanduser('~/catkin_ws/src/sonar_oculus/scripts')
if oculus_viewer_path not in sys.path:
    sys.path.append(oculus_viewer_path)

from oculus_viewer import ping_callback

from sonar_oculus.msg import OculusPing
import yaml

cuda_available = torch.cuda.is_available()
cuda_device_count = torch.cuda.device_count()

print("CUDA Available:", cuda_available)
print("CUDA Device Count:", cuda_device_count)
if cuda_available and cuda_device_count > 0:
    print("Current CUDA Device:", torch.cuda.current_device())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))

to_rad = lambda bearing: bearing * np.pi / 18000
INTENSITY_THRESHOLD = 100

object_names = ['80-20', 'fish-cage', 'hand-net', 'ladder', 'lobster-cages', 'paddle', 'pipe', 'recycling-bin', 'seaweed', 'stairs', 'tire', 'towfish', 'trash-bin']

bearing_queue = deque(maxlen=10)
coord_history = {}
range_history = deque(maxlen=10)
detected_coords = None

res, height, rows, width, cols = None, None, None, None, None
map_x, map_y = None, None
f_bearings = None

def ping_to_range(msg: OculusPing, angle: float) -> float:
    global bridge
    img = bridge.imgmsg_to_cv2(msg.ping, desired_encoding="passthrough")
    print(f"Ping image shape: {img.shape}")
    angle_rad = angle * np.pi / 180 # convert to radians
    az = to_rad(np.asarray(msg.bearings, dtype=np.float32))
    print(f"Angle (radians): {angle_rad}, Azimuth range: {az[0]} to {az[-1]}")
    
    if angle_rad < az[0] or angle_rad > az[-1]:
        print(f"Angle {angle_rad} out of azimuth range")
        return None

    closest_beam = np.argmin(np.abs(az - angle_rad))
    # image is num_ranges x num_beams
    for idx in range(len(img[:, closest_beam])):
        if img[idx, closest_beam] > INTENSITY_THRESHOLD:
            br = idx * msg.range_resolution
            print(f"Return detected: beam={closest_beam}, idx={idx}, range={br}, intensity={img[idx, closest_beam]}")
            return br
    print(f"No return detected for angle: {angle_rad}")
    return None

def smooth_bearing(bearing):
    bearing_queue.append(bearing)
    return sum(bearing_queue) / len(bearing_queue)

def smooth_coordinates(obj_name, coords):
    if obj_name not in coord_history:
        coord_history[obj_name] = deque(maxlen=10)
    coord_history[obj_name].append(coords)
    avg_coords = np.mean(coord_history[obj_name], axis=0)
    return int(avg_coords[0]), int(avg_coords[1])

def smooth_range(range_value):
    range_history.append(range_value)
    return sum(range_history) / len(range_history)

def generate_map_coordinates(range_value, bearing, depth):
    global res, height, rows, width, cols, map_x, map_y, f_bearings
    angle_rad = bearing * np.pi / 180  # convert bearing to radians
    if res is None or height is None:
        raise ValueError("Resolution (res) and height must not be None")
    
    rows = int(np.ceil(height / res))
    _width = np.sin(to_rad(depth.bearings[-1] - depth.bearings[0]) / 2) * height * 2
    width = max(_width, 0.01)
    #width = max(np.sin(angle_rad / 2) * height * 2, 0.01)  # Ensure width is positive
    cols = max(int(np.ceil(width / res)), 1)  # Ensure cols is positive

    bearings = to_rad(np.asarray(depth.bearings, dtype=np.float32))
    f_bearings = interp1d(bearings, np.arange(len(bearings)), kind='linear', fill_value='extrapolate')

    XX, YY = np.meshgrid(np.arange(cols), np.arange(rows))
    x = res * (rows - YY)
    y = res * (-cols / 2.0 + XX + 0.5)
    b = np.arctan2(y, x) * -1  # REVERSE_Z is -1
    r = np.sqrt(np.square(x) + np.square(y))
    map_y = np.asarray(r / res, dtype=np.float32)
    map_x = np.asarray(f_bearings(b), dtype=np.float32)

    print(f"Generated map_x shape: {map_x.shape}, map_y.shape: {map_y.shape}")
    print(f"map_x: {map_x}, map_y: {map_y}")

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
            "sonar_image_topic", "/sonar_vertical/oculus_viewer/image"
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
        global detected_coords, res, height, rows, width, cols, map_x, map_y, f_bearings
        print("Processing images...")

        try:
            image_cv = self.br.imgmsg_to_cv2(rgb, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
            return
        
        CONF_THRESH=0.5
        results = self.model(image_cv, verbose=False, conf=CONF_THRESH)[0]
        
        # Extract segmentation masks
        if results.boxes is None:
            print("NO DETECTIONS")
            return
        
        # Show the results
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
            # im.show()  # show image
            # im.save('results.jpg')  # save image
        class_ids = results.boxes.cls.data.cpu().numpy()
        bboxes = results.boxes.xywh.data.cpu().numpy()
        confs = results.boxes.conf.data.cpu().numpy()

        CAM_FOV = 80 #degrees

        res = depth.range_resolution
        height = depth.num_ranges * res
        #sonar_image_cv = self.br.imgmsg_to_cv2(sonar_image, desired_encoding="bgr8")

        for class_id, bbox, conf in zip(class_ids, bboxes, confs):
            if class_id >= len(object_names):
                print(f"Class ID {class_id} out of range")
                continue
            
            class_id = int(class_id)
            obj_name = object_names[class_id]
            obj_centroid = (bbox[0], bbox[1])  # x, y
            bearing = obj_centroid[0]/rgb.width * CAM_FOV - CAM_FOV/2
            bearing = smooth_bearing(bearing)
            range_value = ping_to_range(depth, bearing)

            if range_value is not None and range_value > 0:
                range_value = smooth_range(range_value)
                #angle = to_rad(bearing)
                
                generate_map_coordinates(range_value, bearing, depth)

                closest_beam = np.argmin(np.abs(np.array(depth.bearings) - (bearing * 100)))
                closest_beam = np.clip(closest_beam, 0, map_x.shape[-1] - 1)
                
                center_x = int(map_x[0, closest_beam] if map_x.ndim == 2 else map_x[closest_beam])
                center_y_idx = int(range_value / depth.range_resolution)
                center_y_idx = np.clip(center_y_idx, 0, map_y.shape[-1] - 1)
                
                print(f"closest_beam: {closest_beam}, center_y_idx: {center_y_idx}, map_x.shape: {map_x.shape}, map_y.shape: {map_y.shape}")
                
                if map_y.ndim == 2:
                    center_y = int(map_y[0, center_y_idx])
                else:
                    center_y = int(map_y[center_y_idx])
                
                print(f"center_x: {center_x}, center_y_idx: {center_y_idx}, map_y.shape: {map_y.shape}")
                
                center_x, center_y = smooth_coordinates(obj_name, (center_x, center_y))

                center_x = np.clip(center_x, 0, 699 - 1)
                center_y = np.clip(center_y, 0, 699 - 1)


                detected_coords = (int(center_x), int(center_y))
                rospy.set_param('/detected_coords', detected_coords)
                print(f"Detected coordinates set to parameter: {detected_coords}")

                ping_callback(depth, self.sonar_img_pub, detected_coords)
        
        try:
            sonar_image_cv = self.br.imgmsg_to_cv2(sonar_image, desired_encoding="bgr8")
            if isinstance(sonar_image_cv, np.ndarray):
                msg_sonar_detections = self.br.cv2_to_imgmsg(sonar_image_cv, encoding="bgr8")
                msg_sonar_detections.header.stamp = sonar_image.header.stamp
                self.sonar_img_pub.publish(msg_sonar_detections)
            else:
                print("Warning: sonar_image is not a valid NumPy array")
        except CvBridgeError as e:
            print(f"Error converting sonar_image: {e}")

if __name__ == "__main__":
    rospy.init_node("closed_set_detector")
    detector = ClosedSetDetector()
    bridge = CvBridge()
    rospy.spin()
