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

oculus_viewer_path = os.path.expanduser('~/catkin_ws/src/sonar_oculus/scripts')
if oculus_viewer_path not in sys.path:
    sys.path.append(oculus_viewer_path)

from oculus_viewer import generate_map_xy, ping_callback

from sonar_oculus.msg import OculusPing

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

def ping_to_range(msg: OculusPing, angle: float) -> float:
    global bridge
    img = bridge.imgmsg_to_cv2(msg.ping, desired_encoding="passthrough")
    print(f"Ping image shape: {img.shape}")
    angle_rad = angle * np.pi / 180 # convert to radians
    r = np.linspace(0, msg.fire_msg.range, num=msg.num_ranges)
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

def smooth_range(range):
    range_history.append(range)
    return sum(range_history) / len(range_history)

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

        sonar_image_cv = self.br.imgmsg_to_cv2(sonar_image, desired_encoding="bgr8")

        for class_id, bbox, conf in zip(class_ids, bboxes, confs):
            if class_id >= len(object_names):
                print(f"Class ID {class_id} out of range")
                continue
            
            class_id = int(class_id)
            obj_name = object_names[class_id]
            obj_centroid = (bbox[0], bbox[1])  # x, y
            bearing = obj_centroid[0]/rgb.width * CAM_FOV - CAM_FOV/2
            bearing = smooth_bearing(bearing)
            range = ping_to_range(depth, bearing)

            if range is not None and range > 0:
                range = smooth_range(range)
                angle = to_rad(bearing)
                x = range * np.cos(angle)
                y = range * np.sin(angle)
                
                center_x = int(x / depth.range_resolution)
                center_y = int((depth.num_ranges - (range / depth.range_resolution)))
                center_x, center_y = smooth_coordinates(obj_name, (center_x, center_y))

                detected_coords = (center_x, center_y)
                rospy.set_param('/detected_coords', detected_coords)
                print(f"Detected coordinates set to parameter: {detected_coords}")

                ping_callback(depth, self.sonar_img_pub, detected_coords)
        
        msg_sonar_detections = self.br.cv2_to_imgmsg(sonar_image_cv, encoding="bgr8")
        msg_sonar_detections.header.stamp = sonar_image.header.stamp
        self.sonar_img_pub.publish(msg_sonar_detections)

if __name__ == "__main__":
    rospy.init_node("closed_set_detector")
    detector = ClosedSetDetector()
    bridge = CvBridge()
    rospy.spin()
