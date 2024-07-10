import pathlib
import message_filters
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import CameraInfo, Image
from ultralytics import YOLO
from PIL import Image as PILImage
from sensor_msgs.msg import Image as RosImage
import sys
print("PYTHONPATH:",sys.path)
from sonar_oculus.msg import OculusPing
import yaml

to_rad = lambda bearing: bearing * np.pi / 18000
INTENSITY_THRESHOLD = 0

def ping_to_range(msg: OculusPing, angle: float) -> float:
    """
    msg: OculusPing message
    angle: angle in degrees
    Convert sonar ping to range (take most intense return on beam) at given angle.
    """
    #img = bridge.compressed_imgmsg_to_cv2(msg.ping, desired_encoding="passthrough")
    img = bridge.imgmsg_to_cv2(msg.ping, desired_encoding="passthrough")
    # pre-process ping
    #ping = self.sonar.deconvolve(img)
    ping = img
    #print(ping)
    angle = angle * np.pi / 180 # convert to radians
    angular_res = 2.268928027592628 / 512 # radians for oculus m1200d lf and m750d hf/lf assuming even spacing
    r = np.linspace(0, msg.fire_msg.range,num=msg.num_ranges)
    az = to_rad(np.asarray(msg.bearings, dtype=np.float32))
    # image is num_ranges x num_beams
    for beam in range(0, len(az)):
        if (az[beam] >= angle - angular_res/2) and (az[beam] <= angle + angular_res/2):
            #print(az[beam], angle, angular_res/2)
            idx = np.argmax(ping[:, beam])
            if ping[idx, beam] > INTENSITY_THRESHOLD:
                # beam range
                br = idx*msg.range_resolution
                return br
            else:
                return False # TODO: return NaN or something else, integrate with gapslam

class ClosedSetDetector:
    """
    This holds an instance of YoloV8 and runs inference
    """
    def __init__(self) -> None:
        assert torch.cuda.is_available()
        model_file = "/home/haydentedrake/Downloads/last.pt"
        self.model = YOLO(model_file)
        rospy.loginfo("Model loaded")
        self.img_pub = rospy.Publisher("/camera/yolo_img", RosImage, queue_size=10)
        self.sonar_img_pub = rospy.Publisher("/camera/sonar_img", RosImage, queue_size=10)
        self.br = CvBridge()
        # Set up synchronized subscriber
        # sdr bluerov params
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

    def forward_pass(self, rgb: Image, depth: OculusPing, sonar_image: Image) -> None:
        """
        Run YOLOv8 on the image and extract all segmented masks
        """
        print("IN CALLBACK")
        image_cv = self.br.imgmsg_to_cv2(rgb, desired_encoding="bgr8")
        # Run inference args: https://docs.ultralytics.com/modes/predict/#inference-arguments
        #results = self.model(image_cv, verbose=False, conf=CONF_THRESH, imgsz=(736, 1280))[0] # do this for realsense (img dim not a multiple of max stride length 32)
        CONF_THRESH=0.2
        results = self.model(image_cv, verbose=False, conf=CONF_THRESH)[0]
        # Extract segmentation masks
        if (results.boxes is None):
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
            # ---- Object Vector ----
            class_id = int(class_id)
            print(conf)
            print(class_id)
            obj_centroid = (bbox[0], bbox[1])  # x, y
            bearing = obj_centroid[0]/rgb.width * CAM_FOV - CAM_FOV/2
            print(obj_centroid)
            print(bearing)
            range = ping_to_range(depth, bearing)
            
            if range:
                angle = to_rad(bearing)
                x = range * np.cos(angle)
                y = range * np.sin(angle)

                center_x = int(x / depth.range_resolution)
                center_y = int((depth.num_ranges - (range / depth.range_resolution)))

                bbox_size = 100
                top_left = (center_x - bbox_size // 2, center_y - bbox_size // 2)
                bottom_right = (center_x + bbox_size // 2, center_y + bbox_size // 2)
            
                cv2.rectangle(sonar_image_cv, top_left, bottom_right, (0, 255, 0), 2)
                label = f"Class {class_id}"
                cv2.putText(sonar_image_cv, label, (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        msg_sonar_detections = self.br.cv2_to_imgmsg(sonar_image_cv, encoding="bgr8")
        msg_sonar_detections.header.stamp = sonar_image.header.stamp
        self.sonar_img_pub.publish(msg_sonar_detections)

if __name__ == "__main__":
    rospy.init_node("closed_set_detector")
    detector = ClosedSetDetector()
    bridge = CvBridge()
    rospy.spin()