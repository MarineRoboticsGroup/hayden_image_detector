import pathlib
import message_filters
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import CameraInfo, Image
from sonar_oculus.msg import OculusPing
from ultralytics import YOLO
from PIL import Image as PILImage
from sensor_msgs.msg import Image as RosImage
import sys
import yaml

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

if __name__ == "__main__":
    rospy.init_node("closed_set_detector")
    detector = ClosedSetDetector()
    bridge = CvBridge()
    rospy.spin()