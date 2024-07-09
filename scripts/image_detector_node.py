#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Hack for now until we set the PYTHONPATH properly
sys.path.append('/home/haydentedrake/Desktop')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../src')
from cosegmentation import find_cosegmentation, draw_cosegmentation_binary_masks, draw_cosegmentation

def figure_to_numpy(fig):
    """Convert a Matplotlib figure to a numpy array"""
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    np_array = np.asarray(buf)
    return cv2.cvtColor(np_array, cv2.COLOR_RGBA2BGR)

class ImageDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.callback)
        self.detection_pub = rospy.Publisher("/detection/image_raw/compressed", CompressedImage, queue_size=10)
       
    def callback(self, data: CompressedImage) -> None:
        try:
            np_arr = np.frombuffer(data.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # Insert your detector processing here
            detections = self.detect_objects(cv_image)
            rospy.loginfo(f"Detections type: {type(detections)}, shape: {detections.shape}")
            compressed_image = CompressedImage()
            compressed_image.header = data.header
            compressed_image.format = "jpeg"
            compressed_image.data = np.array(cv2.imencode('.jpg', detections)[1]).tobytes()
            self.detection_pub.publish(compressed_image)
        except CvBridgeError as e:
            rospy.logerr("CvBridgeError: %s", e)
        except Exception as e:
            rospy.logerr("Error in callback: %s", e)
       
    def detect_objects(self, image: np.ndarray) -> np.ndarray:
        try:
            segmentation_masks, image_pil_list = find_cosegmentation(image, load_size=120, low_res_saliency_maps=False)
            rospy.loginfo(f"Segmentation masks type: {type(segmentation_masks)}, length: {len(segmentation_masks)}")
            rospy.loginfo(f"Image PIL list type: {type(image_pil_list)}, length: {len(image_pil_list)}")
            for i, img_pil in enumerate(image_pil_list):
                rospy.loginfo(f"Image {i} type: {type(img_pil)}")
            
            result_images = draw_cosegmentation(segmentation_masks, image_pil_list)
            rospy.loginfo(f"Result image type: {type(result_images)}")
            
            result_image = None

            if isinstance(result_images, list):
                rospy.loginfo(f"Result images list length: {len(result_images)}")
                for i, img in enumerate(result_images):
                    rospy.loginfo(f"Result image {i} type: {type(img)}")
                result_arrays = [figure_to_numpy(img) for img in result_images if isinstance(img, plt.Figure)]
                if result_arrays:
                    result_image = result_arrays[0]
                    rospy.loginfo(f"Converted result image type: {type(result_image)}, shape: {result_image.shape}")
                else:
                    rospy.logerr("No valid images in result list")
                    return image
            elif isinstance(result_images, Image.Image):
                rospy.loginfo("Converting PIL image to numpy array")
                result_image = figure_to_numpy(result_images)
                rospy.loginfo(f"Converted result image type: {type(result_image)}, shape: {result_image.shape}")
            else:
                rospy.logerr("Result image is not a numpy array")
                rospy.loginfo(f"Result image type: {type(result_image)}")
                result_image = image
            
            return result_image
            
        except Exception as e:
            rospy.logerr("Error in detect_objects: %s", e)
            return image
        

def main():
    rospy.init_node('image_detector', anonymous=True)
    detector = ImageDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")

if __name__ == '__main__':
    main()
