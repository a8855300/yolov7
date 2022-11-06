#! /usr/bin/env python3
import argparse
from cmath import atan
import time
from pathlib import Path
from xml.etree.ElementTree import PI

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# ros libs
import rospy
import rospkg

# ros messages
from sensor_msgs.msg import Image, CompressedImage  
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker,MarkerArray

import numpy as np
import math
import yaml
import os

bridge = CvBridge()
default_devices = ""
ros_pkg_path = rospkg.RosPack().get_path("yolov7")

default_weight = os.path.join(ros_pkg_path, "src/weight/yolov7.pt")
default_img_size = 640
default_augment = False
default_conf_thres = 0.25
default_iou_thres = 0.45
default_agnostic_nms = False
xy = None
    



def callback(msg):
    global xy, xyz, wh, confident, classes, x, y, z, im0, cls, list_classes
    
    img = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
    im0 = img.copy()
    im_input = img.copy()
    # cv2.imshow("Image window", img)
    # cv2.waitKey(1)

    with torch.no_grad():

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1   


        # Detect
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        # list_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 35, 36, 38, 39, 40, 41, 42, 43, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 75, 76, 77, 78]
        # list_classes = [49, 50, 51]
        
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes = list_classes, agnostic=agnostic_nms)
        t3 = time_synchronized()
        bbox_xy = []


        for i, det in enumerate(pred):  # detections per image 
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    label = f'{names[int(cls)]} {conf:.2f}'
                    # if label.split(" ")[0] == "person":
                    #     print(int(cls))
                    # print(label)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    # print(xyxy[0], xyxy[1], xyxy[2], xyxy[3])

                # Stream results
        objects = pred[0].cpu().numpy()
        
        # print(objects)
        wh = (objects[:,2:4]-objects[:,:2]).astype(int)
        xy = ((objects[:,:2]+objects[:,2:4])/2).astype(int)
        confident = objects[:,4]
        classes = objects[:,5]
        # print(int(xyxy[0].item()), int(xyxy[2].item()), int(xyxy[1].item()), int(xyxy[3].item()))

        x0 = 0
        x1 = im0.shape[0]
        y0 = 0
        y1 = im0.shape[1]

        if len(objects) != 0:
            print("person!")
            x0 = int(objects[:,1][0])
            x1 = int(objects[:,3][0])
            y0 = int(objects[:,0][0])
            y1 = int(objects[:,2][0])
            dx = x1-x0
            dy = y1-y0
            # bigger 10 %
            ddx = dx * 0.1/2
            ddy = dy * 0.1/2
            if x0-ddx<0:
                x0 = 0
            else:
                x0 = x0-ddx
            if y0-ddy<0:
                y0 = 0
            else:
                y0 = y0-ddy
            if x1+ddx>im0.shape[0]:
                x1 = im0.shape[0]
            else:
                x1 = x1+ddx
            if y1+ddy>im0.shape[1]:
                y1 = im0.shape[1]
            else:
                y1 = y1+ddy
            
            # square
            dx = x1-x0
            dy = y1-y0
            diff = 0
            if dy<dx:
                diff = dx-dy
                if x0-diff/2<0:
                    x0 = 0
                else:
                    x0 = x0-diff/2
                if x1+diff/2>im0.shape[0]:
                    x1 = im0.shape[0]
                else:
                    x1 = x1+diff/2
            else:
                diff = dy-dx
                if y0-diff/2<0:
                    y0 = 0
                else:
                    y0 = y0-diff/2
                if y1+diff/2>im0.shape[1]:
                    y1 = im0.shape[1]
                else:
                    y1 = y1+diff/2

            cropped_image = im_input[int(x0):int(x1), int(y0):int(y1)]
            # dx = x1-x0
            # dy = y1-y0
            # print(dx, dy)
            # print(xy)
            cv2.imshow("yolov7", cropped_image)
            cv2.waitKey(1)  # 1 millisecond

            detect_human_msg = bridge.cv2_to_compressed_imgmsg(cropped_image)
            detect_human_msg.header = msg.header
            detect_human_pub.publish(detect_human_msg)

        
        

if __name__ == '__main__':
    
    devices = rospy.get_param("detect_devices", default=default_devices)
    weight = rospy.get_param("detect_weight", default=default_weight)
    img_size = rospy.get_param("detect_img_size", default=default_img_size)
    augment=rospy.get_param("detect_augment", default=default_augment)
    conf_thres=rospy.get_param("detect_conf_thres", default=default_conf_thres)
    iou_thres=rospy.get_param("detect_iou_thres", default=default_iou_thres)
    agnostic_nms=rospy.get_param("detect_agnostic_nms", default=default_agnostic_nms)


    #check_requirements(exclude=('pycocotools', 'thop'))


    # Initialize
    set_logging()
    device = select_device(devices)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    weights = weight
    imgsz = img_size

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    ros_pkg_path = rospkg.RosPack().get_path("yolov7")
    
    with open(os.path.join(ros_pkg_path, "config/class_colors.yaml"), 'r') as stream:
        dict_class_colors = yaml.safe_load(stream)
    colors = [dict_class_colors[name] for name in names]

    list_classes = [0] # onlu detect human

    rospy.init_node('yolov7', anonymous=True)
    rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, queue_size = 1, callback=callback)
    detect_human_pub = rospy.Publisher("yolo_detect_human/compressed", CompressedImage, queue_size=1)

    rospy.spin()
