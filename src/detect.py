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
from sensor_msgs.msg import Image   
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker,MarkerArray
from yolov7.msg import BoundBox, BoundBoxArray

import numpy as np
import math

bridge = CvBridge()
default_devices = ""
default_weight = "/home/ariel/catkin_ws/src/yolov7/src/weight/yolov7-e6e.pt"
default_img_size = 640
default_augment = False
default_conf_thres = 0.25
default_iou_thres = 0.45
default_agnostic_nms = False
xy = None

def callback(msg):
    global xy, xyz
    img = bridge.imgmsg_to_cv2(msg, "bgr8")
    im0 = img.copy()
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
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=agnostic_nms)
        t3 = time_synchronized()



        for i, det in enumerate(pred):  # detections per image 
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Stream results
        objects = pred[0].cpu().numpy()
        
        # print(objects)
        xy = ((objects[:,:2]+objects[:,2:4])/2).astype(int)
        # print(xy)


        cv2.imshow("yolov7", im0)
        cv2.waitKey(1)  # 1 millisecond

def callback2(msg):
    global xy, xyz
    img = bridge.imgmsg_to_cv2(msg, "32FC1")
    if xy is not None:
        
        depth = img[xy[:,1],xy[:,0]]
        depth = depth[:, np.newaxis] 
        xyz = np.concatenate((xy,depth),1)

        
        marker_array = MarkerArray()

        fake_marker = Marker()
        fake_marker.action = Marker.DELETEALL
        marker_array.markers.append(fake_marker)

        for id, position in enumerate(xyz):
            object_xyz = Marker()   
            object_xyz.id = id
            object_xyz.header.frame_id = "camera"
            object_xyz.header.stamp = msg.header.stamp
            theta_yz = 3.1415926/2-math.atan((xyz[id,0]-320)/320)
            theta_xz = 3.1415926/2-math.atan((xyz[id,1]-320)/320)
            object_xyz.pose.position.x = xyz[id,2]/math.tan(theta_yz)
            object_xyz.pose.position.y = xyz[id,2]/math.tan(theta_xz)
            object_xyz.pose.position.z = xyz[id,2]
            object_xyz.pose.orientation.w = 1.0
            object_xyz.scale.x = 0.1
            object_xyz.scale.y = 0.1
            object_xyz.scale.z = 0.1
            object_xyz.color.a = 1.0
            object_xyz.color.r = 0.0
            object_xyz.color.g = 1.0
            object_xyz.color.b = 0.0
            object_xyz.type = Marker.SPHERE
            marker_array.markers.append(object_xyz)
        
        object_xyz_pub.publish(marker_array)
        # print(xyz)
        

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
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


    rospy.init_node('yolov7', anonymous=True)
    rospy.Subscriber("image", Image, callback=callback)
    rospy.Subscriber("depth", Image, callback=callback2)
    object_xyz_pub = rospy.Publisher("object_position", MarkerArray, queue_size=1)

    rospy.spin()
