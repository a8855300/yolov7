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
from yolov7.msg import BoundingBox, BoundingBoxArray

import numpy as np
import math
import yaml

bridge = CvBridge()
default_devices = ""
default_weight = "/home/ariel/catkin_ws/src/yolov7/src/weight/yolov7-e6e.pt"
default_img_size = 640
default_augment = False
default_conf_thres = 0.25
default_iou_thres = 0.45
default_agnostic_nms = False
xy = None

def MakeMarker(type, id, msg, marker_array, name):
    global xy, xyz, wh, confident, classes, x, y, z

    marker = Marker()     
    marker.id = id
    marker.ns = name
    marker.header.frame_id = "camera"
    marker.header.stamp = msg.header.stamp
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.type = type
    if type == Marker.TEXT_VIEW_FACING:
        marker.text = f'{names[int(classes[id])]}'
        marker.pose.position.x = x
        marker.pose.position.y = y-0.1
        marker.pose.position.z = z


    marker_array.markers.append(marker)
    



def callback(msg):
    global xy, xyz, wh, confident, classes, x, y, z, im0, cls
    
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
        list_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 35, 36, 38, 39, 40, 41, 42, 43, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 75, 76, 77, 78]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes = list_classes, agnostic=agnostic_nms)
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
        wh = (objects[:,2:4]-objects[:,:2]).astype(int)
        xy = ((objects[:,:2]+objects[:,2:4])/2).astype(int)
        confident = objects[:,4]
        classes = objects[:,5]

        
        # print(xy)
        cv2.imshow("yolov7", im0)
        cv2.waitKey(1)  # 1 millisecond

        

def callback2(msg):
    global xy, xyz, wh, confident, classes, x, y, z, fake_img, im0, cls

    

    img = bridge.imgmsg_to_cv2(msg, "32FC1")
    fake_img = np.zeros((640, 640, 3), dtype=np.uint8)
    

    if xy is not None:
        
        depth = img[xy[:,1],xy[:,0]] #opencv dimension 2 = x
        depth = depth[:, np.newaxis] 
        xyz = np.concatenate((xy,depth),1)

        
        marker_array = MarkerArray()
        bounding_box_array = BoundingBoxArray()

        # add a fake marker to let rviz delete the previous markers
        fake_marker = Marker()
        fake_marker.action = Marker.DELETEALL
        marker_array.markers.append(fake_marker)

        # print(img)

        for id, position in enumerate(xyz):
            
            x = xyz[id,2]*(xyz[id,0]-320)/320
            y = xyz[id,2]*(xyz[id,1]-320)/320
            z = xyz[id,2]
            w = xyz[id,2]*(wh[id,1])/320
            h = xyz[id,2]*(wh[id,0])/320

            # Markers Publish
            MakeMarker(Marker.SPHERE, id, msg, marker_array, "SPHERE")
            MakeMarker(Marker.TEXT_VIEW_FACING, id, msg, marker_array, "TEXT")


            #Bounding Box Publish
            bounding_box = BoundingBox()
            bounding_box.x = x
            bounding_box.y = y
            bounding_box.z = z
            bounding_box.w = w
            bounding_box.h = h
            bounding_box.confident = confident[id]
            bounding_box.classes = f'{names[int(classes[id])]}'
            bounding_box_array.boundingboxes.append(bounding_box) 
            color = colors[int(classes[id])]
            # print(color)
            # class_colors[f'{names[int(classes[id])]}'] = color
            # fake_img[int(xyz[id,1]-wh[id,1]/2):int(xyz[id,1]+wh[id,1]/2), int(xyz[id,0]-wh[id,0]/2):int(xyz[id,0]+wh[id,0]/2)] = 1
            cv2.rectangle(fake_img,
             (int(xyz[id,0]-wh[id,0]/2), int(xyz[id,1]-wh[id,1]/2)),
                (int(xyz[id,0]+wh[id,0]/2), int(xyz[id,1]+wh[id,1]/2)), color,-1)

        # cv2.imshow("fake_img", fake_img)
        # cv2.waitKey(1)  # 1 millisecond
       
        fake_img_msg = bridge.cv2_to_imgmsg(fake_img, encoding="bgr8")
        fake_img_msg.header = msg.header
        
        
        object_xyz_pub.publish(marker_array)
        bounding_box_pub.publish(bounding_box_array)
        fake_image_pub.publish(fake_img_msg)

        
        # print(class_colors)

        
        

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
    with open("/home/ariel/catkin_ws/src/yolov7/config/class_colors.yaml", 'r') as stream:
        dict_class_colors = yaml.safe_load(stream)
    colors = [dict_class_colors[name] for name in names]


    # for num,i in enumerate(names):    
    #     class_colors[f'{i}'] = colors[num]

    rospy.init_node('yolov7', anonymous=True)
    rospy.Subscriber("image", Image, callback=callback)
    rospy.Subscriber("depth", Image, callback=callback2)
    object_xyz_pub = rospy.Publisher("object_position", MarkerArray, queue_size=1)
    bounding_box_pub = rospy.Publisher("bounding_box", BoundingBoxArray, queue_size=1)
    fake_image_pub = rospy.Publisher("fake_image", Image, queue_size=1)

    rospy.spin()
