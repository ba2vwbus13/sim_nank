import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np

class ObjectDetector:
    def __init__(self, model_path):
        self.device = select_device('0')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        self.model = attempt_load(model_path, map_location=self.device)  # load FP32 model
        if self.half:
            self.model.half()  # to FP16

        self.stride = int(self.model.stride.max())  # model stride
        imgsz = 640
        self.img_size = check_img_size(imgsz, s=self.stride)  # check img_size

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = False

    def __call__(self, img0):
        if img0 is None:
            return None
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment='True')[0]
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

        return pred, img, img0

    def print_detections(self, pred, img, im0):
        # Process detections
        det = pred[0]
        s = '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        print("normal gain whwh :{}".format(gn))
        print("det :{}".format(det))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
            print("s :{}".format(s))
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)
        return im0

if __name__ == '__main__':

    model_path = '/home/nakahira/nakahira/catkin_ws/src/sim_nank/weights/yolov5m.pt'
    oc = ObjectDetector(model_path)
    capcher_file = 0
    cap = cv2.VideoCapture(capcher_file)   # capture from camera
    if not cap.isOpened():
        raise ImportError("Couldn't open video file or webcam.")

    while True:
        ret,im = cap.read()
        if im is None:
            continue     
        pred, img, im0 = oc(im)
        im0 = oc.print_detections(pred, img, im0)
        cv2.imshow("yolov5 Result", im0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    cap.release()
    cv2.destroyAllWindows()
