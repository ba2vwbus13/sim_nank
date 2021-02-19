#!/usr/bin/env python3

from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
#from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
#from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import os 

class ObjectDetector:

    def __init__(self, model_path):

        net_type = 'mb1-ssd'
        label_path = os.path.join(os.path.dirname(model_path), 'voc-model-labels.txt')
        self.class_names = [name.strip() for name in open(label_path).readlines()]
        class_length = len(self.class_names)
        if net_type == 'vgg16-ssd':
            net = create_vgg_ssd(class_length, is_test=True)
        elif net_type == 'mb1-ssd':
            net = create_mobilenetv1_ssd(class_length, is_test=True)
        elif net_type == 'mb1-ssd-lite':
            net = create_mobilenetv1_ssd_lite(class_length, is_test=True)
        else:
            print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
            sys.exit(1)
    
        if net_type == 'vgg16-ssd':
            predictor = create_vgg_ssd_predictor(net, candidate_size=200)
        elif net_type == 'mb1-ssd':
            predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
        elif net_type == 'mb1-ssd-lite':
            predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
        else:
            print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
            sys.exit(1)
            
        net.load(model_path)
        self.predictor = predictor


    def __call__(self, image):
        if image is None:
            return None

        (height, width, channel) = image.shape
        boxes, labels, probs = self.predictor.predict(image, 10, 0.2)

        detections = []
        for i in range(boxes.size(0)):
            boxes[i][0] /= width
            boxes[i][1] /= height
            boxes[i][2] /= width
            boxes[i][3] /= height
            list = {'bbox': boxes[i], 'label' : labels[i], 
            'label_name': self.class_names[labels[i]], 'prob' : probs[i]}
            detections.append(list)

        return [detections]

