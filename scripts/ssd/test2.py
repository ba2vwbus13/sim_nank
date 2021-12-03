#!/usr/bin/env python3

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
 
class ObjectDetector:

    def __init__(self):

        net_type = 'mb1-ssd'
        model_path = 'models/mobilenet-v1-ssd-mp-0_675.pth'
        label_path = 'models/voc-model-labels.txt'

        self.class_names = [name.strip() for name in open(label_path).readlines()]
        self.net = None
        self.net_initilize(net_type, model_path, self.class_names)

    def predictor(self, image):
        return None
        
    def net_initilize(self, net_type, model_path, class_names):
        if net_type == 'vgg16-ssd':
            net = create_vgg_ssd(len(class_names), is_test=True)
        elif net_type == 'mb1-ssd':
            net = create_mobilenetv1_ssd(len(class_names), is_test=True)
        elif net_type == 'mb1-ssd-lite':
            net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
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
            
        self.net = net
        self.net.load(model_path)
        self.pdc = predictor


    def predictor(self, image):
        if image is None:
            return None

        (height, width, channel) = image.shape
        boxes, labels, probs = self.pdc.predict(image, 10, 0.4)

        detections = []
        for i in range(boxes.size(0)):
            boxes[i][0] /= width
            boxes[i][1] /= height
            boxes[i][2] /= width
            boxes[i][3] /= height
            list = {'bbox': boxes[i], 'label' : labels[i], 
            'label_name': self.class_names[labels[i]], 'prob' : probs[i]}
            detections.append(list)

        return detections

if __name__ == '__main__':

    oc = ObjectDetector()

    print(oc)
  
    capcher_file = 0
    cap = cv2.VideoCapture(capcher_file)   # capture from camera
    if not cap.isOpened():
        raise ImportError("Couldn't open video file or webcam.")

    timer = Timer()
    while True:
        ret,image = cap.read()
        if image is None:
            continue
        timer.start()        
        detections = oc.predictor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        (height, width, channel) = image.shape
        interval = timer.end()
        for det in detections:
            bbox = det['bbox']
            label = det['label']
            label_name = det['label_name']
            prob = det['prob']
            label = f"{label_name}: {prob:.2f}"
            cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])), (int(width * bbox[2]), int(height * bbox[3])), (255, 0, 0), 2)
            cv2.putText(image, label,
                    (int(width*bbox[0]+10), int(height*bbox[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
        cv2.imshow("SSD Result", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    cap.release()
    cv2.destroyAllWindows()