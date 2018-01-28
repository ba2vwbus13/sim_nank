#!/usr/bin/env python3

import time
from jetbot import ObjectDetector
following_model_path = "/home/jetbot/catkin_ws/src/sim_nank/weights/ssd_mobilenet_v2_coco.engine"
detection_model = ObjectDetector(following_model_path)
print('object make success {}'.format(detection_model))
time.sleep(5)
'''
class Controller:

    def __init__(self):
        from jetbot import ObjectDetector
        following_model_path = "/home/jetbot/catkin_ws/src/sim_nank/weights/ssd_mobilenet_v2_coco.engine"
        self.detection_model = ObjectDetector(following_model_path)   

if __name__ == '__main__':
    oc = Controller()
'''