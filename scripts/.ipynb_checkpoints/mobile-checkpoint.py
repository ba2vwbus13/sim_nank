import threading
import time
import numpy as np
from jetbot import Robot

class MobileController:

    loop = True
    controll_thread = None
    left_v = 0.0
    right_v = 0.0
    max_radius = 60.0
    gradient = 30

    def __init__(self, wheel_distance, robot: Robot):
        self.wheel_distance = wheel_distance
        self.robot = robot

    def _controll_loop(self):
        while self.loop:
            self.robot.set_motors(self.left_v, self.right_v)
            time.sleep(0.1)

    def run(self):
        self.controll_thread = threading.Thread(target=self._controll_loop)
        self.controll_thread.start()
        pass

    def stop(self):
        self.loop = False

    def controll(self, speed, radius):
        self.right_v = (speed + radius)*0.5
        self.left_v = (speed - radius)*0.5

