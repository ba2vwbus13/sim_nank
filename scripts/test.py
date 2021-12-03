import torch
import tensorrt as trt
import atexit

engine_path="/home/jetbot/catkin_ws/src/sim_nank/weights/ssd_mobilenet_v2_coco.engine"
logger = trt.Logger()
runtime = trt.Runtime(logger)
with open(engine_path, 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
