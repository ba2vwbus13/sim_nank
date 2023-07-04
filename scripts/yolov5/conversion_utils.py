#!/usr/bin/env python3
import array
from io import BytesIO
import sys
from PIL import Image
from PIL import ImageOps
import numpy as np
import cv2

try:
    import cairo
except ImportError:
    import cairocffi as cairo

def imgmsg_to_pil(img_msg, rgba=False):
    try:
        if img_msg._type == 'sensor_msgs/CompressedImage':
            pil_img = Image.open(BytesIO(img_msg.data))
            if pil_img.mode.startswith('BGR'):
                pil_img = pil_bgr2rgb(pil_img)
            pil_mode = 'RGB'
        else:
            pil_mode = 'RGB'
            if img_msg.encoding in ['mono8', '8UC1']:
                mode = 'L'
            elif img_msg.encoding == 'rgb8':
                mode = 'RGB'
            elif img_msg.encoding == 'bgr8':
                mode = 'BGR'
            elif img_msg.encoding in ['bayer_rggb8', 'bayer_bggr8', 'bayer_gbrg8', 'bayer_grbg8']:
                mode = 'L'
            elif img_msg.encoding in ['bayer_rggb16', 'bayer_bggr16', 'bayer_gbrg16', 'bayer_grbg16']:
                pil_mode = 'I;16'
                if img_msg.is_bigendian:
                    mode = 'I;16B'
                else:
                    mode = 'I;16L'
            elif img_msg.encoding == 'mono16' or img_msg.encoding == '16UC1':
                pil_mode = 'F'
                if img_msg.is_bigendian:
                    mode = 'F;16B'
                else:
                    mode = 'F;16'
            elif img_msg.encoding == '32FC1':
                pil_mode = 'F'
                if img_msg.is_bigendian:
                    mode = 'F;32BF'
                else:
                    mode = 'F;32F'
            elif img_msg.encoding == 'rgba8':
                mode = 'BGR'
            elif img_msg.encoding == 'bgra8':
                mode = 'RGB'
            else:
                raise Exception("Unsupported image format: %s" % img_msg.encoding)
            pil_img = Image.frombuffer(
                pil_mode, (img_msg.width, img_msg.height), img_msg.data, 'raw', mode, 0, 1)

        # 16 bits conversion to 8 bits
        if pil_mode == 'I;16':
            pil_img = pil_img.convert('I').point(lambda i: i * (1. / 256.)).convert('L')

        if pil_img.mode == 'F':
            pil_img = pil_img.point(lambda i: i * (1. / 256.)).convert('L')
            pil_img = ImageOps.autocontrast(pil_img)
            pil_img = ImageOps.invert(pil_img)

        if rgba and pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')

        return pil_img

    except Exception as ex:
        #print('Can\'t convert image: %s' % ex, file=sys.stderr)
        print('Cant convert image')
        return None


def pil_to_cv(image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  #mono cro
        pass
    elif new_image.shape[2] == 3:  #color
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  #transpalent
        new_image = new_image[:, :, [2, 1, 0, 3]]
    return new_image

def pil_bgr2rgb(pil_img):
    rgb2bgr = (0, 0, 1, 0,
               0, 1, 0, 0,
               1, 0, 0, 0)
    return pil_img.convert('RGB', rgb2bgr)


def pil_to_cairo(pil_img):
    w, h = pil_img.size
    data = array.array('c')
    data.fromstring(pil_img.tostring())

    return cairo.ImageSurface.create_for_data(data, cairo.FORMAT_ARGB32, w, h)


def detection_center(detection):
    """Computes the center x, y coordinates of the object"""
    bbox = detection['bbox']
    center_x = (bbox[0] + bbox[2]) / 2.0 - 0.5
    center_y = (bbox[1] + bbox[3]) / 2.0 - 0.5
    return (center_x, center_y)

def norm(vec):
    """Computes the length of the 2D vector"""
    return np.sqrt(vec[0]**2 + vec[1]**2)

def closest_detection(detections):
    """Finds the detection closest to the image center"""
    closest_detection = None
    for det in detections:
        center = detection_center(det)
        if closest_detection is None:
            closest_detection = det
        elif norm(detection_center(det)) < norm(detection_center(closest_detection)):
            closest_detection = det
    return closest_detection

def rbox_to_box(bbox, image):
    (height, width, channel) = image.shape
    return (int(width * bbox[0]), int(height * bbox[1])), (int(width * bbox[2]), int(height * bbox[3]))

def bbox_to_rbbox(bbox, image):
    (height, width, channel) = image.shape
    return (bbox[0]/width, bbox[1]/height, bbox[2]/width, bbox[3]/height)

def bbox_to_roi(bbox):
    return (bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])

def roi_to_bbox(roi):
    return (roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3])

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)