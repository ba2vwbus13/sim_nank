import array
from io import BytesIO
import sys

from PIL import Image
from PIL import ImageOps
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
        print('Can\'t convert image: %s' % ex, file=sys.stderr)
        return None


def pil_to_cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  # 透過
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
