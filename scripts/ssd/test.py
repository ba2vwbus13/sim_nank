from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
 
from vision.utils.misc import Timer
import cv2
import sys
 
if len(sys.argv) < 4:
    print('Usage: python run_ssd_example.py <net type>  <model path> [video file]')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
 
if len(sys.argv) >= 5:
    cap = cv2.VideoCapture(sys.argv[4])  # capture from file
else:
    cap = cv2.VideoCapture(0)   # capture from camera
    if not cap.isOpened():
        raise ImportError("Couldn't open video file or webcam.")
 
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
 
if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)
 
if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
 
 
timer = Timer()
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    interval = timer.end()
 
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    # Draw FPS
    fps = "FPS:" + str(int(1/interval))
    cv2.rectangle(orig_image, (0, 0), (50, 17), (0, 0, 0), -1)
    cv2.putText(orig_image, fps, (0, 10), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
 
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        box = box.to('cpu').detach().numpy().copy()
        box = [int(i) for i in box]
        print(box)
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        print(label)
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)
        cv2.putText(orig_image, label,
                    (box[0]+10, box[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
 
    cv2.imshow("SSD Result", orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()