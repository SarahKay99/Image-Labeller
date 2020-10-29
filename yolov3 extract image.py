# https://stackoverflow.com/questions/54520996/yolo-v3-how-to-extract-an-image-of-a-detected-object
import cv2
import detect as dt
from darknet import Darknet
from PIL import Image

vidcap = cv2.VideoCapture('your/video/path.mp4')
success, image = vidcap.read()
count = 0

m = Darknet('your/cfg/file/path.cfg')
m.load_weights('weight/file/path.weights')
use_cuda = 1
m.cuda()

while success:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(image)
    im_pil = im_pil.resize((m.width, m.height))
    boxes = dt.do_detect(m, im_pil, 0.5, 0.4, use_cuda)

    result = open('your/save/file/path/frame%04d.txt'%(count), 'w')
    for i in range(len(boxes)):
        result.write(boxes[i])
    count = count + 1
    success, image = vidcap.read()
    result.close()