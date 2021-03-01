import numpy as np
from matplotlib import pyplot as plt

from mxnet import image
from gluoncv import utils

from autogluon.vision import ObjectDetector

import cv2
detector = ObjectDetector.load('detector.ag')

img_file = utils.download( 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/mhpv1_examples/1.jpg')
image_array = image.imread(img_file)

result = detector.predict(image_array)

selected_result = result.query('predict_class=="head" & predict_score > 0.8')

bounding_boxes = [[x[i] for i in x.keys()] for x in selected_result['predict_rois']]

np_image = image_array.asnumpy()

blurred_img = cv2.blur(np_image, (20,20))

mask = np.zeros(shape=(np_image.shape[0], np_image.shape[1], 1)).astype('uint8')

for i in range(len(selected_result)):
    x_min = int(bounding_boxes[i][0]*np_image.shape[1])
    y_min = int(bounding_boxes[i][1]*np_image.shape[0])
    x_max = int(bounding_boxes[i][2]*np_image.shape[1])
    y_max = int(bounding_boxes[i][3]*np_image.shape[0])

    mask = cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)

mask_not = cv2.bitwise_not(mask)
img1 = cv2.bitwise_and(np_image, np_image, mask=mask_not)
img2 = cv2.bitwise_and(blurred_img, blurred_img, mask=mask)

final_img = cv2.add(img1, img2)

plt.imshow(final_img)
plt.show()
