import time
import numpy as np

import cv2

import mxnet as mx
from gluoncv import utils

from autogluon.vision import ObjectDetector

detector = ObjectDetector.load('detector.ag')

# Load the webcam handler
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# letting the camera autofocus
time.sleep(1)

while(True):
    # Load frame from the camera
    ret, frame = cap.read()

    # Image pre-processing
    mx_frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
    
    result = detector.predict(mx_frame)

    selected_result = result.query('predict_class=="head" & predict_score > 0.8')

    bounding_boxes = [[x[i] for i in x.keys()] for x in list(selected_result['predict_rois'])]

    blurred_img = cv2.blur(frame, (20,20))
    mask = np.zeros(shape=(frame.shape[0], frame.shape[1], 1)).astype('uint8')

    for i in range(len(selected_result)):
        x_min = int(bounding_boxes[i][0]*frame.shape[1])
        y_min = int(bounding_boxes[i][1]*frame.shape[0])
        x_max = int(bounding_boxes[i][2]*frame.shape[1])
        y_max = int(bounding_boxes[i][3]*frame.shape[0])

        mask = cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)

    # Display the result
    mask_not = cv2.bitwise_not(mask)
    img1 = cv2.bitwise_and(frame, frame, mask=mask_not)
    img2 = cv2.bitwise_and(blurred_img, blurred_img, mask=mask)

    final_img = cv2.add(img1, img2)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    
    utils.viz.cv_plot_image(final_img)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()