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
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
    
    result = detector.predict(frame)

    class_ids , class_names = result['predict_class'].factorize()

    bounding_boxes = np.array([[x[i] for i in x.keys()] for x in result['predict_rois']])

    scores = np.array(result['predict_score'])

    # Display the result
    img = utils.viz.cv_plot_bbox(frame, bounding_boxes, scores=scores, 
                                labels=class_ids, class_names=class_names,
                                absolute_coordinates=False,
                                thresh=0.7)

    utils.viz.cv_plot_image(img)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()