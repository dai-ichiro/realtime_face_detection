import numpy as np
from matplotlib import pyplot as plt

from mxnet import image
from gluoncv import utils

from autogluon.vision import ObjectDetector

detector = ObjectDetector.load('detector.ag')

img_file = utils.download( 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/mhpv1_examples/1.jpg')
image_array = image.imread(img_file)

result = detector.predict(image_array)
selected_result = result.query('predict_score > 0.8')

class_ids , class_names = selected_result['predict_class'].factorize()

bounding_boxes = np.array([[x[i] for i in x.keys()] for x in selected_result['predict_rois']])

scores = np.array(selected_result['predict_score'])

utils.viz.plot_bbox(image_array, bounding_boxes, scores=scores,
                    labels=class_ids, class_names = class_names, absolute_coordinates=False)

plt.show()