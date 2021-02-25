from gluoncv import utils
from gluoncv.data import VOCDetection
from matplotlib import pyplot as plt

VOCDetection.CLASSES = ['head']
train_dataset = VOCDetection(root='VOCdevkit', splits=((2012,'train'),))

train_image, train_label = train_dataset[10]

bounding_boxes = train_label[:, :4]
class_ids = train_label[:, 4:5]

utils.viz.plot_bbox(train_image.asnumpy(), bounding_boxes, scores=None,
                    labels=class_ids, class_names=['face'])

plt.show()