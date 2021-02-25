from autogluon.vision import ObjectDetector

dataset_train = ObjectDetector.Dataset.from_voc('VOCdevkit/VOC2012', splits='train')

time_limit = 5*60*60  # 5 hour
detector = ObjectDetector()
hyperparameters = {'batch_size':4}
detector.fit(dataset_train, time_limit=time_limit, num_trials=2, hyperparameters=hyperparameters)

detector.save('detector.ag')