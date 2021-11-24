from autogluon.vision import ObjectDetector

dataset_train = ObjectDetector.Dataset.from_voc('VOCdevkit/VOC2012', splits='train')

time_limit = 5*60*60  # 5 hour
detector = ObjectDetector()
hyperparameters = {'batch_size':4}
hyperparameter_tune_kwargs={'num_trials': 2}
detector.fit(dataset_train,
            time_limit = time_limit, 
            hyperparameters = hyperparameters,
            hyperparameter_tune_kwargs = hyperparameter_tune_kwargs)

detector.save('detector.ag')
