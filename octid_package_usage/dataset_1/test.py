from octid import octid
classify_model = octid.octid(model = 'mobilenet_v2', customised_model = False, feature_dimension = 3, outlier_fraction_of_SVM = 0.03, training_dataset='small_samples/training_dataset', validation_dataset='small_samples/validation_dataset', unlabeled_dataset='small_samples/testing_dataset')
classify_model()