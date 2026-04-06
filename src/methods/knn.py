import numpy as np


class KNN(object):
    """
    kNN classifier object.
    """

    def __init__(self, k=1, task_kind="classification", distance_metric="euclidean"):
        """
        Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind
        self.distance_metric = distance_metric

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: Since KNN does not really have parameters to train, you can try saving
        the training_data and training_labels as part of the class. This way, when you
        call the "predict" function with the test_data, you will have already stored
        the training_data and training_labels in the object.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        self.training_data = training_data
        self.training_labels = training_labels

        pred_labels = self.predict(training_data)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        test_labels = []

        for x in test_data: 
            if self.distance_metric == "euclidean": 
                distances = np.sqrt(np.sum((self.training_data - x) ** 2, axis=1))
            elif self.distance_metric == "manhattan":
                distances = np.sum(np.abs(self.training_data - x), axis=1)
            else: 
                raise ValueError("distance_metric must be 'euclidean' or 'manhattan'")

            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.training_labels[nearest_indices]

            if self.task_kind == "classification":
                pred_label = np.bincount(nearest_labels.astype(int)).argmax()
            elif self.task_kind == "regression":
                pred_label = np.mean(nearest_labels)
            else: 
                raise ValueError("task_kind must be 'classification' or 'regression'")

            test_labels.append(pred_label)

        test_labels = np.array(test_labels)
        return test_labels
