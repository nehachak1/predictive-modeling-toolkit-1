import numpy as np

from ..utils import label_to_onehot, onehot_to_label, accuracy_fn


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.weight = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        training_labels = label_to_onehot(training_labels)
        self.weight = self.__logistic_regression_train_multi(training_data, training_labels, max_iters=self.max_iters, lr=self.lr)
        return self.__logistic_regression_predict_multi(training_data, self.weight)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        if self.weight is None: 
            raise ValueError("Model must be fitted before calling predict().")
        
        return self.__logistic_regression_predict_multi(test_data, self.weight)

    @staticmethod
    def __f_softmax(data, W):
        """
        Softmax function for multi-class logistic regression.
        
        Args:
            data (array): Input data of shape (N, D)
            W (array): Weights of shape (D, C) where C is the number of classes
        Returns:
            array of shape (N, C): Probability array where each value is in the
                range [0, 1] and each row sums to 1.
                The row i corresponds to the prediction of the ith data sample, and 
                the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
        """
        z = np.dot(data, W)
        z = z - np.max(z, axis=1, keepdims=True)

        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def __gradient_logistic_multi(self, data, labels, W):
        """
        Compute the gradient of the entropy for multi-class logistic regression.
        
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            W (array): Weights of shape (D, C)
        Returns:
            grad (np.array): Gradients of shape (D, C)
        """
        return (data.T @ (self.__f_softmax(data, W) - labels)) / data.shape[0]

    def __logistic_regression_predict_multi(self, data, W):
        """
        Prediction the label of data for multi-class logistic regression.
        
        Args:
            data (array): Dataset of shape (N, D).
            W (array): Weights of multi-class logistic regression model of shape (D, C)
        Returns:
            array of shape (N,): Label predictions of data.
        """
        soft = self.__f_softmax(data, W)
        return np.argmax(soft, axis=1)

    def __logistic_regression_train_multi(self, data, labels, max_iters, lr):
        """
        Training function for multi class logistic regression.
        
        Args:
            data (array): Dataset of shape (N, D).
            labels (array): Labels of shape (N, C)
            max_iters (int): Maximum number of iterations. Default: 10
            lr (float): The learning rate of  the gradient step. Default: 0.001
        Returns:
            weights (array): weights of the logistic regression model, of shape(D, C)
        """
        D = data.shape[1]
        C = labels.shape[1]

        # Random initialization of the weights
        weights = np.random.normal(0, 0.1, (D, C))

        for it in range(max_iters):
            gradient = self.__gradient_logistic_multi(data, labels, weights)
            old_weights = weights.copy()
            weights = weights - lr * gradient

            if np.linalg.norm(old_weights - weights) < 1e-6:
                print(f"LR has converged after {it+1} iterations!")
                break

            predictions = self.__logistic_regression_predict_multi(data, weights)
            if accuracy_fn(predictions, onehot_to_label(labels)) == 100:
                print(f"LR has converged after {it+1} iterations!")
                break
            
        return weights