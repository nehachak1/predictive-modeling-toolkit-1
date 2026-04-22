import numpy as np


class LinearRegression(object):
    """
    Linear regression.
    """

    def __init__(self, regularization=None, lambda_=0.0):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.
        """
        self.regularization = regularization
        self.lambda_ = lambda_
        self.theta = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: You can use the closed-form solution for linear regression
        (with or without regularization). Remember to handle the bias term.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        X = np.asarray(training_data, dtype=float)
        y = np.asarray(training_labels, dtype=float).reshape(-1)

        if self.regularization is None:
            self.theta = np.linalg.pinv(X) @ y

        elif self.regularization == "l2":
            if self.lambda_ <= 0:
                raise ValueError("lambda_ must be > 0 when regularization is 'l2'.")

            D = X.shape[1]
            I = np.eye(D)

            # do not regularize the bias term if bias was appended as first column
            I[0, 0] = 0

            self.theta = np.linalg.solve(X.T @ X + self.lambda_ * I, X.T @ y)

        else:
            raise ValueError("regularization must be None or 'l2'.")

        pred_labels = self.predict(X)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        if self.theta is None:
            raise ValueError("Model must be fitted before calling predict().")

        X = np.asarray(test_data, dtype=float)
        pred_labels = X @ self.theta
        return pred_labels