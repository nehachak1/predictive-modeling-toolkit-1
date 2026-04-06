import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression
from src.methods.knn import KNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """


    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ## 1. We first load the data.

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'],feature_data['xtest'],feature_data['ytrainreg'],
        feature_data['ytestreg'],feature_data['ytrainclassif'],feature_data['ytestclassif']
    )

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        num_train = int(0.8 * len(train_features))

        # Split features
        xvalid = train_features[num_train:]
        train_features = train_features[:num_train]

        # Split labels (classification)
        yvalid_classif = train_labels_classif[num_train:]
        train_labels_classif = train_labels_classif[:num_train]

        # Split labels (regression)
        yvalid_reg = train_labels_reg[num_train:]
        train_labels_reg = train_labels_reg[:num_train]

        # Replace test set with validation set
        test_features = xvalid
        test_labels_classif = yvalid_classif
        test_labels_reg = yvalid_reg

    means = np.mean(train_features, axis=0)
    stds = np.std(train_features, axis=0)
    stds[stds == 0] = 1  # avoid division by zero

    train_features = normalize_fn(train_features, means, stds)
    test_features = normalize_fn(test_features, means, stds)

    # Add bias term (for linear/logistic regression)
    if args.method in ["logistic_regression", "linear_regression"]:
        train_features = append_bias_term(train_features)
        test_features = append_bias_term(test_features)

    ## 3. Initialize the method you want to use.

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "knn":
        method_obj = KNN(k=args.K, task_kind=args.task, distance_metric=args.distance_metric)

    elif args.method == "logistic_regression":
        ### WRITE YOUR CODE HERE
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)

    elif args.method == "linear_regression":
        ### WRITE YOUR CODE HERE
        method_obj = LinearRegression(lr=args.lr, max_iters=args.max_iters)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method

    if args.task == "classification":
        assert args.method != "linear_regression", f"You should use linear regression as a regression method"
        # Fit the method on training data
        preds_train = method_obj.fit(train_features, train_labels_classif)

        # Predict on unseen data
        preds = method_obj.predict(test_features)

        # Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, train_labels_classif)
        macrof1 = macrof1_fn(preds_train, train_labels_classif)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, test_labels_classif)
        macrof1 = macrof1_fn(preds, test_labels_classif)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    elif args.task == "regression":
        assert args.method != "logistic_regression", f"You should use logistic regression as a classification method"
        # Fit the method on training data
        preds_train = method_obj.fit(train_features, train_labels_reg)

        # Predict on unseen data
        preds = method_obj.predict(test_features)

        # Report results: MSE on train and valid/test sets
        train_mse = mse_fn(preds_train, train_labels_reg)
        print(f"\nTrain set: MSE = {train_mse:.6f}")

        test_mse = mse_fn(preds, test_labels_reg)
        print(f"Test set:  MSE = {test_mse:.6f}")

    else:
        raise ValueError(f"Unknown task: {args.task}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.

    # Hyperparameter search for logistic regression

    if args.method == "logistic_regression" and not args.test:
        print("\n--- Logistic Regression Hyperparameter Search ---")

        max_iters_list = [50, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
        lr_list = [1e-4, 1e-3, 1e-2, 0.1]

        best_f1 = 0
        best_acc = 0
        best_config = None
        results = [] 

        for max_iter in max_iters_list:
            for lr in lr_list:
                model = LogisticRegression(lr=lr, max_iters=max_iter)

                model.fit(train_features, train_labels_classif)
                preds = model.predict(test_features)

                acc = accuracy_fn(preds, test_labels_classif)
                f1 = macrof1_fn(preds, test_labels_classif)

                print(f"max_iter={max_iter}, lr={lr} → Acc={acc:.3f}, F1={f1:.4f}")

                results.append((max_iter, lr, acc, f1)) 

                if f1 > best_f1:
                    best_f1 = f1
                    best_acc = acc
                    best_config = (max_iter, lr)

        print("\n--- Best Configuration ---")
        print(f"max_iter={best_config[0]}, lr={best_config[1]}")
        print(f"Validation Accuracy: {best_acc:.3f}")
        print(f"Validation F1-score: {best_f1:.4f}")

        fixed_iter = best_config[0]

        lrs = []
        val_acc = []
        val_f1 = []

        for max_iter, lr, acc, f1 in results:
            if max_iter == fixed_iter:
                lrs.append(lr)
                val_acc.append(acc)
                val_f1.append(f1 * 100)  

        train_acc = [accuracy_fn(preds_train, train_labels_classif)] * len(lrs)
        train_f1 = [macrof1_fn(preds_train, train_labels_classif) * 100] * len(lrs)

        plt.figure(figsize=(8,6))

        plt.plot(lrs, train_acc, 'o--', label="Train Accuracy (%)")
        plt.plot(lrs, train_f1, 's--', label="Train F1-score (%)")

        plt.plot(lrs, val_f1, 's-', label="Validation F1-score (%)")
        plt.plot(lrs, val_acc, 'o-', label="Validation Accuracy (%)")

        plt.xscale("log")
        plt.xlabel("Value of Learning Rate (lr)")
        plt.ylabel("Score (%)")
        plt.title("Logistic Regression - Accuracy & F1-score vs. Learning Rate")

        plt.legend()
        plt.grid(True)

        plt.show()

        fixed_lr = best_config[1]  

        iters = []
        val_acc_2 = []
        val_f1_2 = []

        for max_iter, lr, acc, f1 in results:
            if lr == fixed_lr:
                iters.append(max_iter)
                val_acc_2.append(acc)
                val_f1_2.append(f1 * 100)  # scale F1

        train_acc_2 = [accuracy_fn(preds_train, train_labels_classif)] * len(iters)
        train_f1_2 = [macrof1_fn(preds_train, train_labels_classif) * 100] * len(iters)

        plt.figure(figsize=(8,6))

        plt.plot(iters, train_acc_2, 'o--', label="Train Accuracy (%)")
        plt.plot(iters, train_f1_2, 's--', label="Train F1-score (%)")

        plt.plot(iters, val_f1_2, 's-', label="Validation F1-score (%)")
        plt.plot(iters, val_acc_2, 'o-', label="Validation Accuracy (%)")

        plt.xlabel("Number of Iterations (max_iters)")
        plt.ylabel("Score (%)")
        plt.title("Logistic Regression - Accuracy & F1-score vs. Max Iterations")

        plt.legend()
        plt.grid(True)

        plt.show()

        return  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression",
    )
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / knn / logistic_regression / linear_regression",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="number of neighboring datapoints used for knn",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, "
             "otherwise use a validation set",
    )
    parser.add_argument(
        "--distance_metric",
        type=str,
        default="euclidean",
        help="euclidean / manhattan",
    )
    # Feel free to add more arguments here if you need!

    args = parser.parse_args()
    main(args)
