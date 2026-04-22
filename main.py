import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

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

    # Normalize features using training statistics only 
    means = np.mean(train_features, axis=0)
    stds = np.std(train_features, axis=0)
    stds[stds == 0] = 1  

    train_features = normalize_fn(train_features, means, stds)
    test_features = normalize_fn(test_features, means, stds)

    # Add bias term for linear/logistic regression 
    if args.method in ["logistic_regression", "linear_regression"]:
        train_features = append_bias_term(train_features)
        test_features = append_bias_term(test_features)

    ## 3. Initialize the method you want to use.

    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "knn":
        method_obj = KNN(k=args.K, task_kind=args.task, distance_metric=args.distance_metric)

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)

    elif args.method == "linear_regression":
        method_obj = LinearRegression(regularization=args.regularization, lambda_=args.lambda_)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method

    if args.task == "classification":
        assert args.method != "linear_regression", f"You should use linear regression as a regression method"
        # Fit the method on training data
        start_train = time.perf_counter()
        preds_train = method_obj.fit(train_features, train_labels_classif)
        end_train = time.perf_counter()

        # Predict on unseen data
        start_pred = time.perf_counter()
        preds = method_obj.predict(test_features)
        end_pred = time.perf_counter()

        print(f"\nTraining time:   {end_train - start_train:.4f} sec")
        print(f"Prediction time: {end_pred - start_pred:.4f} sec")

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
        start_train = time.perf_counter()
        preds_train = method_obj.fit(train_features, train_labels_reg)
        end_train = time.perf_counter()

        # Predict on unseen data
        start_pred = time.perf_counter()
        preds = method_obj.predict(test_features)
        end_pred = time.perf_counter()

        print(f"\nTraining time:   {end_train - start_train:.4f} sec")
        print(f"Prediction time: {end_pred - start_pred:.4f} sec")

        # Report results: MSE on train and valid/test sets
        train_mse = mse_fn(preds_train, train_labels_reg)
        print(f"\nTrain set: MSE = {train_mse:.6f}")

        test_mse = mse_fn(preds, test_labels_reg)
        print(f"Test set:  MSE = {test_mse:.6f}")

        unique_labels = np.unique(test_labels_reg)

        print("\n--- MSE per True Label ---")
        for val in unique_labels:
            mask = (test_labels_reg == val)
            mse_val = np.mean((test_labels_reg[mask] - preds[mask]) ** 2)
            print(f"Label {val}: MSE = {mse_val:.4f}")

    else:
        raise ValueError(f"Unknown task: {args.task}")

    ### Additional outputs / visualizations / hyperparameter search

    if args.show_plots and not args.test: 
        # ---- k-NN classification ----
        if args.method == "knn" and args.task == "classification":
            print("\n--- k-NN Classification Hyperparameter Search ---")

            k_list = list(range(1, 61))
            distance_metrics = ["euclidean", "manhattan"]

            for dist in distance_metrics:
                train_accs = []
                valid_accs = []
                train_f1s = []
                valid_f1s = []

                for k in k_list:
                    model = KNN(k=k, task_kind="classification", distance_metric=dist)

                    preds_train_plot = model.fit(train_features, train_labels_classif)
                    preds_test_plot = model.predict(test_features)

                    train_accs.append(accuracy_fn(preds_train_plot, train_labels_classif))
                    valid_accs.append(accuracy_fn(preds_test_plot, test_labels_classif))
                    train_f1s.append(macrof1_fn(preds_train_plot, train_labels_classif) * 100)
                    valid_f1s.append(macrof1_fn(preds_test_plot, test_labels_classif) * 100)

                plt.figure(figsize=(8, 6))
                plt.plot(k_list, train_accs, 'o--', label="Train Accuracy (%)")
                plt.plot(k_list, valid_accs, 'o-', label="Validation Accuracy (%)")
                plt.plot(k_list, train_f1s, 's--', label="Train F1-score (%)")
                plt.plot(k_list, valid_f1s, 's-', label="Validation F1-score (%)")
                plt.xlabel("Number of Neighbors (k)")
                plt.ylabel("Score (%)")
                plt.title(f"k-NN Classification - Accuracy & F1-score vs. k ({dist.capitalize()} Distance)")
                plt.legend()
                plt.grid(True)
                plt.show()

        # ---- k-NN regression ----
        elif args.method == "knn" and args.task == "regression":
            k_list = list(range(1, 61))
            distance_metrics = ["euclidean", "manhattan"]

            for dist in distance_metrics:
                train_mses = []
                valid_mses = []
            
                for k in k_list:
                    model = KNN(k=k, task_kind="regression", distance_metric=dist)

                    preds_train_plot = model.fit(train_features, train_labels_reg)
                    preds_valid_plot = model.predict(test_features)

                    train_mse = mse_fn(preds_train_plot, train_labels_reg)
                    valid_mse = mse_fn(preds_valid_plot, test_labels_reg)

                    train_mses.append(train_mse)
                    valid_mses.append(valid_mse)

                plt.figure(figsize=(8, 6))
                plt.plot(k_list, train_mses, 'o--', label = "Train MSE")
                plt.plot(k_list, valid_mses, 'o-', label = "Validation MSE")
                plt.xlabel("Number of Neighbors (k)")
                plt.ylabel("MSE")
                plt.title(f"k-NN - MSE vs. k ({dist.capitalize()} Distance)")
                plt.legend()
                plt.grid(True)
                plt.show()
        
        # ---- Logistic regression ----
        if args.method == "logistic_regression" and args.task == "classification":
            max_iters_list = [50, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
            lr_list = [1e-4, 1e-3, 1e-2, 0.1]

            results = []

            for max_iter in max_iters_list:
                for lr in lr_list:
                    model = LogisticRegression(lr=lr, max_iters=max_iter)

                    preds_train_plot = model.fit(train_features, train_labels_classif)
                    preds_valid_plot = model.predict(test_features)

                    train_acc = accuracy_fn(preds_train_plot, train_labels_classif)
                    train_f1 = macrof1_fn(preds_train_plot, train_labels_classif) * 100
                    valid_acc = accuracy_fn(preds_valid_plot, test_labels_classif)
                    valid_f1 = macrof1_fn(preds_valid_plot, test_labels_classif) * 100

                    results.append((max_iter, lr, train_acc, train_f1, valid_acc, valid_f1)) 

            # Plot 1: vary lr for a fixed max_iters
            fixed_iter = args.max_iters
            lrs = []
            train_accs = []
            train_f1s = []
            valid_accs = []
            valid_f1s = []

            for max_iter, lr, train_acc, train_f1, valid_acc, valid_f1 in results:
                if max_iter == fixed_iter:
                    lrs.append(lr)
                    train_accs.append(train_acc)
                    train_f1s.append(train_f1)
                    valid_accs.append(valid_acc)
                    valid_f1s.append(valid_f1)  

            plt.figure(figsize=(8, 6))
            plt.plot(lrs, train_accs, 'o--', label="Train Accuracy (%)")
            plt.plot(lrs, valid_accs, 'o-', label="Validation Accuracy (%)")
            plt.plot(lrs, train_f1s, 's--', label="Train F1-score (%)")
            plt.plot(lrs, valid_f1s, 's-', label="Validation F1-score (%)")
            plt.xscale("log")
            plt.xlabel("Learning Rate (lr)")
            plt.ylabel("Score (%)")
            plt.title(f"Logistic Regression - Performance vs. Learing rate (max_iters = {fixed_iter})")
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plot 2: vary max_iters for a fixed lr 
            fixed_lr = args.lr
            iters = []
            train_accs2 = []
            train_f1s2 = []
            valid_accs2 = []
            valid_f1s2 = []

            for max_iter, lr, train_acc, train_f1, valid_acc, valid_f1 in results:
                if lr == fixed_lr:
                    iters.append(max_iter)
                    train_accs2.append(train_acc)
                    train_f1s2.append(train_f1)
                    valid_accs2.append(valid_acc)
                    valid_f1s2.append(valid_f1) 

            plt.figure(figsize=(8, 6))
            plt.plot(iters, train_accs2, 'o--', label = "Train Accuracy (%)")
            plt.plot(iters, valid_accs2, 'o-', label = "Validation Accuracy (%)")
            plt.plot(iters, train_f1s2, 's--', label = "Train F1-score (%)")
            plt.plot(iters, valid_f1s2, 's-', label = "Validation F1-score (%)")
            plt.xlabel("Number of Iterations (max_iters)")
            plt.ylabel("Score (%)")
            plt.title(f"Logistic Regression - Performance vs. Max Iterations (Lr = {fixed_lr})")
            plt.legend()
            plt.grid(True)
            plt.show()
    
        # ---- Linear regression ----
        if args.method == "linear_regression" and args.task == "regression": 
           
            plt.figure(figsize=(8,6))

            # Scatter: true vs predicted
            plt.scatter(
                test_labels_reg,
                preds,
                color="purple",
                alpha=0.65,
                s=45,
                label="Predictions"
            )

            # Perfect prediction line y = x
            min_val = min(np.min(test_labels_reg), np.min(preds))
            max_val = max(np.max(test_labels_reg), np.max(preds))

            plt.plot(
                [min_val, max_val],
                [min_val, max_val],
                color="black",
                linestyle="--",
                linewidth=2,
                label="Perfect Prediction"
            )

            # 1D regression line fitted on displayed points
            coef = np.polyfit(test_labels_reg, preds, 1)   # slope, intercept
            x_line = np.linspace(min_val, max_val, 200)
            y_line = coef[0] * x_line + coef[1]

            plt.plot(
                x_line,
                y_line,
                color="red",
                linewidth=2.5,
                label=f"L.R line of the graph\ny = {coef[0]:.2f}x + {coef[1]:.2f}\nMSE of the model = {test_mse:.4f}"
            )

            plt.xlabel("True Labels")
            plt.ylabel("Predicted Labels")
            plt.title("Linear Regression Model")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


            configs = [(None, 0.0)]
            for lam in [1e-4,1e-3, 1e-2, 1e-1, 1.0, 10.0]:
                configs.append(("l2", lam))

            lambdas = []
            train_mses = []
            valid_mses = []

            for regularization, lambda_ in configs:
                model = LinearRegression(regularization=regularization, lambda_=lambda_)
                
                preds_train_plot = model.fit(train_features, train_labels_reg)
                preds_valid_plot = model.predict(test_features)

                train_mse = mse_fn(preds_train_plot, train_labels_reg)
                valid_mse = mse_fn(preds_valid_plot, test_labels_reg)

                if regularization == "l2":
                    lambdas.append(lambda_)
                    train_mses.append(train_mse)
                    valid_mses.append(valid_mse)

            if len(lambdas) > 0:
                plt.figure(figsize = (8,6)) 
                plt.plot(lambdas, train_mses, "o--", label="Train MSE")
                plt.plot(lambdas, valid_mses, "o-", label="Validation MSE")
                plt.xscale("log")
                plt.xlabel("lambda")
                plt.ylabel("MSE")
                plt.title("Linear Regression (Ridge) - MSE vs lambda")
                plt.legend()
                plt.grid(True)
                plt.show()

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
    parser.add_argument(
        "--regularization",
        type=str,
        default=None,
        help="None / l2 for linear regression",
    )
    parser.add_argument(
        "--lambda_", 
        type=float,
        default=0.0,
        help="regularization strength for linear regression",
    )
    parser.add_argument(
        "--show_plots", 
        action="store_true", 
        help="show validation plots",
    )
    # Feel free to add more arguments here if you need!

    args = parser.parse_args()

    start_total = time.perf_counter()
    main(args)
    end_total = time.perf_counter()

    print(f"\n⏱ Total runtime: {end_total - start_total:.4f} seconds")