#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Load the diabetes dataset.
    dataset = sklearn.datasets.load_diabetes()

    # The input data are in `dataset.data`, targets are in `dataset.target`.
    x=dataset.data
    y=dataset.target
    # If you want to learn about the dataset, you can print some information
    # about it using `print(dataset.DESCR)`.

    # TODO: Append a constant feature with value 1 to the end of every input data.
    x=np.c_[x,np.ones(x.shape[0])]
    # Then we do not need to explicitly represent bias - it becomes the last weight.

    
    # TODO: Split the dataset into a train set and a test set.
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=args.test_size, random_state=args.seed)
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.

    # TODO: Solve the linear regression using the algorithm from the lecture,
    w = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train
    # explicitly computing the matrix inverse (using `np.linalg.inv`).

    
    # TODO: Predict target values on the test set.
    y_pred = x_test @ w

    
    # TODO: Manually compute root mean square error on the test set predictions
    rmse =np.sqrt(np.mean((y_test - y_pred) ** 2))
    return rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))
