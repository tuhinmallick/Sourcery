#!/usr/bin/env python3
# This line specifies the interpreter to be used to execute the script.

# -*- coding: utf-8 -*-
# This line specifies the encoding of the script.

# Filename : _metrics.py
# Author : Tuhin Mallick
# These lines specify the filename and author of the script.

import pandas as pd
import numpy as np

# These lines import the pandas and numpy libraries.

from ._metrics import ForecastEvaluation

# This line imports the ForecastEvaluation class from the _metrics module in the current package.


def model_evaluation(
    results: list,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    y_test: pd.DataFrame,
    sorting_criteria="test_mae",
):
    """
    Evaluates a list of forecasting models on training, validation and test data.

    :param results: A list of dictionaries containing model information and predictions.
                    Each dictionary should have the following keys:
                    'model': The name of the model.
                    'algorithm': The algorithm used by the model.
                    'category': The category of the model.
                    'dataset': The dataset used to train the model.
                    'y_train_pred': A 1D array-like object containing the model's predictions on the training data.
                    'y_val_pred': A 1D array-like object containing the model's predictions on the validation data.
                    'y_test_pred': A 1D array-like object containing the model's predictions on the test data.
    :type results: list
    :param y_train: A pandas DataFrame containing the training data.
    :type y_train: pd.DataFrame
    :param y_val: A pandas DataFrame containing the validation data.
    :type y_val: pd.DataFrame
    :param y_test: A pandas DataFrame containing the test data.
    :type y_test: pd.DataFrame
    :param sorting_criteria: The column name to sort the resulting DataFrame by. Default is 'test_mae'.
    :type sorting_criteria: str
    :return: A pandas DataFrame containing model information and evaluation metrics for each model in results.
             The columns are:
             'category': The category of the model.
             'algorithm': The algorithm used by the model.
             'dataset': The dataset used to train the model.
             'model': The name of the model.
             'train_acc': The accuracy of the model on the training data.
             'train_mape': The mean absolute percentage error of the model on the training data.
             'train_mae': The mean absolute error of the model on the training data.
             'train_rmse': The root mean squared error of the model on the training data.
             'train_dir': The directional symmetry of the model on the training data.
             'val_acc': The accuracy of the model on the validation data.
             'val_mape': The mean absolute percentage error of the model on the validation data.
             'val_mae': The mean absolute error of the model on the validation data.
             'val_rmse': The root mean squared error of the model on the validation data.
             'val_dir': The directional symmetry of the model on the validation data.
             'test_acc': The accuracy of the model on the test data.
             'test_mape': The mean absolute percentage error of the model on the test data.
             'test_mae': The mean absolute error of the model on the test data.
             'test_rmse': The root mean squared error of the model on the test data.
             'test_dir': The directional symmetry of the model on the test data.

    """
    # This line defines a function named model_evaluation that takes in 5 arguments:
    # results - a list of results
    # y_train - a pandas DataFrame containing training data
    # y_val - a pandas DataFrame containing validation data
    # y_test - a pandas DataFrame containing test data
    # sorting_criteria - a string specifying the sorting criteria, with a default value of 'test_mae'

    (
        train_acc,
        train_mape,
        train_mae,
        train_rmse,
        train_dsymm,
        val_acc,
        val_mape,
        val_mae,
        val_rmse,
        val_dsymm,
        test_acc,
        test_mape,
        test_mae,
        test_rmse,
        test_dsymm,
        models,
        category,
        algorithm,
        dataset,
    ) = ([] for _ in range(19))
    # This block of code initializes 19 lists to store various evaluation metrics for training,
    # validation and test data as well as model information.

    evaluation = ForecastEvaluation()
    # This line creates an instance of the ForecastEvaluation class.

    for result in results:
        # This block of code iterates over each result in the results list.

        models.append(result["model"])
        algorithm.append(result["algorithm"])
        category.append(result["category"])
        dataset.append(result["dataset"])
        # These lines extract model information from the result and append it to the respective lists.

        train_result = evaluation(
            np.array(y_train).flatten(), np.array(
                result["y_train_pred"]).flatten()
        )
        val_result = evaluation(
            np.array(y_val).flatten(), np.array(result["y_val_pred"]).flatten()
        )
        test_result = evaluation(
            np.array(y_test).flatten(), np.array(
                result["y_test_pred"]).flatten()
        )
        # These lines evaluate the model on training, validation and test data using the evaluation instance.

        train_acc.append(train_result[0])
        train_mape.append(train_result[1])
        train_mae.append(train_result[2])
        train_rmse.append(train_result[3])
        train_dsymm.append(train_result[4])

        val_acc.append(val_result[0])
        val_mape.append(val_result[1])
        val_mae.append(val_result[2])
        val_rmse.append(val_result[3])
        val_dsymm.append(val_result[4])

        test_acc.append(test_result[0])
        test_mape.append(test_result[1])
        test_mae.append(test_result[2])
        test_rmse.append(test_result[3])
        test_dsymm.append(test_result[4])
        # This line appends the test directional symmetry metric to the test_dsymm list.

        # end for loop
        # This line indicates the end of the for loop.

        list_of_tuples = list(
            zip(
                category,
                algorithm,
                dataset,
                models,
                train_acc,
                train_mape,
                train_mae,
                train_rmse,
                train_dsymm,
                val_acc,
                val_mape,
                val_mae,
                val_rmse,
                val_dsymm,
                test_acc,
                test_mape,
                test_mae,
                test_rmse,
                test_dsymm,
            )
        )
        # This block of code creates a list of tuples by zipping together all the lists containing model information and evaluation metrics.

        result = pd.DataFrame(
            list_of_tuples,
            columns=[
                "category",
                "algorithm",
                "dataset",
                "model",
                "train_acc",
                "train_mape",
                "train_mae",
                "train_rmse",
                "train_dir",
                "val_acc",
                "val_mape",
                "val_mae",
                "val_rmse",
                "val_dir",
                "test_acc",
                "test_mape",
                "test_mae",
                "test_rmse",
                "test_dir",
            ],
        )
        # This line creates a pandas DataFrame from the list of tuples with specified column names.

        result.sort_values(by=sorting_criteria, ascending=True, inplace=True)
        # This line sorts the DataFrame by the specified sorting criteria in ascending order.

        result.reset_index(inplace=True, drop=True)
        # This line resets the index of the DataFrame and drops the old index.

        return result
        # This line returns the resulting DataFrame.
