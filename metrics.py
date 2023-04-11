#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename : _metrics.py
# Author : Tuhin Mallick

# Import necessary libraries

import sys
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics

# Define an abstract class for custom metric implementations


class AbstractMetric(ABC):
    @staticmethod
    @abstractmethod
    def __call__(pred, label, weights=None):
        pass


# Define a SMAPE class that inherits from the AbstractMetric class


class SMAPE(AbstractMetric):
    name = "SMAPE"

    @staticmethod
    def __call__(preds, labels, weights=None):
        """
        Calculate the Symmetric Mean Absolute Percentage Error (SMAPE) of the predictions.

        Args:
            preds (np.ndarray): Predicted values.
            labels (np.ndarray): True values.
            weights (np.ndarray): Weights for each sample, default is None.

        Returns:
            float: The SMAPE metric value.
        """
        if not weights.size:
            weights = None
        return 100 * np.average(
            2 * np.abs(preds - labels) / (np.abs(labels) + np.abs(preds)),
            weights=weights,
        )


# Define a function to compute the Normalised Quantile Loss


def normalised_quantile_loss(y_pred, y, quantile, weights=None):
    """
    Compute the normalised quantile loss.

    Implementation of the q-Risk function from https://arxiv.org/pdf/1912.09363.pdf.

    Args:
        y_pred (np.ndarray): Predicted values.
        y (np.ndarray): True values.
        quantile (float): Quantile value (between 0 and 1).
        weights (np.ndarray, optional): Weights for each sample, default is None.

    Returns:
        float: The normalised quantile loss.
    """
    # Compute the prediction underflow
    prediction_underflow = y - y_pred
    # Compute the weighted errors
    weighted_errors = quantile * np.maximum(prediction_underflow, 0.0) + (
        1.0 - quantile
    ) * np.maximum(-prediction_underflow, 0.0)
    # Check if weights are provided
    if weights is not None and weights.size:
        # Apply the weights
        weighted_errors = weighted_errors * weights
        # Compute the normaliser
        y = y * weights
    # Compute the normalised quantile loss
    loss = weighted_errors.sum()
    # Sum of the absolute values of the true values
    normaliser = abs(y).sum()
    # Return the normalised quantile loss
    return 2 * loss / normaliser  # Normalised Quantile Loss


class P50_loss(AbstractMetric):
    name = "P50"
    selector = 1

    @staticmethod
    def __call__(labels, preds, weights):
        """
        Calculate the P50 quantile loss of the predictions.

        Args:
            labels (np.ndarray): True values.
            preds (np.ndarray): Predicted values.
            weights (np.ndarray): Weights for each sample.

        Returns:
            float: The P50 quantile loss.
        """
        return normalised_quantile_loss(labels, preds, 0.5, weights)


class P90_loss(AbstractMetric):
    name = "P90"
    selector = 2

    @staticmethod
    def __call__(labels, preds, weights):
        """
        Calculate the P90 quantile loss of the predictions.

        Args:
            labels (np.ndarray): True values.
            preds (np.ndarray): Predicted values.
            weights (np.ndarray): Weights for each sample.

        Returns:
            float: The P90 quantile loss.
        """
        return normalised_quantile_loss(labels, preds, 0.9, weights)


# Normalized Deviation


class ND(AbstractMetric):
    name = "ND"

    @staticmethod
    def __call__(preds, labels, weights):
        """
        Calculate the Normalized Deviation of the predictions.

        Args:
            preds (np.ndarray): Predicted values.
            labels (np.ndarray): True values.
            weights (np.ndarray): Weights for each sample.

        Returns:
            float: The Normalized Deviation metric value.
        """
        diff = np.abs(labels - preds)

        return (
            np.sum(diff * weights) / np.sum(np.abs(labels) * weights)
            if weights.size
            else np.sum(diff) / np.sum(np.abs(labels))
        )


class MAE(AbstractMetric):
    name = "MAE"

    @staticmethod
    def __call__(preds, labels, weights, return_individual=False):
        """
        Calculate the Mean Absolute Error (MAE) of the predictions.

        Args:
            preds (np.ndarray): Predicted values.
            labels (np.ndarray): True values.
            weights (np.ndarray): Weights for each sample, default is None.
            return_individual (bool): If True, return individual errors per sample, default is False.

        Returns:
            float or np.ndarray: The MAE metric value, or individual errors if return_individual is True.
        """
        if not weights.size:
            weights = None
        if return_individual:
            return np.average(np.abs(preds - labels), weights=weights, axis=0)
        else:
            return np.average(np.abs(preds - labels), weights=weights)


class MSE(AbstractMetric):
    name = "MSE"

    @staticmethod
    def __call__(preds, labels, weights, return_individual=False):
        """
        Calculate the Mean Squared Error (MSE) of the predictions.

        Args:
            preds (np.ndarray): Predicted values.
            labels (np.ndarray): True values.
            weights (np.ndarray): Weights for each sample, default is None.
            return_individual (bool): If True, return individual errors per sample, default is False.

        Returns:
            float or np.ndarray: The MSE metric value, or individual errors if return_individual is True.
        """
        if not weights.size:
            weights = None
        if return_individual:
            return np.average((preds - labels) ** 2, weights=weights, axis=0)
        else:
            return np.average((preds - labels) ** 2, weights=weights)


class RMSE(AbstractMetric):
    name = "RMSE"

    @staticmethod
    def __call__(preds, labels, weights):
        """
        Calculate the Root Mean Squared Error (RMSE) of the predictions.

        Args:
            preds (np.ndarray): Predicted values.
            labels (np.ndarray): True values.
            weights (np.ndarray): Weights for each sample.

        Returns:
            float: The RMSE metric value.
        """
        if not weights.size:
            weights = None
        return np.sqrt(np.average((preds - labels) ** 2, weights=weights))


class R_Squared(AbstractMetric):
    name = "R_Squared"

    @staticmethod
    def __call__(preds, labels, weights, return_individual=False):
        """
        Calculate the R-squared (coefficient of determination) of the predictions.

        Args:
            preds (np.ndarray): Predicted values.
            labels (np.ndarray): True values.
            weights (np.ndarray): Weights for each sample, default is None.
            return_individual (bool): If True, return individual R-squared values per sample, default is False.

        Returns:
            float or np.ndarray: The R-squared metric value, or individual R-squared values if return_individual is True.
        """
        if not weights.size:
            return (
                skmetrics.r2_score(preds, labels, multioutput="raw_values")
                if return_individual
                else skmetrics.r2_score(preds, labels)
            )
        values = skmetrics.r2_score(preds, labels, multioutput="raw_values")
        if return_individual:
            return values * weights
        return np.sum(values * weights) / np.sum(weights)


class WMSMAPE(AbstractMetric):
    name = "WMSMAPE"

    @staticmethod
    def __call__(preds, labels, weights, return_individual=False):
        """
        Calculate the Weighted Mean Symmetric Mean Absolute Percentage Error (WMSMAPE) of the predictions.

        Args:
            preds (np.ndarray): Predicted values.
            labels (np.ndarray): True values.
            weights (np.ndarray): Weights for each sample, default is None.
            return_individual (bool): If True, return individual WMSMAPE values per sample, default is False.

        Returns:
            float or np.ndarray: The WMSMAPE metric value, or individual WMSMAPE values if return_individual is True.
        """
        if weights.size:
            if return_individual:
                return (
                    2
                    * weights
                    * np.abs(preds - labels)
                    / (np.maximum(labels, 1) + np.abs(preds))
                )
            else:
                return (
                    100.0
                    / np.sum(weights)
                    * np.sum(
                        2
                        * weights
                        * np.abs(preds - labels)
                        / (np.maximum(labels, 1) + np.abs(preds))
                    )
                )
        if return_individual:
            return 2 * np.abs(preds - labels) / (np.maximum(labels, 1) + np.abs(preds))
        else:
            return (
                100.0
                / len(labels)
                * np.sum(
                    2 * np.abs(preds - labels) / (np.maximum(labels, 1) + np.abs(preds))
                )
            )


class Accuracy(AbstractMetric):
    name = "Accuracy"

    @staticmethod
    def __call__(y_pred, y_true, weights=None):
        """
        Calculate the accuracy of the predictions.

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values.
            weights (np.ndarray): Weights for each sample, not supported in this metric.

        Returns:
            float: The accuracy metric value in percentage.
        """
        try:
            if weights is not None:
                raise NotImplementedError("Weighted accuracy is not supported.")
            # Handle division by zero
            if (np.array(y_true) == 0).sum() > 0:
                y_true = np.where(y_true == 0, 1e-5, y_true)
            acc = 1 - np.mean(
                np.clip(np.abs((y_true - y_pred) / y_true), a_min=0, a_max=1)
            )
            acc_percent = acc * 100
        except Exception as e:
            acc_percent = 0
            print(f"ERROR: calculating accuracy - {str(e)}")
        return acc_percent


class MAPE(AbstractMetric):
    name = "MAPE"

    @staticmethod
    def __call__(y_pred, y_true, weights=None):
        """
        Calculate the Mean Absolute Percentage Error (MAPE) of the predictions.

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values.
            weights (np.ndarray): Weights for each sample, not supported in this metric.

        Returns:
            float: The MAPE metric value in percentage.
        """
        try:
            if weights is not None:
                raise NotImplementedError("Weighted MAPE is not supported.")
            # Handle division by zero
            if (np.array(y_true) == 0).sum() > 0:
                y_true = np.where(y_true == 0, 1e-5, y_true)
                error_percent = np.mean(
                    np.clip(np.abs((y_true - y_pred) / y_true), a_min=0, a_max=1)
                )
            else:
                error_percent = np.mean(np.abs((y_true - y_pred) / y_true))
        except Exception as e:
            error_percent = 100
            print(f"ERROR: calculating MAPE - {str(e)}")
        return error_percent


class MAE(AbstractMetric):
    name = "MAE"

    @staticmethod
    def __call__(y_pred, y_true, weights=None):
        """
        Calculate the Mean Absolute Error (MAE) of the predictions.

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values.
            weights (np.ndarray): Weights for each sample, default is None.

        Returns:
            float: The MAE metric value.
        """
        return skmetrics.mean_absolute_error(
            y_true=y_true, y_pred=y_pred, sample_weight=weights
        )


class RMSE(AbstractMetric):
    name = "RMSE"

    @staticmethod
    def __call__(y_pred, y_true, weights=None):
        """
        Calculate the Root Mean Squared Error (RMSE) of the predictions.

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values.
            weights (np.ndarray): Weights for each sample, default is None.

        Returns:
            float: The RMSE metric value.
        """
        return np.sqrt(
            skmetrics.mean_squared_error(
                y_true=y_true, y_pred=y_pred, sample_weight=weights
            )
        )


class DirectionalSymmetry(AbstractMetric):
    name = "DirectionalSymmetry"

    @staticmethod
    def __call__(y_pred, y_true, tolerance=1, weights=None):
        """
        Calculate the directional symmetry of the predictions.

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values.
            tolerance (int, optional): Tolerance level for percentage change. Default is 1.
            weights (np.ndarray): Weights for each sample, not supported in this metric.

        Returns:
            float: The directional symmetry metric value in percentage.
        """

        name = "DirectionalSymmetry"

        try:
            if weights is not None:
                raise NotImplementedError(
                    "Weighted directional symmetry is not supported."
                )
            # Check if tolerance is valid
            if tolerance < 0:
                raise ValueError("Tolerance cannot be less than zero!")
            # Define common variables for true and predicted differences
            true_diff = np.diff(y_true)
            pred_diff = np.diff(y_pred)

            # Case not zero: modify true_diff and pred_diff
            if tolerance != 0:
                # Scale tolerance
                tolerance /= 100

                # Get %change for true values and update true_diff accordingly
                tmp = pd.Series(y_true).pct_change().iloc[1:]
                true_diff[tmp.abs() < tolerance] = 0

                # Get %change for predicted values and update pred_diff accordingly
                tmp = pd.Series(y_pred).pct_change().iloc[1:]
                pred_diff[tmp.abs() < tolerance] = 0
            # Core formula for directional symmetry
            d = (true_diff * pred_diff) > 0
            d[
                (true_diff == 0) & (pred_diff == 0)
            ] = 1  # Case of plateau for both y_true and y_pred
            dsymm = np.round(100 * d.sum() / len(d), 2)
        except Exception as e:
            dsymm = 0
            print(f"ERROR: calculating directional symmetry - {str(e)}")
        return dsymm


METRICS = {
    "SMAPE": SMAPE,
    "WMSMAPE": WMSMAPE,
    "MSE": MSE,
    "MAE": MAE,
    "P50": P50_loss,
    "P90": P90_loss,
    "RMSE": RMSE,
    "R_Squared": R_Squared,
    "ND": ND,
    "Accuracy": Accuracy,
    "MAPE": MAPE,
    "MAE": MAE,
    "RMSE": RMSE,
    "DirectionalSymmetry": DirectionalSymmetry,
}
