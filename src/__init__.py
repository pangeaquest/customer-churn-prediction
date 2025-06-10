"""
Customer Churn Prediction Package

A comprehensive machine learning pipeline for predicting customer churn
in telecommunications companies.
"""

__version__ = "1.0.0"
__author__ = "Augment Agent"
__email__ = "agent@augmentcode.com"

from . import data, models, evaluation, visualization, utils

__all__ = ["data", "models", "evaluation", "visualization", "utils"]
