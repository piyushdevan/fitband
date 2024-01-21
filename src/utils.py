import os
import sys
import scipy
import math

import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)




def mark_outliers_chauvenet(dataset, col, C=2):
 
    try:
        dataset = dataset.copy()
        # Compute the mean and standard deviation.
        mean = dataset[col].mean()
        std = dataset[col].std()
        N = len(dataset.index)
        criterion = 1.0 / (C * N)

        # Consider the deviation for the data points.
        deviation = abs(dataset[col] - mean) / std

        # Express the upper and lower bounds.
        low = -deviation / math.sqrt(C)
        high = deviation / math.sqrt(C)
        prob = []
        mask = []

        # Pass all rows in the dataset.
        for i in range(0, len(dataset.index)):
            # Determine the probability of observing the point
            prob.append(
                1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
            )
            # And mark as an outlier when the probability is below our criterion.
            mask.append(prob[i] < criterion)
        dataset[col + "_outlier"] = mask
        return dataset
    
    except Exception as e:
        raise CustomException(e, sys)
