import os
import sys
from dataclasses import dataclass


sys.path.append("C:\\Users\\PIYUSH KUMAR\\coding\\fitband")

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import numpy as np
import copy

from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, df):
        try:
            logging.info("Split training and test input data")
            df_train = df.drop(["participant", "category", "set"], axis=1)
            X = df_train.drop("label", axis=1)
            y = df["label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
            logging.info("Splitting done")
            models = {
                # "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                # "Neural Network": MLPClassifier(),
                # "K Nearest Neighbour": KNeighborsClassifier(),
                # "Support vector Without Kernel": LinearSVC(),
                # "Support Vector Kernel": SVC(),
            }
            params = {
                # "Decision Tree": {
                #     "min_samples_leaf": [2, 10, 50, 100, 200],
                #     "criterion": ["gini", "entropy"],
                # },
                "Random Forest": {
                    "min_samples_leaf": [2, 10, 50, 100, 200],
                    "n_estimators": [10, 50, 100],
                    "criterion": ["gini", "entropy"],
                },
                # "Neural Network": {
                #     "hidden_layer_sizes": [
                #         (5,),
                #         (10,),
                #         (25,),
                #         (100,),
                #         (
                #             100,
                #             5,
                #         ),
                #         (
                #             100,
                #             10,
                #         ),
                #     ],
                #     "activation": ["logistic"],
                #     "learning_rate": ["adaptive"],
                #     "max_iter": [1000, 2000],
                #     "alpha": [0.0001],
                # },
                # "K Nearest Neighbour": {"n_neighbors": [1, 2, 5, 10]},
                # "Support vector Without Kernel": {
                #     "max_iter": [1000, 2000],
                #     "tol": [1e-3, 1e-4],
                #     "C": [1, 10, 100],
                # },
                # "Support Vector Kernel": {
                #     "kernel": ["rbf", "poly"],
                #     "gamma": [1e-3, 1e-4],
                #     "C": [1, 10, 100],
                # },
            }
            # selected_features = [
            #     "pca_1",
            #     "acc_x_freq_0.0_Hz_ws_14",
            #     "acc_z_freq_0.0_Hz_ws_14",
            #     "acc_y_temp_mean_ws_5",
            #     "gyr_r",
            #     "gyr_y_freq_1.429_Hz_ws_14",
            #     "gyr_y_freq_0.357_Hz_ws_14",
            #     "acc_r_freq_0.357_Hz_ws_14",
            #     "gyr_z_freq_0.714_Hz_ws_14",
            # ]
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)

            r2_square = accuracy_score(y_test, predicted)
            print(r2_square)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
