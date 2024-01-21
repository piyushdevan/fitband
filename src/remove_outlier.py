import os
import sys

sys.path.append("C:\\Users\\PIYUSH KUMAR\\coding\\fitband")

from src.exception import CustomException
from src.logger import logging
from src.utils import mark_outliers_chauvenet
import pandas as pd
import numpy as np


from dataclasses import dataclass


class RemoveOutlier:
    def __init__(self):
        pass

    def initiate_Outlier_Removal(self):
        logging.info("Entered the outlier detection")
        try:
            df = pd.read_csv("artifacts\test.csv")

            outliers_removed_df = df.copy()

            outlier_columns = df[1:7]
            for col in outlier_columns:
                for label in df["label"].unique():
                    dataset = mark_outliers_chauvenet(df[df["label"] == label], col)
                    dataset.loc[dataset[col + "_outlier"], col] = np.nan
                    outliers_removed_df.loc[
                        (outliers_removed_df["label"] == label), col
                    ] = dataset[col]

            logging.info("outlier detection done saving file in pickle")

            outliers_removed_df.to_csv("artifacts\train.csv")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = Re()
    train_data, test_data = obj.initiate_data_ingestion()
