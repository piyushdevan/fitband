import os
import sys

sys.path.append("C:\\Users\\PIYUSH KUMAR\\coding\\fitband")

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, load_object, mark_outliers_chauvenet
import pandas as pd
import numpy as np


from dataclasses import dataclass


@dataclass
class RemoveOutlierConfig:
    data_path: str = os.path.join("notebook", "study.csv")
    save_path: str = os.path.join("artifacts", "outlier_removed")


class RemoveOutlier:
    def __init__(self):
        self.outlier_config = RemoveOutlierConfig()

    def initiate_Outlier_Removal(self, df):
        try:
            logging.info("Entered the outlier detection")

            df.set_index("epoch (ms)", inplace=True)

            outliers_removed_df = df.copy()
            outlier_columns = list(df.columns[0:6])

            for col in outlier_columns:
                for label in df["label"].unique():
                    dataset = mark_outliers_chauvenet(df[df["label"] == label], col)
                    dataset.loc[dataset[col + "_outlier"], col] = np.nan
                    outliers_removed_df.loc[
                        (outliers_removed_df["label"] == label), col
                    ] = dataset[col]

            logging.info("outlier detection done  -saving file ")
            save_object(self.outlier_config.save_path, outliers_removed_df)
            return outliers_removed_df

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = RemoveOutlier()
    obj.initiate_Outlier_Removal()
