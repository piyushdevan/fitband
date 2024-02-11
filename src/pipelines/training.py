import os
import sys

sys.path.append("C:\\Users\\PIYUSH KUMAR\\coding\\fitband")

from src.exception import CustomException
from src.logger import logging
from src.remove_outlier import RemoveOutlier
from src.build_feature import BuildFeature
from src.data_ingestion import DataIngestion
from src.train_model import ModelTrainer


import pandas as pd
import numpy as np


from dataclasses import dataclass


@dataclass
class TrainingConfig:
    data_path: str = os.path.join("artifacts", "raw_data.csv")


class Training_Model:
    def __init__(self):
        self.ingestion_config = TrainingConfig()

    def initiate_model_training(self):
        try:
            logging.info("fetching data from raw data")
            df = pd.read_csv(self.ingestion_config.data_path)

            logging.info("Removing Outlier")
            removed_outlier_df = RemoveOutlier().initiate_Outlier_Removal(df)

            logging.info("Building Features")
            preprocessed_df = BuildFeature().initiate_Feature_Building(
                removed_outlier_df
            )

            logging.info("Training models")
            score = ModelTrainer().initiate_model_trainer(preprocessed_df)

            logging.info("Model Training Successful")
            return score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = Training_Model()
    score = obj.initiate_model_training()
    print(score)
