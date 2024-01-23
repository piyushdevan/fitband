import os
import sys

sys.path.append("C:\\Users\\PIYUSH KUMAR\\coding\\fitband")

from src.exception import CustomException
from src.logger import logging
from src.remove_outlier import RemoveOutlier
from src.build_feature import BuildFeature
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")
    data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            logging.info("Read the dataset as dataframe ")
            df = pd.read_csv(self.ingestion_config.raw_data_path)

            remove_outlier = RemoveOutlier()
            rm_df = remove_outlier.initiate_Outlier_Removal(df)
            build_feature = BuildFeature()
            df = build_feature.initiate_Feature_Building(rm_df)

            df.to_csv(self.ingestion_config.data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Inmgestion of the data iss completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
