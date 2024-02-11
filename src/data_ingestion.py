import os
import sys

sys.path.append("C:\\Users\\PIYUSH KUMAR\\coding\\fitband")

from src.exception import CustomException
from src.logger import logging


import pandas as pd

from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    save_path: str = os.path.join("artifacts", "fetched.csv")
    data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            logging.info("fetching data ....")
            df = pd.read_csv(self.ingestion_config.data_path)
            # df = df.set_index("epoch (ms)", drop=True)
            # df = df.drop(["participant", "category", "set", "label"], axis=1)
            logging.info("fetching_done saving data")

            df.to_csv(self.ingestion_config.save_path)
            return df

        except Exception as e:
            raise CustomException(e, sys)
