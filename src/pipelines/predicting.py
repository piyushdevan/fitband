import os
import sys

sys.path.append("C:\\Users\\PIYUSH KUMAR\\coding\\fitband")

import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object, save_object
from src.build_feature import BuildFeature
from src.data_ingestion import DataIngestion
from sklearn.metrics import accuracy_score, r2_score
import numpy as np

from dataclasses import dataclass


@dataclass
class PredictPipelineConfig:
    modelPath: str = os.path.join("artifacts", "model.pkl")
    # processingPath: str = os.path.
    saveDataPath: str = os.path.join("artifacts", "predicted.csv")


class PredictPipeline:
    def __init__(self):
        self.ingestion_config = PredictPipelineConfig()

    def Prediction(self):
        try:
            logging.info("Importing fetched Data")
            df = DataIngestion().initiate_data_ingestion()

            df.set_index("epoch (ms)", inplace=True)

            logging.info("Processing Data ........")
            dataset = BuildFeature().initiate_Feature_Building(df)

            logging.info("Predicting.........")
            model = load_object(file_path=self.ingestion_config.modelPath)

            preds = model.predict(dataset)

            df["prediction"] = preds

            df.to_csv(self.ingestion_config.saveDataPath)
            logging.info("Prediction completed see the result>>>>>")

            return preds

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = PredictPipeline()
    preds = obj.Prediction()
    print(preds)
