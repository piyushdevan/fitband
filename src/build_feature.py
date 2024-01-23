import os
import sys

sys.path.append("C:\\Users\\PIYUSH KUMAR\\coding\\fitband")

from src.logger import logging
from src.exception import CustomException
from src.utils import (
    save_object,
    load_object,
    LowPassFilter,
    PrincipalComponentAnalysis,
    NumericalAbstraction,
    FourierTransformation,
)
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


from dataclasses import dataclass


@dataclass
class BuildFeaturesConfig:
    save_path: str = os.path.join("artifacts", "feature_build")
    data_path: str = os.path.join("artifacts", "outlier_removed")
    LowPass = LowPassFilter()
    PCA = PrincipalComponentAnalysis()
    NumAbs = NumericalAbstraction()
    FreqAbs = FourierTransformation()


class BuildFeature:
    def __init__(self):
        self.outlier_config = BuildFeaturesConfig()

    def initiate_Feature_Building(self, df):
        try:
            LowPass = LowPassFilter()
            PCA = PrincipalComponentAnalysis()
            NumAbs = NumericalAbstraction()
            FreqAbs = FourierTransformation()

            logging.info("start feature building")
            predictor_columns = list(df.columns[:6])

            for col in predictor_columns:
                df[col] = df[col].interpolate()

            fs = 1000 / 200
            fc = 1.2

            logging.info("Applying Lowpass")
            df_lowpass = df.copy()
            for col in predictor_columns:
                df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, fc, order=5)
                df_lowpass[col] = df_lowpass[col + "_lowpass"]
                del df_lowpass[col + "_lowpass"]

            logging.info("Applying pca")
            df_pca = df_lowpass.copy()
            pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)
            df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

            logging.info("Adding rms acc and rms gyr")
            df_squared = df_pca.copy()
            acc_r = (
                df_squared["acc_x"] ** 2
                + df_squared["acc_y"] ** 2
                + df_squared["acc_z"] ** 2
            )
            gyr_r = (
                df_squared["gyr_x"] ** 2
                + df_squared["gyr_y"] ** 2
                + df_squared["gyr_z"] ** 2
            )
            df_squared["acc_r"] = np.sqrt(acc_r)
            df_squared["gyr_r"] = np.sqrt(gyr_r)

            logging.info("Adding temporal abstraction")
            df_temporal = df_squared.copy()
            NumAbs = NumericalAbstraction()
            predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

            ws = int(1000 / 200)

            for col in predictor_columns:
                df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
                df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")

            logging.info("Adding Fourier_transform")
            df_freq = df_temporal.copy().reset_index()
            FreqAbs = FourierTransformation()

            fs = int(1000 / 200)
            ws = int(2800 / 200)

            df_freq = FreqAbs.abstract_frequency(df_freq, predictor_columns, ws, fs)
            df_freq = df_freq.set_index("epoch (ms)", drop=True)

            logging.info("Adding cluster")

            df_cluster = df_freq.copy()
            cluster_columns = ["acc_x", "acc_y", "acc_z"]
            k_values = range(2, 10)
            inertias = []

            kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
            subset = df_cluster[cluster_columns]
            df_cluster["cluster"] = kmeans.fit_predict(subset)
            df_cluster.dropna(inplace=True)
            save_object(self.outlier_config.save_path, df_cluster)
            return df_cluster

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = BuildFeature()
    obj.initiate_Feature_Building()
