import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import json
import os
from pathlib import Path

class RFTrain:
    def __init__(self, config):
        self.config = config


    def train(self, features, labels):
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        y = labels.copy()

        regr = RandomForestRegressor(
            criterion=self.config["RF_model_parameters"]["criterion"],
            n_estimators=self.config["RF_model_parameters"]["n_estimators"],
            max_features=self.config["RF_model_parameters"]["max_features"],
            max_depth=self.config["RF_model_parameters"]["max_depth"],
            min_samples_split=self.config["RF_model_parameters"]["min_samples_split"],
            min_samples_leaf=self.config["RF_model_parameters"]["min_samples_leaf"],
            random_state=self.config["RF_model_parameters"]["random_state"],
            n_jobs=self.config["RF_model_parameters"]["n_jobs"], 
            oob_score=True
        )
        regr.fit(X, y)

        self.save(regr, self.config["mapping"]["model_path"])
        self.save(scaler, self.config["mapping"]["scaler_path"])

        return regr, scaler


    def save(self, model, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
