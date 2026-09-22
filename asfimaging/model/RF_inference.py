from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
import numpy as np

class RFInference:
    def __init__(self, config):
        self.config = config


    def load_model(self):
        return pickle.load(open(self.config["mapping"]["model_path"], "rb"))

    def load_scaler(self):
        return pickle.load(open(self.config["mapping"]["scaler_path"], "rb"))

    def inference(self, features):
        model = self.load_model()
        scaler = self.load_scaler()
        X = scaler.transform(features)
        return model.predict(X)

    def save(self, labels):
        path = os.path.join(self.config["mapping"]["directory"], "RF_labels.npy")
        np.save(path, labels)
        return None
