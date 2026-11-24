import joblib

from asfimaging.config import labels_path
import numpy as np

class RFInference:
    def __init__(self, config):
        self.config = config


    # joblib files are pickle-based: loading one can run code embedded in it,
    # so load only models and scalers you trained or obtained from a trusted source.
    def load_model(self):
        return joblib.load(self.config["mapping"]["model_path"])

    def load_scaler(self):
        return joblib.load(self.config["mapping"]["scaler_path"])

    def inference(self, features):
        model = self.load_model()
        scaler = self.load_scaler()
        X = scaler.transform(features)
        return model.predict(X)

    # Forest targets are scaled for training (generator.generate); this converts a
    # target column back to um^2/ms (= 1e-9 m^2/s). ASF is dimensionless.
    TARGET_UNITS = {
        'dpar': (1e-10 / 1e-9, 'um^2/ms'),
        'dperp': (1e-12 / 1e-9, 'um^2/ms'),
        'asf': (1.0, ''),
    }

    def target_names(self):
        names = ['dpar', 'dperp']
        if self.config['features_labels']['predicted_asf']:
            names.append('asf')
        return names

    @staticmethod
    def regression_metrics(y_true, y_pred, unit=''):
        """MAE, RMSE, bias (mean predicted - true) and R^2 for one target."""
        error = y_pred - y_true
        variance = np.sum((y_true - np.mean(y_true))**2)
        return {
            'unit': unit,
            'mae': float(np.mean(np.abs(error))),
            'rmse': float(np.sqrt(np.mean(error**2))),
            'bias': float(np.mean(error)),
            'r2': float(1 - np.sum(error**2) / variance) if variance > 0 else float('nan'),
            'n': int(len(y_true)),
        }

    def evaluate(self, features, labels):
        """Predict `features` and compare with `labels` (scaled training targets).

        Does not save labels, so labels_path (real-data predictions) is untouched.
        Returns (metrics per target in physical units, scaled predictions).
        """
        predictions = np.asarray(self.inference(features)).reshape(len(labels), -1)
        labels = np.asarray(labels).reshape(len(labels), -1)
        names = self.target_names()
        if predictions.shape[1] != len(names) or labels.shape[1] != len(names):
            raise ValueError("Predictions/labels do not match the enabled targets; "
                             "was the model trained with the current features_labels?")
        metrics = {}
        for k, name in enumerate(names):
            scale, unit = self.TARGET_UNITS[name]
            metrics[name] = self.regression_metrics(labels[:, k] * scale, predictions[:, k] * scale, unit)
        return metrics, predictions

    def labels_path(self):
        return labels_path(self.config)

    def save(self, labels):
        path = self.labels_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, labels)
        return path
