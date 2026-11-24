import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class RFTrain:
    def __init__(self, config):
        self.config = config


    def train(self, features, labels):
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        y = labels.copy()

        parameters = self.config["RF_model_parameters"]
        bootstrap = bool(parameters.get("bootstrap", True))
        # OOB estimates need the samples bootstrap leaves out of each tree.
        oob_score = bool(parameters.get("oob_score", bootstrap))
        if oob_score and not bootstrap:
            raise ValueError("RF_model_parameters.oob_score requires bootstrap: true")

        regr = RandomForestRegressor(
            criterion=parameters["criterion"],
            n_estimators=parameters["n_estimators"],
            max_features=parameters["max_features"],
            max_depth=parameters["max_depth"],
            min_samples_split=parameters["min_samples_split"],
            min_samples_leaf=parameters["min_samples_leaf"],
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=parameters["random_state"],
            n_jobs=parameters["n_jobs"],
        )
        regr.fit(X, y)
        if oob_score:
            logger.info("Out-of-bag R^2: %.4f", regr.oob_score_)

        self.save(regr, self.config["mapping"]["model_path"])
        self.save(scaler, self.config["mapping"]["scaler_path"])

        return regr, scaler


    # zlib level 3: much smaller files for large forests at modest save/load cost.
    COMPRESSION = 3

    def save(self, model, file_path):
        """Save with joblib (pickle-based: load only files you trust)."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, file_path, compress=self.COMPRESSION)
