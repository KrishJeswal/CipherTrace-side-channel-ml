import os
import sys
import numpy as np
import joblib
from dataclasses import dataclass, field
from typing import Literal, Tuple

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils import compute_snr


Strategy = Literal["snr", "anova", "pca"]


@dataclass
class DataTransformationConfig:
    strategy: Strategy = "snr"
    k: int = 50
    n_components: int = 50
    save_dir: str = os.path.join("artifacts", "transformed")
    transformer_path: str = field(init=False)

    def __post_init__(self):
        tag = (
            f"{self.strategy}_k{self.k}"
            if self.strategy != "pca"
            else f"pca_n{self.n_components}"
        )

        self.transformer_path = os.path.join(
            "artifacts",
            "models",
            f"transformer_{tag}.joblib",
        )


class POITransformer:
    """
    Fit/transform wrapper for three POI extraction strategies.
    Follows sklearn API so it can be swapped into any pipeline.
    """

    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.strategy = config.strategy
        self._selector = None
        self._snr_indices = None
        self._scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "POITransformer":
        """Fit the transformer on profiling traces."""
        logging.info(f"Fitting POITransformer: strategy={self.strategy}, k={self.config.k}")

        if self.strategy == "snr":
            snr = compute_snr(X, y)
            self._snr_indices = np.argsort(snr)[-self.config.k:]
            self._scaler.fit(X[:, self._snr_indices])

        elif self.strategy == "anova":
            self._selector = SelectKBest(f_classif, k=self.config.k)
            self._selector.fit(X, y)
            self._scaler.fit(self._selector.transform(X))

        elif self.strategy == "pca":
            self._selector = PCA(n_components=self.config.n_components)
            self._scaler.fit(X)
            self._selector.fit(self._scaler.transform(X))

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self.is_fitted = True
        logging.info("POITransformer fitted successfully")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted transformer to any trace array."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform()")

        if self.strategy == "snr":
            return self._scaler.transform(X[:, self._snr_indices])

        elif self.strategy == "anova":
            return self._scaler.transform(self._selector.transform(X))

        elif self.strategy == "pca":
            return self._selector.transform(self._scaler.transform(X))

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @property
    def selected_indices(self) -> np.ndarray:
        """Return original sample indices selected (SNR and ANOVA only)."""
        if self.strategy == "snr":
            return np.sort(self._snr_indices)
        elif self.strategy == "anova":
            return np.where(self._selector.get_support())[0]
        else:
            raise AttributeError("PCA does not produce interpretable sample indices")


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initiate_data_transformation(
        self,
        X_prof: np.ndarray,
        y_prof: np.ndarray,
        X_atk: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, POITransformer]:
        """
        Fit transformer on profiling traces, apply to both splits.
        Saves fitted transformer to artifacts/models/.

        Returns:
            X_prof_t : transformed profiling traces
            X_atk_t  : transformed attack traces
            transformer : fitted POITransformer instance
        """
        logging.info(
            f"Data transformation started — strategy={self.config.strategy}, "
            f"k={self.config.k}"
        )
        try:
            transformer = POITransformer(self.config)
            X_prof_t = transformer.fit_transform(X_prof, y_prof)
            X_atk_t  = transformer.transform(X_atk)

            logging.info(
                f"Transformed shapes — profiling: {X_prof_t.shape}, "
                f"attack: {X_atk_t.shape}"
            )

            os.makedirs(os.path.dirname(self.config.transformer_path), exist_ok=True)
            joblib.dump(transformer, self.config.transformer_path)
            logging.info(f"Transformer saved: {self.config.transformer_path}")

            os.makedirs(self.config.save_dir, exist_ok=True)
            tag = f"{self.config.strategy}_k{self.config.k}"
            np.save(os.path.join(self.config.save_dir, f"X_prof_{tag}.npy"), X_prof_t)
            np.save(os.path.join(self.config.save_dir, f"X_atk_{tag}.npy"),  X_atk_t)

            return X_prof_t, X_atk_t, transformer

        except Exception as e:
            raise CustomException(e, sys)