import os
import sys
import time
import joblib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

from src.logger import logging
from src.exception import CustomException


@dataclass
class ModelTrainerConfig:
    strategy: str = "snr"
    k: int = 50
    models_dir: str = os.path.join("artifacts", "models")
    results_path: str = field(init=False)

    def __post_init__(self):
        tag = f"{self.strategy}_k{self.k}"
        self.results_path = os.path.join(
            self.models_dir, f"results_{tag}.joblib"
        )

    def model_path(self, model_name: str) -> str:
        tag = f"{self.strategy}_k{self.k}"
        return os.path.join(self.models_dir, f"{model_name}_{tag}.joblib")


def get_models() -> Dict[str, Any]:
    """Return all six classifiers with fixed random states."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, C=0.1, solver="lbfgs",
            random_state=42
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=20, min_samples_leaf=5, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=20, n_jobs=-1, random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric="mlogloss",
            n_jobs=-1, random_state=42
        ),
        # "svm": SVC(
        #     kernel="rbf", C=10, gamma="scale",
        #     probability=True, random_state=42
        # ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu", max_iter=200,
            early_stopping=True, random_state=42
        ),
    }


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def _evaluate_model(
        self,
        name: str,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Train one model, run CV, evaluate on test set. Return metrics dict."""
        logging.info(f"Training {name} ...")
        t0 = time.time()

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(
            model, X_train, y_train,
            cv=cv, scoring="f1_macro",
            return_train_score=False, n_jobs=-1
        )

        model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="macro", zero_division=0)

        result = {
            "model"       : model,
            "accuracy"    : round(acc, 4),
            "macro_f1"    : round(f1, 4),
            "cv_mean"     : round(cv_results["test_score"].mean(), 4),
            "cv_std"      : round(cv_results["test_score"].std(), 4),
            "train_time_s": round(train_time, 2),
        }

        logging.info(
            f"{name} — acc={acc:.4f} f1={f1:.4f} "
            f"cv={result['cv_mean']:.4f}±{result['cv_std']:.4f} "
            f"time={train_time:.1f}s"
        )
        return result

    def initiate_model_training(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict]:
        """
        Train all six models. Save each to artifacts/models/.
        Returns a results dict keyed by model name.
        """
        logging.info(
            f"Model training started — strategy={self.config.strategy}, k={self.config.k}"
        )
        try:
            os.makedirs(self.config.models_dir, exist_ok=True)
            models   = get_models()
            all_results = {}

            for name, model in models.items():
                result = self._evaluate_model(
                    name, model, X_train, y_train, X_test, y_test
                )
                joblib.dump(result["model"], self.config.model_path(name))
                all_results[name] = {k: v for k, v in result.items() if k != "model"}

            joblib.dump(all_results, self.config.results_path)
            logging.info(f"All models saved. Results saved: {self.config.results_path}")

            _print_results_table(all_results)
            return all_results

        except Exception as e:
            raise CustomException(e, sys)


def _print_results_table(results: Dict[str, Dict]) -> None:
    header = f"{'Model':<22} {'Accuracy':>8} {'Macro F1':>9} {'CV Mean':>8} {'CV Std':>7} {'Time(s)':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name, r in results.items():
        print(
            f"{name:<22} {r['accuracy']:>8.4f} {r['macro_f1']:>9.4f} "
            f"{r['cv_mean']:>8.4f} {r['cv_std']:>7.4f} {r['train_time_s']:>8.1f}"
        )
    print("=" * len(header))