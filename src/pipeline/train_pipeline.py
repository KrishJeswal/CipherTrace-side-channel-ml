import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer


def run_training_pipeline(strategy: str = "snr", k: int = 50) -> dict:
    """
    Full training pipeline: ingest → transform → train.

    Args:
        strategy : poi strategy — 'snr', 'anova', or 'pca'
        k        : number of POIs (or PCA components)

    Returns:
        results dict from ModelTrainer (accuracy, f1, cv scores, train time)
    """
    try:
        logging.info(f"=== Training pipeline START | strategy={strategy} k={k} ===")

        logging.info("Step 1: Data ingestion")
        di = DataIngestion()
        X_prof, y_prof, pt_prof, X_atk, y_atk, pt_atk = di.initiate_data_ingestion()

        logging.info("Step 2: Data transformation")
        cfg_t = DataTransformationConfig(strategy=strategy, k=k, n_components=k)
        dt    = DataTransformation(cfg_t)
        X_prof_t, X_atk_t, _ = dt.initiate_data_transformation(X_prof, y_prof, X_atk)

        logging.info("Step 3: Model training")
        cfg_m   = ModelTrainerConfig(strategy=strategy, k=k)
        trainer = ModelTrainer(cfg_m)
        results = trainer.initiate_model_training(X_prof_t, y_prof, X_atk_t, y_atk)

        logging.info("=== Training pipeline COMPLETE ===")
        return results

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SCA training pipeline")
    parser.add_argument("--strategy", type=str, default="snr",
                        choices=["snr", "anova", "pca"],
                        help="POI extraction strategy")
    parser.add_argument("--k", type=int, default=50,
                        help="Number of POIs / PCA components")
    args = parser.parse_args()

    run_training_pipeline(strategy=args.strategy, k=args.k)