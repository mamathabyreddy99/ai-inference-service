"""Runtime configuration, read from environment variables at call time."""
import os


def get_model_path() -> str:
    return os.environ.get("MODEL_PATH", "models/sentiment.joblib")


def get_db_path() -> str:
    return os.environ.get("DB_PATH", "data/predictions.db")
