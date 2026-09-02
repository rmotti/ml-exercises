import pandas as pd

from loguru import logger

from module_olist.modeling.pipeline import (
    create_gradient_boosting_pipeline,
    create_xgboost_pipeline,
    create_lightgbm_pipeline,
)

PIPELINES = {
    "Gradient Boosting": create_gradient_boosting_pipeline,
    "XGBoost": create_xgboost_pipeline,
    "LightGBM": create_lightgbm_pipeline,
}


def train_best_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """
    Treina o modelo vencedor da Cross Validation em todo o conjunto de treino.

    A comparacao entre modelos ja foi feita na validacao cruzada.
    Aqui treinamos apenas o vencedor, usando 100% dos dados de treino.

    Args:
        model_name (str): Nome do modelo escolhido na Cross Validation.
        X_train (pd.DataFrame): Features de treino.
        y_train (pd.Series): Alvo de treino.

    Returns:
        Pipeline: Modelo treinado.
    """

    logger.info(f"Treinando modelo final: {model_name}")

    pipeline = PIPELINES[model_name]()
    pipeline.fit(X_train, y_train)

    logger.success(f"Modelo {model_name} treinado.")

    return pipeline
