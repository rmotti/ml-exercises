import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def evaluate_model(model, X_test, y_test, threshold):
    """
    Avalia o modelo final no conjunto de teste.

    O threshold NÃO é escolhido aqui: ele vem da validação cruzada,
    calculado sobre as probabilidades Out-of-Fold do treino.
    Assim o conjunto de teste permanece intocado e serve como
    estimativa honesta do desempenho em produção.

    Args:
        model: Pipeline já treinado.
        X_test: Features de teste.
        y_test: Rótulos verdadeiros de teste.
        threshold (float): Ponto de corte definido na Cross Validation.

    Returns:
        dict: Métricas do modelo no conjunto de teste.
    """

    # Probabilidade de o pedido atrasar.
    y_proba = model.predict_proba(X_test)[:, 1]

    # Aplica o ponto de corte definido na Cross Validation.
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }

    logger.success("AVALIACAO NO CONJUNTO DE TESTE")
    logger.info(f"Threshold (vindo da CV): {metrics['threshold']:.2f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
    logger.info(f"Precision: {metrics['precision']:.3f}")
    logger.info(f"Recall: {metrics['recall']:.3f}")
    logger.info(f"F1: {metrics['f1']:.3f}")
    logger.info(f"ROC AUC: {metrics['roc_auc']:.3f}")
    logger.info(f"PR AUC: {metrics['pr_auc']:.3f}")

    # Leitura operacional: de cada 100 pedidos priorizados,
    # quantos realmente atrasariam, e quantos atrasos capturamos.
    n_priorizados = int(y_pred.sum())
    n_atrasos = int(np.asarray(y_test).sum())

    logger.info(
        f"Pedidos priorizados: {n_priorizados:,} de {len(y_pred):,} "
        f"({n_priorizados / len(y_pred):.1%} da operacao)"
    )
    logger.info(
        f"Atrasos capturados: {int(metrics['recall'] * n_atrasos):,} "
        f"de {n_atrasos:,}"
    )

    return metrics
