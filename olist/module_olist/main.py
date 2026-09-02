"""Pipeline executavel do caso Olist.

Objetivo: logo apos a aprovacao do pagamento, estimar quais pedidos tem
maior risco de serem entregues depois do prazo prometido, apoiando a
priorizacao operacional e o acompanhamento preventivo.

Fluxo:
1. Constroi o dataset a nivel de pedido;
2. Separa treino e teste;
3. Cross Validation no TREINO -> escolhe modelo e threshold;
4. Treina o vencedor em todo o treino;
5. Avalia uma unica vez no TESTE;
6. Salva o modelo e o threshold.
"""

import json

import joblib
from loguru import logger

from module_olist.config import INTERIM_DATA_DIR, MODELS_DIR, RAW_DATA_DIR
from module_olist.dataset import create_dataset, load_data, save_dataset
from module_olist.features import create_features
from module_olist.modeling.cross_validation import cross_validate_models
from module_olist.modeling.evaluate import evaluate_model
from module_olist.modeling.split import split_data
from module_olist.modeling.train import train_best_model

ORDERS_PATH = RAW_DATA_DIR / "olist_orders_dataset.csv"
ITEMS_PATH = RAW_DATA_DIR / "olist_order_items_dataset.csv"
CUSTOMERS_PATH = RAW_DATA_DIR / "olist_customers_dataset.csv"
OUTPUT_PATH = INTERIM_DATA_DIR / "dataset.csv"

MODEL_PATH = MODELS_DIR / "model.pkl"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"


def main() -> None:
    """Constroi o dataset, seleciona o modelo por CV e avalia no teste."""

    # -------------------------------------------------
    # 1. Dataset
    # -------------------------------------------------

    logger.info("Carregando as tabelas brutas da Olist...")
    orders, items, customers = load_data(ORDERS_PATH, ITEMS_PATH, CUSTOMERS_PATH)

    logger.info("Construindo a base analitica a nivel de pedido...")
    dataset = create_dataset(orders, items, customers)
    dataset = create_features(dataset)

    save_dataset(dataset, OUTPUT_PATH)
    logger.success(f"Dataset criado com {len(dataset):,} pedidos.")

    # -------------------------------------------------
    # 2. Separacao treino / teste
    # -------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(dataset)

    logger.info(f"Treino: {len(X_train):,} pedidos")
    logger.info(f"Teste: {len(X_test):,} pedidos")

    # -------------------------------------------------
    # 3. Cross Validation (somente no treino)
    #
    # A escolha do modelo e do threshold acontece aqui,
    # com probabilidades Out-of-Fold. O conjunto de teste
    # nao participa dessa decisao.
    # -------------------------------------------------

    best_model_name, best_threshold = cross_validate_models(X_train, y_train)

    # -------------------------------------------------
    # 4. Treino do modelo vencedor
    # -------------------------------------------------

    model = train_best_model(best_model_name, X_train, y_train)

    # -------------------------------------------------
    # 5. Avaliacao final (uma unica vez)
    # -------------------------------------------------

    evaluate_model(model, X_test, y_test, best_threshold)

    # -------------------------------------------------
    # 6. Persistencia
    # -------------------------------------------------

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    THRESHOLD_PATH.write_text(
        json.dumps(
            {"model": best_model_name, "threshold": best_threshold},
            indent=2,
        )
    )

    logger.success(f"Modelo salvo em {MODEL_PATH}")
    logger.success(f"Threshold salvo em {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()
