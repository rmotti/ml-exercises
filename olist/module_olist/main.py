from module_olist.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from module_olist.dataset import create_dataset, load_data, save_dataset
from module_olist.features import create_features

ORDERS_PATH = RAW_DATA_DIR / "olist_orders_dataset.csv"
ITEMS_PATH = RAW_DATA_DIR / "olist_order_items_dataset.csv"
CUSTOMERS_PATH = RAW_DATA_DIR / "olist_customers_dataset.csv"
OUTPUT_PATH = INTERIM_DATA_DIR / "dataset.csv"


def main() -> None:
    """Carrega os dados brutos, aplica os tratamentos e salva a base intermediaria."""
    orders, items, customers = load_data(ORDERS_PATH, ITEMS_PATH, CUSTOMERS_PATH)

    dataset = create_dataset(orders, items, customers)
    dataset = create_features(dataset)

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_dataset(dataset, OUTPUT_PATH)


if __name__ == "__main__":
    main()
