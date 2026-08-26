import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    "purchase_hour",
    "purchase_weekday",
    "purchase_month",
    "promised_days",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
    "costumer_state",
]

TARGET = "is_late"

def split_data(data: pd.DataFrame):
    """
    Split the data into train and test sets.

    Args:
        data (pd.DataFrame): The input data.
    """
    X = data[FEATURES]
    y = data[TARGET]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    
