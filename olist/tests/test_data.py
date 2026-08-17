from pathlib import Path

import pandas as pd
import pandera.pandas as pa

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
ORDERS_PATH = RAW_DATA_DIR / "olist_orders_dataset.csv"
VALID_ORDER_STATUSES = {
    "created",
    "approved",
    "invoiced",
    "processing",
    "shipped",
    "delivered",
    "unavailable",
    "canceled",
}


def test_orders_have_unique_non_null_ids():
    orders = pd.read_csv(
        ORDERS_PATH,
        usecols=["order_id"],
    )
    schema = pa.DataFrameSchema(
        {
            "order_id": pa.Column(
                str,
                nullable=False,
                unique=True,
            )
        }
    )

    schema.validate(orders)


def test_orders_have_non_null_customer_ids():
    orders = pd.read_csv(
        ORDERS_PATH,
        usecols=["customer_id"],
    )
    schema = pa.DataFrameSchema(
        {
            "customer_id": pa.Column(
                str,
                nullable=False,
            )
        }
    )

    schema.validate(orders)


def test_orders_have_valid_statuses():
    orders = pd.read_csv(
        ORDERS_PATH,
        usecols=["order_status"],
    )
    schema = pa.DataFrameSchema(
        {
            "order_status": pa.Column(
                str,
                pa.Check.isin(VALID_ORDER_STATUSES),
                nullable=False,
            )
        }
    )

    schema.validate(orders)


def test_order_approval_is_not_before_purchase():
    date_columns = ["order_purchase_timestamp", "order_approved_at"]
    orders = pd.read_csv(
        ORDERS_PATH,
        usecols=date_columns,
        parse_dates=date_columns,
    )
    schema = pa.DataFrameSchema(
        {
            "order_purchase_timestamp": pa.Column(
                "datetime64[ns]",
                nullable=False,
            ),
            "order_approved_at": pa.Column(
                "datetime64[ns]",
                nullable=True,
            ),
        },
        checks=pa.Check(
            lambda dataframe: dataframe["order_approved_at"].isna()
            | (
                dataframe["order_approved_at"]
                >= dataframe["order_purchase_timestamp"]
            )
        ),
    )

    schema.validate(orders)
