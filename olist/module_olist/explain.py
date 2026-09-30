import pandas as pd
import shap
import matplotlib.pyplot as plt
from loguru import logger

from module_olist.config import(
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
)

from module_olist.modeling.predict import(
    load_model,
)

from module_olist.modeling.split import(
    split_data,
)

from module_olist.modeling.interpret import(
    prepare_data_for_shap,
    create_explainer,
    calculate_shap_values,
)

def main():
    # Load the trained model pipeline
    pipeline, _, _ = load_model(
        model_path=MODELS_DIR / "best_model.joblib",
        metadata_path=MODELS_DIR / "metadata.json",
    )

    # Load the dataset for SHAP analysis (same test set used in main.py)
    data = pd.read_csv(INTERIM_DATA_DIR / "orders_dataset_refined.csv")
    _, X, _, _ = split_data(data)

    # Prepare the data for SHAP analysis
    X_transformed = prepare_data_for_shap(pipeline, X)

    # Create a SHAP explainer
    explainer = create_explainer(pipeline)

    # Calculate SHAP values
    shap_values = calculate_shap_values(explainer, X_transformed)

    # Plot SHAP summary plot
    plt.figure(figsize=(10, 6))
    # show=False keeps the plot open so savefig doesn't write a blank figure
    shap.summary_plot(shap_values, X_transformed, show=False)
    plt.savefig(FIGURES_DIR / "shap_summary_plot.png", bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot saved.")

if __name__ == "__main__":
    main()
