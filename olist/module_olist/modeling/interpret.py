import pandas as pd 
import shap 
from scipy import sparse

def prepare_data_for_shap(pipeline, X):
    """
    Prepares the data for SHAP analysis by transforming it using the provided pipeline.

    Args:
        pipeline: A fitted sklearn pipeline that includes preprocessing steps.
        X (pd.DataFrame): The input features DataFrame.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    X_transformed = preprocessor.transform(X)

    if sparse.issparse(X_transformed):
        X_transformed = X_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    X_transformed = pd.DataFrame(
        X_transformed, 
        columns=feature_names,
        index=X.index,
    )

    return X_transformed

def create_explainer(pipeline):
    """
    Creates a SHAP explainer for the model in the provided pipeline.

    Args:
        pipeline: A fitted sklearn pipeline that includes a model.

    Returns:
        shap.TreeExplainer: A SHAP explainer object.
    """
    model = pipeline.named_steps['model']
    explainer = shap.TreeExplainer(model)
    return explainer

def calculate_shap_values(explainer, X_transformed):
    """
    Calculates SHAP values for the transformed data using the provided explainer.

    Args:
        explainer: A SHAP explainer object.
        X_transformed (pd.DataFrame): The transformed input features DataFrame.

    Returns:
        np.ndarray: The SHAP values for the transformed data.
    """
    shap_values = explainer.shap_values(X_transformed)
    
    return shap_values                  