from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

def _create_model_instance(model_name, params):
    """
    Create a machine learning model instance with the specified parameters.

    This function initializes an instance of the specified model with the given
    parameters or default configurations. It ensures reusability and consistency
    in model creation, particularly for pipelines or repetitive experiments.

    Parameters:
    -----------
    model_name : str
        The name of the model to instantiate. Must be one of the supported models.
    params : dict or None
        A dictionary of hyperparameters to configure the model. If None, the model
        is initialized with default parameters.

    Returns:
    --------
    object
        An instance of the specified model with the provided parameters.

    Raises:
    -------
    ValueError
        If the specified model name is not supported.
    """
    if model_name == "RandomForestClassifier":
        return RandomForestClassifier(**params) if params else RandomForestClassifier()
    elif model_name == "LogisticRegression":
        return LogisticRegression(**params) if params else LogisticRegression()
    elif model_name == "XGBClassifier":
        return XGBClassifier(**params) if params else XGBClassifier()
    elif model_name == "LGBMClassifier":
        return LGBMClassifier(**params) if params else LGBMClassifier(verbose=-1)
    elif model_name == "CatBoostClassifier":
        return CatBoostClassifier(**params) if params else CatBoostClassifier(verbose=0)
    elif model_name == "SVC":
        return SVC(**params) if params else SVC()
    elif model_name == "KNeighborsClassifier":
        return KNeighborsClassifier(**params) if params else KNeighborsClassifier()
    elif model_name == "LinearDiscriminantAnalysis":
        return LinearDiscriminantAnalysis(**params) if params else LinearDiscriminantAnalysis()
    elif model_name == "GaussianNB":
        return GaussianNB(**params) if params else GaussianNB()
    elif model_name == "GradientBoostingClassifier":
        return GradientBoostingClassifier(**params) if params else GradientBoostingClassifier()
    elif model_name == "GaussianProcessClassifier":
        return GaussianProcessClassifier(**params) if params else GaussianProcessClassifier()
    elif model_name == "ElasticNet":
        return LogisticRegression(**params) if params else LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5)
    else:
        raise ValueError(f"Unsupported model: {model_name}")