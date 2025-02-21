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

def _create_model_instance(model_name: str, params: dict = None) -> object:
    """
    Create and configure a machine learning model instance with specified parameters.

    This factory function instantiates a machine learning model based on the provided
    model name and configures it with the given parameters. If no parameters are provided,
    the model is created with default settings.

    Parameters
    ----------
    model_name : str
        The name of the model to instantiate. Must be one of the supported model types.
    params : dict, optional
        Dictionary of hyperparameters to configure the model. If None, default parameters
        are used.

    Returns
    -------
    object
        An initialized instance of the specified model type with the given parameters.
    """
    # Dictionary mapping model names to their instantiation functions
    model_factories = {
        "RandomForestClassifier": lambda p: RandomForestClassifier(**p) if p else RandomForestClassifier(),
        "LogisticRegression": lambda p: LogisticRegression(**p) if p else LogisticRegression(),
        "XGBClassifier": lambda p: XGBClassifier(**p) if p else XGBClassifier(),
        "LGBMClassifier": lambda p: LGBMClassifier(**p) if p else LGBMClassifier(verbose=-1),
        "CatBoostClassifier": lambda p: CatBoostClassifier(**p) if p else CatBoostClassifier(verbose=0),
        "SVC": lambda p: SVC(**p) if p else SVC(),
        "KNeighborsClassifier": lambda p: KNeighborsClassifier(**p) if p else KNeighborsClassifier(),
        "LinearDiscriminantAnalysis": lambda p: LinearDiscriminantAnalysis(**p) if p else LinearDiscriminantAnalysis(),
        "GaussianNB": lambda p: GaussianNB(**p) if p else GaussianNB(),
        "GradientBoostingClassifier": lambda p: GradientBoostingClassifier(**p) if p else GradientBoostingClassifier(),
        "GaussianProcessClassifier": lambda p: GaussianProcessClassifier(**p) if p else GaussianProcessClassifier(),
        "ElasticNet": lambda p: LogisticRegression(**p) if p else LogisticRegression(
            penalty="elasticnet", 
            solver="saga", 
            l1_ratio=0.5
        )
    }

    # Check if the requested model is supported
    if model_name not in model_factories:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Supported models are: {', '.join(model_factories.keys())}"
        )

    # Create and return the model instance
    return model_factories[model_name](params)