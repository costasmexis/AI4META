# Model instance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

def _create_model_instance(model_name, params):
    """
    This function creates a model instance with the given parameters.
    It is used in order to prevent fitting of an already fitted model from previous runs.

    Args:
        model_name (str): The name of the model to create.
        params (dict): The parameters to use when creating the model.

    Returns:
        object: An instance of the specified model with the given parameters.

    Raises:
        ValueError: If the model name is not recognized.
    """
    if model_name == "RandomForestClassifier":
        if params is None:
            return RandomForestClassifier()
        else:
            return RandomForestClassifier(**params)
    elif model_name == "LogisticRegression":
        if params is None:
            return LogisticRegression()
        else:
            return LogisticRegression(**params)
    elif model_name == "XGBClassifier":
        if params is None:
            return XGBClassifier()
        else:
            return XGBClassifier(**params)
    elif model_name == "LGBMClassifier":
        if params is None:
            return LGBMClassifier(verbose=-1)
        else:
            return LGBMClassifier(**params)
    elif model_name == "CatBoostClassifier":
        if params is None:
            return CatBoostClassifier(verbose=0)
        else:
            return CatBoostClassifier(**params)
    elif model_name == "SVC":
        if params is None:
            return SVC()
        else:
            return SVC(**params)
    elif model_name == "KNeighborsClassifier":
        if params is None:
            return KNeighborsClassifier()
        else:
            return KNeighborsClassifier(**params)
    elif model_name == "LinearDiscriminantAnalysis":
        if params is None:
            return LinearDiscriminantAnalysis()
        else:
            return LinearDiscriminantAnalysis(**params)
    elif model_name == "GaussianNB":
        if params is None:
            return GaussianNB()
        else:
            return GaussianNB(**params)
    elif model_name == "GradientBoostingClassifier":
        if params is None:
            return GradientBoostingClassifier()
        else:
            return GradientBoostingClassifier(**params)
    elif model_name == "GaussianProcessClassifier":
        if params is None:
            return GaussianProcessClassifier()
        else:
            return GaussianProcessClassifier(**params)
    elif model_name == "ElasticNet":
        if params is None:
            return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5)
        else:
            return LogisticRegression(**params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")