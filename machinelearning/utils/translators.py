from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


AVAILABLE_CLFS = {
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
            "LogisticRegression": LogisticRegression(),
            "ElasticNet": LogisticRegression(penalty="elasticnet", solver="saga"),
            "XGBClassifier": XGBClassifier(),
            "GaussianNB": GaussianNB(),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "SVC": SVC(),
            "LGBMClassifier": LGBMClassifier(),
            "GaussianProcessClassifier": GaussianProcessClassifier(),
            "CatBoostClassifier": CatBoostClassifier(),
        }

METRIC_ADDREVIATIONS = {
        'roc_auc': 'AUC',
        'accuracy': 'ACC',
        'balanced_accuracy': 'BAL_ACC',
        'recall': 'REC',
        'precision': 'PREC',
        'f1': 'F1',
        'average_precision': 'AVG_PREC',
        'specificity': 'SPEC',
        'matthews_corrcoef': 'MCC'
    }