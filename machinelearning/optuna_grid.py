from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from catboost import CatBoostClassifier
import optuna

optuna_grid = {
    "NestedCV": {
        "RandomForestClassifier": {
            "n_estimators": optuna.distributions.IntDistribution(2, 200),
            "criterion": optuna.distributions.CategoricalDistribution(
                ["gini", "entropy"]
            ),
            "max_depth": optuna.distributions.IntDistribution(1, 50),
            "min_samples_leaf": optuna.distributions.IntDistribution(1, 10),
            "min_samples_split": optuna.distributions.IntDistribution(2, 10),
            "bootstrap": optuna.distributions.CategoricalDistribution([True, False]),
        },
        "KNeighborsClassifier": {
            "n_neighbors": optuna.distributions.IntDistribution(2, 15),
            "weights": optuna.distributions.CategoricalDistribution(
                ["uniform", "distance"]
            ),
            "algorithm": optuna.distributions.CategoricalDistribution(
                ["auto", "ball_tree", "kd_tree", "brute"]
            ),
            "p": optuna.distributions.IntDistribution(1, 2),
            "leaf_size": optuna.distributions.IntDistribution(5, 50),
        },
        "DecisionTreeClassifier": {
            "criterion": optuna.distributions.CategoricalDistribution(
                ["gini", "entropy"]
            ),
            "splitter": optuna.distributions.CategoricalDistribution(
                ["best", "random"]
            ),
            "max_depth": optuna.distributions.IntDistribution(1, 100),
            "min_samples_split": optuna.distributions.IntDistribution(2, 10),
            "min_weight_fraction_leaf": optuna.distributions.IntDistribution(0.0, 0.5),
        },
        "SVC": {
            "C": optuna.distributions.IntDistribution(1, 10),
            "kernel": optuna.distributions.CategoricalDistribution(
                ["linear", "rbf", "sigmoid", "poly"]
            ),
            "degree": optuna.distributions.IntDistribution(1, 10),
            "probability": optuna.distributions.CategoricalDistribution([True, False]),
            "shrinking": optuna.distributions.CategoricalDistribution([True, False]),
            "decision_function_shape": optuna.distributions.CategoricalDistribution(
                ["ovo", "ovr"]
            ),
        },
        "GradientBoostingClassifier": {
            "loss": optuna.distributions.CategoricalDistribution(
                ["log_loss", "exponential"]
            ),
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.5),
            "n_estimators": optuna.distributions.IntDistribution(2, 200),
            "criterion": optuna.distributions.CategoricalDistribution(
                ["friedman_mse", "squared_error"]
            ),
            "max_depth": optuna.distributions.IntDistribution(1, 50),
            "min_samples_split": optuna.distributions.IntDistribution(2, 10),
            "min_samples_leaf": optuna.distributions.IntDistribution(1, 10),
        },
        "XGBClassifier": {
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 1),
            "n_estimators": optuna.distributions.IntDistribution(2, 500),
            "max_depth": optuna.distributions.IntDistribution(1, 50),
            "min_child_weight": optuna.distributions.IntDistribution(1, 10),
            "gamma": optuna.distributions.FloatDistribution(0, 10),
            "subsample": optuna.distributions.FloatDistribution(0.001, 1.0),
            "colsample_bytree": optuna.distributions.FloatDistribution(0.1, 1.0),
            "n_jobs": optuna.distributions.CategoricalDistribution([-1]),
            "booster": optuna.distributions.CategoricalDistribution(
                ["gbtree", "dart"]
            ),  #'gblinear',
            "tree_method": optuna.distributions.CategoricalDistribution(
                ["auto", "exact", "approx", "hist"]
            ),
            "reg_alpha": optuna.distributions.FloatDistribution(0, 5),
            "reg_lambda": optuna.distributions.FloatDistribution(0, 5),
            "scale_pos_weight": optuna.distributions.FloatDistribution(0, 5),
            "objective": optuna.distributions.CategoricalDistribution(
                ["binary:logistic"]
            ),
        },
        "LinearDiscriminantAnalysis": {
            "solver": optuna.distributions.CategoricalDistribution(["lsqr", "eigen"]),
            "shrinkage": optuna.distributions.FloatDistribution(0.0, 1.0),
            "tol": optuna.distributions.CategoricalDistribution([1e-3, 1e-4, 1e-5]),
            "store_covariance": optuna.distributions.CategoricalDistribution(
                [True, False]
            ),
        },
        "LogisticRegression": {
            "penalty": optuna.distributions.CategoricalDistribution(
                ["l1", "l2", None]#, "elasticnet"]
            ),
            "C": optuna.distributions.FloatDistribution(0.1, 10.0),
            "solver": optuna.distributions.CategoricalDistribution(
                ["newton-cg", "lbfgs", "sag", "saga", "newton-cholesky", "liblinear"]
            ),
            "max_iter": optuna.distributions.IntDistribution(100, 1000),
        },
        "GaussianNB": {
            "var_smoothing": optuna.distributions.FloatDistribution(1e-9, 1e-5)
        },
        'LGBMClassifier': {
            'boosting_type': optuna.distributions.CategoricalDistribution(['gbdt', 'dart', 'goss']),
            'num_leaves': optuna.distributions.IntDistribution(2, 256),
            'learning_rate': optuna.distributions.FloatDistribution(0.01, 0.5),
            'n_estimators': optuna.distributions.IntDistribution(2, 200),
            # 'min_child_samples': optuna.distributions.IntDistribution(5, 100),
            # 'reg_alpha': optuna.distributions.FloatDistribution(0.0, 1.0),
            # 'reg_lambda': optuna.distributions.FloatDistribution(0.0, 1.0),
            'subsample_for_bin': optuna.distributions.IntDistribution(100000, 400000),
            'objective': optuna.distributions.CategoricalDistribution(['binary']),
            'min_split_gain': optuna.distributions.FloatDistribution(0.0, 1.0),
            'n_jobs': optuna.distributions.CategoricalDistribution([-1]),
            'verbose': optuna.distributions.CategoricalDistribution([-1]),
        }
    },
    "ManualSearch": {
        "RandomForestClassifier": lambda trial: RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 2, 200),
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            max_depth=trial.suggest_int("max_depth", 1, 50),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
            n_jobs=-1,
        ),
        "KNeighborsClassifier": lambda trial: KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors", 2, 15),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            algorithm=trial.suggest_categorical(
                "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
            ),
            p=trial.suggest_int("p", 1, 2),
            leaf_size=trial.suggest_int("leaf_size", 5, 50),
            n_jobs=-1,
        ),
        "DecisionTreeClassifier": lambda trial: DecisionTreeClassifier(
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            splitter=trial.suggest_categorical("splitter", ["best", "random"]),
            max_depth=trial.suggest_int("max_depth", 1, 100),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_weight_fraction_leaf=trial.suggest_float(
                "min_weight_fraction_leaf", 0.0, 0.5
            ),
        ),
        "SVC": lambda trial: SVC(
            C=trial.suggest_int("C", 1, 10),
            kernel=trial.suggest_categorical(
                "kernel", ["linear", "rbf", "sigmoid", "poly"]
            ),
            degree=(
                trial.suggest_int("degree", 1, 10)
                if trial.params.get("kernel") == "poly"
                else 3
            ),
            probability=trial.suggest_categorical("probability", [True, False]),
            shrinking=trial.suggest_categorical("shrinking", [True, False]),
            decision_function_shape=trial.suggest_categorical(
                "decision_function_shape", ["ovo", "ovr"]
            ),
        ),
        "GradientBoostingClassifier": lambda trial: GradientBoostingClassifier(
            loss=trial.suggest_categorical("loss", ["log_loss", "exponential"]),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.5),
            n_estimators=trial.suggest_int("n_estimators", 2, 200),
            criterion=trial.suggest_categorical(
                "criterion", ["friedman_mse", "squared_error"]
            ),
            max_depth=trial.suggest_int("max_depth", 1, 50),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        ),
        "XGBClassifier": lambda trial: XGBClassifier(
            learning_rate=trial.suggest_float("learning_rate", 0.001, 0.5),
            n_estimators=trial.suggest_int("n_estimators", 2, 500),
            max_depth=trial.suggest_int("max_depth", 1, 50),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0, 5),
            subsample=trial.suggest_float("subsample", 0.001, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.001, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0, 1),
            reg_lambda=trial.suggest_float("reg_lambda", 0.001, 1),
            n_jobs=-1,
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1, 100, log=True),
            objective="binary:logistic",  # trial.suggest_categorical('objective', ['binary:logistic', 'multi:softprob'])
            booster=trial.suggest_categorical(
                "booster", ["gbtree", "dart"]
            ),  #'gblinear'
            tree_method=trial.suggest_categorical(
                "tree_method", ["auto", "exact", "approx", "hist"]
            ),
        ),
        "LinearDiscriminantAnalysis": lambda trial: LinearDiscriminantAnalysis(
            solver=trial.suggest_categorical("solver", ["svd", "lsqr", "eigen"]),
            shrinkage=(
                trial.suggest_float("shrinkage", 0.0, 1.0)
                if trial.params.get("solver") != "svd"
                else None
            ),
            store_covariance=(
                trial.suggest_categorical("store_covariance", [True, False])
                if trial.params.get("solver") == "svd"
                else False
            ),
            tol=trial.suggest_categorical("tol", [1e-3, 1e-4, 1e-5]),
        ),
        "LogisticRegression": lambda trial: LogisticRegression(
            penalty=trial.suggest_categorical("penalty", ["l1", "l2", "none"]),
            C=trial.suggest_float("C", 0.1, 10.0),
            solver=(
                trial.suggest_categorical("solver_1", ["liblinear", "saga"])
                if trial.params["penalty"] == "l1"
                else (
                    trial.suggest_categorical(
                        "solver_2",
                        [
                            "newton-cg",
                            "lbfgs",
                            "sag",
                            "saga",
                            "newton-cholesky",
                            "liblinear",
                        ],
                    )
                    if trial.params["penalty"] == "l2"
                    else trial.suggest_categorical(
                        "solver_3",
                        ["lbfgs", "newton-cg", "sag", "saga", "newton-cholesky"],
                    )
                )
            ),
            max_iter=trial.suggest_int("max_iter", 100, 1000),
            fit_intercept=trial.suggest_categorical("fit_intercept", [True, False]),
            n_jobs=-1,
        ),
        "GaussianNB": lambda trial: GaussianNB(
            var_smoothing=trial.suggest_float("var_smoothing", 1e-9, 1e-5)
        ),
        'LGBMClassifier': lambda trial: LGBMClassifier(
        boosting_type=trial.suggest_categorical('boosting_type', ['gbdt', 'dart','goss', 'rf']),
        num_leaves=trial.suggest_int('num_leaves', 2, 256),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.5),
        n_estimators=trial.suggest_int('n_estimators', 2, 200),
        subsample_for_bin=trial.suggest_int('subsample_for_bin', 100000, 400000),
        objective='binary',
        min_split_gain=trial.suggest_float('min_split_gain', 0.0, 1.0),
        n_jobs=-1,
        bagging_fraction=trial.suggest_float('bagging_fraction', 0.1, 0.9) if trial.params.get('boosting_type', 'gbdt') not in ['goss', 'rf'] else 1.0,
        bagging_freq=trial.suggest_int('bagging_freq', 1, 7) if trial.params.get('boosting_type', 'gbdt') == 'rf' else 0,
        feature_fraction=trial.suggest_float('feature_fraction', 0.1, 0.9) if trial.params.get('boosting_type', 'gbdt') != 'goss' else 1.0,
        verbose=-1
    )

    },
}
