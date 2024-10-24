from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from catboost import CatBoostClassifier
import optuna

optuna_grid = {
    "NestedCV": {
        "RandomForestClassifier": {
            "n_estimators": optuna.distributions.IntDistribution(2, 300),
            "criterion": optuna.distributions.CategoricalDistribution(
                ["gini", "entropy"]
            ),
            "max_depth": optuna.distributions.IntDistribution(2, 100),
            "min_samples_leaf": optuna.distributions.IntDistribution(2,15),
            "bootstrap": optuna.distributions.CategoricalDistribution([True, False]),
            "n_jobs": optuna.distributions.CategoricalDistribution([1]),
        },
        "KNeighborsClassifier": {
            "n_neighbors": optuna.distributions.IntDistribution(1, 15),
            "weights": optuna.distributions.CategoricalDistribution(
                ["uniform", "distance"]
            ),
            "algorithm": optuna.distributions.CategoricalDistribution(
                ["auto", "ball_tree", "kd_tree", "brute"]
            ),
            "p": optuna.distributions.IntDistribution(1, 2),
            "leaf_size": optuna.distributions.IntDistribution(5, 50),
            "n_jobs": optuna.distributions.CategoricalDistribution([1]),
        },
        "SVC": {
            "C": optuna.distributions.FloatDistribution(0.01, 1.0),
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
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.2),
            "n_estimators": optuna.distributions.IntDistribution(2, 300),
            "criterion": optuna.distributions.CategoricalDistribution(
                ["friedman_mse", "squared_error"]
            ),
            "max_depth": optuna.distributions.IntDistribution(2, 100),
            "min_samples_leaf": optuna.distributions.IntDistribution(2,15),
        },
        "XGBClassifier": {
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.2),
            "n_estimators": optuna.distributions.IntDistribution(50, 500),
            "max_depth": optuna.distributions.IntDistribution(2, 12),
            "subsample": optuna.distributions.FloatDistribution(0.5, 1.0),
            "n_jobs": optuna.distributions.CategoricalDistribution([1]),
            "booster": optuna.distributions.CategoricalDistribution(["gbtree", "dart"]),
            "tree_method": optuna.distributions.CategoricalDistribution(
                ["auto", "exact", "approx", "hist"]
            ),
            "reg_alpha": optuna.distributions.FloatDistribution(0, 0.5),
            "reg_lambda": optuna.distributions.FloatDistribution(0.5, 1),
            "objective": optuna.distributions.CategoricalDistribution(
                ["binary:logistic"]
            ),
            "eval_metric": optuna.distributions.CategoricalDistribution(["error"]),
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
            "penalty": optuna.distributions.CategoricalDistribution(["l1", "l2", None]),
            "C": optuna.distributions.FloatDistribution(0.01, 1.0),
            "solver": optuna.distributions.CategoricalDistribution(
                ["newton-cg", "lbfgs", "sag", "saga", "liblinear"]
            ),
            "max_iter": optuna.distributions.IntDistribution(100, 1500),
            "n_jobs": optuna.distributions.CategoricalDistribution([None]),
        },
        "ElasticNet": {
            "penalty": optuna.distributions.CategoricalDistribution(["elasticnet"]),
            "C": optuna.distributions.FloatDistribution(0.01, 1.0),
            "solver": optuna.distributions.CategoricalDistribution(["saga"]),
            "max_iter": optuna.distributions.IntDistribution(100, 1500),
            "n_jobs": optuna.distributions.CategoricalDistribution([None]),
            "l1_ratio": optuna.distributions.FloatDistribution(0.0, 1.0),
        },
        "GaussianNB": {
            "var_smoothing": optuna.distributions.FloatDistribution(1e-10, 1e-5)
        },
        "LGBMClassifier": {
            "boosting_type": optuna.distributions.CategoricalDistribution(
                ["gbdt", "dart", "goss"]
            ),
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.2),
            "n_estimators": optuna.distributions.IntDistribution(2, 300),
            "max_depth": optuna.distributions.IntDistribution(2, 50),
            "objective": optuna.distributions.CategoricalDistribution(["binary"]),
            "min_child_samples": optuna.distributions.IntDistribution(2, 15),
            "n_jobs": optuna.distributions.CategoricalDistribution([1]),
            "verbose": optuna.distributions.CategoricalDistribution([-1]),
            "reg_lambda": optuna.distributions.FloatDistribution(0.5, 1.0),
            "reg_alpha": optuna.distributions.FloatDistribution(0,0.5),
        },
        "GaussianProcessClassifier": {
            "optimizer": optuna.distributions.CategoricalDistribution(
                ["fmin_l_bfgs_b", None]
            ),
            "max_iter_predict": optuna.distributions.IntDistribution(50, 200),
            "warm_start": optuna.distributions.CategoricalDistribution([True, False]),
            "n_jobs": optuna.distributions.CategoricalDistribution([1]),
            "n_restarts_optimizer": optuna.distributions.IntDistribution(0, 10),
        },
        "CatBoostClassifier": {
            # "iterations": optuna.distributions.IntDistribution(50, 1000),
            "n_estimators": optuna.distributions.IntDistribution(50, 300),
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.2),
            "depth": optuna.distributions.IntDistribution(2, 16),
            "l2_leaf_reg": optuna.distributions.FloatDistribution(1e-8, 2),
            "border_count": optuna.distributions.IntDistribution(1, 255),
            "bagging_temperature": optuna.distributions.FloatDistribution(0.0, 10.0),
            "random_strength": optuna.distributions.FloatDistribution(0.0, 10.0),
            "leaf_estimation_method": optuna.distributions.CategoricalDistribution(
                ["Newton", "Gradient"]
            ),
            "verbose": optuna.distributions.CategoricalDistribution([0]),
            "model_size_reg": optuna.distributions.FloatDistribution(1e-3, 1),
            "min_data_in_leaf": optuna.distributions.IntDistribution(2, 15),
            "thread_count": optuna.distributions.CategoricalDistribution([1]),
            "loss_function": optuna.distributions.CategoricalDistribution(
                ["Logloss", "CrossEntropy"]
            ),
        },
        # "Lasso": {
        #     "alpha": optuna.distributions.FloatDistribution(0.01, 1.0),
        #     "selection": optuna.distributions.CategoricalDistribution([True, False]),
        #     "warm_start": optuna.distributions.CategoricalDistribution([True, False]),
        #     "positive": optuna.distributions.CategoricalDistribution([True, False]),
        #     "precompute": optuna.distributions.CategoricalDistribution([True, False]),
        # },
    },
 
    "ManualSearch": {
        "RandomForestClassifier": lambda trial: RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 2, 300),
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2,15),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
            max_depth=trial.suggest_int("max_depth", 2, 100),
            n_jobs=-1,
        ),
        "KNeighborsClassifier": lambda trial: KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors", 1, 15),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            algorithm=trial.suggest_categorical(
                "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
            ),
            p=trial.suggest_int("p", 1, 2),
            leaf_size=trial.suggest_int("leaf_size", 5, 50),
            n_jobs=-1,
        ),
        "SVC": lambda trial: SVC(
            C=trial.suggest_float("C", 0.01, 1.0),
            kernel=trial.suggest_categorical(
                "kernel", ["linear", "rbf", "sigmoid", "poly"]
            ),
            degree=(
                trial.suggest_int("degree", 1, 10)
                if trial.params.get("kernel") == "poly"
                else 3
            ),
            probability=trial.suggest_categorical("probability", [True]),
            shrinking=trial.suggest_categorical("shrinking", [True, False]),
            decision_function_shape=trial.suggest_categorical(
                "decision_function_shape", ["ovo", "ovr"]
            ),
        ),
        "GradientBoostingClassifier": lambda trial: GradientBoostingClassifier(
            loss=trial.suggest_categorical("loss", ["log_loss", "exponential"]),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
            n_estimators=trial.suggest_int("n_estimators", 2, 400),
            criterion=trial.suggest_categorical(
                "criterion", ["friedman_mse", "squared_error"]
            ),
            max_depth=trial.suggest_int("max_depth", 2, 100),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2,15),
        ),
        "XGBClassifier": lambda trial: XGBClassifier(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 15),
            # min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            subsample=trial.suggest_float("subsample", 0.01, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0, 0.5),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 1.0),
            n_jobs=-1,
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1, 100, log=True),
            objective="binary:logistic",
            booster=trial.suggest_categorical("booster", ["gbtree", "dart"]),
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
            penalty=trial.suggest_categorical("penalty", [ "l2", None]),
            solver=trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "sag", "saga","newton-cholesky"]),
            C=trial.suggest_float("C", 0.01, 1.0),
            max_iter=trial.suggest_int("max_iter", 100, 1500),
            fit_intercept=trial.suggest_categorical("fit_intercept", [True, False]),
            n_jobs=-1,
        ),
        "ElasticNet": lambda trial: LogisticRegression(
            penalty=trial.suggest_categorical("penalty", ["elasticnet"]),
            C=trial.suggest_float("C", 0.01, 1.0),
            solver=trial.suggest_categorical("solver", ["saga"]),
            fit_intercept=trial.suggest_categorical("fit_intercept", [True, False]),
            l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
            max_iter=trial.suggest_int("max_iter", 100, 1500),
            n_jobs=-1,
        ),
        "GaussianNB": lambda trial: GaussianNB(
            var_smoothing=trial.suggest_float("var_smoothing", 1e-10, 1e-5)
        ),
        "LGBMClassifier": lambda trial: LGBMClassifier(
            boosting_type=trial.suggest_categorical(
                "boosting_type", ["gbdt", "dart", "goss", "rf"]
            ),
            num_leaves=trial.suggest_int("num_leaves", 2, 256),
            learning_rate=trial.suggest_float("learning_rate", 0.001, 0.5),
            n_estimators=trial.suggest_int("n_estimators", 2, 300),
            subsample_for_bin=trial.suggest_int("subsample_for_bin", 100000, 400000),
            objective="binary",
            min_child_samples=trial.suggest_int("min_child_samples", 2, 15),
            n_jobs=-1,
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 0.5),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.1, 0.9)
            if trial.params.get("boosting_type", "gbdt") not in ["goss", "rf"]
            else 1.0,
            bagging_freq=trial.suggest_int("bagging_freq", 1, 7)
            if trial.params.get("boosting_type", "gbdt") == "rf"
            else 0,
            feature_fraction=trial.suggest_float("feature_fraction", 0.1, 0.9)
            if trial.params.get("boosting_type", "gbdt") != "goss"
            else 1.0,
            verbose=-1,
            max_depth=trial.suggest_int("max_depth", 2, 50),
        ),
        "GaussianProcessClassifier": lambda trial: GaussianProcessClassifier(
            optimizer=trial.suggest_categorical("optimizer", ["fmin_l_bfgs_b", None]),
            max_iter_predict=trial.suggest_int("max_iter_predict", 50, 1500),
            warm_start=trial.suggest_categorical("warm_start", [True, False]),
            n_restarts_optimizer=trial.suggest_int("n_restarts_optimizer", 0, 10),
            n_jobs=-1,
        ),
        "CatBoostClassifier": lambda trial: CatBoostClassifier(
            # iterations=trial.suggest_int("iterations", 50, 1000),
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 0.001, 0.5),
            depth=trial.suggest_int("depth", 2, 16),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
            border_count=trial.suggest_int("border_count", 1, 255),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 10.0),
            random_strength=trial.suggest_float("random_strength", 0.0, 10.0),
            leaf_estimation_method=trial.suggest_categorical(
                "leaf_estimation_method", ["Newton", "Gradient"]
            ),
            logging_level=trial.suggest_categorical("logging_level", ["Silent"]),  # "Silent",
            model_size_reg=trial.suggest_float("model_size_reg", 0.01, 10.0, log=True),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf",  2, 15),
            loss_function=trial.suggest_categorical(
                "loss_function", ["Logloss", "CrossEntropy", None]
            ),
        ),
    },
    "SklearnParameterGrid": {
        "RandomForestClassifier": {
            "n_estimators": [2, 10, 20, 50, 80, 100, 200, 300],
            "criterion": ["gini", "entropy"],
            "max_depth": [2, 8, 15, 25, 30, 45,55,80,100],
            "min_samples_leaf": [1, 2, 3, 5, 8, 10, 12],
            "bootstrap": [True, False],
        },
        "KNeighborsClassifier": {
            "n_neighbors": list(range(1, 16)),
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "p": [1, 2],
            "leaf_size": list(range(5, 51, 5)),
        },
        "SVC": {
            "C": [0.01, 0.1, 0.5, 1.0],
            "kernel": ["linear", "rbf", "sigmoid", "poly"],
            "degree": [1, 3, 5, 7, 10],
            "probability": [True],
            "shrinking": [True, False],
            "decision_function_shape": ["ovo", "ovr"],
        },
        "GradientBoostingClassifier": {
            "loss": ["log_loss", "exponential"],
            "learning_rate": [0.001, 0.01, 0.1, 0.3, 0.5],
            "n_estimators": [2, 50, 100, 200, 400],
            "criterion": ["friedman_mse", "squared_error"],
            "max_depth": [2, 8, 15, 25, 30, 45,55,80,100],
            "min_samples_leaf": [1, 5, 10],
        },
        "XGBClassifier": {
            "learning_rate": [0.01, 0.1, 0.3, 0.5],
            "n_estimators": [2, 50, 100, 200, 400],
            "max_depth": [2, 4, 6, 8, 10, 12, 14, 15],
            "min_child_weight": [1, 3, 5, 7, 10],
            "subsample": [0.5, 0.7, 1.0],
            "reg_alpha": [0, 0.1, 0.2, 0.3, 0.5],
            "reg_lambda": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "booster": ["gbtree", "dart"],
            "tree_method": ["auto", "exact", "approx", "hist"],
        },
        "LinearDiscriminantAnalysis": {
            "solver": ["svd", "lsqr", "eigen"],
            "shrinkage": [None, 0.1, 0.5, 1.0],
            "store_covariance": [False, True],
            "tol": [1e-3, 1e-4, 1e-5],
        },
        "LogisticRegression": {
            "penalty": ["l1", "l2", None],
            "C": [0.01, 0.1, 0.5, 1.0],
            "solver": ["newton-cg", "lbfgs", "sag", "saga", "liblinear"],
            "fit_intercept": [True, False],
        },
        "ElasticNet": {
            "penalty": ["elasticnet"],
            "C": [0.01, 0.1, 0.5, 1.0],
            "solver": ["saga"],
            "fit_intercept": [True, False],
            "l1_ratio": [0.0, 0.2, 0.5, 0.8, 1.0],
        },
        "GaussianNB": {
            "var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
        },
        "LGBMClassifier": {
            "boosting_type": ["gbdt", "dart", "goss", "rf"],
            "num_leaves": [2, 31, 64, 128, 256],
            "learning_rate": [0.001, 0.01, 0.1, 0.3, 0.5],
            "n_estimators": [2, 50, 100, 200, 300],
            "subsample_for_bin": [100000, 200000, 300000, 400000],
            "min_split_gain": [0.0, 0.1, 0.3, 0.5, 1.0],
            "reg_alpha": [0.0, 1.0, 3.0, 10.0],
            "reg_lambda": [0.0, 1.0, 3.0, 10.0],
            "bagging_fraction": [0.1, 0.3, 0.5, 0.7, 0.9],
            "bagging_freq": [1, 3, 5, 7],
            "feature_fraction": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        "GaussianProcessClassifier": {
            "optimizer": ["fmin_l_bfgs_b", None],
            "max_iter_predict": [50, 100, 500, 1000, 1500],
            "warm_start": [True, False],
        },
        "CatBoostClassifier": {
            "iterations": [50, 100, 200, 500, 1000],
            "learning_rate": [0.01, 0.1, 0.3, 0.5],
            "depth": [2, 4, 6, 8, 10,12, 14, 16],
            "l2_leaf_reg": [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0],
            "random_strength": [0.0, 0.1, 0.5, 1.0, 5.0, 10.0],
            "leaf_estimation_method": ["Newton", "Gradient"],
            "model_size_reg": [0.01, 0.1, 1.0, 5.0, 10.0],
            "min_data_in_leaf": [1,3,5,8,10,15],
            "loss_function": ["Logloss", "CrossEntropy"],
        },
    },
}

# hyper_coml will be used for one-sem calculations
hyper_compl = {
    "RandomForestClassifier": {
        "max_depth": True,
        "n_estimators": True,
        "min_samples_leaf": False,
    },
    "LogisticRegression": {
        "C": True,
        "max_iter": True
    },
    "XGBClassifier": {
        "max_depth": True,
        "n_estimators": True,
        "learning_rate": False
    },
    "LGBMClassifier": {
        "max_depth": True,
        "n_estimators": True,
        'min_child_samples': False,
        'learning_rate': True,
    },
    "CatBoostClassifier": {
        'depth': True,
        'n_estimators': True,
        "min_data_in_leaf": False,
        'learning_rate': False,
    },
    "SVC": {
        "C": True,
    },
    "KNeighborsClassifier": {
        "leaf_size": True,
        'n_neighbors': True,
    },
    "LinearDiscriminantAnalysis": {"shrinkage": True,
                                   'tol': False},
    "GaussianNB": {"var_smoothing": True},
    "GradientBoostingClassifier": {
        "max_depth": True,
        "n_estimators": True,
        "min_samples_leaf": False,
        
    },
    "GaussianProcessClassifier": {
        "n_restarts_optimizer": True,
        "max_iter_predict": True,
    },
    "ElasticNet": {
        "C": True,
        "max_iter": True
        }
}