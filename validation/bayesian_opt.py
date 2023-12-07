import optuna
import sklearn
from sklearn.metrics import get_scorer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class BayesianOptimization:
    def __init__(self, X_train, y_train, X_test, y_test, scoring='accuracy', direction='maximize'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        self.scoring = scoring
        self.direction = direction

        self.clf_name = None
        self.clf = None
        self.model = None

        if self.scoring not in sklearn.metrics.SCORERS.keys():
            raise ValueError(f'Invalid scoring metric: {self.scoring}. Select one of the following: {list(sklearn.metrics.SCORERS.keys())}')

        self.available_clf = {
            'RandomForestClassifier': RandomForestClassifier,
            'KNeighborsClassifier': KNeighborsClassifier,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'SVC_poly': SVC,
            'SVC_rest': SVC,
            'GradientBoostingClassifier': GradientBoostingClassifier
        }

        self.bayesian_clfs = {
            'RandomForestClassifier': lambda trial: RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 2, 200),
                criterion='gini',  # or trial.suggest_categorical('criterion', ['gini', 'entropy'])
                max_depth=trial.suggest_int('max_depth', 1, 50),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
                n_jobs=-1,
            ),
            'KNeighborsClassifier': lambda trial: KNeighborsClassifier(
                n_neighbors=trial.suggest_int('n_neighbors', 2, 15),
                weights=trial.suggest_categorical('weights', ['uniform', 'distance']),
                algorithm=trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                p=trial.suggest_int('p', 1, 2),
                leaf_size=trial.suggest_int('leaf_size', 5, 50),
                n_jobs=-1
            ),
            'DecisionTreeClassifier': lambda trial: DecisionTreeClassifier(
                criterion='gini',  # or trial.suggest_categorical('criterion', ['gini', 'entropy'])
                splitter=trial.suggest_categorical('splitter', ['best', 'random']),
                max_depth=trial.suggest_int('max_depth', 1, 100),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                min_weight_fraction_leaf=trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
            ),
            'SVC_poly': lambda trial: SVC(
                C=trial.suggest_int('C', 1, 10),
                kernel='poly',
                degree=trial.suggest_int('degree', 2, 5),
                coef0=trial.suggest_float('coef0', 0, 1),
                probability=trial.suggest_categorical('probability', [True, False]),
                shrinking=trial.suggest_categorical('shrinking', [True, False]),
                decision_function_shape=trial.suggest_categorical('decision_function_shape', ['ovo', 'ovr'])
            ),
            'SVC_rest': lambda trial: SVC(
                C=trial.suggest_int('C', 1, 10),
                kernel=trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid']),
                probability=trial.suggest_categorical('probability', [True, False]),
                shrinking=trial.suggest_categorical('shrinking', [True, False]),
                decision_function_shape=trial.suggest_categorical('decision_function_shape', ['ovo', 'ovr'])
            ),
            'GradientBoostingClassifier': lambda trial: GradientBoostingClassifier(
                loss=trial.suggest_categorical('loss', ['log_loss', 'exponential']),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.5),
                n_estimators=trial.suggest_int('n_estimators', 2, 200),
                criterion=trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error']),
                max_depth=trial.suggest_int('max_depth', 1, 50),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            )
    }

    def create_model(self, trial, classifier):
        if classifier in self.bayesian_clfs:
            model = self.bayesian_clfs[classifier](trial)
        else:
            raise ValueError('Classifier not supported')
        return model
    
    def objective(self, trial):
        model = self.create_model(trial=trial, classifier=self.clf_name)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        eval_metric = get_scorer(self.scoring)._score_func(self.y_test, y_pred)
        return eval_metric

    def run_optimization(self, classifier, n_trials=10):
        self.clf_name = classifier
        self.clf = self.available_clf[self.clf_name]

        study = optuna.create_study(direction=self.direction)
        study.optimize(self.objective, n_trials=n_trials)

        if self.clf_name in self.bayesian_clfs:
            self.model = self.available_clf[self.clf_name](**study.best_params)
        else:
            raise ValueError('Classifier not supported')

        return study

