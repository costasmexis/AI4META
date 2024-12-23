import tqdm
import copy
from sklearn import metrics
from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    StratifiedKFold
)
import numpy as np
import pandas as pd
from sklearn.utils import resample
from machinelearning.utils.calc_hlp_fnc import _calc_shap

def _bootstrap_validation(
        X, y, model,scoring, extra_metrics=None):#, calculate_shap=False
    """Performs bootstrap validation for model evaluation.
    :return: A tuple of (bootstrap_scores, extra_metrics_scores).
    :rtype: tuple
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True
    )

    bootstrap_scores = []
    extra_metrics_scores = {extra: [] for extra in extra_metrics} if extra_metrics else {}    
    
    for i in tqdm(range(100), desc="Bootstrap validation"):
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.25, shuffle=True, random_state=i
        #     )
        model_bootstrap = copy.deepcopy(model)
        X_train_res, y_train_res = resample(X_train, y_train, random_state=i)
        # model_bootstrap.fit(X_train_res, y_train_res)
        # y_pred = model_bootstrap.predict(X_test)
        model_bootstrap.fit(X_train_res, y_train_res)
        y_pred = model_bootstrap.predict(X_test)
        
        # Calculate the main scoring metric
        score = metrics.get_scorer(scoring)._score_func(y_test, y_pred)
        bootstrap_scores.append(score)
        
        # Calculate and store extra metrics
        if extra_metrics is not None:
            for extra in extra_metrics:
                extra_score = metrics.get_scorer(extra)._score_func(y_test, y_pred)
                extra_metrics_scores[extra].append(extra_score)
        
        # # Calculate and accumulate SHAP values
        # if calculate_shap:
        #     shap_values = self.calc_shap(X_train, X_test, model_bootstrap)
        #     all_shap_values[X_test.index] += shap_values.values
        #     counts[X_test.index] += 1
    
    # Calculate the mean SHAP values by dividing accumulated SHAP values by counts
    # if calculate_shap:
    #     mean_shap_values = all_shap_values / counts[:, None]        
    #     return bootstrap_scores, extra_metrics_scores, mean_shap_values
    # else:
    return bootstrap_scores, extra_metrics_scores

def _oob_validation(
        X, y, model, scoring, extra_metrics=None
    ):

    oob_scores = []
    extra_metrics_scores = {extra: [] for extra in extra_metrics} if extra_metrics else {}
    
    for i in tqdm(range(100), desc="OOB validation"):
        # Generate a random bootstrap sample with replacement
        rng = np.random.default_rng(i)  # New random generator for each seed
        indices = rng.choice(range(X.shape[0]), size=X.shape[0], replace=True)

        X_train, y_train = X.iloc[indices, :], y[indices]
        
        # Determine the OOB indices
        oob_indices = list(set(range(X.shape[0])) - set(indices))
        X_test, y_test = X.iloc[oob_indices, :], y[oob_indices]
        
        model_oob = copy.deepcopy(model)
        model_oob = model_oob.fit(X_train, y_train)
        
        y_pred = model_oob.predict(X_test)
    
        score = metrics.get_scorer(scoring)._score_func(y_test, y_pred)
        oob_scores.append(score)
        
        # Calculate and store extra metrics
        if extra_metrics is not None:
            for extra in extra_metrics:
                extra_score = metrics.get_scorer(extra)._score_func(y_test, y_pred)
                extra_metrics_scores[extra].append(extra_score)

    return oob_scores, extra_metrics_scores

def _train_test_validation(X, y, model, scoring, extra_metrics=None):
    tt_prop_scores = []
    extra_metrics_scores = {extra: [] for extra in extra_metrics} if extra_metrics else {}
    sss = StratifiedShuffleSplit(n_splits=100, test_size=0.3, random_state=0)

    for i, (train_index, test_index) in tqdm(enumerate(sss.split(X, y)), desc="TT prop validation"):
        # Use .iloc for DataFrame X and NumPy indexing for array y
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Deepcopy the model and fit it on the train set
        model_tt_prop = copy.deepcopy(model)
        model_tt_prop = model_tt_prop.fit(X_train, y_train)  
        y_pred = model_tt_prop.predict(X_test)

        # Evaluate the primary metric
        score = metrics.get_scorer(scoring)._score_func(y_test, y_pred)
        tt_prop_scores.append(score)

        # Calculate and store extra metrics
        if extra_metrics is not None:
            for extra in extra_metrics:
                extra_score = metrics.get_scorer(extra)._score_func(y_test, y_pred)
                extra_metrics_scores[extra].append(extra_score)

    return tt_prop_scores, extra_metrics_scores

def _evaluate(
    X, y, scoring, cv, evaluation, rounds, best_model, best_params, way, calculate_shap, extra_metrics, training_method, estimator_name,  features_names_list, num_features
):
    """
    Evaluate the performance of a machine learning model using cross-validation or bootstrap methods.

    :param X: The input features.
    :type X: pandas.DataFrame
    :param y: The target variable.
    :type y: pandas.Series
    :param cv: The number of cross-validation folds.
    :type cv: int
    :param evaluation: The evaluation method to use. Must be either 'cv_simple', 'bootstrap', or 'cv_rounds'.
    :type evaluation: str
    :param rounds: The number of rounds for cross-validation or bootstrap evaluation.
    :type rounds: int
    :param best_model: The best model obtained from model selection.
    :type best_model: object
    :param best_params: The best hyperparameters obtained from model selection.
    :type best_params: dict
    :param way: The model selection method used. Only required if evaluation is 'cv_rounds' or 'cv_simple'.
    :type way: object
    :param calculate_shap: Whether to calculate SHAP values or not.
    :type calculate_shap: bool
    :raises ValueError: If cv is less than 2.
    :raises ValueError: If evaluation method is not one of 'cv_simple', 'bootstrap', or 'cv_rounds'.
    :return: The best model, evaluation results, and SHAP values (if calculate_shap is True).
    :rtype: tuple
    """
    
    local_data_full_outer = pd.DataFrame()
    if calculate_shap:
        x_shap = np.zeros((X.shape[0], X.shape[1]))

    if cv < 2:
        raise ValueError("Cross-validation rounds must be greater than 1")

    list_train_test_indices = []
    list_x_train = []
    list_x_test = []
    list_y_train = []
    list_y_test = []
    scores = []
    scores_per_cv = []

    if evaluation == "cv_rounds":# or evaluation == "cv_simple":
        # split the train and test sets for cv and rounds cv evaluations
        for i in range(rounds):
            cv_splits = StratifiedKFold(n_splits=cv, shuffle=True, random_state=i)
            temp_train_test_indices = list(cv_splits.split(X, y))
            list_train_test_indices.append(temp_train_test_indices)
            for train_index, test_index in temp_train_test_indices:
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]
                list_x_train.append(X_train)
                list_x_test.append(X_test)
                list_y_train.append(y_train)
                list_y_test.append(y_test)

        scores_per_cv = []
        metric_lists = {}
        if extra_metrics is not None:
            for extra in extra_metrics:
                metric_lists[extra] = []

        for i in range(rounds):
            scores = []
            for train_index, test_index in list_train_test_indices[i]:
                # temp_model = copy.deepcopy(best_model)
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]
                best_model.fit(X_train, y_train)
                scores.append(scoring(best_model, X_test, y_test))
                
                # Calculate and store scores for each extra metric
                if extra_metrics is not None:
                    for extra in extra_metrics:
                        extra_score = metrics.get_scorer(extra)._score_func(y_test, best_model.predict(X_test))
                        metric_lists[extra].append(extra_score)

                if calculate_shap:
                    shap_values = _calc_shap(X_train, X_test, best_model)
                    x_shap[test_index, :] = np.add(
                        x_shap[test_index, :], shap_values.values
                    )
                # del(temp_model)
            scores_per_cv.append(scores)
        
        if calculate_shap:
            x_shap = x_shap / (rounds)
                    
        for round_num in range(rounds):
            row = {}
            scores = []
            for cv_trial in range(cv):
                scores.append(scores_per_cv[round_num][cv_trial])
                
            row['Scores'] = scores 
                
            row["mean_test_score"] = np.mean(scores_per_cv[round_num])
            row["std_test_score"] = np.std(scores_per_cv[round_num])

            valid_scores = [
                score for score in scores_per_cv[round_num] if score is not None
            ]
            if valid_scores:
                sem = np.std(valid_scores) / np.sqrt(len(valid_scores))
            else:
                sem = 0

            row["sem_test_score"] = sem
            row["params"] = best_params
            row["round"] = "round_cv"
            row['train_mthd'] = training_method
            row['estimator'] = estimator_name
            if features_names_list is not None:
                row['features'] = len(features_names_list)
            elif num_features is not None:
                row['features'] = num_features
            else:
                row['features'] = 'all'

            # Calculate and add extra metrics to the row
            if extra_metrics is not None:
                for extra in extra_metrics:
                    extra_metric_scores = metric_lists[extra][round_num * cv:(round_num + 1) * cv]
                    row[f"{extra}"] = extra_metric_scores

            row_df = pd.DataFrame([row])
            local_data_full_outer = pd.concat(
                [local_data_full_outer, row_df], axis=0
            )

        local_data_full_outer.reset_index(drop=True, inplace=True)
    
        # if calculate_shap:
        #     return best_model, local_data_full_outer, x_shap

    elif (evaluation == "bootstrap") or (evaluation == "oob"):
        if evaluation == "bootstrap":
            bootstrap_scores, extra_metrics_scores = _bootstrap_validation(X, y, scoring, best_model, extra_metrics)#, calculate_shap=False)
        else:
            bootstrap_scores, extra_metrics_scores = _oob_validation(X, y, scoring, best_model, extra_metrics)#, calculate_shap=False)
        local_data_full_outer["Scores"] = bootstrap_scores
        local_data_full_outer["mean_test_score"] = np.mean(bootstrap_scores)
        local_data_full_outer["std_test_score"] = np.std(bootstrap_scores)
        
        valid_scores = [score for score in bootstrap_scores if score is not None]
        if valid_scores:
            sem = np.std(valid_scores) / np.sqrt(len(valid_scores))
        else:
            sem = 0
        local_data_full_outer["sem_test_score"] = sem
        # print(best_params, best_model.get_params())
        local_data_full_outer["params"] =  local_data_full_outer.apply(lambda row: best_params.copy(), axis=1)
        if evaluation == "oob":
            local_data_full_outer["round"] = "oob"
        else:
            local_data_full_outer["round"] = "bootstrap"
        local_data_full_outer['train_mthd'] = training_method
        local_data_full_outer['estimator'] = estimator_name
        if features_names_list is not None:
            local_data_full_outer['features'] = len(features_names_list)
        elif num_features is not None:
            local_data_full_outer['features'] = num_features
        else:
            local_data_full_outer['features'] = 'all'

        # Calculate and add extra metrics for bootstrap validation
        if extra_metrics is not None:
            for extra in extra_metrics:
                extra_metric_scores = extra_metrics_scores[extra]
                local_data_full_outer[f"{extra}"] = extra_metric_scores
    
    elif evaluation == "train_test":
        tt_scores, extra_metrics_scores = _train_test_validation(X, y, scoring, best_model, extra_metrics)
        
        local_data_full_outer["Scores"] = tt_scores
        local_data_full_outer["mean_test_score"] = np.mean(tt_scores)
        local_data_full_outer["std_test_score"] = np.std(tt_scores)
        
        valid_scores = [score for score in tt_scores if score is not None]
        if valid_scores:
            sem = np.std(valid_scores) / np.sqrt(len(valid_scores))
        else:
            sem = 0
        local_data_full_outer["sem_test_score"] = sem
        local_data_full_outer["params"] =  local_data_full_outer.apply(lambda row: best_params.copy(), axis=1)
        local_data_full_outer["round"] = "train_test"
        local_data_full_outer['train_mthd'] = training_method
        local_data_full_outer['estimator'] = estimator_name
        if features_names_list is not None:
            local_data_full_outer['features'] = len(features_names_list)
        elif num_features is not None:
            local_data_full_outer['features'] = num_features
        else:
            local_data_full_outer['features'] = 'all'

        # Calculate and add extra metrics for train_test validation
        if extra_metrics is not None:
            for extra in extra_metrics:
                extra_metric_scores = extra_metrics_scores[extra]
                local_data_full_outer[f"{extra}"] = extra_metric_scores

    # return best_model, local_data_full_outer
    if calculate_shap:
        return best_model, local_data_full_outer, x_shap
    else:
        empty_array = np.array([])
        return best_model, local_data_full_outer, empty_array
