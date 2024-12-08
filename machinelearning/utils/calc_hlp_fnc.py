import numpy as np
from sklearn.feature_selection import SelectFromModel
from scipy.stats import sem
from sklearn.metrics import get_scorer, confusion_matrix, get_scorer_names
from sklearn.metrics import average_precision_score, roc_auc_score
from .translators import METRIC_ADDREVIATIONS

def _calculate_metrics(config, results, clf, X_test, y_test):
    for metric in config["extra_metrics"]:
        if metric == 'specificity':
            results[f"{metric}"].append(
                _specificity_scorer(clf, X_test, y_test)
            )
        else:
            # print(res_model)
            try:                                 
                results[f"{metric}"].append(
                    get_scorer(metric)(clf, X_test, y_test)
                )
            except AttributeError:
                # Handle metrics like roc_auc and average_precision explicitly
                if metric in ['roc_auc', 'average_precision']:
                    if hasattr(clf, 'predict_proba'):
                        # Use decision_function if available
                        y_pred = clf.predict_proba(X_test)[:, 1]
                    else:
                        raise AttributeError(
                            f"Model {type(clf).__name__} does not support `predict_proba`, "
                            f"which are required for {metric}."
                        )

                    # Compute the score using the selected y_pred
                    if metric == 'roc_auc':
                        score = roc_auc_score(y_test, y_pred)
                    elif metric == 'average_precision':
                        score = average_precision_score(y_test, y_pred)

                results[f"{metric}"].append(score)
    return results
    
def _scoring_check(scoring: str) -> None:
    """This function is used to check if the scoring string metric is valid"""
    if (scoring not in get_scorer_names()) and (scoring != "specificity"):
        raise ValueError(
            f"Invalid scoring metric: {scoring}. Select one of the following: {list(get_scorer_names())} and specificity"
        )
    
def _specificity_scorer(estimator, X, y):
    """_This function is used to calculate the specificity score"""
    y_pred = estimator.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def _bootstrap_ci(data, type='median'):
    """
    Calculate the confidence interval of the mean or median using bootstrapping.

    Args:
        data (array-like): Input data to calculate the confidence interval for.
        type (str): Type of central tendency to compute ('mean' or 'median'). Defaults to 'median'.

    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """
    ms = []
    for _ in range(1000):
        # Generate a bootstrap sample
        sample = np.random.choice(data, size=len(data), replace=True)
        
        # Compute the desired central tendency
        if type == 'median':
            ms.append(np.median(sample))
        elif type == 'mean':
            ms.append(np.mean(sample))
    
    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = np.percentile(ms, (1 - 0.95) / 2 * 100)
    upper_bound = np.percentile(ms, (1 + 0.95) / 2 * 100)
    
    return lower_bound, upper_bound

def _sfm(estimator, X_train, X_test, y_train, num_feature2_use=None, threshold="mean"):
    """
    Select features using SelectFromModel with either a predefined number of features 
    or using the threshold if num_feature2_use is not provided.

    Args:
        estimator: The estimator object to use for feature selection. Must support `fit` and `feature_importances_`.
        X_train: Training feature set.
        X_test: Testing feature set.
        y_train: Training target variable.
        num_feature2_use: Number of features to select, defaults to None.
        threshold: Threshold value for feature selection, defaults to "mean".

    Returns:
        X_train_selected: The training set with selected features.
        X_test_selected: The testing set with selected features.
        num_feature2_use: The number of features selected.
    """
    
    # Fit the estimator on the training data
    estimator.fit(X_train, y_train)

    # Initialize SelectFromModel based on num_feature2_use or threshold
    if num_feature2_use is None:
        sfm = SelectFromModel(estimator, threshold=threshold)
    else:
        sfm = SelectFromModel(estimator, max_features=num_feature2_use)

    # Fit SelectFromModel to identify important features
    sfm.fit(X_train, y_train)
    
    # Get indices of selected features
    selected_features = sfm.get_support(indices=True)
    selected_columns = X_train.columns[selected_features].to_list()
    
    # Select the features for training and testing sets
    X_train_selected = X_train[selected_columns]
    X_test_selected = X_test[selected_columns]

    return X_train_selected, X_test_selected, num_feature2_use
 
def _parameters_check(config, main_type, X, csv_dir, label, available_clfs):
    """
    This function checks the parameters of the pipeline and returns the final parameters config for the class pipeline.
    """
    # Missing values manipulation
    if config['missing_values_method'] == "drop":
        print(
            "Values cannot be dropped at ncv because of inconsistent shapes. \nThe missing values with automaticly replaced by the median of each feature."
        )
        config['missing_values_method'] = "median"
    elif (config['missing_values_method'] != "mean") and (config['missing_values_method'] != "median"):
        raise ValueError(
            "The missing values method should be 'mean' or 'median'."
        )
    if X.isnull().values.any():
        print(
            f"Your Dataset contains NaN values. Some estimators does not work with NaN values.\nThe {config['missing_values_method']} method will be used for the missing values manipulation.\n"
        )
        
    # Checks for reliability of parameters
    if isinstance(config["num_features"], int):
        config["num_features"] = [config["num_features"]]
    elif isinstance(config["num_features"], list):
        pass
    elif config["num_features"] is None:
        config["num_features"] = [X.shape[1]]
    else:
        raise ValueError("num_features must be an integer or a list or None")
    
    if config['extra_metrics'] is not None:
        if type(config['extra_metrics']) is not list:
            config['extra_metrics'] = [config['extra_metrics']]
        for metric in config['extra_metrics']:
            _scoring_check(metric)
        
    if main_type == 'rncv':    
        if config['outer_scoring'] not in config['extra_metrics']:
            config['extra_metrics'].insert(0, config['outer_scoring'])
        elif config['outer_scoring'] in config['extra_metrics'] and config['extra_metrics'].index(config['outer_scoring']) != 0:
            # Remove it from its current position
            config['extra_metrics'].remove(config['outer_scoring'])
            # Insert it at the first index
            config['extra_metrics'].insert(0, config['outer_scoring'])
        elif config['outer_scoring'] in config['extra_metrics'] and config['extra_metrics'].index(config['outer_scoring']) == 0:
            pass
        else:
            config['extra_metrics'] = [config['outer_scoring']]
            
        config['model_selection_type'] = 'rncv'
        
        # Checks for reliability of parameters
        if (config['inner_scoring'] not in get_scorer_names()) and (config['inner_scoring'] != "specificity"):
            raise ValueError(
                f"Invalid inner scoring metric: {config['inner_scoring']}. Select one of the following: {list(get_scorer_names())} and specificity"
            )
        if (config['outer_scoring'] not in get_scorer_names()) and (config['outer_scoring'] != "specificity"):
            raise ValueError(
                f"Invalid outer scoring metric: {config['outer_scoring']}. Select one of the following: {list(get_scorer_names())} and specificity"
            )
        for inner_selection in config['inner_selection_lst']:
            if inner_selection not in ["validation_score", "one_sem", "gso_1", "gso_2","one_sem_grd"]:
                raise ValueError(
                    f'Invalid inner method: {inner_selection}. Select one of the following: ["validation_score", "one_sem", "one_sem_grd", "gso_1", "gso_2"]'
                )
                
        if (config['parallel'] not in ['thread_per_round', 'freely_parallel']) and (config['parallel'] is not None):
            raise ValueError(
                f'Invalid parallel method: {config["parallel"]}. Select one of the following: ["thread_per_round", "freely_parallel"]'
        )
        elif (config['parallel'] == None):
            config['parallel'] = 'thread_per_round'
            print('Parallel method is set to "thread_per_round"')
        
    else:
        if config['scoring'] not in config['extra_metrics']:
            config['extra_metrics'].insert(0, config['scoring'])
        elif config['scoring'] in config['extra_metrics'] and config['extra_metrics'].index(config['scoring']) != 0:
            # Remove it from its current position
            config['extra_metrics'].remove(config['scoring'])
            # Insert it at the first index
            config['extra_metrics'].insert(0, config['scoring'])
        elif config['scoring'] in config['extra_metrics'] and config['extra_metrics'].index(config['scoring']) == 0:
            pass
        else:
            config['extra_metrics'] = [config['scoring']]
        
        config['model_selection_type'] = 'rcv'
        
        if (config['scoring'] not in get_scorer_names()) and (config['scoring'] != "specificity"):
            raise ValueError(
                f"Invalid scoring metric: {config['scoring']}. Select one of the following: {list(get_scorer_names())} and specificity"
            )
            
    if config['class_balance'] not in ['auto','smote','borderline_smote','tomek', None]:
        raise ValueError("class_balance must be one of the following: 'auto','smote','smotenn','adasyn','borderline_smote','tomek', or None")
    elif config['class_balance'] == None:
        config['class_balance'] = 'auto'
        print('Class balance is set to "auto"')
        
    config['dataset_name'] = csv_dir
    config['dataset_label'] = label
    config['features_name'] = None if config['num_features'] == [X.shape[1]] else config['num_features']
            
    # Set available classifiers
    if config['search_on'] is not None:
        classes = config['search_on']  # 'search_on' is a list of classifier names as strings
        exclude_classes = [
            clf for clf in available_clfs.keys() if clf not in classes
        ]
    elif config['exclude'] is not None:
            exclude_classes = (
            config['exclude']  # 'exclude' is a list of classifier names as strings
        )
    else:
        exclude_classes = []

    # Filter classifiers based on the exclude_classes list
    clfs = [clf for clf in available_clfs.keys() if clf not in exclude_classes]
    config["clfs"] = clfs

    return config  
 
def _input_renamed_metrics( extra_metrics, results, indices):
    """
    Add renamed metrics to the results dataframe.

    Parameters
    ----------
    extra_metrics : list
        List of extra metrics to be added.
    results : list
        The results dataframe to which the metrics will be added.
    indices : DataFrame
        DataFrame containing the metric values.

    Returns
    -------
    list
        Updated results with renamed metrics.
    """
    # Metric abbreviation mapping
    metric_abbreviations = METRIC_ADDREVIATIONS
    
    # Iterate over each metric and calculate statistics
    for metric in extra_metrics:
        # Get the abbreviated metric name
        qck_mtrc = metric_abbreviations[f"{metric}"]
        # Extract metric values from indices
        metric_values = indices[f"{metric}"].values

        # Store the raw metric values in the results
        results[-1][f"{metric}"] = metric_values

        # Calculate mean, standard deviation, and standard error of the mean
        results[-1][f"{qck_mtrc}_mean"] = round(np.mean(metric_values), 3)
        results[-1][f"{qck_mtrc}_std"] = round(np.std(metric_values), 3)
        results[-1][f"{qck_mtrc}_sem"] = round(sem(metric_values), 3)

        # Compute and store the 5th and 95th percentiles
        lower_percentile = np.percentile(metric_values, 5)
        upper_percentile = np.percentile(metric_values, 95)
        results[-1][f"{qck_mtrc}_lowerCI"] = round(lower_percentile, 3)
        results[-1][f"{qck_mtrc}_upperCI"] = round(upper_percentile, 3)

        # Calculate and store the median
        results[-1][f"{qck_mtrc}_med"] = round(np.median(metric_values), 3)

        # Bootstrap confidence intervals for median and mean
        lomed, upmed = _bootstrap_ci(metric_values, type='median')
        lomean, upmean = _bootstrap_ci(metric_values, type='mean')
        results[-1][f"{qck_mtrc}_lomean"] = round(lomean, 3)
        results[-1][f"{qck_mtrc}_upmean"] = round(upmean, 3)
        results[-1][f"{qck_mtrc}_lomed"] = round(lomed, 3)
        results[-1][f"{qck_mtrc}_upmed"] = round(upmed, 3)
    
    return results