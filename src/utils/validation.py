from sklearn.metrics import get_scorer, confusion_matrix, get_scorer_names
from sklearn.metrics import average_precision_score, roc_auc_score

def _scoring_check(scoring: str) -> None:
    """This function is used to check if the scoring string metric is valid"""
    if (scoring not in get_scorer_names()) and (scoring != "specificity"):
        raise ValueError(
            f"Invalid scoring metric: {scoring}. Select one of the following: {list(get_scorer_names())} and specificity"
        )
        
def _validation(config, main_type, X, csv_dir, label, available_clfs):
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
        
    if (main_type == 'rncv') or (main_type == 'rcv'):
        # Checks for reliability of parameters
        if isinstance(config["num_features"], int):
            config["num_features"] = [config["num_features"]]
        elif isinstance(config["num_features"], list):
            pass
        elif config["num_features"] is None:
            config["num_features"] = [X.shape[1]]
        else:
            raise ValueError("num_features must be an integer or a list or None")
    else: 
        if (config["num_features"] is None) or (config["num_features"] > X.shape[1]):
            config["num_features"] = X.shape[1]
        elif isinstance(config["num_features"], int):
            pass
        else:
            raise ValueError("num_features must be an integer or None")

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
                    
        # Checks for reliability of parameters
        if (config['inner_scoring'] not in get_scorer_names()) and (config['inner_scoring'] != "specificity"):
            raise ValueError(
                f"Invalid inner scoring metric: {config['inner_scoring']}. Select one of the following: {list(get_scorer_names())} and specificity"
            )
        if (config['outer_scoring'] not in get_scorer_names()) and (config['outer_scoring'] != "specificity"):
            raise ValueError(
                f"Invalid outer scoring metric: {config['outer_scoring']}. Select one of the following: {list(get_scorer_names())} and specificity"
            )
        for inner_selection in config['inner_selection']:
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
        if 'evaluation' not in config.keys():
            config['evaluation'] = None
        if 'param_grid' not in config.keys():
            config['param_grid'] = None
        if 'features_names_list' not in config.keys():  
            config['features_names_list'] = None
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
        
        if (config['scoring'] not in get_scorer_names()) and (config['scoring'] != "specificity"):
            raise ValueError(
                f"Invalid scoring metric: {config['scoring']}. Select one of the following: {list(get_scorer_names())} and specificity"
            )
            
    config['model_selection_type'] = main_type
    
    if config['class_balance'] not in ['smote','borderline_smote','tomek', None]:
        raise ValueError("class_balance must be one of the following: 'smote','smotenn','adasyn','borderline_smote','tomek', or None")
    elif config['class_balance'] == None:
        config['class_balance'] = 'None'
        print('No class balancing will be applied')
        
    config['dataset_name'] = csv_dir
    config['dataset_label'] = label
    config['features_name'] = None if (config['num_features'] == [X.shape[1]]) or (config['num_features'] == X.shape[1]) else config['num_features']
    config['all_features'] = X.shape[1]
            
    if (main_type == 'rncv') or (main_type == 'rcv'):
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
    else: 
        if config['param_grid'] is None:
            config['param_grid'] = 'None'
        else:
            config['param_grid'] = str(config['param_grid'])
        if config['features_names_list'] is None:
            config['features_names_list'] = 'None'
        else:
            config['features_names_list'] = str(config['features_names_list'])
    return config 