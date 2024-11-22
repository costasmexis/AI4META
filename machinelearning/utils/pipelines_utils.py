import sklearn
from sklearn.metrics import get_scorer, confusion_matrix, make_scorer
import numpy as np
from sklearn.feature_selection import SelectFromModel
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
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from itertools import chain
from collections import Counter
import pandas as pd
from datetime import datetime
import os
import psycopg2
from psycopg2.extras import execute_values
from scipy.stats import sem
import json

def _scoring_check(scoring: str) -> None:
    """This function is used to check if the scoring string metric is valid"""
    if (scoring not in sklearn.metrics.get_scorer_names()) and (scoring != "specificity"):
        raise ValueError(
            f"Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())} and specificity"
        )
    
def _specificity_scorer(estimator, X, y):
    """_This function is used to calculate the specificity score"""
    # try:
    y_pred = estimator.predict(X)
    # except AttributeError:
    #     y_pred_proba = estimator.predict_proba(X)
    #     y_pred =y_pred_proba[:,1]
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
    
def _plot(
    scores_dataframe: pd.DataFrame, 
    plot: str, 
    scorer: str, 
    final_dataset_name: str
) -> None:
    """
    This function creates a box or violin plot of the outer cross-validation scores for each classifier

    Parameters:
    scores_dataframe (DataFrame): A dataframe containing the results of the outer loop.
    plot (str): The type of plot to create ("box" or "violin").
    scorer (str): The name of the scorer to plot.
    final_dataset_name (str): The name of the dataset.

    Returns:
    None
    """
    scores_long = scores_dataframe.explode(f"{scorer}")
    scores_long[f"{scorer}"] = scores_long[f"{scorer}"].astype(float)
    fig = go.Figure()
    
    classifiers = scores_long["Clf"].unique()

    if plot == "box":
        # Add box plots for each classifier within each Inner_Selection method
        for classifier in classifiers:
            data = scores_long[scores_long["Clf"] == classifier][
                f"{scorer}"
            ]
            median = np.median(data)
            fig.add_trace(
                go.Box(
                    y=data,
                    name=f"{classifier} (Median: {median:.2f})",
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                )
            )

            # Calculate and add 95% CI for the median
            lower, upper = _bootstrap_ci(data, type='median')
            fig.add_trace(
                go.Scatter(
                    x=[f"{classifier} (Median: {median:.2f})",
                    f"{classifier} (Median: {median:.2f})"],
                    y=[lower, upper],
                    mode="lines",
                    line=dict(color="black", dash="dash"),
                    showlegend=False,
                )
            )

    elif plot == "violin":
        for classifier in classifiers:
            data = scores_long[scores_long["Clf"] == classifier][
                f"{scorer}"
            ]
            median = np.median(data)
            fig.add_trace(
                go.Violin(
                    y=data,
                    name=f"{classifier} (Median: {median:.2f})",
                    box_visible=False,
                    points="all",
                    jitter=0.3,
                    pointpos=-1.8,
                )
            )
    else:
        raise ValueError(
            f'The "{plot}" is not a valid option for plotting. Choose between "box" or "violin".'
        )

    # Update layout for better readability
    fig.update_layout(
        autosize = False,
        width=1500,
        height=1200,
        title="Model Selection Results by Classifier",
        yaxis_title=f"Scores {scorer}",
        xaxis_title="Classifier",
        xaxis_tickangle=-45,
        template="plotly_white",
    )
    
    # Save the figure as an image in the "Results" directory
    image_path = f"{final_dataset_name}_model_selection_plot.png"
    fig.write_image(image_path)
            
def _histogram(scores_dataframe, final_dataset_name, freq_feat, clfs, max_features):
    """
    Function to create a histogram of the selected features counts.

    Parameters:
    scores_dataframe (DataFrame): The dataframe containing the results of the outer loop.
    final_dataset_name (str): The name of the dataset.
    freq_feat (int): The number of features to show in the histogram. If None, it will be set to max_features.
    clfs (list): The list of classifiers used.
    max_features (int): The maximum number of features.

    Returns:
    None
    """
    if freq_feat is None:
        freq_feat = max_features
    elif freq_feat > max_features:
        freq_feat = max_features

    # Plot histogram of features
    feature_counts = Counter()
    for idx, row in scores_dataframe.iterrows():
        if row["Sel_way"] != "full":  # If no features were selected, skip
            features = list(
                chain.from_iterable(
                    [list(index_obj) for index_obj in row["Sel_feat"]]
                )
            )
            feature_counts.update(features)

    sorted_features_counts = feature_counts.most_common()

    if len(sorted_features_counts) == 0:
        print("No features were selected.")
    else:
        features, counts = zip(*sorted_features_counts[:freq_feat])
        counts = [x / len(clfs) for x in counts]  # Normalize counts
        print(f"Selected {freq_feat} features")

        # Create the bar chart using Plotly
        fig = go.Figure()

        # Add bars to the figure
        fig.add_trace(go.Bar(
            x=features,
            y=counts,
            marker=dict(color="skyblue"),
            text=[f"{count:.2f}" for count in counts],  # Show normalized counts as text
            textposition='auto'
        ))

        # Set axis labels and title
        fig.update_layout(
            title="Histogram of Selected Features",
            xaxis_title="Features",
            yaxis_title="Counts",
            xaxis_tickangle=-90,  # Rotate x-ticks to avoid overlap
            bargap=0.2,
            template="plotly_white",
            width=min(max(1000, freq_feat * 20), 2000),  # Dynamically adjust plot width
            height=700  # Set plot height
        )

        # Save the plot to 'Results/histogram.png'
        save_path = f"{final_dataset_name}_histogram.png"
        fig.write_image(save_path)

def _return_csv( final_dataset_name, scores_dataframe, extra_metrics=None, filter_csv=None):
    """
    Function to save the results to a csv file

    Parameters
    ----------
    final_dataset_name : str
        Name of the dataset
    scores_dataframe : DataFrame
        DataFrame containing the results
    extra_metrics : list, optional
        List of extra metrics to include in the csv file
    filter_csv : dict, optional
        Dictionary containing the filters to apply to the csv file

    Returns
    -------
    DataFrame
        The filtered DataFrame
    """
    results_path = f"{final_dataset_name}_outerloops_results.csv"
    cols_drop = ["Classif_rates", "Clf", "Hyp", "Sel_feat"]
    if extra_metrics is not None:
        for metric in extra_metrics:
            cols_drop.append(f"{metric}") 
    statistics_dataframe = scores_dataframe.drop(cols_drop, axis=1)
    if filter_csv is not None:
        try:
            # Apply filters to the csv file
            for mtrc_stat in filter_csv:
                if 'h' in filter_csv[mtrc_stat]:
                    statistics_dataframe = statistics_dataframe[statistics_dataframe[mtrc_stat] >= filter_csv[mtrc_stat]['h']]
                elif 'l' in filter_csv[mtrc_stat]:
                    statistics_dataframe = statistics_dataframe[statistics_dataframe[mtrc_stat] <= filter_csv[mtrc_stat]['l']]
        except Exception as e:
            print(f'An error occurred while filtering the final csv file: {e}\nThe final csv file will not be filtered.')
    statistics_dataframe.to_csv(results_path, index=False)
    print(f"Statistics results saved to {results_path}")
    return statistics_dataframe

def _file_name(config):
    """
    Function to set the name of the result nested cv and rcv_accel file with respect to the dataset name

    Parameters
    ----------
    config : dict
        Dictionary containing the configuration of the experiment

    Returns
    -------
    str
        The name of the result file
    """
    # Default values of the configuration
    default_values = {
        "rounds": 10,
        "n_trials": 100,
        "feature_selection_type": "mrmr",
        "feature_selection_method": "chi2",
        "inner_scoring": "matthews_corrcoef",
        "outer_scoring": "matthews_corrcoef",
        "inner_splits": 5,
        "outer_splits": 5,
        "normalization": "minmax",
        "class_balance": "auto",    
        "sfm": False,
        "missing_values": "median",
        "num_features": None,
        "scoring": "matthews_corrcoef",
        "splits": 5
        
    }
    # Generate the name of the result file
    name_add = ""
    for conf in config:
        if conf in default_values.keys():
            if config[conf] != default_values[conf]:
                name_add += f"_{conf}_{config[conf]}"
    name_add += f"_{datetime.now().strftime('%Y%m%d_%H%M')}"
    return name_add

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
    metric_abbreviations = {
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
@staticmethod
def _calculate_complexity(trial, model_name, samples):
    """
    This function calculates the complexity of a model based on its hyperparameters for each estimator.
    
    Parameters
    ----------
    trial : dict
        Trial to calculate complexity for.
    model_name : str
        Name of the model to calculate complexity for.
    samples : int
        Number of samples in the dataset.
    
    Returns
    -------
    float
        Calculated complexity.
    """
    params = trial["params"]
    if model_name in ['RandomForestClassifier', 'XGBClassifier', 'GradientBoostingClassifier', 'LGBMClassifier']:
        max_depth = params["max_depth"]
        if model_name == 'RandomForestClassifier' or model_name == 'GradientBoostingClassifier':
            actual_depth = min((samples / params["min_samples_leaf"]), max_depth)
        elif model_name == 'XGBClassifier':
            actual_depth = max_depth  # Assuming XGBClassifier does not use min_samples_leaf
        elif model_name == 'LGBMClassifier':
            actual_depth = min((samples / params["min_child_samples"]), max_depth)
        complexity = params["n_estimators"] * (2 ** (actual_depth - 1))
    elif model_name == 'CatBoostClassifier':
        max_depth = params["depth"]
        actual_depth = min((samples / params["min_data_in_leaf"]), max_depth)
        complexity = params["n_estimators"] * (2 ** (actual_depth - 1))#*params["iterations"]
    elif model_name == 'LogisticRegression' or model_name == 'ElasticNet':
        complexity = params["C"] * params["max_iter"]
    elif model_name == 'SVC':
        if params["kernel"] == 'poly':
            complexity = params["C"] * params["degree"]
        else:
            complexity = params["C"]
    elif model_name == 'KNeighborsClassifier':
        complexity = params["leaf_size"]
    elif model_name == 'LinearDiscriminantAnalysis':
        complexity = params["shrinkage"]
    elif model_name == 'GaussianNB':
        complexity = params["var_smoothing"]
    elif model_name == 'GaussianProcessClassifier':
        complexity = params["max_iter_predict"]*params["n_restarts_optimizer"]
    else:
        complexity = float('inf')  # If model not recognized, set high complexity
    return complexity

def _name_outputs( config, results_dir, csv_dir):
    """ Function to set the name of the result nested cv file with respect to the dataset name """
    try:
        dataset_name = _set_result_csv_name(csv_dir)
        name_add  = _file_name(config)
        results_name = f"{dataset_name}_{name_add}_{config['model_selection_type']}"
        final_dataset_name = os.path.join(results_dir, results_name)
    except Exception as e:
        name_add = _file_name(config)
        results_name = f"results_{name_add}_{config['model_selection_type']}"
        final_dataset_name = os.path.join(
            results_dir, results_name
        )
    return final_dataset_name

def _set_result_csv_name( csv_dir):
    """This function is used to set the name of the result nested cv file with respect to the dataset name"""
    # Split the name of the dataset with the ".csv" part and keep the first part
    data_name = os.path.basename(csv_dir).split(".")[0]
    print(data_name)
    return data_name

def _insert_data_into_db( scores_dataframe, config):
    """
    This function is used to insert new data into the database schema.
    """
    try:
        try:
            # Get the directory of the current file (pipeline_utils.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Navigate to the machinelearning directory
            base_dir = os.path.abspath(os.path.join(current_dir, '..'))

            # Construct the full path to credentials.json
            credentials_path = os.path.join(base_dir, "db_credentials", "credentials.json")

            # Check if the file exists
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(f"Database credentials not found at {credentials_path}")

        except Exception as e:
            print(f"Error locating database credentials: {e}")
            raise

        try: 
            with open(credentials_path, "r") as file:
                db_credentials = json.load(file)

            # Establish a connection to the PostgreSQL database
            connection = psycopg2.connect(
                dbname=db_credentials["db_name"],
                user=db_credentials["db_user"],
                password=db_credentials["db_password"],
                host=db_credentials["db_host"],
                port=db_credentials["db_port"]
            )
            cursor = connection.cursor()
        
        except Exception as e:
            print(f"Connection error with the credentials: {e}")
            raise

    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise
    try:
        # Insert dataset or fetch existing dataset_id
        dataset_query = """
            INSERT INTO Datasets (dataset_name)
            VALUES (%s) ON CONFLICT (dataset_name) DO NOTHING;
        """
        cursor.execute(dataset_query, (config['dataset_name'],))
        
        # Fetch the dataset_id, whether it was inserted or already exists
        cursor.execute("SELECT dataset_id FROM Datasets WHERE dataset_name = %s;", (config['dataset_name'],))
        dataset_id = cursor.fetchone()

        if dataset_id is None:
            raise ValueError(f"Unable to find or insert dataset with name: {config['dataset_name']}")
        
        dataset_id = dataset_id[0]  # Extract the ID from the result tuple

        # Insert job parameters or fetch existing job_id
        if config['model_selection_type'] == 'rncv':
            # Query for RN-CV model selection
            job_parameters_query = """
                INSERT INTO Job_Parameters (
                    n_trials, rounds, feature_selection_type, feature_selection_method, 
                    inner_scoring, outer_scoring, inner_splits, outer_splits, normalization, 
                    missing_values_method, class_balance
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING;
            """
            cursor.execute(
                job_parameters_query,
                (
                    config['n_trials'], config['rounds'], config['feature_selection_type'],
                    config['feature_selection_method'], config['inner_scoring'], config['outer_scoring'],
                    config['inner_splits'], config['outer_splits'], config['normalization'],
                    config['missing_values_method'], config['class_balance']
                )
            )
        else:
            # Query for other model selection types
            job_parameters_query = """
                INSERT INTO Job_Parameters (
                    rounds, feature_selection_type, feature_selection_method, 
                    outer_scoring, outer_splits, normalization, 
                    missing_values_method, class_balance
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING;
            """
            cursor.execute(
                job_parameters_query,
                (
                    config['rounds'], config['feature_selection_type'],
                    config['feature_selection_method'], config['scoring'],
                    config['splits'], config['normalization'],
                    config['missing_values_method'], config['class_balance']
                )
            )

        # Fetch the job_id (whether it was inserted or already exists)
        job_parameters_select_query = """
            SELECT job_id FROM Job_Parameters
            WHERE 
                rounds = %s AND feature_selection_type = %s AND feature_selection_method = %s
                AND outer_scoring = %s AND outer_splits = %s AND normalization = %s
                AND missing_values_method = %s AND class_balance = %s
                {extra_conditions};
        """
        if config['model_selection_type'] == 'rncv':
            job_parameters_select_query = job_parameters_select_query.format(extra_conditions="""
                AND n_trials = %s AND inner_scoring = %s AND inner_splits = %s
            """)
            cursor.execute(
                job_parameters_select_query,
                (
                    config['rounds'], config['feature_selection_type'], config['feature_selection_method'],
                    config['outer_scoring'], config['outer_splits'], config['normalization'],
                    config['missing_values_method'], config['class_balance'],
                    config['n_trials'], config['inner_scoring'], config['inner_splits']
                )
            )
        else:
            job_parameters_select_query = job_parameters_select_query.format(extra_conditions="")
            cursor.execute(
                job_parameters_select_query,
                (
                    config['rounds'], config['feature_selection_type'], config['feature_selection_method'],
                    config['scoring'], config['splits'], config['normalization'],
                    config['missing_values_method'], config['class_balance']
                )
            )

        job_id_result = cursor.fetchone()
        job_id = job_id_result[0]

        # Insert classifiers and associated data
        for _, row in scores_dataframe.iterrows():
            # Check if the classifier combination already exists
            try:
                check_query = """
                    SELECT classifier_id FROM Classifiers
                    WHERE estimator = %s AND inner_selection = %s;
                """
                cursor.execute(check_query, (row["Est"], row["In_sel"]))
                classifier_result = cursor.fetchone()

                if classifier_result:
                    classifier_id = classifier_result[0]
                else:
                    classifier_query = """
                        INSERT INTO Classifiers (estimator, inner_selection)
                        VALUES (%s, %s) RETURNING classifier_id;
                    """
                    cursor.execute(classifier_query, (row["Est"], row["In_sel"]))
                    classifier_id = cursor.fetchone()[0]
            except Exception as e:
                print(f"Error with Classifiers table: {e}")
                raise

            # Insert hyperparameters
            try:
                hyperparameters_query = """
                    INSERT INTO Hyperparameters (hyperparameters)
                    VALUES (%s) RETURNING hyperparameter_id;
                """
                hyperparameters = row["Hyp"]
                if isinstance(hyperparameters, np.ndarray):
                    hyperparameters = [dict(item) for item in hyperparameters]
                cursor.execute(hyperparameters_query, (json.dumps(hyperparameters),))
                hyperparameter_id = cursor.fetchone()[0]
            except Exception as e:
                print(f"Error with Hyperparameters table: {e}")
                raise

            # Check if the feature selection data already exists
            try:
                feature_selection_check_query = """
                    SELECT selection_id FROM Feature_Selection
                    WHERE way_of_selection = %s AND numbers_of_features = %s;
                """
                cursor.execute(feature_selection_check_query, (row["Sel_way"], row["Fs_num"]))
                feature_selection_result = cursor.fetchone()

                if feature_selection_result:
                    selection_id = feature_selection_result[0]
                else:
                    feature_selection_query = """
                        INSERT INTO Feature_Selection (way_of_selection, numbers_of_features)
                        VALUES (%s, %s) RETURNING selection_id;
                    """
                    cursor.execute(feature_selection_query, (row["Sel_way"], row["Fs_num"]))
                    selection_id = cursor.fetchone()[0]
            except Exception as e:
                print(f"Error with Feature_Selection table: {e}")
                raise

            # Insert performance metrics
            try:
                metrics_to_add = ', '.join([f'{metric}' for metric in config['extra_metrics']])
                list_of_s = [f'%s' for metric in config['extra_metrics']]
                count_s = ', '.join(list_of_s)
                performance_metrics_query = f"""
                    INSERT INTO Performance_Metrics ({metrics_to_add})
                    VALUES ({count_s}) RETURNING performance_id;
                """

                metrics = [
                    json.dumps(
                        metric.tolist() if isinstance(metric, np.ndarray) else (metric if metric is not None else None)
                    )
                    for metric in [row.get(metric, None) for metric in config['extra_metrics']]
                ]

                cursor.execute(performance_metrics_query, metrics)
                performance_results = cursor.fetchone()
                performance_id = performance_results[0]
            except Exception as e:
                print(f"Error with Performance_Metrics table: {e}")
                raise

            # Insert samples classification rates
            try:
                samples_classification_query = """
                    INSERT INTO Samples_Classification_Rates (samples_classification_rates)
                    VALUES (%s) RETURNING sample_rate_id;
                """
                cursor.execute(samples_classification_query, (json.dumps(row["Classif_rates"]),))
                sample_rate_id = cursor.fetchone()[0]
            except Exception as e:
                print(f"Error with Samples_Classification_Rates table: {e}")
                raise

            # Insert data into job combinations
            try:
                job_combinations_query = """
                    INSERT INTO Job_Combinations (
                        job_id, classifier_id, dataset_id, selection_id, hyperparameter_id, performance_id, sample_rate_id, model_selection_type
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING combination_id;
                """
                cursor.execute(
                    job_combinations_query,
                    (job_id, classifier_id, dataset_id, selection_id, hyperparameter_id, performance_id, sample_rate_id, config['model_selection_type'])
                )
                combination_id = cursor.fetchone()[0]
            except Exception as e:
                print(f"Error with Job_Combinations table: {e}")
                raise

            # Insert feature counts and associate with combination_id
            try:
                if row["Sel_feat"] is not None:
                    selected_features = row["Sel_feat"]
                    if isinstance(selected_features, np.ndarray):
                        selected_features = selected_features.tolist()

                    if any(isinstance(i, list) for i in selected_features):
                        selected_features = [item for sublist in selected_features for item in sublist]

                    # Count occurrences of each feature
                    feature_counts = Counter([feature for feature in selected_features if feature])
                    feature_counts_query = """
                        INSERT INTO Feature_Counts (feature_name, count, combination_id)
                        VALUES (%s, %s, %s) RETURNING count_id;
                    """

                    # Prepare feature values with combination_id included
                    feature_values = []
                    for feat, count in feature_counts.items():
                        cursor.execute(feature_counts_query, (feat, count, combination_id))
                        feature_values.append(cursor.fetchone()[0])

                    # Associate feature_count_ids with job combination
                    feature_list_query = """
                        UPDATE Job_Combinations
                        SET feature_count_ids = %s
                        WHERE combination_id = %s;
                    """
                    cursor.execute(feature_list_query, (json.dumps(feature_values), combination_id))
            except Exception as e:
                print(f"Error with Feature_Counts table: {e}")

        # Commit the transaction
        connection.commit()
        print("Data inserted into the database successfully.")
    except Exception as e:
        connection.rollback()
        print(f"An error occurred while inserting data into the database: {e}")
    finally:
        cursor.close()
        connection.close()
