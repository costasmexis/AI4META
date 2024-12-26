from src.database.database_manager import DatabaseManager
import json
from collections import Counter
import numpy as np

def insert_to_db(scores_dataframe, config, database_name="ai4meta.db"):
    """
    Insert new data into the SQLite database schema using DatabaseManager.
    """
    # Initialize DatabaseManager
    db_manager = DatabaseManager(db_name=database_name)
    
    # Insert dataset and retrieve dataset_id
    dataset_id = db_manager.insert_dataset(config['dataset_name'])
    
    # Insert job parameters and retrieve job_id
    if config['model_selection_type'] == 'rncv':
        job_id = db_manager.insert_job_parameters_rncv(config)
    elif config['model_selection_type'] == 'rcv':
        job_id = db_manager.insert_job_parameters_rcv(config)
    else:
        job_id = db_manager.insert_job_parameters_cv(config)
    
    # Iterate over scores dataframe to insert related data
    for _, row in scores_dataframe.iterrows():
        # Insert classifier and retrieve classifier_id
        classifier_id = db_manager.insert_classifier(row["Est"], row["In_sel"])
        
        # Insert hyperparameters and retrieve hyperparameter_id
        hyperparameters = row["Hyp"]
        if isinstance(hyperparameters, np.ndarray):  # Convert numpy arrays to lists
            hyperparameters = hyperparameters.tolist()
        hyperparameter_id = db_manager.insert_hyperparameters(json.dumps(hyperparameters))
        
        # Insert feature selection and retrieve selection_id
        selection_id = db_manager.insert_feature_selection(row["Sel_way"], row["Fs_num"])
        
        # Insert performance metrics and retrieve performance_id
        metrics = {
            metric: json.dumps(
                metric.tolist() if isinstance(metric, np.ndarray) else metric
            )
            for metric in config['extra_metrics']
        }
        performance_id = db_manager.insert_performance_metrics(metrics)
        
        # Insert sample classification rates and retrieve sample_rate_id
        sample_rate_id = db_manager.insert_sample_classification_rates(json.dumps(row["Classif_rates"]))
        
        # Insert job combination and retrieve combination_id
        combination_id = db_manager.insert_job_combination(
            job_id, classifier_id, dataset_id, selection_id,
            hyperparameter_id, performance_id, sample_rate_id,
            config['model_selection_type']
        )
        
        # Insert feature counts if features are available
        if config['features_name'] is not None:
            if all(val is None for val in row["Sel_feat"]) and (row["Sel_feat"].size > 0):
                selected_features = row["Sel_feat"]

                # Convert numpy array to list if necessary
                if isinstance(selected_features, np.ndarray):
                    selected_features = selected_features.tolist()

                # Flatten nested lists if needed
                if any(isinstance(i, list) for i in selected_features):
                    selected_features = [item for sublist in selected_features for item in sublist]

                # Insert feature counts
                db_manager.insert_feature_counts(selected_features, combination_id)

    print("Data inserted into the SQLite database successfully.")

#             # Insert hyperparameters
#             hyperparameters_query = """
#                 INSERT INTO Hyperparameters (hyperparameters)
#                 VALUES (?);
#             """
#             hyperparameters = row["Hyp"]
#             if isinstance(hyperparameters, np.ndarray):  # Convert numpy arrays to lists
#                 hyperparameters = hyperparameters.tolist()
#             cursor.execute(hyperparameters_query, (json.dumps(hyperparameters),))
#             hyperparameter_id = cursor.lastrowid



#             # Insert performance metrics
#             metrics_to_add = ', '.join([f'{metric}' for metric in config['extra_metrics']])
#             list_of_s = [f'?' for metric in config['extra_metrics']]
#             count_q = ', '.join(list_of_s)
#             performance_metrics_query = f"""
#                 INSERT INTO Performance_Metrics ({metrics_to_add})
#                 VALUES ({count_q}) RETURNING performance_id;
#             """
            
#             metrics = [
#                 json.dumps(
#                     metric.tolist() if isinstance(metric, np.ndarray) else metric
#                 )
#                 for metric in [row.get(metric, None) for metric in config['extra_metrics']]
#             ]
#             cursor.execute(performance_metrics_query, metrics)
#             performance_id = cursor.lastrowid

#             # Insert samples classification rates
#             samples_classification_query = """
#                 INSERT INTO Samples_Classification_Rates (samples_classification_rates)
#                 VALUES (?);
#             """
#             cursor.execute(samples_classification_query, (json.dumps(row["Classif_rates"]),))
#             sample_rate_id = cursor.lastrowid

#             # Insert data into job combinations
#             job_combinations_query = """
#                 INSERT INTO Job_Combinations (
#                     job_id, classifier_id, dataset_id, selection_id, hyperparameter_id, performance_id, sample_rate_id, model_selection_type
#                 )
#                 VALUES (?, ?, ?, ?, ?, ?, ?, ?);
#             """
#             cursor.execute(
#                 job_combinations_query,
#                 (job_id, classifier_id, dataset_id, selection_id, hyperparameter_id, performance_id, sample_rate_id, config['model_selection_type'])
#             )
#             combination_id = cursor.lastrowid

#             if config['features_name'] == None:
#                 continue
#             else:
#                 # Insert feature counts
#                 if all(val is None for val in row["Sel_feat"]) and (row["Sel_feat"].size > 0):
#                     selected_features = row["Sel_feat"]

#                     # Convert numpy array to a list if necessary
#                     if isinstance(selected_features, np.ndarray):
#                         selected_features = selected_features.tolist()

#                     # Flatten nested lists if needed
#                     if any(isinstance(i, list) for i in selected_features):
#                         selected_features = [item for sublist in selected_features for item in sublist]

#                     # Count occurrences of each feature
#                     feature_counts = Counter(selected_features)
#                     for feature, count in feature_counts.items():
#                         if feature is None:
#                             continue
#                         feature_counts_query = """
#                             INSERT INTO Feature_Counts (feature_name, count, combination_id)
#                             VALUES (?, ?, ?);
#                         """
#                         cursor.execute(feature_counts_query, (feature, count, combination_id))
