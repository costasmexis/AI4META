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
        
        if row["Fs_num"] != config.get('all_features'):    
            # Insert feature selection and retrieve selection_id
            selection_id = db_manager.insert_feature_selection(row["Sel_way"], row["Fs_num"])

            # Insert sample classification rates and retrieve sample_rate_id
            sample_rate_id = db_manager.insert_sample_classification_rates(json.dumps(row["Classif_rates"]))
            
        else:
            selection_id = None
            sample_rate_id = None
        
        # Insert performance metrics and retrieve performance_id
        metrics_values = [
            json.dumps(row.get(metric, None).tolist() if isinstance(row.get(metric, None), np.ndarray) else row.get(metric, None))
            for metric in config['extra_metrics']
        ]
        performance_id = db_manager.insert_performance_metrics(metrics_values, config['extra_metrics'])
        
        # Insert job combination and retrieve combination_id
        combination_id = db_manager.insert_job_combination(
            job_id, classifier_id, dataset_id, selection_id,
            hyperparameter_id, performance_id, sample_rate_id,
            config['model_selection_type']
        )
        
        if row["Fs_num"] != config.get('all_features'):
        
            # Insert feature counts and retrieve combination_id
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
