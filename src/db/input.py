from src.db.manager import DatabaseManager
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
    dataset_id = db_manager.insert_dataset(config['csv_dir'])
    
    # Insert job parameters and retrieve job_id
    if config['model_selection_type'] == 'rncv':
        job_id = db_manager.insert_job_parameters_rncv(config)
    else:
        job_id = db_manager.insert_job_parameters_cv(config)

    # Iterate over scores dataframe to insert related data
    for _, row in scores_dataframe.iterrows():
        # Insert classifier and retrieve classifier_id
        if ('Est' in scores_dataframe.columns) and ('In_sel' in scores_dataframe.columns):
            classifier_id = db_manager.insert_classifier(row["Est"], row["In_sel"])
        else: 
            classifier_id = db_manager.insert_classifier(config['estimator_name'], config['inner_selection'])
        
        # Insert hyperparameters and retrieve hyperparameter_id
        if 'Hyp' in scores_dataframe.columns:   
            hyperparameters = row["Hyp"]
        else:
            hyperparameters = config['hyperparameters']
        if isinstance(hyperparameters, np.ndarray):  # Convert numpy arrays to lists
            hyperparameters = hyperparameters.tolist()
        hyperparameter_id = db_manager.insert_hyperparameters(json.dumps(hyperparameters))
        
        if 'Fs_num' in scores_dataframe.columns:
            # Insert sample classification rates and retrieve sample_rate_id
            classif_rates_str = np.array2string(
                row["Classif_rates"], 
                separator=',', 
                suppress_small=True,  
                max_line_width=np.inf 
            )
            sample_rate_id = db_manager.insert_sample_classification_rates(classif_rates_str)
            if row["Fs_num"] != config.get('all_features'):    
                # Insert feature selection and retrieve selection_id
                selection_id = db_manager.insert_feature_selection(row["Sel_way"], row["Fs_num"])
            else:
                selection_id = None
                # sample_rate_id = None
        else: 
            if config['num_features'] != config.get('all_features'):
                # Insert feature selection and retrieve selection_id
                selection_id = db_manager.insert_feature_selection(config['feature_selection_type'], config['num_features'])
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
        
        if 'Fs_num' in scores_dataframe.columns:
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

    # Close the database connection
    db_manager.close_connection()

    print("Data inserted into the SQLite database successfully.")
