import os
import json
import numpy as np
from collections import Counter
import sqlite3

def _insert_data_into_sqlite_db(scores_dataframe, config, database_name="ai4meta.db"):
    """
    This function is used to insert new data into the SQLite database schema.
    """
    db_path = "database/" + database_name
    try:
        # Establish a connection to the SQLite database
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        # Enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys = ON;")
    except Exception as e:
        print(f"Error connecting to SQLite database: {e}")
        raise

    try:
        # Insert dataset or fetch existing dataset_id
        dataset_query = """
            INSERT INTO Datasets (dataset_name)
            VALUES (?)
            ON CONFLICT(dataset_name) DO NOTHING;
        """
        cursor.execute(dataset_query, (config['dataset_name'],))
        
        # Fetch the dataset_id, whether it was inserted or already exists
        cursor.execute("SELECT dataset_id FROM Datasets WHERE dataset_name = ?;", (config['dataset_name'],))
        dataset_id = cursor.fetchone()

        if dataset_id is None:
            raise ValueError(f"Unable to find or insert dataset with name: {config['dataset_name']}")
        
        dataset_id = dataset_id[0]  # Extract the ID from the result tuple

        # Insert job parameters or fetch existing job_id
        if config['model_selection_type'] == 'rncv':
            job_parameters_query = """
                INSERT INTO Job_Parameters (
                    n_trials, rounds, feature_selection_type, feature_selection_method, 
                    inner_scoring, outer_scoring, inner_splits, outer_splits, normalization, 
                    missing_values_method, class_balance
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            job_parameters_query = """
                INSERT INTO Job_Parameters (
                    rounds, feature_selection_type, feature_selection_method, 
                    outer_scoring, outer_splits, normalization, 
                    missing_values_method, class_balance
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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

        # Fetch the job_id
        job_parameters_select_query = """
            SELECT job_id FROM Job_Parameters
            WHERE 
                rounds = ? AND feature_selection_type = ? AND feature_selection_method = ?
                AND outer_scoring = ? AND outer_splits = ? AND normalization = ?
                AND missing_values_method = ? AND class_balance = ?;
        """
        if config['model_selection_type'] == 'rncv':
            cursor.execute(
                job_parameters_select_query,
                (
                    config['rounds'], config['feature_selection_type'], config['feature_selection_method'],
                    config['outer_scoring'], config['outer_splits'], config['normalization'],
                    config['missing_values_method'], config['class_balance']
                )
            )
        else:
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
            check_query = """
                SELECT classifier_id FROM Classifiers
                WHERE estimator = ? AND inner_selection = ?;
            """
            cursor.execute(check_query, (row["Est"], row["In_sel"]))
            classifier_result = cursor.fetchone()

            if classifier_result:
                classifier_id = classifier_result[0]
            else:
                classifier_query = """
                    INSERT INTO Classifiers (estimator, inner_selection)
                    VALUES (?, ?);
                """
                cursor.execute(classifier_query, (row["Est"], row["In_sel"]))
                classifier_id = cursor.lastrowid

            # Insert hyperparameters
            hyperparameters_query = """
                INSERT INTO Hyperparameters (hyperparameters)
                VALUES (?);
            """
            hyperparameters = row["Hyp"]
            if isinstance(hyperparameters, np.ndarray):  # Convert numpy arrays to lists
                hyperparameters = hyperparameters.tolist()
            cursor.execute(hyperparameters_query, (json.dumps(hyperparameters),))
            hyperparameter_id = cursor.lastrowid

            # Insert feature selection
            feature_selection_query = """
                INSERT INTO Feature_Selection (way_of_selection, numbers_of_features)
                VALUES (?, ?);
            """
            cursor.execute(feature_selection_query, (row["Sel_way"], row["Fs_num"]))
            selection_id = cursor.lastrowid

            # Insert performance metrics
            metrics_to_add = ', '.join([f'{metric}' for metric in config['extra_metrics']])
            list_of_s = [f'?' for metric in config['extra_metrics']]
            count_q = ', '.join(list_of_s)
            performance_metrics_query = f"""
                INSERT INTO Performance_Metrics ({metrics_to_add})
                VALUES ({count_q}) RETURNING performance_id;
            """
            
            metrics = [
                json.dumps(
                    metric.tolist() if isinstance(metric, np.ndarray) else metric
                )
                for metric in [row.get(metric, None) for metric in config['extra_metrics']]
            ]
            cursor.execute(performance_metrics_query, metrics)
            performance_id = cursor.lastrowid

            # Insert samples classification rates
            samples_classification_query = """
                INSERT INTO Samples_Classification_Rates (samples_classification_rates)
                VALUES (?);
            """
            cursor.execute(samples_classification_query, (json.dumps(row["Classif_rates"]),))
            sample_rate_id = cursor.lastrowid

            # Insert data into job combinations
            job_combinations_query = """
                INSERT INTO Job_Combinations (
                    job_id, classifier_id, dataset_id, selection_id, hyperparameter_id, performance_id, sample_rate_id, model_selection_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """
            cursor.execute(
                job_combinations_query,
                (job_id, classifier_id, dataset_id, selection_id, hyperparameter_id, performance_id, sample_rate_id, config['model_selection_type'])
            )
            combination_id = cursor.lastrowid

            if config['features_name'] == None:
                continue
            else:
                # Insert feature counts
                if all(val is None for val in row["Sel_feat"]) and (row["Sel_feat"].size > 0):
                    selected_features = row["Sel_feat"]

                    # Convert numpy array to a list if necessary
                    if isinstance(selected_features, np.ndarray):
                        selected_features = selected_features.tolist()

                    # Flatten nested lists if needed
                    if any(isinstance(i, list) for i in selected_features):
                        selected_features = [item for sublist in selected_features for item in sublist]

                    # Count occurrences of each feature
                    feature_counts = Counter(selected_features)
                    for feature, count in feature_counts.items():
                        if feature is None:
                            continue
                        feature_counts_query = """
                            INSERT INTO Feature_Counts (feature_name, count, combination_id)
                            VALUES (?, ?, ?);
                        """
                        cursor.execute(feature_counts_query, (feature, count, combination_id))

        # Commit the transaction
        connection.commit()
        print("Data inserted into the SQLite database successfully.")
    except Exception as e:
        connection.rollback()
        print(f"An error occurred while inserting data into the SQLite database: {e}")
    finally:
        cursor.close()
        connection.close()