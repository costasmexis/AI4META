import os
import json
import numpy as np
# import psycopg2
# from psycopg2.extras import execute_values
from collections import Counter
import sqlite3

def _insert_data_into_sqlite_db(scores_dataframe, config, database_name="ai4meta.db"):
    """
    This function is used to insert new data into the SQLite database schema.
    """
    db_path = "Database/" + database_name
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
        cursor.execute(
            job_parameters_select_query,
            (
                config['rounds'], config['feature_selection_type'], config['feature_selection_method'],
                config['outer_scoring'], config['outer_splits'], config['normalization'],
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
            performance_metrics_query = """
                INSERT INTO Performance_Metrics (matthews_corrcoef, roc_auc, accuracy, balanced_accuracy, recall, precision, f1, specificity, average_precision)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
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

            # Insert feature counts
            if row["Sel_feat"] is not None and row["Sel_feat"].size > 0:
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


# def _insert_data_into_db(scores_dataframe, config):
#     """
#     This function is used to insert new data into the database schema.
#     """
#     try:
#         try:
#             # Get the directory of the current file (pipeline_utils.py)
#             current_dir = os.path.dirname(os.path.abspath(__file__))

#             # Navigate to the machinelearning directory
#             base_dir = os.path.abspath(os.path.join(current_dir, '..'))

#             # Construct the full path to credentials.json
#             credentials_path = os.path.join(base_dir, "db_credentials", "credentials.json")

#             # Check if the file exists
#             if not os.path.exists(credentials_path):
#                 raise FileNotFoundError(f"Database credentials not found at {credentials_path}")

#         except Exception as e:
#             print(f"Error locating database credentials: {e}")
#             raise

#         try: 
#             with open(credentials_path, "r") as file:
#                 db_credentials = json.load(file)

#             # Establish a connection to the PostgreSQL database
#             connection = psycopg2.connect(
#                 dbname=db_credentials["db_name"],
#                 user=db_credentials["db_user"],
#                 password=db_credentials["db_password"],
#                 host=db_credentials["db_host"],
#                 port=db_credentials["db_port"]
#             )
#             cursor = connection.cursor()
        
#         except Exception as e:
#             print(f"Connection error with the credentials: {e}")
#             raise

#     except Exception as e:
#         print(f"Error connecting to database: {e}")
#         raise
#     try:
#         # Insert dataset or fetch existing dataset_id
#         dataset_query = """
#             INSERT INTO Datasets (dataset_name)
#             VALUES (%s) ON CONFLICT (dataset_name) DO NOTHING;
#         """
#         cursor.execute(dataset_query, (config['dataset_name'],))
        
#         # Fetch the dataset_id, whether it was inserted or already exists
#         cursor.execute("SELECT dataset_id FROM Datasets WHERE dataset_name = %s;", (config['dataset_name'],))
#         dataset_id = cursor.fetchone()

#         if dataset_id is None:
#             raise ValueError(f"Unable to find or insert dataset with name: {config['dataset_name']}")
        
#         dataset_id = dataset_id[0]  # Extract the ID from the result tuple

#         # Insert job parameters or fetch existing job_id
#         if config['model_selection_type'] == 'rncv':
#             # Query for RN-CV model selection
#             job_parameters_query = """
#                 INSERT INTO Job_Parameters (
#                     n_trials, rounds, feature_selection_type, feature_selection_method, 
#                     inner_scoring, outer_scoring, inner_splits, outer_splits, normalization, 
#                     missing_values_method, class_balance
#                 )
#                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#                 ON CONFLICT DO NOTHING;
#             """
#             cursor.execute(
#                 job_parameters_query,
#                 (
#                     config['n_trials'], config['rounds'], config['feature_selection_type'],
#                     config['feature_selection_method'], config['inner_scoring'], config['outer_scoring'],
#                     config['inner_splits'], config['outer_splits'], config['normalization'],
#                     config['missing_values_method'], config['class_balance']
#                 )
#             )
#         else:
#             # Query for other model selection types
#             job_parameters_query = """
#                 INSERT INTO Job_Parameters (
#                     rounds, feature_selection_type, feature_selection_method, 
#                     outer_scoring, outer_splits, normalization, 
#                     missing_values_method, class_balance
#                 )
#                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
#                 ON CONFLICT DO NOTHING;
#             """
#             cursor.execute(
#                 job_parameters_query,
#                 (
#                     config['rounds'], config['feature_selection_type'],
#                     config['feature_selection_method'], config['scoring'],
#                     config['splits'], config['normalization'],
#                     config['missing_values_method'], config['class_balance']
#                 )
#             )

#         # Fetch the job_id (whether it was inserted or already exists)
#         job_parameters_select_query = """
#             SELECT job_id FROM Job_Parameters
#             WHERE 
#                 rounds = %s AND feature_selection_type = %s AND feature_selection_method = %s
#                 AND outer_scoring = %s AND outer_splits = %s AND normalization = %s
#                 AND missing_values_method = %s AND class_balance = %s
#                 {extra_conditions};
#         """
#         if config['model_selection_type'] == 'rncv':
#             job_parameters_select_query = job_parameters_select_query.format(extra_conditions="""
#                 AND n_trials = %s AND inner_scoring = %s AND inner_splits = %s
#             """)
#             cursor.execute(
#                 job_parameters_select_query,
#                 (
#                     config['rounds'], config['feature_selection_type'], config['feature_selection_method'],
#                     config['outer_scoring'], config['outer_splits'], config['normalization'],
#                     config['missing_values_method'], config['class_balance'],
#                     config['n_trials'], config['inner_scoring'], config['inner_splits']
#                 )
#             )
#         else:
#             job_parameters_select_query = job_parameters_select_query.format(extra_conditions="")
#             cursor.execute(
#                 job_parameters_select_query,
#                 (
#                     config['rounds'], config['feature_selection_type'], config['feature_selection_method'],
#                     config['scoring'], config['splits'], config['normalization'],
#                     config['missing_values_method'], config['class_balance']
#                 )
#             )

#         job_id_result = cursor.fetchone()
#         job_id = job_id_result[0]

#         # Insert classifiers and associated data
#         for _, row in scores_dataframe.iterrows():
#             # Check if the classifier combination already exists
#             try:
#                 check_query = """
#                     SELECT classifier_id FROM Classifiers
#                     WHERE estimator = %s AND inner_selection = %s;
#                 """
#                 cursor.execute(check_query, (row["Est"], row["In_sel"]))
#                 classifier_result = cursor.fetchone()

#                 if classifier_result:
#                     classifier_id = classifier_result[0]
#                 else:
#                     classifier_query = """
#                         INSERT INTO Classifiers (estimator, inner_selection)
#                         VALUES (%s, %s) RETURNING classifier_id;
#                     """
#                     cursor.execute(classifier_query, (row["Est"], row["In_sel"]))
#                     classifier_id = cursor.fetchone()[0]
#             except Exception as e:
#                 print(f"Error with Classifiers table: {e}")
#                 raise

#             # Insert hyperparameters
#             try:
#                 hyperparameters_query = """
#                     INSERT INTO Hyperparameters (hyperparameters)
#                     VALUES (%s) RETURNING hyperparameter_id;
#                 """
#                 hyperparameters = row["Hyp"]
#                 if isinstance(hyperparameters, np.ndarray):
#                     hyperparameters = [dict(item) for item in hyperparameters]
#                 cursor.execute(hyperparameters_query, (json.dumps(hyperparameters),))
#                 hyperparameter_id = cursor.fetchone()[0]
#             except Exception as e:
#                 print(f"Error with Hyperparameters table: {e}")
#                 raise

#             # Check if the feature selection data already exists
#             try:
#                 feature_selection_check_query = """
#                     SELECT selection_id FROM Feature_Selection
#                     WHERE way_of_selection = %s AND numbers_of_features = %s;
#                 """
#                 cursor.execute(feature_selection_check_query, (row["Sel_way"], row["Fs_num"]))
#                 feature_selection_result = cursor.fetchone()

#                 if feature_selection_result:
#                     selection_id = feature_selection_result[0]
#                 else:
#                     feature_selection_query = """
#                         INSERT INTO Feature_Selection (way_of_selection, numbers_of_features)
#                         VALUES (%s, %s) RETURNING selection_id;
#                     """
#                     cursor.execute(feature_selection_query, (row["Sel_way"], row["Fs_num"]))
#                     selection_id = cursor.fetchone()[0]
#             except Exception as e:
#                 print(f"Error with Feature_Selection table: {e}")
#                 raise

#             # Insert performance metrics
#             try:
#                 metrics_to_add = ', '.join([f'{metric}' for metric in config['extra_metrics']])
#                 list_of_s = [f'%s' for metric in config['extra_metrics']]
#                 count_s = ', '.join(list_of_s)
#                 performance_metrics_query = f"""
#                     INSERT INTO Performance_Metrics ({metrics_to_add})
#                     VALUES ({count_s}) RETURNING performance_id;
#                 """

#                 metrics = [
#                     json.dumps(
#                         metric.tolist() if isinstance(metric, np.ndarray) else (metric if metric is not None else None)
#                     )
#                     for metric in [row.get(metric, None) for metric in config['extra_metrics']]
#                 ]

#                 cursor.execute(performance_metrics_query, metrics)
#                 performance_results = cursor.fetchone()
#                 performance_id = performance_results[0]
#             except Exception as e:
#                 print(f"Error with Performance_Metrics table: {e}")
#                 raise

#             # Insert samples classification rates
#             try:
#                 samples_classification_query = """
#                     INSERT INTO Samples_Classification_Rates (samples_classification_rates)
#                     VALUES (%s) RETURNING sample_rate_id;
#                 """
#                 cursor.execute(samples_classification_query, (json.dumps(row["Classif_rates"]),))
#                 sample_rate_id = cursor.fetchone()[0]
#             except Exception as e:
#                 print(f"Error with Samples_Classification_Rates table: {e}")
#                 raise

#             # Insert data into job combinations
#             try:
#                 job_combinations_query = """
#                     INSERT INTO Job_Combinations (
#                         job_id, classifier_id, dataset_id, selection_id, hyperparameter_id, performance_id, sample_rate_id, model_selection_type
#                     )
#                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
#                     RETURNING combination_id;
#                 """
#                 cursor.execute(
#                     job_combinations_query,
#                     (job_id, classifier_id, dataset_id, selection_id, hyperparameter_id, performance_id, sample_rate_id, config['model_selection_type'])
#                 )
#                 combination_id = cursor.fetchone()[0]
#             except Exception as e:
#                 print(f"Error with Job_Combinations table: {e}")
#                 raise

#             # Insert feature counts and associate with combination_id
#             try:
#                 if row["Sel_feat"] is not None:
#                     selected_features = row["Sel_feat"]
#                     if isinstance(selected_features, np.ndarray):
#                         selected_features = selected_features.tolist()

#                     if any(isinstance(i, list) for i in selected_features):
#                         selected_features = [item for sublist in selected_features for item in sublist]

#                     # Count occurrences of each feature
#                     feature_counts = Counter([feature for feature in selected_features if feature])
#                     feature_counts_query = """
#                         INSERT INTO Feature_Counts (feature_name, count, combination_id)
#                         VALUES (%s, %s, %s) RETURNING count_id;
#                     """

#                     # Prepare feature values with combination_id included
#                     feature_values = []
#                     for feat, count in feature_counts.items():
#                         cursor.execute(feature_counts_query, (feat, count, combination_id))
#                         feature_values.append(cursor.fetchone()[0])

#                     # Associate feature_count_ids with job combination
#                     feature_list_query = """
#                         UPDATE Job_Combinations
#                         SET feature_count_ids = %s
#                         WHERE combination_id = %s;
#                     """
#                     cursor.execute(feature_list_query, (json.dumps(feature_values), combination_id))
#             except Exception as e:
#                 print(f"Error with Feature_Counts table: {e}")

#         # Commit the transaction
#         connection.commit()
#         print("Data inserted into the database successfully.")
#     except Exception as e:
#         connection.rollback()
#         print(f"An error occurred while inserting data into the database: {e}")
#     finally:
#         cursor.close()
#         connection.close()

