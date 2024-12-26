import os
import sqlite3
import json
from collections import Counter

class DatabaseManager():
    def __init__(self, db_name="ai4meta.db", db_folder="database"):
        self.db_path = os.path.join(db_folder, db_name)
        os.makedirs(db_folder, exist_ok=True)
        self._enable_foreign_keys()

    def _enable_foreign_keys(self):
        """Enable foreign key support for SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")

    def execute_query(self, query, params=None, fetch_one=False, fetch_all=False):
        """Execute a query on the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            if fetch_one:
                return cursor.fetchone()
            if fetch_all:
                return cursor.fetchall()
            conn.commit()

    def initialize_schema(self, create_table_queries):
        """Initialize the database schema."""
        for query in create_table_queries:
            self.execute_query(query)

    def insert_dataset(self, dataset_name):
        """Insert or fetch a dataset ID."""
        query = """
            INSERT INTO Datasets (dataset_name) VALUES (?)
            ON CONFLICT(dataset_name) DO NOTHING;
        """
        self.execute_query(query, (dataset_name,))
        dataset_id_query = "SELECT dataset_id FROM Datasets WHERE dataset_name = ?;"
        return self.execute_query(dataset_id_query, (dataset_name,), fetch_one=True)[0]
    
    def insert_job_parameters_rncv(self, config):
        # Fetch existing job_id first
        job_parameters_select_query = """
            SELECT job_id 
            FROM Job_Parameters
            WHERE 
                rounds = ? AND feature_selection_type = ? AND feature_selection_method = ?
                AND outer_scoring = ? AND outer_splits = ? AND normalization = ?
                AND missing_values_method = ? AND class_balance = ?;
        """
        job_id_result = self.execute_query(
            job_parameters_select_query,
            (
                config.get('rounds'), config.get('feature_selection_type'), config.get('feature_selection_method'),
                config.get('outer_scoring'), config.get('outer_splits'), config.get('normalization'),
                config.get('missing_values_method'), config.get('class_balance')
            ),
            fetch_one=True
        )

        if job_id_result:
            return job_id_result[0]  # Return existing job_id

        # If no matching job_id found, insert new parameters
        job_parameters_query = """
            INSERT INTO Job_Parameters (
                n_trials, rounds, feature_selection_type, feature_selection_method, 
                inner_scoring, outer_scoring, inner_splits, outer_splits, normalization, 
                missing_values_method, class_balance, evaluation_mthd, param_grid, features_names_list
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING job_id;  # Retrieve job_id after insertion
        """
        params = (
            config.get("n_trials"), config.get("rounds"), config.get("feature_selection_type"),
            config.get("feature_selection_method"), config.get("inner_scoring"), config.get("outer_scoring"),
            config.get("inner_splits"), config.get("outer_splits"), config.get("normalization"),
            config.get("missing_values_method"), config.get('class_balance'), None, None, None
        )

        # Execute the insert query and retrieve the job_id
        job_id_result = self.execute_query(job_parameters_query, params, fetch_one=True)

        if job_id_result:
            return job_id_result[0]
        else:
            raise ValueError("Failed to insert job parameters and fetch job_id.")

    def insert_job_parameters_rcv(self, config):
        # Fetch existing job_id first
        job_parameters_select_query = """
            SELECT job_id 
            FROM Job_Parameters
            WHERE 
                rounds = ? AND feature_selection_type = ? AND feature_selection_method = ?
                AND outer_scoring = ? AND outer_splits = ? AND normalization = ?
                AND missing_values_method = ? AND class_balance = ?;
        """
        job_id_result = self.execute_query(
            job_parameters_select_query,
            (
                config.get('rounds'), config.get('feature_selection_type'), config.get('feature_selection_method'),
                config.get('scoring'), config.get('splits'), config.get('normalization'),
                config.get('missing_values_method'), config.get('class_balance')
            ),
            fetch_one=True
        )

        if job_id_result:
            return job_id_result[0]  # Return existing job_id

        # If no matching job_id found, insert new parameters
        job_parameters_query = """
            INSERT INTO Job_Parameters (
                rounds, feature_selection_type, feature_selection_method, 
                outer_scoring, outer_splits, normalization, 
                missing_values_method, class_balance
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING job_id;  # Retrieve job_id after insertion
        """
        params = (
            config.get("rounds"), config.get("feature_selection_type"), config.get("feature_selection_method"),
            config.get("scoring"), config.get("splits"), config.get("normalization"),
            config.get("missing_values_method"), config.get("class_balance")
        )

        # Execute the insert query and retrieve the job_id
        job_id_result = self.execute_query(job_parameters_query, params, fetch_one=True)

        if job_id_result:
            return job_id_result[0]
        else:
            raise ValueError("Failed to insert job parameters and fetch job_id.")

    def insert_classifier(self, estimator, inner_selection):
        # Check if the classifier already exists
        select_query = """
            SELECT classifier_id 
            FROM Classifiers 
            WHERE estimator = ? AND inner_selection = ?;
        """
        existing_id = self.execute_query(select_query, (estimator, inner_selection), fetch_one=True)

        if existing_id:
            return existing_id[0]  # Return existing classifier_id

        # If the classifier doesn't exist, insert it
        insert_query = """
            INSERT INTO Classifiers (estimator, inner_selection) 
            VALUES (?, ?)
        """
        self.execute_query(insert_query, (estimator, inner_selection))

        # Retrieve the newly inserted classifier_id
        select_query = "SELECT classifier_id FROM Classifiers WHERE estimator = ? AND inner_selection = ?;"
        return self.execute_query(select_query, (estimator, inner_selection), fetch_one=True)[0]
  
    def insert_feature_selection(self, way_of_selection, numbers_of_features):
        # Check if the feature selection already exists
        select_query = """
            SELECT selection_id 
            FROM Feature_Selection 
            WHERE way_of_selection = ? AND numbers_of_features = ?;
        """
        existing_id = self.execute_query(select_query, (way_of_selection, numbers_of_features), fetch_one=True)

        if existing_id:
            return existing_id[0]  # Return existing feature_selection_id

        # If the feature selection doesn't exist, insert it
        insert_query = """
            INSERT INTO Feature_Selection (way_of_selection, numbers_of_features) 
            VALUES (?, ?);
        """
        self.execute_query(insert_query, (way_of_selection, numbers_of_features))

        # Retrieve the newly inserted feature_selection_id
        select_query = "SELECT last_insert_rowid();"
        return self.execute_query(select_query, fetch_one=True)[0]

    def insert_performance_metrics(self, metrics_dict):
        """Insert performance metrics and return the ID."""
        columns = ', '.join(metrics_dict.keys())
        placeholders = ', '.join(['?'] * len(metrics_dict))
        query = f"INSERT INTO Performance_Metrics ({columns}) VALUES ({placeholders});"
        self.execute_query(query, list(metrics_dict.values()))
        return self.execute_query("SELECT last_insert_rowid();", fetch_one=True)[0]

    def insert_job_combination(self, job_id, classifier_id, dataset_id, selection_id, 
                            hyperparameter_id, performance_id, sample_rate_id, 
                            model_selection_type):
        select_query = """
            SELECT combination_id 
            FROM Job_Combinations 
            WHERE job_id = ? AND classifier_id = ? AND dataset_id = ? AND selection_id = ? 
            AND hyperparameter_id = ? AND performance_id = ? AND sample_rate_id = ? 
            AND model_selection_type = ?;
        """
        existing_id = self.execute_query(select_query, (job_id, classifier_id, dataset_id, selection_id, 
                                                    hyperparameter_id, performance_id, sample_rate_id, 
                                                    model_selection_type), fetch_one=True)

        if existing_id:
            return existing_id[0]  # Return existing job_combination_id

        insert_query = """
            INSERT INTO Job_Combinations (
                job_id, classifier_id, dataset_id, selection_id, hyperparameter_id,
                performance_id, sample_rate_id, model_selection_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.execute_query(insert_query, (job_id, classifier_id, dataset_id, selection_id, 
                                hyperparameter_id, performance_id, sample_rate_id, 
                                model_selection_type))

        return self.execute_query("SELECT last_insert_rowid();", fetch_one=True)[0]

    def insert_feature_counts(self, selected_features, combination_id):
        feature_counts = Counter(selected_features)
        for feature, count in feature_counts.items():
            select_query = """
                SELECT count_id 
                FROM Feature_Counts 
                WHERE feature_name = ? AND count = ? AND combination_id = ?;
            """
            existing_id = self.execute_query(select_query, (feature, count, combination_id), fetch_one=True)

            if existing_id:
                continue  # If the entry already exists, skip insertion

            insert_query = """
                INSERT INTO Feature_Counts (feature_name, count, combination_id)
                VALUES (?, ?, ?);
            """
            self.execute_query(insert_query, (feature, count, combination_id))

    def insert_hyperparameters(self, hyperparameters):
        select_query = "SELECT hyperparameter_id FROM Hyperparameters WHERE hyperparameters = ?;"
        existing_id = self.execute_query(select_query, (hyperparameters,), fetch_one=True)

        if existing_id:
            return existing_id[0]  # Return existing hyperparameter_id

        query = "INSERT INTO Hyperparameters (hyperparameters) VALUES (?);"
        self.execute_query(query, (hyperparameters,))
        return self.execute_query("SELECT last_insert_rowid();", fetch_one=True)[0]

    def insert_sample_classification_rates(self, classification_rates):
        select_query = """
            SELECT sample_rate_id FROM Samples_Classification_Rates 
            WHERE samples_classification_rates = ?;
        """
        existing_id = self.execute_query(select_query, (classification_rates,), fetch_one=True)

        if existing_id:
            return existing_id[0]  # Return existing sample_rate_id

        query = """
            INSERT INTO Samples_Classification_Rates (samples_classification_rates)
            VALUES (?);
        """
        self.execute_query(query, (classification_rates,))
        return self.execute_query("SELECT last_insert_rowid();", fetch_one=True)[0]
