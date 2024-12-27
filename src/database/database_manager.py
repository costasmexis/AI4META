import os
import sqlite3
import json
from collections import Counter

class DatabaseManager():
    def __init__(self, db_name="ai4meta.db", db_folder="database"):
        self.db_path = os.path.join(db_folder, db_name)
        os.makedirs(db_folder, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute("PRAGMA foreign_keys = ON;")  # Enable foreign keys
        self.connection.row_factory = sqlite3.Row  # Optional: fetch results as dictionaries

    def execute_query(self, query, params=(), fetch_one=False, fetch_all=False):
        """Execute a query on the SQLite database."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)

            if fetch_one:
                return cursor.fetchone()
            if fetch_all:
                return cursor.fetchall()

            self.connection.commit()

            # Return lastrowid for INSERT queries
            if query.strip().lower().startswith("insert"):
                return cursor.lastrowid

        except sqlite3.IntegrityError as e:
            raise sqlite3.IntegrityError(f"Integrity Error: {e}")
        except sqlite3.Error as e:
            raise sqlite3.Error(f"Database Error: {e}")

        return None

    def close_connection(self):
        """Close the database connection."""
        self.connection.close()

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
            RETURNING job_id;  
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

    def insert_performance_metrics(self, metrics_values, metrics_names):
        """Insert performance metrics and return the ID."""
        # Join column names and placeholders
        columns = ', '.join(metrics_names)
        placeholders = ', '.join(['?'] * len(metrics_names))
        query = f"INSERT INTO Performance_Metrics ({columns}) VALUES ({placeholders});"
        # Execute the query with ordered metrics
        self.execute_query(query, metrics_values)
        return self.execute_query("SELECT last_insert_rowid();", fetch_one=True)[0]

    def insert_job_combination(self, job_id, classifier_id, dataset_id, selection_id, 
                                hyperparameter_id, performance_id, sample_rate_id, 
                                model_selection_type):
        """Insert or fetch a job combination and return its ID."""
        
        # Construct WHERE clause dynamically
        conditions = []
        params = []

        # Mapping column names to their values
        columns = [
            ("job_id", job_id),
            ("classifier_id", classifier_id),
            ("dataset_id", dataset_id),
            ("selection_id", selection_id),
            ("hyperparameter_id", hyperparameter_id),
            ("performance_id", performance_id),
            ("sample_rate_id", sample_rate_id),
            ("model_selection_type", model_selection_type),
        ]

        for col, val in columns:
            if val is None:
                conditions.append(f"{col} IS NULL")
            else:
                conditions.append(f"{col} = ?")
                params.append(val)

        where_clause = " AND ".join(conditions)

        # Check if the combination already exists
        select_query = f"""
            SELECT combination_id 
            FROM Job_Combinations 
            WHERE {where_clause};
        """
        existing_id = self.execute_query(select_query, params, fetch_one=True)

        if existing_id:
            return existing_id[0]  # Return the existing combination ID

        # Insert a new combination
        insert_query = """
            INSERT INTO Job_Combinations (
                job_id, classifier_id, dataset_id, selection_id, hyperparameter_id,
                performance_id, sample_rate_id, model_selection_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.execute_query(insert_query, (
            job_id, classifier_id, dataset_id, selection_id, 
            hyperparameter_id, performance_id, sample_rate_id, 
            model_selection_type))

        # Fetch the newly inserted combination_id
        new_combination_id = self.execute_query("SELECT last_insert_rowid();", fetch_one=True)
        if new_combination_id:
            return new_combination_id[0]
        else:
            raise ValueError("Failed to insert or fetch combination_id.")

    def insert_feature_counts(self, selected_features, combination_id):
        """Insert feature counts for a job combination."""
        # Filter out invalid features (e.g., None or empty strings)
        selected_features = [feature for feature in selected_features if feature]

        if not selected_features:
            raise ValueError("No valid features to insert for combination_id: {}".format(combination_id))

        feature_counts = Counter(selected_features)
        for feature, count in feature_counts.items():
            select_query = """
                SELECT count_id 
                FROM Feature_Counts 
                WHERE feature_name = ? AND count = ? AND combination_id = ?;
            """
            existing_id = self.execute_query(select_query, (feature, count, combination_id), fetch_one=True)

            if existing_id:
                # print(f"Feature '{feature}' with count '{count}' already exists for combination_id {combination_id}")
                continue  # Skip existing rows

            insert_query = """
                INSERT INTO Feature_Counts (feature_name, count, combination_id)
                VALUES (?, ?, ?);
            """
            self.execute_query(insert_query, (feature, count, combination_id))

    def insert_hyperparameters(self, hyperparameters):
        """Insert or fetch a hyperparameter ID."""
        select_query = "SELECT hyperparameter_id FROM Hyperparameters WHERE hyperparameters = ?;"
        existing_id = self.execute_query(select_query, (hyperparameters,), fetch_one=True)

        if existing_id:
            return existing_id[0]

        # If the hyperparameters don't exist, insert them
        insert_query = "INSERT INTO Hyperparameters (hyperparameters) VALUES (?);"
        hyperparameter_id = self.execute_query(insert_query, (hyperparameters,))

        if hyperparameter_id is None:
            raise ValueError("Failed to insert or fetch hyperparameter_id.")
        return hyperparameter_id

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
