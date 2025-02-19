import os
import sqlite3
import json
import logging
from typing import Optional, Any, List, Dict, Union, Tuple
from collections import Counter

class DatabaseManager:
    """Manager for database operations including creation, queries, and data insertion."""

    def __init__(self, db_name: str = "ai4meta.db", db_folder: str = "database"):
        """Initialize DatabaseManager with specified database name and folder."""
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.db_path = os.path.join(db_folder, db_name)
        os.makedirs(db_folder, exist_ok=True)
        
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.execute("PRAGMA foreign_keys = ON;")
            self.connection.row_factory = sqlite3.Row
            self.logger.info(f"✓ Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            self.logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def execute_query(self, query: str, params: tuple = (), 
                     fetch_one: bool = False, fetch_all: bool = False) -> Optional[Any]:
        """Execute an SQL query with optional parameters and fetching options."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)

            if fetch_one:
                result = cursor.fetchone()
                return result
            if fetch_all:
                result = cursor.fetchall()
                return result

            self.connection.commit()

            if query.strip().lower().startswith("insert"):
                return cursor.lastrowid

        except sqlite3.IntegrityError as e:
            self.logger.error(f"Database integrity error: {str(e)}")
            raise sqlite3.IntegrityError(f"Integrity Error: {e}")
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {str(e)}")
            raise sqlite3.Error(f"Database Error: {e}")

        return None

    def close_connection(self) -> None:
        """Close the database connection."""
        self.connection.close()
        self.logger.info("✓ Database connection closed")

    def initialize_schema(self, create_table_queries: List[str]) -> None:
        """Initialize database schema using provided queries."""
        for i, query in enumerate(create_table_queries, 1):
            table_name = query.split('CREATE TABLE IF NOT EXISTS')[1].split('(')[0].strip()
            self.logger.info(f"Creating table {i}/{len(create_table_queries)}: {table_name}")
            self.execute_query(query)
        self.logger.info(f"✓ Successfully created {len(create_table_queries)} tables")

    def insert_dataset(self, dataset_name: str) -> int:
        """Insert or fetch a dataset ID."""
        query = """
            INSERT INTO Datasets (dataset_name) VALUES (?)
            ON CONFLICT(dataset_name) DO NOTHING;
        """
        self.execute_query(query, (dataset_name,))
        
        dataset_id_query = "SELECT dataset_id FROM Datasets WHERE dataset_name = ?;"
        result = self.execute_query(dataset_id_query, (dataset_name,), fetch_one=True)
        # self.logger.info(f"✓ Dataset processed with ID: {result[0]}")
        return result[0]

    def insert_job_parameters_rncv(self, config: Dict) -> int:
        """Insert job parameters for nested CV and return job_id."""
        
        # Check for existing job parameters
        select_query = """
            SELECT job_id 
            FROM Job_Parameters
            WHERE 
                rounds = ? AND feature_selection_type = ? AND feature_selection_method = ?
                AND outer_scoring = ? AND outer_splits = ? AND normalization = ?
                AND missing_values_method = ? AND class_balance = ?;
        """
        params = (
            config.get('rounds'), config.get('feature_selection_type'),
            config.get('feature_selection_method'), config.get('outer_scoring'),
            config.get('outer_splits'), config.get('normalization'),
            config.get('missing_values_method'), config.get('class_balance')
        )
        
        existing_job = self.execute_query(select_query, params, fetch_one=True)
        
        # if existing_job:
        #     self.logger.info(f"✓ Found existing job parameters with ID: {existing_job[0]}")
        #     return existing_job[0]

        # Insert new parameters
        insert_query = """
            INSERT INTO Job_Parameters (
                n_trials, rounds, feature_selection_type, feature_selection_method, 
                inner_scoring, outer_scoring, inner_splits, outer_splits, normalization, 
                missing_values_method, class_balance
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            config.get("n_trials"), config.get("rounds"), config.get("feature_selection_type"),
            config.get("feature_selection_method"), config.get("inner_scoring"),
            config.get("outer_scoring"), config.get("inner_splits"),
            config.get("outer_splits"), config.get("normalization"),
            config.get("missing_values_method"), config.get('class_balance')
        )
        
        job_id = self.execute_query(insert_query, params)
        self.logger.info(f"✓ Created new job parameters with ID: {job_id}")
        return job_id
    
    def insert_job_parameters_cv(self, config):
        """Insert job parameters for rcv and return the job_id."""
        
        # Check for existing job parameters
        job_parameters_query = """
            INSERT INTO Job_Parameters (
                rounds, feature_selection_type, feature_selection_method, 
                outer_scoring, outer_splits, normalization, 
                missing_values_method, class_balance, evaluation_mthd, param_grid, features_names_list
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            config.get("rounds"), config.get("feature_selection_type"), config.get("feature_selection_method"),
            config.get("scoring"), config.get("splits"), config.get("normalization"),
            config.get("missing_values_method"), config.get("class_balance"), config.get("evaluation"),
            config.get("param_grid"),config.get('features_names_list')
        )

        # Execute the insert query
        job_id = self.execute_query(job_parameters_query, params)
        # self.logger.info(f"✓ Created new job parameters with ID: {job_id}")
        return job_id

    def insert_classifier(self, estimator: str, inner_selection: Optional[str]) -> int:
        """Insert or fetch a classifier ID."""
        
        # Check for existing classifier
        select_query = """
            SELECT classifier_id 
            FROM Classifiers 
            WHERE estimator = ? AND inner_selection = ?;
        """
        existing_id = self.execute_query(select_query, (estimator, inner_selection), fetch_one=True)

        # if existing_id:
        #     self.logger.info(f"✓ Found existing classifier with ID: {existing_id[0]}")
        #     return existing_id[0]

        # Insert new classifier
        insert_query = "INSERT INTO Classifiers (estimator, inner_selection) VALUES (?, ?);"
        self.execute_query(insert_query, (estimator, inner_selection))
        
        classifier_id = self.execute_query(select_query, (estimator, inner_selection), fetch_one=True)[0]
        # self.logger.info(f"✓ Created new classifier with ID: {classifier_id}")
        return classifier_id

    def insert_feature_selection(self, way_of_selection: str, numbers_of_features: int) -> int:
        """Insert or fetch a feature selection ID."""
        
        # Check for existing feature selection
        select_query = """
            SELECT selection_id 
            FROM Feature_Selection 
            WHERE way_of_selection = ? AND numbers_of_features = ?;
        """
        existing_id = self.execute_query(select_query, 
                                       (way_of_selection, numbers_of_features), 
                                       fetch_one=True)

        # if existing_id:
        #     self.logger.info(f"✓ Found existing feature selection with ID: {existing_id[0]}")
        #     return existing_id[0]

        # Insert new feature selection
        insert_query = """
            INSERT INTO Feature_Selection (way_of_selection, numbers_of_features) 
            VALUES (?, ?);
        """
        self.execute_query(insert_query, (way_of_selection, numbers_of_features))
        
        selection_id = self.execute_query("SELECT last_insert_rowid();", fetch_one=True)[0]
        # self.logger.info(f"✓ Created new feature selection with ID: {selection_id}")
        return selection_id

    def insert_performance_metrics(self, metrics_values: List, metrics_names: List[str]) -> int:
        """Insert performance metrics and return the ID."""
        
        columns = ', '.join(metrics_names)
        placeholders = ', '.join(['?'] * len(metrics_names))
        query = f"INSERT INTO Performance_Metrics ({columns}) VALUES ({placeholders});"
        
        self.execute_query(query, metrics_values)
        metrics_id = self.execute_query("SELECT last_insert_rowid();", fetch_one=True)[0]
        
        # self.logger.info(f"✓ Inserted performance metrics with ID: {metrics_id}")
        return metrics_id

    def insert_hyperparameters(self, hyperparameters: str) -> int:
        """Insert or fetch a hyperparameter ID."""
                
        select_query = "SELECT hyperparameter_id FROM Hyperparameters WHERE hyperparameters = ?;"
        existing_id = self.execute_query(select_query, (hyperparameters,), fetch_one=True)

        # if existing_id:
        #     self.logger.info(f"✓ Found existing hyperparameters with ID: {existing_id[0]}")
        #     return existing_id[0]

        insert_query = "INSERT INTO Hyperparameters (hyperparameters) VALUES (?);"
        self.execute_query(insert_query, (hyperparameters,))
        
        hyperparameter_id = self.execute_query("SELECT last_insert_rowid();", fetch_one=True)[0]
        # self.logger.info(f"✓ Created new hyperparameters with ID: {hyperparameter_id}")
        return hyperparameter_id

    def insert_sample_classification_rates(self, classification_rates: str) -> int:
        """Insert or fetch a sample classification rates ID."""

        select_query = """
            SELECT sample_rate_id FROM Samples_Classification_Rates 
            WHERE samples_classification_rates = ?;
        """
        existing_id = self.execute_query(select_query, (classification_rates,), fetch_one=True)

        # if existing_id:
        #     self.logger.info(f"✓ Found existing classification rates with ID: {existing_id[0]}")
        #     return existing_id[0]

        query = """
            INSERT INTO Samples_Classification_Rates (samples_classification_rates)
            VALUES (?);
        """
        self.execute_query(query, (classification_rates,))
        
        rate_id = self.execute_query("SELECT last_insert_rowid();", fetch_one=True)[0]
        # self.logger.info(f"✓ Created new classification rates with ID: {rate_id}")
        return rate_id

    def insert_job_combination(self, job_id: int, classifier_id: int, dataset_id: int,
                             selection_id: Optional[int], hyperparameter_id: int,
                             performance_id: int, sample_rate_id: Optional[int],
                             model_selection_type: str) -> int:
        """Insert or fetch a job combination ID."""        
        # Construct dynamic query conditions
        conditions = []
        params = []
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
        select_query = f"SELECT combination_id FROM Job_Combinations WHERE {where_clause};"
        
        existing_id = self.execute_query(select_query, params, fetch_one=True)
        # if existing_id:
        #     self.logger.info(f"✓ Found existing job combination with ID: {existing_id[0]}")
        #     return existing_id[0]

        # Insert new combination
        insert_query = """
            INSERT INTO Job_Combinations (
                job_id, classifier_id, dataset_id, selection_id, hyperparameter_id,
                performance_id, sample_rate_id, model_selection_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        insert_params = (job_id, classifier_id, dataset_id, selection_id,
                        hyperparameter_id, performance_id, sample_rate_id,
                        model_selection_type)
        
        self.execute_query(insert_query, insert_params)
        combination_id = self.execute_query("SELECT last_insert_rowid();", fetch_one=True)[0]
        self.logger.info(f"✓ Created new job combination with ID: {combination_id}")
        return combination_id

    def insert_feature_counts(self, selected_features: List[str], combination_id: int) -> None:
        """Insert feature counts for a job combination."""        
        # Filter out invalid features
        selected_features = [feature for feature in selected_features if feature]
        if not selected_features:
            raise ValueError(f"No valid features to insert for combination_id: {combination_id}")

        feature_counts = Counter(selected_features)
        total_features = len(feature_counts)
        processed = 0
        
        for feature, count in feature_counts.items():
            processed += 1
            select_query = """
                SELECT count_id 
                FROM Feature_Counts 
                WHERE feature_name = ? AND count = ? AND combination_id = ?;
            """
            existing_id = self.execute_query(select_query, 
                                           (feature, count, combination_id), 
                                           fetch_one=True)

            if not existing_id:
                insert_query = """
                    INSERT INTO Feature_Counts (feature_name, count, combination_id)
                    VALUES (?, ?, ?);
                """
                self.execute_query(insert_query, (feature, count, combination_id))

        # self.logger.info(f"✓ Processed {total_features} feature counts")