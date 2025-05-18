import os
import logging
from typing import Optional, Any, List, Dict, Union, Tuple, Type
from collections import Counter
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import numpy as np
import json
from collections import Counter

from sqlalchemy import func
# Import your dataclass models
from src.database.dataclasses import Base, Dataset, Classifier, FeatureSelection, Hyperparameters
from src.database.dataclasses import PerformanceMetrics, SamplesClassificationRates, ShapValues, Experiment, FeatureCount

class DatabaseManager:
    """Manager for database operations using SQLAlchemy ORM with dataclasses."""

    def __init__(self, db_name: str = "ai4meta.db", db_folder: str = "database"):
        """Initialize DatabaseManager with specified database name and folder."""
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        os.makedirs(db_folder, exist_ok=True)
        self.db_path = os.path.join(db_folder, db_name)
        
        try:
            # Create engine and session
            self.engine = create_engine(f"sqlite:///{self.db_path}")
            self.Session = sessionmaker(bind=self.engine)
            self.logger.info(f"✓ Connected to database: {self.db_path}")
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def initialize_schema(self) -> None:
        """Initialize database schema using SQLAlchemy models."""
        try:
            # Create all tables defined in Base
            Base.metadata.create_all(self.engine)
            
            # Get all table names for logging
            inspector = inspect(self.engine)
            table_names = inspector.get_table_names()
            
            self.logger.info(f"✓ Successfully created {len(table_names)} tables: {', '.join(table_names)}")
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to initialize schema: {str(e)}")
            raise

    def close_connection(self) -> None:
        """Close the database connection."""
        # SQLAlchemy handles connection pooling, so we don't need to explicitly close anything
        self.logger.info("✓ Database connection closed")

    def _get_or_create(self, session: Session, model: Type[Base], **kwargs) -> Tuple[Any, bool]:
        """Get an existing instance or create a new one."""
        instance = session.query(model).filter_by(**kwargs).first()
        if instance:
            return instance, False
        else:
            instance = model(**kwargs)
            session.add(instance)
            session.flush()  # Flush to get the ID without committing
            return instance, True

    def insert_dataset(self, dataset_name: str) -> int:
        """Insert or fetch a dataset ID."""
        with self.Session() as session:
            dataset, created = self._get_or_create(session, Dataset, dataset_name=dataset_name)
            if created:
                session.commit()
                self.logger.info(f"✓ Created new dataset with ID: {dataset.dataset_id}")
            return dataset.dataset_id

    def insert_classifier(self, estimator: str, inner_selection: Optional[str]) -> int:
        """Insert or fetch a classifier ID."""
        with self.Session() as session:
            classifier, created = self._get_or_create(
                session, Classifier, estimator=estimator, inner_selection=inner_selection
            )
            if created:
                session.commit()
                self.logger.info(f"✓ Created new classifier with ID: {classifier.classifier_id}")
            return classifier.classifier_id

    def insert_feature_selection(self, way_of_selection: str, numbers_of_features: int, 
                                way_of_inner_selection: str) -> int:
        """Insert or fetch a feature selection ID."""
        with self.Session() as session:
            selection, created = self._get_or_create(
                session, FeatureSelection, 
                way_of_selection=way_of_selection, 
                numbers_of_features=numbers_of_features,
                way_of_inner_selection=way_of_inner_selection
            )
            if created:
                session.commit()
                self.logger.info(f"✓ Created new feature selection with ID: {selection.selection_id}")
            return selection.selection_id

    def insert_hyperparameters(self, hyperparameters: str) -> int:
        """Insert or fetch hyperparameters ID."""
        with self.Session() as session:
            params, created = self._get_or_create(
                session, Hyperparameters, hyperparameters=hyperparameters
            )
            if created:
                session.commit()
                self.logger.info(f"✓ Created new hyperparameters with ID: {params.hyperparameter_id}")
            return params.hyperparameter_id

    def insert_performance_metrics(self, metrics_dict: Dict[str, str]) -> int:
        """Insert performance metrics and return the ID."""
        with self.Session() as session:
            try:
                # Create a new PerformanceMetrics instance
                metrics = PerformanceMetrics(**metrics_dict)
                session.add(metrics)
                session.flush()
                metrics_id = metrics.performance_id
                session.commit()
                self.logger.info(f"✓ Inserted performance metrics with ID: {metrics_id}")
                return metrics_id
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Failed to insert performance metrics: {str(e)}")
                raise

    def insert_sample_classification_rates(self, classification_rates: str) -> int:
        """Insert or fetch a sample classification rates ID."""
        with self.Session() as session:
            rates, created = self._get_or_create(
                session, SamplesClassificationRates, 
                samples_classification_rates=classification_rates
            )
            if created:
                session.commit()
                self.logger.info(f"✓ Created new classification rates with ID: {rates.sample_rate_id}")
            return rates.sample_rate_id

    def insert_shap_values(self, shap_values: str) -> int:
        """Insert SHAP values and return the ID."""
        with self.Session() as session:
            try:
                shap = ShapValues(shap_values=shap_values)
                session.add(shap)
                session.flush()
                shap_id = shap.shap_values_id
                session.commit()
                self.logger.info(f"✓ Inserted SHAP values with ID: {shap_id}")
                return shap_id
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Failed to insert SHAP values: {str(e)}")
                raise

    def insert_experiment(self, experiment_data: Dict[str, Any]) -> int:
        """Insert an experiment and return its ID."""
        with self.Session() as session:
            try:
                experiment = Experiment(**experiment_data)
                session.add(experiment)
                session.flush()
                experiment_id = experiment.experiment_id
                session.commit()
                self.logger.info(f"✓ Created new experiment with ID: {experiment_id}")
                return experiment_id
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Failed to insert experiment: {str(e)}")
                raise

    def insert_feature_counts(self, feature_counts: Dict[str, int], experiment_id: int) -> None:
        """Insert feature counts for an experiment."""
        with self.Session() as session:
            try:
                # Check if we have any features
                if feature_counts == {'All': 1}:
                    self.logger.warning(f"No features to insert for experiment_id: {experiment_id}")
                    return
                
                # Get the experiment to ensure it exists
                experiment = session.query(Experiment).get(experiment_id)
                if not experiment:
                    raise ValueError(f"Experiment with ID {experiment_id} does not exist")
                
                # Insert feature counts
                for feature_name, count in feature_counts.items():
                    # Check if this feature count already exists
                    existing = session.query(FeatureCount).filter_by(
                        feature_name=feature_name, count=count, experiment_id=experiment_id
                    ).first()
                    
                    if not existing:
                        feature_count = FeatureCount(
                            feature_name=feature_name, 
                            count=count,
                            experiment_id=experiment_id  # Add this line to set the experiment_id
                        )
                        session.add(feature_count)
                
                session.commit()
                self.logger.info(f"✓ Inserted {len(feature_counts)} feature counts for experiment {experiment_id}")
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Failed to insert feature counts: {str(e)}")
                raise
            
    def insert_experiment_data(self, scores_dataframe, config, database_name=None):
        """
        Insert experiment data from model selection or evaluation into the database.
        
        This method handles the insertion of experiment results from either MLSelector 
        (model selection) or MachineLearningEstimator (model evaluation) into the database,
        correctly mapping the dataframe and configuration objects to the database schema.
        
        Parameters:
        -----------
        scores_dataframe : pandas.DataFrame
            DataFrame containing experiment scores and metadata
        config : ModelSelectionConfig or ModelEvaluationConfig
            Configuration object with experiment parameters
        database_name : str, optional
            Name of the database (default: uses instance database_name)
        """
        with self.Session() as session:
            try:
                # Determine the type of experiment
                is_model_selection = hasattr(config, 'model_selection_type')
                is_model_evaluation = hasattr(config, 'search_type')

                max_combination_id = session.query(func.max(Experiment.combination_id)).scalar()
            
                # If no combination_id exists yet or it's None, start with 1
                if max_combination_id is None:
                    combination_id = 1
                else:
                    combination_id = max_combination_id + 1
                    
                self.logger.info(f"Using combination_id: {combination_id} for this batch of experiments")
                
                # Insert dataset
                dataset_id = self.insert_dataset(config.dataset_name)
                
                # Process each row in scores_dataframe
                for _, row in scores_dataframe.iterrows():
                    # Get classifier info
                    estimator = row.get("Est") if "Est" in scores_dataframe.columns else config.estimator_name
                    inner_selection = row.get("Inner_selection") if "Inner_selection" in scores_dataframe.columns else getattr(config, 'inner_selection', None)
                    classifier_id = self.insert_classifier(estimator, inner_selection)
                    
                    # Get hyperparameters
                    hyperparameters = row.get("Hyp") if "Hyp" in scores_dataframe.columns else None
                    if isinstance(hyperparameters, np.ndarray):
                        hyperparameters = hyperparameters.tolist()
                    hyperparameter_id = self.insert_hyperparameters(json.dumps(hyperparameters))
                    
                    # Process feature selection - FIXED to not use way_of_inner_selection
                    selection_id = None
                    # Call insert_feature_selection with only two parameters
                    selection_id = self.insert_feature_selection(row.get("Sel_way"), row.get("Fs_num"), row.get('Fs_inner'))

                    # Process sample classification rates
                    sample_rate_id = None
                    if "Classif_rates" in scores_dataframe.columns:
                        classif_rates_str = np.array2string(
                            row["Classif_rates"], separator=',', suppress_small=True, max_line_width=np.inf
                        )
                        sample_rate_id = self.insert_sample_classification_rates(classif_rates_str)
                    
                    # Process SHAP values if available
                    shap_values_id = None
                    if hasattr(config, 'calculate_shap') and config.calculate_shap and hasattr(row, 'Shap'):
                        shap_values_str = json.dumps(row['Shap'].tolist() if isinstance(row['Shap'], np.ndarray) else row['Shap'])
                        shap_values_id = self.insert_shap_values(shap_values_str)
                    
                    # Process performance metrics
                    metrics_dict = {}
                    for metric in config.extra_metrics:
                        metric_value = row.get(metric)
                        if isinstance(metric_value, np.ndarray):
                            metric_value = json.dumps(metric_value.tolist())
                        elif metric_value is not None:
                            metric_value = json.dumps(metric_value)
                        metrics_dict[metric] = metric_value
                    performance_id = self.insert_performance_metrics(metrics_dict)
                    
                    # Create experiment data dictionary
                    experiment_data = {
                        'n_trials': getattr(config, 'n_trials', None),
                        'rounds': getattr(config, 'rounds', None),
                        'feature_selection_type': row.get("Sel_way"),
                        'feature_selection_method': row.get("Fs_inner"),
                        'inner_scoring': getattr(config, 'inner_scoring', None),
                        'scoring': getattr(config, 'scoring', None),
                        'inner_splits': getattr(config, 'inner_splits', None),
                        'splits': getattr(config, 'splits', None),
                        'normalization': getattr(config, 'normalization', None),
                        'missing_values_method': getattr(config, 'missing_values_method', None),
                        'class_balance': getattr(config, 'class_balance', None),
                        'evaluation_mthd': getattr(config, 'evaluation', None),
                        'classifier_id': classifier_id,
                        'dataset_id': dataset_id,
                        'selection_id': selection_id,
                        'hyperparameter_id': hyperparameter_id,
                        'performance_id': performance_id,
                        'sample_rate_id': sample_rate_id,
                        'shap_values_id': shap_values_id,
                        'combination_id': combination_id
                    }
                    
                    # Add specific fields based on experiment type
                    if is_model_selection:
                        experiment_data.update({
                            'model_selection_type': getattr(config, 'model_selection_type', None),
                            'output_csv_path': getattr(config, 'dataset_csv_name', None),
                            'output_histogram_path': getattr(config, 'dataset_histogram_name', None),
                            'output_plot_path': getattr(config, 'dataset_plot_name', None),
                        })
                    
                    if is_model_evaluation:
                        experiment_data.update({
                            'search_type': getattr(config, 'search_type', None),
                            'output_csv_path': getattr(config, 'dataset_csv_name', None),
                            'output_plot_path': getattr(config, 'dataset_plot_name', None),
                            'output_model_path': getattr(config, 'model_path', None),
                            'output_metadata_path': getattr(config, 'metadata_path', None),
                            'output_parameters_path': getattr(config, 'params_path', None),
                        })
                    
                    # Insert experiment
                    experiment_id = self.insert_experiment(experiment_data)
                    
                    # Process feature counts if needed
                    if "Sel_feat" in row and row.get("Sel_way") != 'none':  
                        selected_features = row["Sel_feat"]
                        if isinstance(selected_features, np.ndarray):
                            selected_features = selected_features.tolist()
                        if any(isinstance(i, list) for i in selected_features):
                            selected_features = [item for sublist in selected_features for item in sublist]
                        
                        # Filter out None values, only proceed if there are valid features
                        valid_features = [f for f in selected_features if f is not None]
                        if valid_features:
                            feature_counts = Counter(valid_features)
                            self.insert_feature_counts(feature_counts, experiment_id)
                        else:
                            self.insert_feature_counts({'All': 1}, experiment_id)

                self.logger.info("✓ Data inserted into the database successfully.")
            except Exception as e:
                session.rollback()
                self.logger.error(f"Failed to insert experiment data: {str(e)}")
                raise
