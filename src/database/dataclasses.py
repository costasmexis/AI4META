from dataclasses import dataclass, field, InitVar
from typing import Optional, Dict, Any, List, ClassVar
import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, Boolean, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from src.constants.translators import INNER_SELECTION_METHODS, AVAILABLE_CLFS

Base = declarative_base()

@dataclass
class Dataset(Base):
    """Dataset information table"""
    __tablename__ = 'datasets'
    
    dataset_id: int = field(default=None)
    dataset_name: str = field()
    
    # Columns
    dataset_id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_name = Column(String, nullable=False, unique=True)
    
    # Relationships
    experiments = relationship("Experiment", back_populates="dataset")

@dataclass
class Classifier(Base):
    """
    Classifier information table
    
    This table stores information about machine learning classifiers and their inner selection methods.
    Each record represents a unique combination of estimator and inner selection method.
    """
    __tablename__ = 'classifiers'
    
    classifier_id: int = field(default=None)
    estimator: str = field()
    inner_selection: Optional[str] = field(default=None)
    
    # Columns
    classifier_id = Column(Integer, primary_key=True, autoincrement=True)
    estimator = Column(String, nullable=False)  # Should match one of AVAILABLE_CLFS keys
    inner_selection = Column(String) # Should match one of INNER_SELECTION_METHODS
    
    # Relationships
    experiments = relationship("Experiment", back_populates="classifier")
    
    # Class-level constants
    valid_estimators: ClassVar[List[str]] = AVAILABLE_CLFS.keys()
    valid_inner_selections: ClassVar[List[str]] = INNER_SELECTION_METHODS

    @validates('estimator')
    def validate_estimator(self, key, estimator):
        """Ensure estimator is one of the valid choices"""
        if estimator not in self.valid_estimators:
            raise ValueError(f"Invalid estimator: {estimator}. Must be one of: {', '.join(self.valid_estimators)}")
        return estimator
    
    @validates('inner_selection')
    def validate_inner_selection(self, key, inner_selection):
        """Ensure inner_selection is one of the valid choices if provided"""
        if inner_selection is not None and inner_selection not in self.valid_inner_selections:
            raise ValueError(
                f"Invalid inner_selection: {inner_selection}. Must be one of: {', '.join(self.valid_inner_selections)}"
            )
        return inner_selection

@dataclass
class FeatureSelection(Base):
    """Feature selection information table"""
    __tablename__ = 'feature_selection'
    
    selection_id: int = field(default=None)
    way_of_selection: Optional[str] = field(default=None)
    numbers_of_features: Optional[int] = field(default=None)
    way_of_inner_selection: Optional[str] = field(default=None)
    
    # Columns
    selection_id = Column(Integer, primary_key=True, autoincrement=True)
    way_of_selection = Column(String)
    numbers_of_features = Column(Integer)
    way_of_inner_selection = Column(String)
    
    # Relationships
    experiments = relationship("Experiment", back_populates="feature_selection")
    
@dataclass
class Hyperparameters(Base):
    """Hyperparameters information table"""
    __tablename__ = 'hyperparameters'
    
    hyperparameter_id: int = field(default=None)
    hyperparameters: str = field()
    
    # Columns
    hyperparameter_id = Column(Integer, primary_key=True, autoincrement=True)
    hyperparameters = Column(Text, nullable=False)
    
    # Relationships
    experiments = relationship("Experiment", back_populates="hyperparameters")

@dataclass
class PerformanceMetrics(Base):
    """Performance metrics table"""
    __tablename__ = 'performance_metrics'
    
    performance_id: int = field(default=None)
    matthews_corrcoef: Optional[str] = field(default=None)
    roc_auc: Optional[str] = field(default=None)
    accuracy: Optional[str] = field(default=None)
    balanced_accuracy: Optional[str] = field(default=None)
    recall: Optional[str] = field(default=None)
    precision: Optional[str] = field(default=None)
    f1: Optional[str] = field(default=None)
    specificity: Optional[str] = field(default=None)
    average_precision: Optional[str] = field(default=None)
    
    # Columns
    performance_id = Column(Integer, primary_key=True, autoincrement=True)
    matthews_corrcoef = Column(Text)
    roc_auc = Column(Text)
    accuracy = Column(Text)
    balanced_accuracy = Column(Text)
    recall = Column(Text)
    precision = Column(Text)
    f1 = Column(Text)
    specificity = Column(Text)
    average_precision = Column(Text)
    
    # Relationships
    experiments = relationship("Experiment", back_populates="performance_metrics")

@dataclass
class SamplesClassificationRates(Base):
    """Sample classification rates table"""
    __tablename__ = 'samples_classification_rates'
    
    sample_rate_id: int = field(default=None)
    samples_classification_rates: str = field()
    
    # Columns
    sample_rate_id = Column(Integer, primary_key=True, autoincrement=True)
    samples_classification_rates = Column(Text)
    
    # Relationships
    experiments = relationship("Experiment", back_populates="sample_rates")

@dataclass
class ShapValues(Base):
    """SHAP values table"""
    __tablename__ = 'shap_values'
    
    shap_values_id: int = field(default=None)
    shap_values: str = field()
    
    # Columns
    shap_values_id = Column(Integer, primary_key=True, autoincrement=True)
    shap_values = Column(Text)
    
    # Relationships
    experiments = relationship("Experiment", back_populates="shap_values")

@dataclass
class Experiment(Base):
    """Experiment table (combines JobParameters and the former job_combinations)"""
    __tablename__ = 'experiments'
    
    experiment_id: int = field(default=None)
    n_trials: Optional[int] = field(default=None)
    rounds: Optional[int] = field(default=None)
    feature_selection_type: Optional[str] = field(default=None)
    feature_selection_method: Optional[str] = field(default=None)
    inner_scoring: Optional[str] = field(default=None)
    scoring: Optional[str] = field(default=None)
    inner_splits: Optional[int] = field(default=None)
    splits: Optional[int] = field(default=None)
    normalization: Optional[str] = field(default=None)
    missing_values_method: Optional[str] = field(default=None)
    class_balance: Optional[str] = field(default=None)
    evaluation_mthd: Optional[str] = field(default=None)
    param_grid: Optional[str] = field(default=None)
    features_names_list: Optional[str] = field(default=None)
    classifier_id: Optional[int] = field(default=None)
    dataset_id: Optional[int] = field(default=None)
    selection_id: Optional[int] = field(default=None)
    hyperparameter_id: Optional[int] = field(default=None)
    performance_id: Optional[int] = field(default=None)
    sample_rate_id: Optional[int] = field(default=None)
    shap_values_id: Optional[int] = field(default=None)
    model_selection_type: Optional[str] = field(default=None)
    search_type: Optional[str] = field(default=None)
    timestamp: Optional[datetime.date] = field(default=None)
    output_csv_path: Optional[str] = field(default=None)
    output_model_path: Optional[str] = field(default=None)
    output_metadata_path: Optional[str] = field(default=None)
    output_plot_path: Optional[str] = field(default=None)
    output_histogram_path: Optional[str] = field(default=None)
    output_parameters_path: Optional[str] = field(default=None)
    output_json_frfs_path: Optional[str] = field(default=None)
    combination_id: Optional[int] = field(default=None)
    # Columns
    experiment_id = Column(Integer, primary_key=True, autoincrement=True)
    n_trials = Column(Integer)
    rounds = Column(Integer)
    feature_selection_type = Column(String)
    feature_selection_method = Column(String)
    inner_scoring = Column(String)
    scoring = Column(String)
    inner_splits = Column(Integer)
    splits = Column(Integer)
    normalization = Column(String)
    missing_values_method = Column(String)
    class_balance = Column(String)
    evaluation_mthd = Column(String)
    param_grid = Column(Text)
    features_names_list = Column(Text)
    classifier_id = Column(Integer, ForeignKey('classifiers.classifier_id', ondelete='CASCADE'))
    dataset_id = Column(Integer, ForeignKey('datasets.dataset_id', ondelete='CASCADE'))
    selection_id = Column(Integer, ForeignKey('feature_selection.selection_id', ondelete='CASCADE'))
    hyperparameter_id = Column(Integer, ForeignKey('hyperparameters.hyperparameter_id', ondelete='CASCADE'))
    performance_id = Column(Integer, ForeignKey('performance_metrics.performance_id', ondelete='CASCADE'))
    sample_rate_id = Column(Integer, ForeignKey('samples_classification_rates.sample_rate_id', ondelete='CASCADE'))
    shap_values_id = Column(Integer, ForeignKey('shap_values.shap_values_id', ondelete='CASCADE'))
    model_selection_type = Column(String)
    search_type = Column(String)
    timestamp = Column(DateTime, default=datetime.date.today)
    output_csv_path = Column(String)
    output_model_path = Column(String)
    output_metadata_path = Column(String)
    output_plot_path = Column(String)
    output_histogram_path = Column(String)
    output_parameters_path = Column(String)
    output_json_frfs_path = Column(String)
    combination_id = Column(Integer)
    
    # Relationships
    classifier = relationship("Classifier", back_populates="experiments")
    dataset = relationship("Dataset", back_populates="experiments")
    feature_selection = relationship("FeatureSelection", back_populates="experiments")
    hyperparameters = relationship("Hyperparameters", back_populates="experiments")
    performance_metrics = relationship("PerformanceMetrics", back_populates="experiments")
    sample_rates = relationship("SamplesClassificationRates", back_populates="experiments")
    shap_values = relationship("ShapValues", back_populates="experiments")
    feature_counts = relationship("FeatureCount", back_populates="experiment")

@dataclass
class FeatureCount(Base):
    """Feature counts table"""
    __tablename__ = 'feature_counts'
    
    feature_count_id: int = field(default=None)
    feature_name: str = field()
    count: int = field()
    experiment_id: int = field(default=None)
    
    # Columns
    feature_count_id = Column(Integer, primary_key=True, autoincrement=True)
    feature_name = Column(String, nullable=False)
    count = Column(Integer, nullable=False)
    experiment_id = Column(Integer, ForeignKey('experiments.experiment_id', ondelete='CASCADE'))
    
    # Relationships
    experiment = relationship("Experiment", back_populates="feature_counts")