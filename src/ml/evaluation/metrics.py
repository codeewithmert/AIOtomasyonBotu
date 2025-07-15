"""
Metrics Calculator

Provides comprehensive metrics calculation for machine learning model evaluation
in the AI Automation Bot.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

from ...core.exceptions import MLModelError
from ...core.logger import get_logger

logger = get_logger(__name__)


class MetricsCalculator:
    """Calculator for machine learning performance metrics."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.metrics_history = []
    
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary of classification metrics
        """
        try:
            metrics = {}
            
            # Basic classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Per-class metrics
            unique_classes = np.unique(y_true)
            if len(unique_classes) > 2:
                metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
                metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
                metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
                
                metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
                metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
                metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
            
            # ROC AUC (if probabilities provided)
            if y_prob is not None:
                try:
                    if len(unique_classes) == 2:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                    else:
                        # Multi-class ROC AUC
                        y_true_bin = label_binarize(y_true, classes=unique_classes)
                        metrics['roc_auc'] = roc_auc_score(y_true_bin, y_prob, average='weighted', multi_class='ovr')
                except Exception as e:
                    logger.warning(f"ROC AUC calculation failed: {e}")
                    metrics['roc_auc'] = None
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Classification report
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics['classification_report'] = report
            
            # Additional metrics
            metrics['total_samples'] = len(y_true)
            metrics['unique_classes'] = len(unique_classes)
            metrics['class_distribution'] = self._calculate_class_distribution(y_true)
            
            logger.info(f"Classification metrics calculated: accuracy = {metrics['accuracy']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Classification metrics calculation error: {e}")
            raise MLModelError(f"Failed to calculate classification metrics: {str(e)}")
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of regression metrics
        """
        try:
            metrics = {}
            
            # Basic regression metrics
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2_score'] = r2_score(y_true, y_pred)
            metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
            metrics['max_error'] = max_error(y_true, y_pred)
            
            # Additional metrics
            metrics['mean_absolute_percentage_error'] = self._calculate_mape(y_true, y_pred)
            metrics['symmetric_mean_absolute_percentage_error'] = self._calculate_smape(y_true, y_pred)
            
            # Residuals analysis
            residuals = y_true - y_pred
            metrics['residuals_mean'] = np.mean(residuals)
            metrics['residuals_std'] = np.std(residuals)
            metrics['residuals_skewness'] = self._calculate_skewness(residuals)
            metrics['residuals_kurtosis'] = self._calculate_kurtosis(residuals)
            
            # Data statistics
            metrics['total_samples'] = len(y_true)
            metrics['y_true_mean'] = np.mean(y_true)
            metrics['y_true_std'] = np.std(y_true)
            metrics['y_pred_mean'] = np.mean(y_pred)
            metrics['y_pred_std'] = np.std(y_pred)
            
            logger.info(f"Regression metrics calculated: R² = {metrics['r2_score']:.4f}, RMSE = {metrics['rmse']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Regression metrics calculation error: {e}")
            raise MLModelError(f"Failed to calculate regression metrics: {str(e)}")
    
    def calculate_feature_importance_metrics(self, feature_importance: np.ndarray, 
                                           feature_names: List[str]) -> Dict[str, Any]:
        """
        Calculate feature importance metrics.
        
        Args:
            feature_importance: Feature importance scores
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance metrics
        """
        try:
            metrics = {}
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            metrics['feature_importance'] = importance_df.to_dict('records')
            metrics['top_features'] = importance_df.head(10).to_dict('records')
            metrics['total_features'] = len(feature_names)
            
            # Calculate importance statistics
            metrics['importance_mean'] = np.mean(feature_importance)
            metrics['importance_std'] = np.std(feature_importance)
            metrics['importance_max'] = np.max(feature_importance)
            metrics['importance_min'] = np.min(feature_importance)
            
            # Calculate cumulative importance
            cumulative_importance = np.cumsum(importance_df['importance'])
            metrics['cumulative_importance'] = cumulative_importance.tolist()
            
            # Find number of features for different importance thresholds
            thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
            for threshold in thresholds:
                n_features = np.sum(cumulative_importance <= threshold)
                metrics[f'features_for_{int(threshold*100)}%_importance'] = n_features
            
            logger.info(f"Feature importance metrics calculated: {len(feature_names)} features")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Feature importance metrics calculation error: {e}")
            raise MLModelError(f"Failed to calculate feature importance metrics: {str(e)}")
    
    def calculate_model_complexity_metrics(self, model, X_train: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate model complexity metrics.
        
        Args:
            model: Trained model
            X_train: Training data
            
        Returns:
            Dictionary of model complexity metrics
        """
        try:
            metrics = {}
            
            # Model parameters
            if hasattr(model, 'n_estimators'):
                metrics['n_estimators'] = model.n_estimators
            if hasattr(model, 'max_depth'):
                metrics['max_depth'] = model.max_depth
            if hasattr(model, 'n_features_in_'):
                metrics['n_features_in'] = model.n_features_in_
            
            # Training data complexity
            metrics['n_samples'] = X_train.shape[0]
            metrics['n_features'] = X_train.shape[1]
            metrics['sparsity'] = self._calculate_sparsity(X_train)
            
            # Model size estimation
            try:
                import sys
                model_size = sys.getsizeof(model)
                metrics['model_size_bytes'] = model_size
                metrics['model_size_mb'] = model_size / (1024 * 1024)
            except Exception:
                metrics['model_size_bytes'] = None
                metrics['model_size_mb'] = None
            
            logger.info(f"Model complexity metrics calculated")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model complexity metrics calculation error: {e}")
            raise MLModelError(f"Failed to calculate model complexity metrics: {str(e)}")
    
    def _calculate_class_distribution(self, y: np.ndarray) -> Dict[str, int]:
        """Calculate class distribution."""
        try:
            unique, counts = np.unique(y, return_counts=True)
            return dict(zip(unique.astype(str), counts.tolist()))
        except Exception:
            return {}
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        try:
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        except Exception:
            return np.nan
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        try:
            return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
        except Exception:
            return np.nan
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        try:
            mean = np.mean(data)
            std = np.std(data)
            return np.mean(((data - mean) / std) ** 3)
        except Exception:
            return np.nan
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        try:
            mean = np.mean(data)
            std = np.std(data)
            return np.mean(((data - mean) / std) ** 4) - 3
        except Exception:
            return np.nan
    
    def _calculate_sparsity(self, X: pd.DataFrame) -> float:
        """Calculate sparsity of data."""
        try:
            return 1.0 - (np.count_nonzero(X) / X.size)
        except Exception:
            return 0.0
    
    def create_metrics_report(self, metrics: Dict[str, Any], task_type: str) -> str:
        """
        Create a formatted metrics report.
        
        Args:
            metrics: Calculated metrics
            task_type: Type of task ('classification' or 'regression')
            
        Returns:
            Formatted report string
        """
        try:
            report_lines = []
            report_lines.append("=" * 50)
            report_lines.append(f"MODEL EVALUATION REPORT - {task_type.upper()}")
            report_lines.append("=" * 50)
            report_lines.append("")
            
            if task_type == 'classification':
                report_lines.append("CLASSIFICATION METRICS:")
                report_lines.append(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
                report_lines.append(f"  Precision: {metrics.get('precision', 0):.4f}")
                report_lines.append(f"  Recall: {metrics.get('recall', 0):.4f}")
                report_lines.append(f"  F1 Score: {metrics.get('f1_score', 0):.4f}")
                
                if metrics.get('roc_auc') is not None:
                    report_lines.append(f"  ROC AUC: {metrics.get('roc_auc', 0):.4f}")
                
                report_lines.append("")
                report_lines.append(f"Total Samples: {metrics.get('total_samples', 0)}")
                report_lines.append(f"Unique Classes: {metrics.get('unique_classes', 0)}")
                
            elif task_type == 'regression':
                report_lines.append("REGRESSION METRICS:")
                report_lines.append(f"  R² Score: {metrics.get('r2_score', 0):.4f}")
                report_lines.append(f"  RMSE: {metrics.get('rmse', 0):.4f}")
                report_lines.append(f"  MAE: {metrics.get('mae', 0):.4f}")
                report_lines.append(f"  Explained Variance: {metrics.get('explained_variance', 0):.4f}")
                report_lines.append(f"  Max Error: {metrics.get('max_error', 0):.4f}")
                
                report_lines.append("")
                report_lines.append(f"Total Samples: {metrics.get('total_samples', 0)}")
                report_lines.append(f"Target Mean: {metrics.get('y_true_mean', 0):.4f}")
                report_lines.append(f"Target Std: {metrics.get('y_true_std', 0):.4f}")
            
            report_lines.append("")
            report_lines.append("=" * 50)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Metrics report creation error: {e}")
            return f"Error creating metrics report: {str(e)}"
    
    def save_metrics(self, metrics: Dict[str, Any], file_path: str):
        """
        Save metrics to file.
        
        Args:
            metrics: Metrics to save
            file_path: Path to save file
        """
        try:
            import json
            
            with open(file_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            logger.info(f"Metrics saved to: {file_path}")
            
        except Exception as e:
            logger.error(f"Metrics saving error: {e}")
            raise MLModelError(f"Failed to save metrics: {str(e)}")
    
    def load_metrics(self, file_path: str) -> Dict[str, Any]:
        """
        Load metrics from file.
        
        Args:
            file_path: Path to metrics file
            
        Returns:
            Loaded metrics
        """
        try:
            import json
            
            with open(file_path, 'r') as f:
                metrics = json.load(f)
            
            logger.info(f"Metrics loaded from: {file_path}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics loading error: {e}")
            raise MLModelError(f"Failed to load metrics: {str(e)}") 