"""
Unit Tests for ML Models

Tests for machine learning models including RandomForest and base model functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from src.ml.models.base_model import BaseModel
from src.ml.models.random_forest_model import RandomForestModel
from src.core.exceptions import MLModelError


class TestBaseModel(unittest.TestCase):
    """Test cases for BaseModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = BaseModel()
        self.X, self.y = make_classification(n_samples=100, n_features=10, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
    
    def test_init(self):
        """Test BaseModel initialization."""
        self.assertIsNotNone(self.model)
        self.assertIsNone(self.model.model)
        self.assertFalse(self.model.is_trained)
    
    def test_train_not_implemented(self):
        """Test that train method raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.model.train(self.X_train, self.y_train)
    
    def test_predict_not_implemented(self):
        """Test that predict method raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.model.predict(self.X_test)
    
    def test_evaluate_not_implemented(self):
        """Test that evaluate method raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.model.evaluate(self.X_test, self.y_test)
    
    def test_save_model_not_implemented(self):
        """Test that save_model method raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.model.save_model("test.pkl")
    
    def test_load_model_not_implemented(self):
        """Test that load_model method raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.model.load_model("test.pkl")
    
    def test_get_feature_importance_not_implemented(self):
        """Test that get_feature_importance method raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.model.get_feature_importance()
    
    def test_get_sklearn_model_not_implemented(self):
        """Test that get_sklearn_model method raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.model.get_sklearn_model()


class TestRandomForestModel(unittest.TestCase):
    """Test cases for RandomForestModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = RandomForestModel()
        
        # Create classification dataset
        self.X_clf, self.y_clf = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        self.X_clf_train, self.X_clf_test, self.y_clf_train, self.y_clf_test = train_test_split(
            self.X_clf, self.y_clf, test_size=0.2, random_state=42
        )
        
        # Create regression dataset
        self.X_reg, self.y_reg = make_regression(
            n_samples=100, n_features=10, random_state=42
        )
        self.X_reg_train, self.X_reg_test, self.y_reg_train, self.y_reg_test = train_test_split(
            self.X_reg, self.y_reg, test_size=0.2, random_state=42
        )
    
    def test_init(self):
        """Test RandomForestModel initialization."""
        self.assertIsNotNone(self.model)
        self.assertIsNone(self.model.model)
        self.assertFalse(self.model.is_trained)
        self.assertIsNone(self.model.task_type)
    
    def test_configure_classification(self):
        """Test classification configuration."""
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        
        self.model.configure_classification(**params)
        
        self.assertEqual(self.model.task_type, 'classification')
        self.assertIsNotNone(self.model.model)
        self.assertEqual(self.model.model.n_estimators, 100)
        self.assertEqual(self.model.model.max_depth, 10)
    
    def test_configure_regression(self):
        """Test regression configuration."""
        params = {
            'n_estimators': 50,
            'max_depth': 5,
            'random_state': 42
        }
        
        self.model.configure_regression(**params)
        
        self.assertEqual(self.model.task_type, 'regression')
        self.assertIsNotNone(self.model.model)
        self.assertEqual(self.model.model.n_estimators, 50)
        self.assertEqual(self.model.model.max_depth, 5)
    
    def test_train_classification(self):
        """Test classification model training."""
        self.model.configure_classification()
        self.model.train(self.X_clf_train, self.y_clf_train)
        
        self.assertTrue(self.model.is_trained)
        self.assertIsNotNone(self.model.model)
    
    def test_train_regression(self):
        """Test regression model training."""
        self.model.configure_regression()
        self.model.train(self.X_reg_train, self.y_reg_train)
        
        self.assertTrue(self.model.is_trained)
        self.assertIsNotNone(self.model.model)
    
    def test_train_without_configuration(self):
        """Test training without configuration raises error."""
        with self.assertRaises(MLModelError):
            self.model.train(self.X_clf_train, self.y_clf_train)
    
    def test_predict_classification(self):
        """Test classification prediction."""
        self.model.configure_classification()
        self.model.train(self.X_clf_train, self.y_clf_train)
        
        predictions = self.model.predict(self.X_clf_test)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.y_clf_test))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_predict_regression(self):
        """Test regression prediction."""
        self.model.configure_regression()
        self.model.train(self.X_reg_train, self.y_reg_train)
        
        predictions = self.model.predict(self.X_reg_test)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.y_reg_test))
        self.assertTrue(all(isinstance(pred, (int, float)) for pred in predictions))
    
    def test_predict_without_training(self):
        """Test prediction without training raises error."""
        self.model.configure_classification()
        
        with self.assertRaises(MLModelError):
            self.model.predict(self.X_clf_test)
    
    def test_evaluate_classification(self):
        """Test classification model evaluation."""
        self.model.configure_classification()
        self.model.train(self.X_clf_train, self.y_clf_train)
        
        metrics = self.model.evaluate(self.X_clf_test, self.y_clf_test)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        
        # Check that accuracy is reasonable
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
    
    def test_evaluate_regression(self):
        """Test regression model evaluation."""
        self.model.configure_regression()
        self.model.train(self.X_reg_train, self.y_reg_train)
        
        metrics = self.model.evaluate(self.X_reg_test, self.y_reg_test)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2_score', metrics)
        
        # Check that R² score is reasonable
        self.assertGreaterEqual(metrics['r2_score'], -1.0)
        self.assertLessEqual(metrics['r2_score'], 1.0)
    
    def test_evaluate_with_custom_metrics(self):
        """Test evaluation with custom metrics."""
        self.model.configure_classification()
        self.model.train(self.X_clf_train, self.y_clf_train)
        
        custom_metrics = ['accuracy', 'precision']
        metrics = self.model.evaluate(self.X_clf_test, self.y_clf_test, metrics=custom_metrics)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertNotIn('recall', metrics)
    
    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        self.model.configure_classification()
        self.model.train(self.X_clf_train, self.y_clf_train)
        
        importance = self.model.get_feature_importance()
        
        self.assertIsInstance(importance, np.ndarray)
        self.assertEqual(len(importance), self.X_clf_train.shape[1])
        self.assertTrue(all(imp >= 0 for imp in importance))
    
    def test_get_feature_importance_without_training(self):
        """Test feature importance without training raises error."""
        self.model.configure_classification()
        
        with self.assertRaises(MLModelError):
            self.model.get_feature_importance()
    
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        import tempfile
        import os
        
        self.model.configure_classification()
        self.model.train(self.X_clf_train, self.y_clf_train)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            self.model.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Load model
            loaded_model = RandomForestModel()
            loaded_model.load_model(model_path)
            
            # Test that loaded model works
            original_predictions = self.model.predict(self.X_clf_test)
            loaded_predictions = loaded_model.predict(self.X_clf_test)
            
            np.testing.assert_array_equal(original_predictions, loaded_predictions)
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_save_model_without_training(self):
        """Test saving model without training raises error."""
        self.model.configure_classification()
        
        with self.assertRaises(MLModelError):
            self.model.save_model("test.pkl")
    
    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning."""
        self.model.configure_classification()
        
        param_grid = {
            'n_estimators': [10, 50],
            'max_depth': [3, 5]
        }
        
        best_params, best_score = self.model.tune_hyperparameters(
            self.X_clf_train, self.y_clf_train, param_grid
        )
        
        self.assertIsInstance(best_params, dict)
        self.assertIsInstance(best_score, float)
        self.assertIn('n_estimators', best_params)
        self.assertIn('max_depth', best_params)
    
    def test_cross_validation(self):
        """Test cross-validation."""
        self.model.configure_classification()
        
        cv_scores = self.model.cross_validate(self.X_clf_train, self.y_clf_train, cv=3)
        
        self.assertIsInstance(cv_scores, list)
        self.assertEqual(len(cv_scores), 3)
        self.assertTrue(all(isinstance(score, float) for score in cv_scores))
    
    def test_feature_selection(self):
        """Test feature selection."""
        self.model.configure_classification()
        self.model.train(self.X_clf_train, self.y_clf_train)
        
        # Select top 5 features
        selected_features = self.model.select_features(self.X_clf_train, k=5)
        
        self.assertIsInstance(selected_features, np.ndarray)
        self.assertEqual(len(selected_features), 5)
        self.assertTrue(all(idx < self.X_clf_train.shape[1] for idx in selected_features))
    
    def test_model_visualization(self):
        """Test model visualization methods."""
        self.model.configure_classification()
        self.model.train(self.X_clf_train, self.y_clf_train)
        
        # Test feature importance plot
        with patch('matplotlib.pyplot.show'):
            self.model.plot_feature_importance()
        
        # Test learning curve
        with patch('matplotlib.pyplot.show'):
            self.model.plot_learning_curve(self.X_clf_train, self.y_clf_train)
    
    def test_model_performance_comparison(self):
        """Test model performance comparison."""
        self.model.configure_classification()
        self.model.train(self.X_clf_train, self.y_clf_train)
        
        # Create another model for comparison
        other_model = RandomForestModel()
        other_model.configure_classification(n_estimators=200)
        other_model.train(self.X_clf_train, self.y_clf_train)
        
        comparison = self.model.compare_performance(
            other_model, self.X_clf_test, self.y_clf_test
        )
        
        self.assertIsInstance(comparison, dict)
        self.assertIn('model1_metrics', comparison)
        self.assertIn('model2_metrics', comparison)
        self.assertIn('improvement', comparison)


class TestMLModelsIntegration(unittest.TestCase):
    """Integration tests for ML models."""
    
    def test_end_to_end_classification_pipeline(self):
        """Test end-to-end classification pipeline."""
        model = RandomForestModel()
        
        # Create dataset
        X, y = make_classification(n_samples=200, n_features=15, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Configure and train
        model.configure_classification(n_estimators=100, random_state=42)
        model.train(X_train, y_train)
        
        # Predict and evaluate
        predictions = model.predict(X_test)
        metrics = model.evaluate(X_test, y_test)
        
        # Verify results
        self.assertTrue(model.is_trained)
        self.assertGreater(metrics['accuracy'], 0.7)  # Should perform reasonably well
        
        # Test feature importance
        importance = model.get_feature_importance()
        self.assertEqual(len(importance), X_train.shape[1])
    
    def test_end_to_end_regression_pipeline(self):
        """Test end-to-end regression pipeline."""
        model = RandomForestModel()
        
        # Create dataset
        X, y = make_regression(n_samples=200, n_features=15, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Configure and train
        model.configure_regression(n_estimators=100, random_state=42)
        model.train(X_train, y_train)
        
        # Predict and evaluate
        predictions = model.predict(X_test)
        metrics = model.evaluate(X_test, y_test)
        
        # Verify results
        self.assertTrue(model.is_trained)
        self.assertGreater(metrics['r2_score'], 0.5)  # Should perform reasonably well
        
        # Test feature importance
        importance = model.get_feature_importance()
        self.assertEqual(len(importance), X_train.shape[1])


if __name__ == '__main__':
    unittest.main() 