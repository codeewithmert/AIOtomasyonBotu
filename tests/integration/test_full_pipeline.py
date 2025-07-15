"""
Integration Tests for Full Pipeline

Integration tests for the complete AI automation bot pipeline including
data collection, processing, ML training, and automation.
"""

import unittest
import tempfile
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

from src.core.config import Config
from src.core.logger import get_logger
from src.data.collectors.api_collector import APICollector
from src.data.collectors.web_scraper import WebScraper
from src.data.processors.cleaner import DataCleaner
from src.ml.models.random_forest_model import RandomForestModel
from src.ml.pipeline.ml_pipeline import MLPipeline
from src.automation.scheduler import AutomationScheduler
from src.reporting.generators.html_generator import HTMLReportGenerator


logger = get_logger(__name__)


class TestFullPipeline(unittest.TestCase):
    """Integration tests for the complete AI automation bot pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Save test data
        self.test_data_path = os.path.join(self.temp_dir, 'test_data.csv')
        self.test_data.to_csv(self.test_data_path, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_data_collection_pipeline(self):
        """Test complete data collection pipeline."""
        try:
            # Test API collector
            api_collector = APICollector()
            
            # Mock API response
            with unittest.mock.patch('requests.get') as mock_get:
                mock_response = unittest.mock.Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    'data': [
                        {'id': 1, 'value': 10},
                        {'id': 2, 'value': 20}
                    ]
                }
                mock_response.raise_for_status.return_value = None
                mock_get.return_value = mock_response
                
                api_data = api_collector.get_data('https://api.test.com/data')
                self.assertIsInstance(api_data, dict)
                self.assertIn('data', api_data)
            
            # Test web scraper
            web_scraper = WebScraper()
            
            with unittest.mock.patch('requests.get') as mock_get:
                mock_response = unittest.mock.Mock()
                mock_response.status_code = 200
                mock_response.text = '<html><body><h1>Test Title</h1><p>Test content</p></body></html>'
                mock_response.raise_for_status.return_value = None
                mock_get.return_value = mock_response
                
                web_data = web_scraper.scrape_page('https://test.com')
                self.assertIsInstance(web_data, dict)
                self.assertIn('title', web_data)
            
            logger.info("Data collection pipeline test completed successfully")
            
        except Exception as e:
            logger.error(f"Data collection pipeline test failed: {e}")
            raise
    
    def test_data_processing_pipeline(self):
        """Test complete data processing pipeline."""
        try:
            # Create test data with issues
            dirty_data = pd.DataFrame({
                'feature1': [1, 2, np.nan, 4, 5],
                'feature2': ['a', 'b', 'c', 'd', 'e'],
                'feature3': [1.1, 2.2, 3.3, 4.4, 5.5],
                'target': [0, 1, 0, 1, 0]
            })
            
            # Clean data
            cleaner = DataCleaner()
            cleaning_config = {
                'remove_duplicates': True,
                'handle_missing': True,
                'clean_text': True,
                'standardize_types': True
            }
            
            cleaned_data = cleaner.clean_dataframe(dirty_data, cleaning_config)
            
            # Verify cleaning results
            self.assertIsInstance(cleaned_data, pd.DataFrame)
            self.assertLess(len(cleaned_data), len(dirty_data))  # Should remove some rows
            
            # Get cleaning report
            report = cleaner.get_cleaning_report()
            self.assertIsInstance(report, dict)
            self.assertIn('cleaning_stats', report)
            
            logger.info("Data processing pipeline test completed successfully")
            
        except Exception as e:
            logger.error(f"Data processing pipeline test failed: {e}")
            raise
    
    def test_ml_training_pipeline(self):
        """Test complete ML training pipeline."""
        try:
            # Create ML pipeline
            pipeline_config = {
                'test_size': 0.2,
                'random_state': 42,
                'models_dir': self.temp_dir
            }
            
            pipeline = MLPipeline(pipeline_config)
            
            # Run pipeline
            results = pipeline.run_pipeline(
                data=self.test_data,
                target_column='target',
                task_type='classification',
                model_type='random_forest'
            )
            
            # Verify results
            self.assertIsInstance(results, dict)
            self.assertIn('task_type', results)
            self.assertIn('model_type', results)
            self.assertIn('evaluation', results)
            self.assertIn('model_path', results)
            
            # Check evaluation metrics
            evaluation = results['evaluation']
            self.assertIn('accuracy', evaluation)
            self.assertGreaterEqual(evaluation['accuracy'], 0.0)
            self.assertLessEqual(evaluation['accuracy'], 1.0)
            
            # Check model file exists
            self.assertTrue(os.path.exists(results['model_path']))
            
            logger.info("ML training pipeline test completed successfully")
            
        except Exception as e:
            logger.error(f"ML training pipeline test failed: {e}")
            raise
    
    def test_model_prediction_pipeline(self):
        """Test model prediction pipeline."""
        try:
            # Train a model first
            model = RandomForestModel()
            model.configure_classification(n_estimators=10, random_state=42)
            
            X = self.test_data.drop('target', axis=1)
            y = self.test_data['target']
            
            model.train(X, y)
            
            # Save model
            model_path = os.path.join(self.temp_dir, 'test_model.pkl')
            model.save_model(model_path)
            
            # Load model and make predictions
            loaded_model = RandomForestModel()
            loaded_model.load_model(model_path)
            
            # Create test data for prediction
            test_X = X.head(5)
            predictions = loaded_model.predict(test_X)
            
            # Verify predictions
            self.assertIsInstance(predictions, np.ndarray)
            self.assertEqual(len(predictions), len(test_X))
            self.assertTrue(all(pred in [0, 1] for pred in predictions))
            
            logger.info("Model prediction pipeline test completed successfully")
            
        except Exception as e:
            logger.error(f"Model prediction pipeline test failed: {e}")
            raise
    
    def test_automation_scheduler_pipeline(self):
        """Test automation scheduler pipeline."""
        try:
            # Create scheduler
            scheduler = AutomationScheduler()
            
            # Define a simple task
            def test_task():
                return {"status": "completed", "timestamp": datetime.now().isoformat()}
            
            # Schedule task
            task_id = scheduler.schedule_task(
                func=test_task,
                trigger='interval',
                seconds=5,
                max_instances=1
            )
            
            # Verify task is scheduled
            self.assertIsNotNone(task_id)
            self.assertTrue(scheduler.is_task_scheduled(task_id))
            
            # Get task info
            task_info = scheduler.get_task_info(task_id)
            self.assertIsInstance(task_info, dict)
            self.assertIn('id', task_info)
            self.assertIn('next_run_time', task_info)
            
            # Remove task
            scheduler.remove_task(task_id)
            self.assertFalse(scheduler.is_task_scheduled(task_id))
            
            logger.info("Automation scheduler pipeline test completed successfully")
            
        except Exception as e:
            logger.error(f"Automation scheduler pipeline test failed: {e}")
            raise
    
    def test_reporting_pipeline(self):
        """Test reporting pipeline."""
        try:
            # Create report data
            report_data = {
                'report_type': 'ml_evaluation',
                'metrics': {
                    'accuracy': 0.85,
                    'precision': 0.82,
                    'recall': 0.88,
                    'f1_score': 0.85
                },
                'model_info': {
                    'model_type': 'random_forest',
                    'task_type': 'classification',
                    'created_at': datetime.now().isoformat(),
                    'n_features': 3,
                    'n_samples': 100
                },
                'feature_importance': {
                    'top_features': [
                        {'feature': 'feature1', 'importance': 0.4},
                        {'feature': 'feature2', 'importance': 0.35},
                        {'feature': 'feature3', 'importance': 0.25}
                    ]
                }
            }
            
            # Generate HTML report
            html_generator = HTMLReportGenerator()
            html_content = html_generator.generate_report(report_data)
            
            # Verify HTML content
            self.assertIsInstance(html_content, str)
            self.assertIn('<!DOCTYPE html>', html_content)
            self.assertIn('ML Model Evaluation Report', html_content)
            self.assertIn('0.85', html_content)  # Should contain accuracy
            
            # Save report
            report_path = os.path.join(self.temp_dir, 'test_report.html')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Verify file exists
            self.assertTrue(os.path.exists(report_path))
            
            logger.info("Reporting pipeline test completed successfully")
            
        except Exception as e:
            logger.error(f"Reporting pipeline test failed: {e}")
            raise
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        try:
            # 1. Data Collection
            api_collector = APICollector()
            with unittest.mock.patch('requests.get') as mock_get:
                mock_response = unittest.mock.Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    'data': self.test_data.to_dict('records')
                }
                mock_response.raise_for_status.return_value = None
                mock_get.return_value = mock_response
                
                collected_data = api_collector.get_data('https://api.test.com/data')
            
            # 2. Data Processing
            cleaner = DataCleaner()
            df = pd.DataFrame(collected_data['data'])
            cleaned_data = cleaner.clean_dataframe(df, {
                'remove_duplicates': True,
                'handle_missing': True
            })
            
            # 3. ML Training
            pipeline = MLPipeline({'models_dir': self.temp_dir})
            results = pipeline.run_pipeline(
                data=cleaned_data,
                target_column='target',
                task_type='classification',
                model_type='random_forest'
            )
            
            # 4. Model Prediction
            model = RandomForestModel()
            model.load_model(results['model_path'])
            test_data = cleaned_data.head(10).drop('target', axis=1)
            predictions = model.predict(test_data)
            
            # 5. Report Generation
            report_data = {
                'report_type': 'ml_evaluation',
                'metrics': results['evaluation'],
                'model_info': {
                    'model_type': 'random_forest',
                    'task_type': 'classification',
                    'created_at': datetime.now().isoformat()
                }
            }
            
            html_generator = HTMLReportGenerator()
            html_content = html_generator.generate_report(report_data)
            
            # 6. Automation Scheduling
            scheduler = AutomationScheduler()
            
            def automated_task():
                return {
                    'status': 'completed',
                    'predictions_count': len(predictions),
                    'timestamp': datetime.now().isoformat()
                }
            
            task_id = scheduler.schedule_task(
                func=automated_task,
                trigger='interval',
                seconds=10
            )
            
            # Verify complete pipeline
            self.assertIsInstance(collected_data, dict)
            self.assertIsInstance(cleaned_data, pd.DataFrame)
            self.assertIsInstance(results, dict)
            self.assertIsInstance(predictions, np.ndarray)
            self.assertIsInstance(html_content, str)
            self.assertIsNotNone(task_id)
            
            # Clean up
            scheduler.remove_task(task_id)
            
            logger.info("End-to-end pipeline test completed successfully")
            
        except Exception as e:
            logger.error(f"End-to-end pipeline test failed: {e}")
            raise
    
    def test_error_handling_pipeline(self):
        """Test error handling throughout the pipeline."""
        try:
            # Test API collector with invalid URL
            api_collector = APICollector()
            with self.assertRaises(Exception):
                api_collector.get_data('invalid-url')
            
            # Test web scraper with invalid URL
            web_scraper = WebScraper()
            with self.assertRaises(Exception):
                web_scraper.scrape_page('invalid-url')
            
            # Test ML pipeline with invalid data
            pipeline = MLPipeline()
            with self.assertRaises(Exception):
                pipeline.run_pipeline(
                    data=pd.DataFrame(),  # Empty dataframe
                    target_column='nonexistent',
                    task_type='classification',
                    model_type='random_forest'
                )
            
            # Test model with invalid configuration
            model = RandomForestModel()
            with self.assertRaises(Exception):
                model.train(None, None)
            
            logger.info("Error handling pipeline test completed successfully")
            
        except Exception as e:
            logger.error(f"Error handling pipeline test failed: {e}")
            raise
    
    def test_performance_pipeline(self):
        """Test pipeline performance with larger datasets."""
        try:
            # Create larger test dataset
            large_data = pd.DataFrame({
                'feature1': np.random.randn(1000),
                'feature2': np.random.randn(1000),
                'feature3': np.random.randn(1000),
                'feature4': np.random.randn(1000),
                'feature5': np.random.randn(1000),
                'target': np.random.randint(0, 2, 1000)
            })
            
            # Test data processing performance
            cleaner = DataCleaner()
            start_time = datetime.now()
            cleaned_data = cleaner.clean_dataframe(large_data, {
                'remove_duplicates': True,
                'handle_missing': True
            })
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Test ML training performance
            pipeline = MLPipeline({'models_dir': self.temp_dir})
            start_time = datetime.now()
            results = pipeline.run_pipeline(
                data=cleaned_data,
                target_column='target',
                task_type='classification',
                model_type='random_forest'
            )
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Verify performance
            self.assertLess(processing_time, 10.0)  # Should process in under 10 seconds
            self.assertLess(training_time, 30.0)    # Should train in under 30 seconds
            
            # Verify results quality
            self.assertGreater(results['evaluation']['accuracy'], 0.5)  # Should perform reasonably well
            
            logger.info(f"Performance pipeline test completed: processing={processing_time:.2f}s, training={training_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Performance pipeline test failed: {e}")
            raise


if __name__ == '__main__':
    unittest.main() 