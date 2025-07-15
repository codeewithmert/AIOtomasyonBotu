"""
Base Model Class
Tüm ML modelleri için temel sınıf
"""

import pickle
import json
import joblib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from ...core.logger import get_logger, performance_context
from ...core.exceptions import ModelTrainingError, ModelPredictionError, ModelEvaluationError
from ...core.utils import write_file, read_file


class BaseModel(ABC):
    """
    Temel model sınıfı
    Tüm ML modelleri için ortak arayüz
    """
    
    def __init__(self, name: str = None, model_type: str = None, **kwargs):
        self.name = name or self.__class__.__name__
        self.model_type = model_type or "base"
        self.logger = get_logger(f"{__name__}.{self.name}")
        
        # Model durumu
        self.is_trained = False
        self.training_history = []
        self.model_params = kwargs
        self.feature_names = None
        self.target_name = None
        
        # Performans metrikleri
        self.metrics = {}
        self.feature_importance = None
        
        # Model dosyası
        self.model_path = None
        self.created_at = datetime.now()
        self.updated_at = None
        
        self.logger.info(f"Model oluşturuldu: {self.name}")
    
    @abstractmethod
    def _create_model(self, **kwargs) -> Any:
        """
        Model nesnesini oluştur
        Alt sınıflar tarafından implement edilmeli
        """
        pass
    
    @abstractmethod
    def _train_model(self, X: Union[np.ndarray, pd.DataFrame], 
                    y: Union[np.ndarray, pd.Series], **kwargs) -> Dict[str, Any]:
        """
        Modeli eğit
        Alt sınıflar tarafından implement edilmeli
        """
        pass
    
    @abstractmethod
    def _predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Tahmin yap
        Alt sınıflar tarafından implement edilmeli
        """
        pass
    
    def train(self, X: Union[np.ndarray, pd.DataFrame], 
              y: Union[np.ndarray, pd.Series], 
              validation_data: Optional[Tuple] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Modeli eğit
        
        Args:
            X: Eğitim verisi
            y: Hedef değişken
            validation_data: Doğrulama verisi (X_val, y_val)
            **kwargs: Ek parametreler
            
        Returns:
            Eğitim sonuçları
        """
        try:
            self.logger.info(f"Model eğitimi başlatılıyor: {self.name}")
            
            # Veri doğrulama
            self._validate_training_data(X, y)
            
            # Feature names'i kaydet
            if isinstance(X, pd.DataFrame):
                self.feature_names = list(X.columns)
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            if isinstance(y, pd.Series):
                self.target_name = y.name
            else:
                self.target_name = "target"
            
            # Model oluştur
            self.model = self._create_model(**self.model_params)
            
            # Eğitim
            with performance_context(self.logger.name, f"model_training_{self.name}"):
                training_results = self._train_model(X, y, **kwargs)
            
            # Model durumunu güncelle
            self.is_trained = True
            self.updated_at = datetime.now()
            self.training_history.append({
                'timestamp': self.updated_at.isoformat(),
                'data_shape': X.shape,
                'results': training_results
            })
            
            # Validation metrikleri
            if validation_data:
                X_val, y_val = validation_data
                val_metrics = self.evaluate(X_val, y_val)
                training_results['validation_metrics'] = val_metrics
            
            self.logger.info(f"Model eğitimi tamamlandı: {self.name}")
            return training_results
            
        except Exception as e:
            raise ModelTrainingError(
                f"Model eğitimi hatası: {e}",
                model_name=self.name,
                training_data_size=X.shape[0] if hasattr(X, 'shape') else len(X)
            )
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Tahmin yap
        
        Args:
            X: Tahmin verisi
            
        Returns:
            Tahminler
        """
        if not self.is_trained:
            raise ModelPredictionError(
                "Model henüz eğitilmemiş",
                model_name=self.name
            )
        
        try:
            # Veri doğrulama
            self._validate_prediction_data(X)
            
            with performance_context(self.logger.name, f"model_prediction_{self.name}"):
                predictions = self._predict(X)
            
            self.logger.debug(f"Tahmin tamamlandı: {self.name} - {len(predictions)} örnek")
            return predictions
            
        except Exception as e:
            raise ModelPredictionError(
                f"Tahmin hatası: {e}",
                model_name=self.name,
                input_data_shape=X.shape if hasattr(X, 'shape') else None
            )
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Olasılık tahminleri yap (sınıflandırma için)
        
        Args:
            X: Tahmin verisi
            
        Returns:
            Olasılık tahminleri
        """
        if not self.is_trained:
            raise ModelPredictionError(
                "Model henüz eğitilmemiş",
                model_name=self.name
            )
        
        try:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                raise ModelPredictionError(
                    "Bu model olasılık tahmini desteklemiyor",
                    model_name=self.name
                )
        except Exception as e:
            raise ModelPredictionError(
                f"Olasılık tahmin hatası: {e}",
                model_name=self.name
            )
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Modeli değerlendir
        
        Args:
            X: Test verisi
            y: Gerçek değerler
            
        Returns:
            Değerlendirme metrikleri
        """
        if not self.is_trained:
            raise ModelEvaluationError(
                "Model henüz eğitilmemiş",
                model_name=self.name
            )
        
        try:
            predictions = self.predict(X)
            metrics = self._calculate_metrics(y, predictions)
            
            # Metrikleri kaydet
            self.metrics.update(metrics)
            
            self.logger.info(f"Model değerlendirildi: {self.name} - {metrics}")
            return metrics
            
        except Exception as e:
            raise ModelEvaluationError(
                f"Model değerlendirme hatası: {e}",
                model_name=self.name
            )
    
    def _calculate_metrics(self, y_true: Union[np.ndarray, pd.Series], 
                          y_pred: np.ndarray) -> Dict[str, float]:
        """
        Metrikleri hesapla
        Alt sınıflar tarafından override edilebilir
        """
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score,
            accuracy_score, precision_score, recall_score, f1_score,
            classification_report, confusion_matrix
        )
        
        metrics = {}
        
        # Regresyon metrikleri
        if len(np.unique(y_true)) > 10:  # Regresyon varsayımı
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
        # Sınıflandırma metrikleri
        else:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            # Binary sınıflandırma için ek metrikler
            if len(np.unique(y_true)) == 2:
                metrics['precision'] = precision_score(y_true, y_pred, average='binary')
                metrics['recall'] = recall_score(y_true, y_pred, average='binary')
                metrics['f1'] = f1_score(y_true, y_pred, average='binary')
            else:
                metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
                metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
                metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        
        return metrics
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Feature importance al
        """
        if not self.is_trained:
            return None
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importance = np.abs(self.model.coef_)
            else:
                return None
            
            if self.feature_names:
                return dict(zip(self.feature_names, importance))
            else:
                return {f"feature_{i}": imp for i, imp in enumerate(importance)}
                
        except Exception as e:
            self.logger.warning(f"Feature importance alınamadı: {e}")
            return None
    
    def save_model(self, filepath: str, format: str = 'pickle') -> str:
        """
        Modeli kaydet
        
        Args:
            filepath: Kaydedilecek dosya yolu
            format: Kaydetme formatı (pickle, joblib, json)
            
        Returns:
            Kaydedilen dosya yolu
        """
        if not self.is_trained:
            raise ModelTrainingError(
                "Eğitilmemiş model kaydedilemez",
                model_name=self.name
            )
        
        try:
            filepath = Path(filepath)
            
            # Model metadata
            metadata = {
                'name': self.name,
                'model_type': self.model_type,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names,
                'target_name': self.target_name,
                'model_params': self.model_params,
                'metrics': self.metrics,
                'created_at': self.created_at.isoformat(),
                'updated_at': self.updated_at.isoformat() if self.updated_at else None,
                'training_history': self.training_history
            }
            
            if format == 'pickle':
                # Model ve metadata'yı pickle ile kaydet
                model_data = {
                    'model': self.model,
                    'metadata': metadata
                }
                with open(filepath, 'wb') as f:
                    pickle.dump(model_data, f)
                    
            elif format == 'joblib':
                # Model ve metadata'yı joblib ile kaydet
                model_data = {
                    'model': self.model,
                    'metadata': metadata
                }
                joblib.dump(model_data, filepath)
                
            elif format == 'json':
                # Sadece metadata'yı JSON olarak kaydet
                write_file(metadata, filepath, '.json')
                
            else:
                raise ValueError(f"Desteklenmeyen format: {format}")
            
            self.model_path = str(filepath)
            self.logger.info(f"Model kaydedildi: {filepath}")
            return str(filepath)
            
        except Exception as e:
            raise ModelTrainingError(
                f"Model kaydetme hatası: {e}",
                model_name=self.name
            )
    
    @classmethod
    def load_model(cls, filepath: str, format: str = 'pickle') -> 'BaseModel':
        """
        Modeli yükle
        
        Args:
            filepath: Model dosyası yolu
            format: Yükleme formatı
            
        Returns:
            Yüklenen model
        """
        try:
            filepath = Path(filepath)
            
            if format == 'pickle':
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                    
            elif format == 'joblib':
                model_data = joblib.load(filepath)
                
            elif format == 'json':
                # JSON sadece metadata içerir
                metadata = read_file(filepath, '.json')
                model_data = {'metadata': metadata, 'model': None}
                
            else:
                raise ValueError(f"Desteklenmeyen format: {format}")
            
            # Model instance'ı oluştur
            metadata = model_data['metadata']
            model_instance = cls(
                name=metadata['name'],
                model_type=metadata['model_type'],
                **metadata.get('model_params', {})
            )
            
            # Model durumunu geri yükle
            model_instance.model = model_data['model']
            model_instance.is_trained = metadata['is_trained']
            model_instance.feature_names = metadata['feature_names']
            model_instance.target_name = metadata['target_name']
            model_instance.metrics = metadata.get('metrics', {})
            model_instance.created_at = datetime.fromisoformat(metadata['created_at'])
            if metadata.get('updated_at'):
                model_instance.updated_at = datetime.fromisoformat(metadata['updated_at'])
            model_instance.training_history = metadata.get('training_history', [])
            model_instance.model_path = str(filepath)
            
            model_instance.logger.info(f"Model yüklendi: {filepath}")
            return model_instance
            
        except Exception as e:
            raise ModelTrainingError(
                f"Model yükleme hatası: {e}",
                model_name=filepath.name
            )
    
    def _validate_training_data(self, X: Union[np.ndarray, pd.DataFrame], 
                               y: Union[np.ndarray, pd.Series]):
        """Eğitim verisini doğrula"""
        if X is None or y is None:
            raise ValueError("X ve y None olamaz")
        
        if len(X) != len(y):
            raise ValueError("X ve y uzunlukları eşit olmalı")
        
        if len(X) == 0:
            raise ValueError("Veri boş olamaz")
    
    def _validate_prediction_data(self, X: Union[np.ndarray, pd.DataFrame]):
        """Tahmin verisini doğrula"""
        if X is None:
            raise ValueError("X None olamaz")
        
        if len(X) == 0:
            raise ValueError("Veri boş olamaz")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Model bilgilerini al"""
        return {
            'name': self.name,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'model_params': self.model_params,
            'metrics': self.metrics,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'model_path': self.model_path,
            'training_history_count': len(self.training_history)
        }
    
    def __str__(self) -> str:
        return f"{self.name} ({self.model_type}) - Trained: {self.is_trained}"
    
    def __repr__(self) -> str:
        return self.__str__()


if __name__ == "__main__":
    # Test
    logger = get_logger(__name__)
    
    # BaseModel abstract olduğu için doğrudan test edilemez
    # Alt sınıflar tarafından implement edilmeli
    logger.info("BaseModel sınıfı hazır") 