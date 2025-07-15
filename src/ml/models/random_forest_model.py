"""
Random Forest Model
Random Forest algoritması için özel model sınıfı
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .base_model import BaseModel
from ...core.logger import get_logger, performance_context
from ...core.exceptions import ModelTrainingError, ModelPredictionError


class RandomForestModel(BaseModel):
    """
    Random Forest Model
    Hem sınıflandırma hem regresyon desteği
    """
    
    def __init__(self, name: str = None, task_type: str = 'classification', **kwargs):
        """
        Random Forest Model başlat
        
        Args:
            name: Model adı
            task_type: Görev türü ('classification' veya 'regression')
            **kwargs: Random Forest parametreleri
        """
        self.task_type = task_type.lower()
        if self.task_type not in ['classification', 'regression']:
            raise ValueError("task_type 'classification' veya 'regression' olmalı")
        
        # Varsayılan parametreler
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Kullanıcı parametreleri ile varsayılanları birleştir
        model_params = {**default_params, **kwargs}
        
        super().__init__(
            name=name or f"RandomForest_{self.task_type.capitalize()}",
            model_type=f"random_forest_{self.task_type}",
            **model_params
        )
        
        self.model = None
        self.classes_ = None
        self.feature_importance_ = None
        
        self.logger.info(f"RandomForest model oluşturuldu: {self.name} ({self.task_type})")
    
    def _create_model(self, **kwargs) -> Union[RandomForestClassifier, RandomForestRegressor]:
        """Random Forest model nesnesini oluştur"""
        if self.task_type == 'classification':
            return RandomForestClassifier(**kwargs)
        else:
            return RandomForestRegressor(**kwargs)
    
    def _train_model(self, X: Union[np.ndarray, pd.DataFrame], 
                    y: Union[np.ndarray, pd.Series], **kwargs) -> Dict[str, Any]:
        """Modeli eğit"""
        try:
            # Veriyi numpy array'e çevir
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = X
            
            if isinstance(y, pd.Series):
                y_array = y.values
            else:
                y_array = y
            
            # Model eğitimi
            self.model.fit(X_array, y_array)
            
            # Sınıflandırma için classes_ kaydet
            if self.task_type == 'classification':
                self.classes_ = self.model.classes_
            
            # Feature importance kaydet
            self.feature_importance_ = self.model.feature_importances_
            
            # Cross-validation skorları
            cv_scores = cross_val_score(
                self.model, X_array, y_array, 
                cv=5, scoring='accuracy' if self.task_type == 'classification' else 'r2'
            )
            
            training_results = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist(),
                'n_features': X_array.shape[1],
                'n_samples': X_array.shape[0],
                'feature_importance': self.get_feature_importance()
            }
            
            return training_results
            
        except Exception as e:
            raise ModelTrainingError(
                f"RandomForest eğitim hatası: {e}",
                model_name=self.name,
                training_data_size=X.shape[0] if hasattr(X, 'shape') else len(X)
            )
    
    def _predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Tahmin yap"""
        try:
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = X
            
            return self.model.predict(X_array)
            
        except Exception as e:
            raise ModelPredictionError(
                f"RandomForest tahmin hatası: {e}",
                model_name=self.name,
                input_data_shape=X.shape if hasattr(X, 'shape') else None
            )
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Olasılık tahminleri yap (sadece sınıflandırma için)"""
        if self.task_type != 'classification':
            raise ModelPredictionError(
                "Olasılık tahmini sadece sınıflandırma için kullanılabilir",
                model_name=self.name
            )
        
        if not self.is_trained:
            raise ModelPredictionError(
                "Model henüz eğitilmemiş",
                model_name=self.name
            )
        
        try:
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = X
            
            return self.model.predict_proba(X_array)
            
        except Exception as e:
            raise ModelPredictionError(
                f"Olasılık tahmin hatası: {e}",
                model_name=self.name
            )
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Feature importance al"""
        if not self.is_trained or self.feature_importance_ is None:
            return None
        
        if self.feature_names:
            return dict(zip(self.feature_names, self.feature_importance_))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(self.feature_importance_)}
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """En önemli feature'ları al"""
        importance = self.get_feature_importance()
        if not importance:
            return []
        
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]
    
    def plot_feature_importance(self, n_features: int = 20, figsize: Tuple[int, int] = (12, 8),
                               save_path: Optional[str] = None) -> plt.Figure:
        """Feature importance grafiği çiz"""
        if not self.is_trained:
            raise ModelTrainingError("Eğitilmemiş model için feature importance çizilemez")
        
        importance = self.get_feature_importance()
        if not importance:
            raise ModelTrainingError("Feature importance bulunamadı")
        
        # En önemli n feature'ı al
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:n_features]
        feature_names, importance_values = zip(*top_features)
        
        # Grafik çiz
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, importance_values)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Feature Importance - {self.name}')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance grafiği kaydedildi: {save_path}")
        
        return fig
    
    def plot_confusion_matrix(self, X: Union[np.ndarray, pd.DataFrame], 
                            y: Union[np.ndarray, pd.Series],
                            figsize: Tuple[int, int] = (8, 6),
                            save_path: Optional[str] = None) -> plt.Figure:
        """Confusion matrix çiz (sadece sınıflandırma için)"""
        if self.task_type != 'classification':
            raise ModelTrainingError("Confusion matrix sadece sınıflandırma için kullanılabilir")
        
        if not self.is_trained:
            raise ModelTrainingError("Eğitilmemiş model için confusion matrix çizilemez")
        
        # Tahmin yap
        y_pred = self.predict(X)
        
        # Confusion matrix hesapla
        cm = confusion_matrix(y, y_pred)
        
        # Grafik çiz
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {self.name}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix kaydedildi: {save_path}")
        
        return fig
    
    def get_classification_report(self, X: Union[np.ndarray, pd.DataFrame], 
                                y: Union[np.ndarray, pd.Series]) -> str:
        """Sınıflandırma raporu al (sadece sınıflandırma için)"""
        if self.task_type != 'classification':
            raise ModelTrainingError("Classification report sadece sınıflandırma için kullanılabilir")
        
        if not self.is_trained:
            raise ModelTrainingError("Eğitilmemiş model için classification report alınamaz")
        
        y_pred = self.predict(X)
        return classification_report(y, y_pred)
    
    def hyperparameter_tuning(self, X: Union[np.ndarray, pd.DataFrame], 
                            y: Union[np.ndarray, pd.Series],
                            param_grid: Optional[Dict[str, List]] = None,
                            cv: int = 5, scoring: str = None) -> Dict[str, Any]:
        """Hiperparametre optimizasyonu"""
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        if scoring is None:
            scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
        
        try:
            # Grid search
            grid_search = GridSearchCV(
                self._create_model(**self.model_params),
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
            
            # Grid search çalıştır
            grid_search.fit(X, y)
            
            # En iyi parametreleri güncelle
            self.model_params.update(grid_search.best_params_)
            
            # En iyi modeli kullan
            self.model = grid_search.best_estimator_
            self.is_trained = True
            
            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            self.logger.info(f"Hiperparametre optimizasyonu tamamlandı: {results['best_score']:.4f}")
            return results
            
        except Exception as e:
            raise ModelTrainingError(
                f"Hiperparametre optimizasyonu hatası: {e}",
                model_name=self.name
            )
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Model özeti al"""
        summary = super().get_model_info()
        
        # RandomForest özel bilgileri
        if self.is_trained:
            summary.update({
                'task_type': self.task_type,
                'n_estimators': self.model_params.get('n_estimators'),
                'max_depth': self.model_params.get('max_depth'),
                'feature_importance_available': self.feature_importance_ is not None,
                'classes_available': self.classes_ is not None if self.task_type == 'classification' else False
            })
            
            if self.feature_importance_ is not None:
                summary['top_features'] = self.get_top_features(5)
        
        return summary
    
    def _calculate_metrics(self, y_true: Union[np.ndarray, pd.Series], 
                          y_pred: np.ndarray) -> Dict[str, float]:
        """Metrikleri hesapla (override)"""
        metrics = super()._calculate_metrics(y_true, y_pred)
        
        # RandomForest özel metrikleri
        if self.task_type == 'classification':
            # Sınıflandırma için ek metrikler
            from sklearn.metrics import roc_auc_score, log_loss
            
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary sınıflandırma
                    y_proba = self.predict_proba(y_true.reshape(-1, 1))[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                    metrics['log_loss'] = log_loss(y_true, y_proba)
                else:
                    # Multi-class sınıflandırma
                    y_proba = self.predict_proba(y_true.reshape(-1, 1))
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                    metrics['log_loss'] = log_loss(y_true, y_proba)
            except Exception as e:
                self.logger.warning(f"Ek sınıflandırma metrikleri hesaplanamadı: {e}")
        
        return metrics


# Convenience functions
def create_random_forest_classifier(n_estimators: int = 100, **kwargs) -> RandomForestModel:
    """Random Forest sınıflandırıcı oluştur"""
    return RandomForestModel(task_type='classification', n_estimators=n_estimators, **kwargs)


def create_random_forest_regressor(n_estimators: int = 100, **kwargs) -> RandomForestModel:
    """Random Forest regresör oluştur"""
    return RandomForestModel(task_type='regression', n_estimators=n_estimators, **kwargs)


if __name__ == "__main__":
    # Test
    logger = get_logger(__name__)
    
    try:
        # Test verisi oluştur
        from sklearn.datasets import make_classification, make_regression
        from sklearn.model_selection import train_test_split
        
        # Sınıflandırma testi
        X_clf, y_clf = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
            X_clf, y_clf, test_size=0.2, random_state=42
        )
        
        # RandomForest sınıflandırıcı
        rf_clf = create_random_forest_classifier(n_estimators=50)
        rf_clf.train(X_train_clf, y_train_clf)
        
        # Değerlendirme
        metrics_clf = rf_clf.evaluate(X_test_clf, y_test_clf)
        logger.info(f"Sınıflandırma metrikleri: {metrics_clf}")
        
        # Regresyon testi
        X_reg, y_reg = make_regression(n_samples=1000, n_features=20, random_state=42)
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        
        # RandomForest regresör
        rf_reg = create_random_forest_regressor(n_estimators=50)
        rf_reg.train(X_train_reg, y_train_reg)
        
        # Değerlendirme
        metrics_reg = rf_reg.evaluate(X_test_reg, y_test_reg)
        logger.info(f"Regresyon metrikleri: {metrics_reg}")
        
        # Feature importance
        importance_clf = rf_clf.get_feature_importance()
        logger.info(f"Feature importance (sınıflandırma): {importance_clf}")
        
        importance_reg = rf_reg.get_feature_importance()
        logger.info(f"Feature importance (regresyon): {importance_reg}")
        
    except Exception as e:
        logger.error(f"Test hatası: {e}") 