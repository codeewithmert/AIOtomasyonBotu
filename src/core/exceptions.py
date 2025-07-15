"""
Custom Exception Classes
AI Automation Bot için özel exception sınıfları
"""

from typing import Any, Dict, Optional, Union


class AutomationBotException(Exception):
    """Ana exception sınıfı"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Exception'ı dictionary'e çevir"""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details
        }


class ConfigurationError(AutomationBotException):
    """Konfigürasyon hatası"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 config_value: Optional[Any] = None):
        super().__init__(message, "CONFIG_ERROR")
        self.config_key = config_key
        self.config_value = config_value
        self.details.update({
            'config_key': config_key,
            'config_value': str(config_value) if config_value is not None else None
        })


class DataCollectionError(AutomationBotException):
    """Veri toplama hatası"""
    
    def __init__(self, message: str, source: Optional[str] = None, 
                 url: Optional[str] = None, status_code: Optional[int] = None):
        super().__init__(message, "DATA_COLLECTION_ERROR")
        self.source = source
        self.url = url
        self.status_code = status_code
        self.details.update({
            'source': source,
            'url': url,
            'status_code': status_code
        })


class DataProcessingError(AutomationBotException):
    """Veri işleme hatası"""
    
    def __init__(self, message: str, operation: Optional[str] = None, 
                 data_shape: Optional[tuple] = None, column: Optional[str] = None):
        super().__init__(message, "DATA_PROCESSING_ERROR")
        self.operation = operation
        self.data_shape = data_shape
        self.column = column
        self.details.update({
            'operation': operation,
            'data_shape': str(data_shape) if data_shape else None,
            'column': column
        })


class ModelTrainingError(AutomationBotException):
    """Model eğitimi hatası"""
    
    def __init__(self, message: str, model_name: Optional[str] = None, 
                 hyperparameters: Optional[Dict[str, Any]] = None, 
                 training_data_size: Optional[int] = None):
        super().__init__(message, "MODEL_TRAINING_ERROR")
        self.model_name = model_name
        self.hyperparameters = hyperparameters
        self.training_data_size = training_data_size
        self.details.update({
            'model_name': model_name,
            'hyperparameters': hyperparameters,
            'training_data_size': training_data_size
        })


class ModelPredictionError(AutomationBotException):
    """Model tahmin hatası"""
    
    def __init__(self, message: str, model_name: Optional[str] = None, 
                 input_data_shape: Optional[tuple] = None, 
                 prediction_type: Optional[str] = None):
        super().__init__(message, "MODEL_PREDICTION_ERROR")
        self.model_name = model_name
        self.input_data_shape = input_data_shape
        self.prediction_type = prediction_type
        self.details.update({
            'model_name': model_name,
            'input_data_shape': str(input_data_shape) if input_data_shape else None,
            'prediction_type': prediction_type
        })


class ModelEvaluationError(AutomationBotException):
    """Model değerlendirme hatası"""
    
    def __init__(self, message: str, model_name: Optional[str] = None, 
                 metrics: Optional[Dict[str, float]] = None, 
                 evaluation_method: Optional[str] = None):
        super().__init__(message, "MODEL_EVALUATION_ERROR")
        self.model_name = model_name
        self.metrics = metrics
        self.evaluation_method = evaluation_method
        self.details.update({
            'model_name': model_name,
            'metrics': metrics,
            'evaluation_method': evaluation_method
        })


class TaskSchedulingError(AutomationBotException):
    """Görev zamanlama hatası"""
    
    def __init__(self, message: str, task_name: Optional[str] = None, 
                 schedule: Optional[str] = None, max_retries: Optional[int] = None):
        super().__init__(message, "TASK_SCHEDULING_ERROR")
        self.task_name = task_name
        self.schedule = schedule
        self.max_retries = max_retries
        self.details.update({
            'task_name': task_name,
            'schedule': schedule,
            'max_retries': max_retries
        })


class TaskExecutionError(AutomationBotException):
    """Görev çalıştırma hatası"""
    
    def __init__(self, message: str, task_name: Optional[str] = None, 
                 execution_time: Optional[float] = None, 
                 retry_count: Optional[int] = None):
        super().__init__(message, "TASK_EXECUTION_ERROR")
        self.task_name = task_name
        self.execution_time = execution_time
        self.retry_count = retry_count
        self.details.update({
            'task_name': task_name,
            'execution_time': execution_time,
            'retry_count': retry_count
        })


class ReportGenerationError(AutomationBotException):
    """Rapor oluşturma hatası"""
    
    def __init__(self, message: str, report_type: Optional[str] = None, 
                 template: Optional[str] = None, output_format: Optional[str] = None):
        super().__init__(message, "REPORT_GENERATION_ERROR")
        self.report_type = report_type
        self.template = template
        self.output_format = output_format
        self.details.update({
            'report_type': report_type,
            'template': template,
            'output_format': output_format
        })


class NotificationError(AutomationBotException):
    """Bildirim gönderme hatası"""
    
    def __init__(self, message: str, notification_type: Optional[str] = None, 
                 recipient: Optional[str] = None, channel: Optional[str] = None):
        super().__init__(message, "NOTIFICATION_ERROR")
        self.notification_type = notification_type
        self.recipient = recipient
        self.channel = channel
        self.details.update({
            'notification_type': notification_type,
            'recipient': recipient,
            'channel': channel
        })


class DatabaseError(AutomationBotException):
    """Veritabanı hatası"""
    
    def __init__(self, message: str, operation: Optional[str] = None, 
                 table: Optional[str] = None, query: Optional[str] = None):
        super().__init__(message, "DATABASE_ERROR")
        self.operation = operation
        self.table = table
        self.query = query
        self.details.update({
            'operation': operation,
            'table': table,
            'query': query
        })


class APIError(AutomationBotException):
    """API hatası"""
    
    def __init__(self, message: str, endpoint: Optional[str] = None, 
                 method: Optional[str] = None, status_code: Optional[int] = None,
                 response_data: Optional[Dict[str, Any]] = None):
        super().__init__(message, "API_ERROR")
        self.endpoint = endpoint
        self.method = method
        self.status_code = status_code
        self.response_data = response_data
        self.details.update({
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'response_data': response_data
        })


class ValidationError(AutomationBotException):
    """Veri doğrulama hatası"""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Optional[Any] = None, expected_type: Optional[str] = None):
        super().__init__(message, "VALIDATION_ERROR")
        self.field = field
        self.value = value
        self.expected_type = expected_type
        self.details.update({
            'field': field,
            'value': str(value) if value is not None else None,
            'expected_type': expected_type
        })


class SecurityError(AutomationBotException):
    """Güvenlik hatası"""
    
    def __init__(self, message: str, security_type: Optional[str] = None, 
                 user: Optional[str] = None, ip_address: Optional[str] = None):
        super().__init__(message, "SECURITY_ERROR")
        self.security_type = security_type
        self.user = user
        self.ip_address = ip_address
        self.details.update({
            'security_type': security_type,
            'user': user,
            'ip_address': ip_address
        })


class ResourceError(AutomationBotException):
    """Kaynak hatası (bellek, disk, CPU)"""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, 
                 current_usage: Optional[float] = None, 
                 max_available: Optional[float] = None):
        super().__init__(message, "RESOURCE_ERROR")
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.max_available = max_available
        self.details.update({
            'resource_type': resource_type,
            'current_usage': current_usage,
            'max_available': max_available
        })


class TimeoutError(AutomationBotException):
    """Zaman aşımı hatası"""
    
    def __init__(self, message: str, operation: Optional[str] = None, 
                 timeout_duration: Optional[float] = None):
        super().__init__(message, "TIMEOUT_ERROR")
        self.operation = operation
        self.timeout_duration = timeout_duration
        self.details.update({
            'operation': operation,
            'timeout_duration': timeout_duration
        })


class RetryableError(AutomationBotException):
    """Yeniden deneme yapılabilir hata"""
    
    def __init__(self, message: str, max_retries: int = 3, 
                 retry_delay: float = 1.0, backoff_factor: float = 2.0):
        super().__init__(message, "RETRYABLE_ERROR")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.details.update({
            'max_retries': max_retries,
            'retry_delay': retry_delay,
            'backoff_factor': backoff_factor
        })


class CriticalError(AutomationBotException):
    """Kritik hata - sistem durdurulmalı"""
    
    def __init__(self, message: str, component: Optional[str] = None, 
                 requires_restart: bool = True):
        super().__init__(message, "CRITICAL_ERROR")
        self.component = component
        self.requires_restart = requires_restart
        self.details.update({
            'component': component,
            'requires_restart': requires_restart
        })


# Exception handler utilities
def handle_exception(func):
    """Exception handler decorator"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AutomationBotException as e:
            # Log the exception
            from .logger import get_logger
            logger = get_logger(func.__module__)
            logger.error(f"AutomationBotException: {e}", extra=e.details)
            raise
        except Exception as e:
            # Convert to AutomationBotException
            from .logger import get_logger
            logger = get_logger(func.__module__)
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise AutomationBotException(f"Unexpected error: {str(e)}", "UNEXPECTED_ERROR")
    
    return wrapper


def is_retryable_error(exception: Exception) -> bool:
    """Hatanın yeniden deneme yapılabilir olup olmadığını kontrol et"""
    if isinstance(exception, RetryableError):
        return True
    
    # Belirli exception türleri için retryable kontrolü
    retryable_exceptions = (
        TimeoutError,
        ConnectionError,
        OSError,
    )
    
    return isinstance(exception, retryable_exceptions)


def get_error_summary(exception: Exception) -> Dict[str, Any]:
    """Exception'dan özet bilgi al"""
    if isinstance(exception, AutomationBotException):
        return exception.to_dict()
    
    return {
        'error_type': exception.__class__.__name__,
        'message': str(exception),
        'error_code': 'UNKNOWN_ERROR',
        'details': {}
    }


# Exception mapping
EXCEPTION_MAPPING = {
    'config': ConfigurationError,
    'data_collection': DataCollectionError,
    'data_processing': DataProcessingError,
    'model_training': ModelTrainingError,
    'model_prediction': ModelPredictionError,
    'model_evaluation': ModelEvaluationError,
    'task_scheduling': TaskSchedulingError,
    'task_execution': TaskExecutionError,
    'report_generation': ReportGenerationError,
    'notification': NotificationError,
    'database': DatabaseError,
    'api': APIError,
    'validation': ValidationError,
    'security': SecurityError,
    'resource': ResourceError,
    'timeout': TimeoutError,
    'retryable': RetryableError,
    'critical': CriticalError,
}


def create_exception(exception_type: str, message: str, **kwargs) -> AutomationBotException:
    """Exception türüne göre exception oluştur"""
    exception_class = EXCEPTION_MAPPING.get(exception_type, AutomationBotException)
    return exception_class(message, **kwargs)


if __name__ == "__main__":
    # Test
    try:
        raise ConfigurationError("Geçersiz konfigürasyon", "database.host", "invalid_host")
    except ConfigurationError as e:
        print(f"Error: {e}")
        print(f"Details: {e.to_dict()}")
    
    try:
        raise DataCollectionError("API çağrısı başarısız", "external_api", "https://api.example.com", 500)
    except DataCollectionError as e:
        print(f"Error: {e}")
        print(f"Details: {e.to_dict()}")
    
    try:
        raise ModelTrainingError("Model eğitimi başarısız", "RandomForest", {"n_estimators": 100}, 1000)
    except ModelTrainingError as e:
        print(f"Error: {e}")
        print(f"Details: {e.to_dict()}") 