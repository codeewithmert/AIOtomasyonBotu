"""
Advanced Logging System
Gelişmiş loglama sistemi - çoklu handler, formatter ve log seviyeleri
"""

import logging
import logging.handlers
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from functools import wraps
import time
import threading
from contextlib import contextmanager

from .config import get_logging_config


class JSONFormatter(logging.Formatter):
    """JSON formatında log mesajları"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': record.process,
            'thread_id': record.thread,
        }
        
        # Exception bilgisi varsa ekle
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Extra fields varsa ekle
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Renkli konsol çıktısı için formatter"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Renk kodunu ekle
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Orijinal format string'ini güncelle
        record.levelname = f"{color}{record.levelname}{reset}"
        
        return super().format(record)


class PerformanceLogger:
    """Performans loglama için özel logger"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._timers: Dict[str, float] = {}
    
    def start_timer(self, name: str):
        """Timer başlat"""
        self._timers[name] = time.time()
        self.logger.debug(f"Timer başlatıldı: {name}")
    
    def end_timer(self, name: str, extra_info: Optional[Dict[str, Any]] = None):
        """Timer bitir ve süreyi logla"""
        if name in self._timers:
            duration = time.time() - self._timers[name]
            info = f"Timer tamamlandı: {name} - {duration:.4f}s"
            if extra_info:
                info += f" - {extra_info}"
            self.logger.info(info)
            del self._timers[name]
        else:
            self.logger.warning(f"Timer bulunamadı: {name}")
    
    @contextmanager
    def timer(self, name: str, extra_info: Optional[Dict[str, Any]] = None):
        """Context manager olarak timer kullanımı"""
        self.start_timer(name)
        try:
            yield
        finally:
            self.end_timer(name, extra_info)


class LoggerManager:
    """Merkezi logger yöneticisi"""
    
    def __init__(self, config=None):
        self.config = config or get_logging_config()
        self._loggers: Dict[str, logging.Logger] = {}
        self._performance_loggers: Dict[str, PerformanceLogger] = {}
        self._setup_root_logger()
    
    def _setup_root_logger(self):
        """Root logger'ı kur"""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # Mevcut handler'ları temizle
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Handler'ları ekle
        self._add_console_handler(root_logger)
        self._add_file_handler(root_logger)
        self._add_rotating_file_handler(root_logger)
    
    def _add_console_handler(self, logger: logging.Logger):
        """Konsol handler'ı ekle"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        # Renkli formatter
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    def _add_file_handler(self, logger: logging.Logger):
        """Dosya handler'ı ekle"""
        log_file = Path(self.config.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def _add_rotating_file_handler(self, logger: logging.Logger):
        """Dönen dosya handler'ı ekle"""
        log_file = Path(self.config.file)
        
        rotating_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.max_size,
            backupCount=self.config.backup_count,
            encoding='utf-8'
        )
        rotating_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        rotating_handler.setFormatter(formatter)
        logger.addHandler(rotating_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Logger al veya oluştur"""
        if name not in self._loggers:
            logger = logging.getLogger(name)
            self._loggers[name] = logger
        return self._loggers[name]
    
    def get_performance_logger(self, name: str) -> PerformanceLogger:
        """Performance logger al veya oluştur"""
        if name not in self._performance_loggers:
            logger = self.get_logger(f"{name}.performance")
            self._performance_loggers[name] = PerformanceLogger(logger)
        return self._performance_loggers[name]
    
    def add_json_handler(self, logger_name: str, file_path: str):
        """JSON formatında log dosyası ekle"""
        logger = self.get_logger(logger_name)
        
        json_file = Path(file_path)
        json_file.parent.mkdir(parents=True, exist_ok=True)
        
        json_handler = logging.FileHandler(json_file, encoding='utf-8')
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(JSONFormatter())
        
        logger.addHandler(json_handler)
    
    def set_level(self, logger_name: str, level: str):
        """Logger seviyesini ayarla"""
        logger = self.get_logger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))
    
    def log_with_context(self, logger_name: str, level: str, message: str, 
                        context: Optional[Dict[str, Any]] = None):
        """Context ile log mesajı"""
        logger = self.get_logger(logger_name)
        
        if context:
            # Extra fields ekle
            record = logger.makeRecord(
                logger_name, getattr(logging, level.upper()),
                "", 0, message, (), None
            )
            record.extra_fields = context
            logger.handle(record)
        else:
            getattr(logger, level.lower())(message)


# Global logger manager
_logger_manager: Optional[LoggerManager] = None


def get_logger(name: str) -> logging.Logger:
    """Global logger al"""
    global _logger_manager
    if _logger_manager is None:
        _logger_manager = LoggerManager()
    return _logger_manager.get_logger(name)


def get_performance_logger(name: str) -> PerformanceLogger:
    """Global performance logger al"""
    global _logger_manager
    if _logger_manager is None:
        _logger_manager = LoggerManager()
    return _logger_manager.get_performance_logger(name)


def init_logger(config=None) -> LoggerManager:
    """Logger'ı başlat"""
    global _logger_manager
    _logger_manager = LoggerManager(config)
    return _logger_manager


# Decorator'lar
def log_function_call(logger_name: Optional[str] = None):
    """Fonksiyon çağrılarını logla"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            
            # Fonksiyon başlangıcı
            logger.debug(f"Fonksiyon çağrıldı: {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Fonksiyon tamamlandı: {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Fonksiyon hatası: {func.__name__} - {str(e)}")
                raise
        
        return wrapper
    return decorator


def log_execution_time(logger_name: Optional[str] = None):
    """Fonksiyon çalışma süresini logla"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            perf_logger = get_performance_logger(logger_name or func.__module__)
            
            with perf_logger.timer(func.__name__):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def log_exceptions(logger_name: Optional[str] = None, reraise: bool = True):
    """Exception'ları logla"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Exception yakalandı: {func.__name__}",
                    exc_info=True,
                    extra={
                        'function': func.__name__,
                        'args': str(args),
                        'kwargs': str(kwargs)
                    }
                )
                if reraise:
                    raise
        
        return wrapper
    return decorator


# Context manager'lar
@contextmanager
def log_context(logger_name: str, context_name: str, level: str = "INFO"):
    """Context manager ile loglama"""
    logger = get_logger(logger_name)
    
    logger.log(
        getattr(logging, level.upper()),
        f"Context başladı: {context_name}"
    )
    
    try:
        yield
        logger.log(
            getattr(logging, level.upper()),
            f"Context tamamlandı: {context_name}"
        )
    except Exception as e:
        logger.error(f"Context hatası: {context_name} - {str(e)}")
        raise


@contextmanager
def performance_context(logger_name: str, operation_name: str):
    """Performans context manager"""
    perf_logger = get_performance_logger(logger_name)
    
    with perf_logger.timer(operation_name):
        yield


# Utility functions
def log_data_sample(logger_name: str, data_name: str, data: Any, max_items: int = 10):
    """Veri örneğini logla"""
    logger = get_logger(logger_name)
    
    if isinstance(data, (list, tuple)):
        sample = data[:max_items]
        logger.info(f"{data_name} örneği ({len(data)} öğe): {sample}")
    elif isinstance(data, dict):
        sample = dict(list(data.items())[:max_items])
        logger.info(f"{data_name} örneği ({len(data)} anahtar): {sample}")
    else:
        logger.info(f"{data_name}: {data}")


def log_model_metrics(logger_name: str, model_name: str, metrics: Dict[str, float]):
    """Model metriklerini logla"""
    logger = get_logger(logger_name)
    
    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"Model metrikleri - {model_name}: {metrics_str}")


def log_api_request(logger_name: str, method: str, url: str, status_code: int, 
                   response_time: float, extra: Optional[Dict[str, Any]] = None):
    """API isteğini logla"""
    logger = get_logger(logger_name)
    
    log_data = {
        'method': method,
        'url': url,
        'status_code': status_code,
        'response_time': f"{response_time:.3f}s"
    }
    
    if extra:
        log_data.update(extra)
    
    if status_code >= 400:
        logger.warning(f"API isteği: {log_data}")
    else:
        logger.info(f"API isteği: {log_data}")


# Thread-safe logging
class ThreadSafeLogger:
    """Thread-safe logger wrapper"""
    
    def __init__(self, logger_name: str):
        self.logger = get_logger(logger_name)
        self._lock = threading.Lock()
    
    def log(self, level: str, message: str, **kwargs):
        """Thread-safe log"""
        with self._lock:
            getattr(self.logger, level.lower())(message, **kwargs)


if __name__ == "__main__":
    # Test
    logger = get_logger("test")
    perf_logger = get_performance_logger("test")
    
    logger.info("Test mesajı")
    logger.warning("Test uyarısı")
    logger.error("Test hatası")
    
    with perf_logger.timer("test_operation"):
        time.sleep(0.1)
    
    # JSON handler test
    logger_manager = LoggerManager()
    logger_manager.add_json_handler("test", "logs/test.json")
    
    test_logger = logger_manager.get_logger("test")
    test_logger.info("JSON formatında test mesajı") 