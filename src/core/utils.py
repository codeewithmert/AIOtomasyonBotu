"""
Utility Functions
Yardımcı fonksiyonlar - veri doğrulama, dosya işlemleri, tarih/zaman işlemleri
"""

import os
import json
import yaml
import csv
import pickle
import hashlib
import base64
import uuid
import re
import time
import random
import string
from datetim e import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from functools import wraps, lru_cache
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import logging

from .logger import get_logger
from .exceptions import ValidationError, DataProcessingError


logger = get_logger(__name__)


# =============================================================================
# Data Validation Functions
# =============================================================================

def validate_data_type(data: Any, expected_type: type, field_name: str = "data") -> bool:
    """Veri tipini doğrula"""
    if not isinstance(data, expected_type):
        raise ValidationError(
            f"{field_name} beklenen tip {expected_type.__name__} değil, {type(data).__name__}",
            field=field_name,
            value=data,
            expected_type=expected_type.__name__
        )
    return True


def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None, 
                      min_rows: int = 0, max_rows: int = None) -> bool:
    """DataFrame'i doğrula"""
    if not isinstance(df, pd.DataFrame):
        raise ValidationError("DataFrame değil", field="dataframe", value=type(df))
    
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValidationError(
                f"Eksik sütunlar: {missing_columns}",
                field="columns",
                value=list(df.columns),
                expected_type="required_columns"
            )
    
    if len(df) < min_rows:
        raise ValidationError(
            f"Satır sayısı çok az: {len(df)} < {min_rows}",
            field="row_count",
            value=len(df),
            expected_type=f"min_{min_rows}"
        )
    
    if max_rows and len(df) > max_rows:
        raise ValidationError(
            f"Satır sayısı çok fazla: {len(df)} > {max_rows}",
            field="row_count",
            value=len(df),
            expected_type=f"max_{max_rows}"
        )
    
    return True


def validate_numeric_range(value: Union[int, float], min_val: float = None, 
                          max_val: float = None, field_name: str = "value") -> bool:
    """Sayısal değer aralığını doğrula"""
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"{field_name} sayısal değil",
            field=field_name,
            value=value,
            expected_type="numeric"
        )
    
    if min_val is not None and value < min_val:
        raise ValidationError(
            f"{field_name} çok küçük: {value} < {min_val}",
            field=field_name,
            value=value,
            expected_type=f"min_{min_val}"
        )
    
    if max_val is not None and value > max_val:
        raise ValidationError(
            f"{field_name} çok büyük: {value} > {max_val}",
            field=field_name,
            value=value,
            expected_type=f"max_{max_val}"
        )
    
    return True


def validate_string_length(text: str, min_length: int = 0, max_length: int = None, 
                          field_name: str = "text") -> bool:
    """String uzunluğunu doğrula"""
    if not isinstance(text, str):
        raise ValidationError(
            f"{field_name} string değil",
            field=field_name,
            value=text,
            expected_type="string"
        )
    
    if len(text) < min_length:
        raise ValidationError(
            f"{field_name} çok kısa: {len(text)} < {min_length}",
            field=field_name,
            value=text,
            expected_type=f"min_length_{min_length}"
        )
    
    if max_length and len(text) > max_length:
        raise ValidationError(
            f"{field_name} çok uzun: {len(text)} > {max_length}",
            field=field_name,
            value=text,
            expected_type=f"max_length_{max_length}"
        )
    
    return True


def validate_email(email: str) -> bool:
    """Email formatını doğrula"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValidationError(
            "Geçersiz email formatı",
            field="email",
            value=email,
            expected_type="valid_email"
        )
    return True


def validate_url(url: str) -> bool:
    """URL formatını doğrula"""
    pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
    if not re.match(pattern, url):
        raise ValidationError(
            "Geçersiz URL formatı",
            field="url",
            value=url,
            expected_type="valid_url"
        )
    return True


# =============================================================================
# File Operations
# =============================================================================

def ensure_directory(path: Union[str, Path]) -> Path:
    """Dizinin var olduğundan emin ol"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(filename: str) -> str:
    """Güvenli dosya adı oluştur"""
    # Geçersiz karakterleri kaldır
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Çoklu alt çizgileri tek alt çizgiye çevir
    filename = re.sub(r'_+', '_', filename)
    # Başındaki ve sonundaki boşlukları kaldır
    filename = filename.strip('._')
    return filename


def get_file_extension(filepath: Union[str, Path]) -> str:
    """Dosya uzantısını al"""
    return Path(filepath).suffix.lower()


def is_supported_file_format(filepath: Union[str, Path], 
                           supported_formats: List[str] = None) -> bool:
    """Dosya formatının desteklenip desteklenmediğini kontrol et"""
    if supported_formats is None:
        supported_formats = ['.csv', '.json', '.xlsx', '.xls', '.parquet', '.pickle']
    
    extension = get_file_extension(filepath)
    return extension in supported_formats


def read_file(filepath: Union[str, Path], file_type: str = None) -> Any:
    """Dosyayı oku"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {filepath}")
    
    if file_type is None:
        file_type = get_file_extension(filepath)
    
    try:
        if file_type == '.csv':
            return pd.read_csv(filepath)
        elif file_type == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif file_type in ['.xlsx', '.xls']:
            return pd.read_excel(filepath)
        elif file_type == '.parquet':
            return pd.read_parquet(filepath)
        elif file_type == '.pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif file_type == '.yaml' or file_type == '.yml':
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Desteklenmeyen dosya formatı: {file_type}")
    
    except Exception as e:
        logger.error(f"Dosya okuma hatası: {filepath} - {e}")
        raise


def write_file(data: Any, filepath: Union[str, Path], file_type: str = None, 
               **kwargs) -> Path:
    """Dosyaya yaz"""
    filepath = Path(filepath)
    
    if file_type is None:
        file_type = get_file_extension(filepath)
    
    # Dizini oluştur
    ensure_directory(filepath.parent)
    
    try:
        if file_type == '.csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False, **kwargs)
            else:
                raise ValueError("CSV için DataFrame gerekli")
        
        elif file_type == '.json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, **kwargs)
        
        elif file_type in ['.xlsx', '.xls']:
            if isinstance(data, pd.DataFrame):
                data.to_excel(filepath, index=False, **kwargs)
            else:
                raise ValueError("Excel için DataFrame gerekli")
        
        elif file_type == '.parquet':
            if isinstance(data, pd.DataFrame):
                data.to_parquet(filepath, **kwargs)
            else:
                raise ValueError("Parquet için DataFrame gerekli")
        
        elif file_type == '.pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, **kwargs)
        
        elif file_type == '.yaml' or file_type == '.yml':
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, **kwargs)
        
        else:
            raise ValueError(f"Desteklenmeyen dosya formatı: {file_type}")
        
        logger.info(f"Dosya yazıldı: {filepath}")
        return filepath
    
    except Exception as e:
        logger.error(f"Dosya yazma hatası: {filepath} - {e}")
        raise


def get_file_size(filepath: Union[str, Path]) -> int:
    """Dosya boyutunu al (bytes)"""
    return Path(filepath).stat().st_size


def format_file_size(size_bytes: int) -> str:
    """Dosya boyutunu okunabilir formata çevir"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


# =============================================================================
# Date and Time Functions
# =============================================================================

def parse_date(date_string: str, formats: List[str] = None) -> datetime:
    """Tarih string'ini parse et"""
    if formats is None:
        formats = [
            '%Y-%m-%d',
            '%Y-%m-%d %H:%M:%S',
            '%d/%m/%Y',
            '%d/%m/%Y %H:%M:%S',
            '%Y%m%d',
            '%Y%m%d%H%M%S'
        ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Tarih formatı tanınmadı: {date_string}")


def format_date(date: datetime, format_string: str = '%Y-%m-%d %H:%M:%S') -> str:
    """Tarihi formatla"""
    return date.strftime(format_string)


def get_date_range(start_date: datetime, end_date: datetime, 
                   step: timedelta = timedelta(days=1)) -> List[datetime]:
    """Tarih aralığı oluştur"""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += step
    return dates


def is_weekend(date: datetime) -> bool:
    """Hafta sonu mu kontrol et"""
    return date.weekday() >= 5


def is_business_day(date: datetime) -> bool:
    """İş günü mü kontrol et"""
    return not is_weekend(date)


def add_business_days(date: datetime, days: int) -> datetime:
    """İş günü ekle"""
    current = date
    remaining_days = abs(days)
    direction = 1 if days > 0 else -1
    
    while remaining_days > 0:
        current += timedelta(days=direction)
        if is_business_day(current):
            remaining_days -= 1
    
    return current


# =============================================================================
# Data Processing Functions
# =============================================================================

def clean_text(text: str) -> str:
    """Metni temizle"""
    if not isinstance(text, str):
        return str(text)
    
    # Boşlukları normalize et
    text = re.sub(r'\s+', ' ', text)
    # Başındaki ve sonundaki boşlukları kaldır
    text = text.strip()
    # Özel karakterleri temizle
    text = re.sub(r'[^\w\s\-.,!?]', '', text)
    
    return text


def remove_duplicates(data: List[Any], key: Callable = None) -> List[Any]:
    """Tekrarlanan öğeleri kaldır"""
    if key is None:
        return list(dict.fromkeys(data))
    else:
        seen = set()
        result = []
        for item in data:
            key_value = key(item)
            if key_value not in seen:
                seen.add(key_value)
                result.append(item)
        return result


def chunk_data(data: List[Any], chunk_size: int) -> List[List[Any]]:
    """Veriyi parçalara böl"""
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def flatten_list(nested_list: List[Any]) -> List[Any]:
    """İç içe listeyi düzleştir"""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def group_by(data: List[Dict[str, Any]], key: str) -> Dict[Any, List[Dict[str, Any]]]:
    """Veriyi anahtara göre grupla"""
    grouped = defaultdict(list)
    for item in data:
        grouped[item[key]].append(item)
    return dict(grouped)


def sort_dict_by_value(d: Dict[Any, Any], reverse: bool = False) -> Dict[Any, Any]:
    """Dictionary'i değere göre sırala"""
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=reverse))


def get_top_n(d: Dict[Any, Any], n: int = 10) -> Dict[Any, Any]:
    """En üst N öğeyi al"""
    sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_items[:n])


# =============================================================================
# Security and Encryption Functions
# =============================================================================

def generate_random_string(length: int = 16, 
                          include_digits: bool = True, 
                          include_symbols: bool = False) -> str:
    """Rastgele string oluştur"""
    chars = string.ascii_letters
    if include_digits:
        chars += string.digits
    if include_symbols:
        chars += string.punctuation
    
    return ''.join(random.choice(chars) for _ in range(length))


def generate_uuid() -> str:
    """UUID oluştur"""
    return str(uuid.uuid4())


def hash_string(text: str, algorithm: str = 'sha256') -> str:
    """String'i hash'le"""
    hash_func = getattr(hashlib, algorithm)
    return hash_func(text.encode('utf-8')).hexdigest()


def encode_base64(data: Union[str, bytes]) -> str:
    """Base64 encode"""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return base64.b64encode(data).decode('utf-8')


def decode_base64(encoded_data: str) -> bytes:
    """Base64 decode"""
    return base64.b64decode(encoded_data.encode('utf-8'))


def mask_sensitive_data(data: str, mask_char: str = '*', 
                       visible_chars: int = 4) -> str:
    """Hassas veriyi maskele"""
    if len(data) <= visible_chars:
        return mask_char * len(data)
    
    return data[:visible_chars] + mask_char * (len(data) - visible_chars)


# =============================================================================
# Performance and Monitoring Functions
# =============================================================================

def measure_execution_time(func: Callable) -> Callable:
    """Fonksiyon çalışma süresini ölç"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} çalışma süresi: {execution_time:.4f} saniye")
        
        return result
    return wrapper


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, 
                    backoff_factor: float = 2.0, 
                    exceptions: tuple = (Exception,)) -> Callable:
    """Hata durumunda yeniden dene"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        sleep_time = delay * (backoff_factor ** attempt)
                        logger.warning(
                            f"{func.__name__} denemesi {attempt + 1} başarısız, "
                            f"{sleep_time:.2f} saniye sonra tekrar deneniyor: {e}"
                        )
                        time.sleep(sleep_time)
            
            logger.error(f"{func.__name__} {max_retries} deneme sonrası başarısız")
            raise last_exception
        
        return wrapper
    return decorator


@lru_cache(maxsize=128)
def cached_function(func: Callable) -> Callable:
    """Fonksiyonu cache'le"""
    return func


# =============================================================================
# Data Analysis Functions
# =============================================================================

def calculate_statistics(data: List[Union[int, float]]) -> Dict[str, float]:
    """Temel istatistikleri hesapla"""
    if not data:
        return {}
    
    data_array = np.array(data)
    
    return {
        'count': len(data),
        'mean': float(np.mean(data_array)),
        'median': float(np.median(data_array)),
        'std': float(np.std(data_array)),
        'min': float(np.min(data_array)),
        'max': float(np.max(data_array)),
        'q25': float(np.percentile(data_array, 25)),
        'q75': float(np.percentile(data_array, 75))
    }


def detect_outliers(data: List[Union[int, float]], 
                   method: str = 'iqr', threshold: float = 1.5) -> List[int]:
    """Aykırı değerleri tespit et"""
    if method == 'iqr':
        # IQR method
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outliers = []
        for i, value in enumerate(data):
            if value < lower_bound or value > upper_bound:
                outliers.append(i)
        
        return outliers
    
    elif method == 'zscore':
        # Z-score method
        mean = np.mean(data)
        std = np.std(data)
        
        outliers = []
        for i, value in enumerate(data):
            z_score = abs((value - mean) / std)
            if z_score > threshold:
                outliers.append(i)
        
        return outliers
    
    else:
        raise ValueError(f"Desteklenmeyen method: {method}")


def calculate_correlation_matrix(df: pd.DataFrame, 
                               method: str = 'pearson') -> pd.DataFrame:
    """Korelasyon matrisini hesapla"""
    return df.corr(method=method)


def calculate_missing_percentage(df: pd.DataFrame) -> Dict[str, float]:
    """Eksik veri yüzdesini hesapla"""
    missing_counts = df.isnull().sum()
    total_rows = len(df)
    
    return {
        column: (count / total_rows) * 100 
        for column, count in missing_counts.items()
    }


# =============================================================================
# Utility Decorators
# =============================================================================

def validate_inputs(*validators: Callable) -> Callable:
    """Input validasyonu için decorator"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validasyonları uygula
            for validator in validators:
                validator(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_function_info(func: Callable) -> Callable:
    """Fonksiyon bilgilerini logla"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Fonksiyon çağrıldı: {func.__name__}")
        logger.debug(f"Argümanlar: {args}")
        logger.debug(f"Keyword argümanlar: {kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Fonksiyon tamamlandı: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Fonksiyon hatası: {func.__name__} - {e}")
            raise
    
    return wrapper


def cache_result(ttl: int = 3600) -> Callable:
    """Sonucu cache'le (TTL ile)"""
    cache = {}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Cache key oluştur
            key = str((args, tuple(sorted(kwargs.items()))))
            
            # Cache'den kontrol et
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < ttl:
                    return result
            
            # Fonksiyonu çalıştır ve cache'e kaydet
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            
            return result
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test functions
    logger.info("Utils modülü test ediliyor...")
    
    # Data validation test
    try:
        validate_numeric_range(5, min_val=0, max_val=10)
        logger.info("Numeric validation başarılı")
    except ValidationError as e:
        logger.error(f"Validation hatası: {e}")
    
    # File operations test
    test_data = {"test": "data"}
    test_file = "test_output.json"
    try:
        write_file(test_data, test_file)
        loaded_data = read_file(test_file)
        logger.info(f"File operations başarılı: {loaded_data}")
        os.remove(test_file)
    except Exception as e:
        logger.error(f"File operations hatası: {e}")
    
    # Date functions test
    today = datetime.now()
    logger.info(f"Bugün: {format_date(today)}")
    logger.info(f"Hafta sonu mu: {is_weekend(today)}")
    
    # Data processing test
    test_list = [1, 2, 2, 3, 3, 3, 4, 5]
    unique_list = remove_duplicates(test_list)
    logger.info(f"Tekrarlar kaldırıldı: {unique_list}")
    
    # Security functions test
    random_str = generate_random_string(10)
    logger.info(f"Rastgele string: {random_str}")
    
    # Statistics test
    stats = calculate_statistics([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    logger.info(f"İstatistikler: {stats}") 