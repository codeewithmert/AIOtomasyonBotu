"""
API Data Collector
REST API'lerden veri toplama modülü
"""

import asyncio
import aiohttp
import requests
import json
import time
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import logging

from ...core.logger import get_logger, log_api_request, performance_context
from ...core.exceptions import DataCollectionError, APIError, TimeoutError, RetryableError
from ...core.utils import retry_on_failure, validate_url, validate_data_type
from ...core.config import get_api_config


class APICollector:
    """
    API veri toplayıcı
    Çoklu endpoint, authentication ve format desteği
    """
    
    def __init__(self, config=None):
        self.config = config or get_api_config()
        self.logger = get_logger(__name__)
        self.session = None
        self._auth_headers = {}
        self._rate_limit_delay = 0
        self._last_request_time = 0
        
        self._setup_session()
    
    def _setup_session(self):
        """HTTP session kur"""
        self.session = requests.Session()
        self.session.timeout = self.config.timeout
        
        # Default headers
        self.session.headers.update({
            'User-Agent': 'AI-Automation-Bot/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        # API key varsa ekle
        if self.config.api_key:
            self._auth_headers['Authorization'] = f'Bearer {self.config.api_key}'
            self.session.headers.update(self._auth_headers)
    
    @retry_on_failure(max_retries=3, delay=1.0, backoff_factor=2.0)
    def get_data(self, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                 headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        GET isteği ile veri al
        
        Args:
            endpoint: API endpoint
            params: Query parametreleri
            headers: Özel headers
            
        Returns:
            API response data
        """
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        # Rate limiting
        self._handle_rate_limiting()
        
        start_time = time.time()
        
        try:
            # Headers'ı birleştir
            request_headers = {**self.session.headers}
            if headers:
                request_headers.update(headers)
            
            response = self.session.get(
                url,
                params=params,
                headers=request_headers
            )
            
            response_time = time.time() - start_time
            
            # Response'u logla
            log_api_request(
                self.logger.name,
                'GET',
                url,
                response.status_code,
                response_time,
                {'params': params}
            )
            
            # Status code kontrolü
            if response.status_code >= 400:
                raise APIError(
                    f"API isteği başarısız: {response.status_code}",
                    endpoint=endpoint,
                    method='GET',
                    status_code=response.status_code,
                    response_data=response.text
                )
            
            # JSON parse et
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise APIError(
                    f"JSON parse hatası: {e}",
                    endpoint=endpoint,
                    method='GET',
                    status_code=response.status_code,
                    response_data=response.text
                )
            
            self.logger.info(f"API verisi alındı: {endpoint} - {len(str(data))} bytes")
            return data
            
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"API isteği zaman aşımı: {endpoint}",
                operation="api_request",
                timeout_duration=self.config.timeout
            )
        except requests.exceptions.ConnectionError as e:
            raise RetryableError(
                f"Bağlantı hatası: {e}",
                max_retries=3,
                retry_delay=2.0
            )
        except Exception as e:
            raise DataCollectionError(
                f"API veri toplama hatası: {e}",
                source="api",
                url=url
            )
    
    @retry_on_failure(max_retries=3, delay=1.0, backoff_factor=2.0)
    def post_data(self, endpoint: str, data: Dict[str, Any], 
                  params: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        POST isteği ile veri gönder
        
        Args:
            endpoint: API endpoint
            data: Gönderilecek veri
            params: Query parametreleri
            headers: Özel headers
            
        Returns:
            API response data
        """
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        # Rate limiting
        self._handle_rate_limiting()
        
        start_time = time.time()
        
        try:
            # Headers'ı birleştir
            request_headers = {**self.session.headers}
            if headers:
                request_headers.update(headers)
            
            response = self.session.post(
                url,
                json=data,
                params=params,
                headers=request_headers
            )
            
            response_time = time.time() - start_time
            
            # Response'u logla
            log_api_request(
                self.logger.name,
                'POST',
                url,
                response.status_code,
                response_time,
                {'data_size': len(str(data))}
            )
            
            # Status code kontrolü
            if response.status_code >= 400:
                raise APIError(
                    f"API isteği başarısız: {response.status_code}",
                    endpoint=endpoint,
                    method='POST',
                    status_code=response.status_code,
                    response_data=response.text
                )
            
            # JSON parse et
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                raise APIError(
                    f"JSON parse hatası: {e}",
                    endpoint=endpoint,
                    method='POST',
                    status_code=response.status_code,
                    response_data=response.text
                )
            
            self.logger.info(f"API verisi gönderildi: {endpoint}")
            return response_data
            
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"API isteği zaman aşımı: {endpoint}",
                operation="api_request",
                timeout_duration=self.config.timeout
            )
        except requests.exceptions.ConnectionError as e:
            raise RetryableError(
                f"Bağlantı hatası: {e}",
                max_retries=3,
                retry_delay=2.0
            )
        except Exception as e:
            raise DataCollectionError(
                f"API veri gönderme hatası: {e}",
                source="api",
                url=url
            )
    
    def get_paginated_data(self, endpoint: str, 
                          page_param: str = 'page',
                          size_param: str = 'size',
                          page_size: int = 100,
                          max_pages: int = None,
                          data_key: str = None,
                          params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Sayfalanmış veriyi topla
        
        Args:
            endpoint: API endpoint
            page_param: Sayfa parametresi adı
            size_param: Sayfa boyutu parametresi adı
            page_size: Sayfa başına kayıt sayısı
            max_pages: Maksimum sayfa sayısı
            data_key: Response'daki veri anahtarı
            params: Ek parametreler
            
        Returns:
            Tüm sayfalardan toplanan veri
        """
        all_data = []
        page = 1
        
        with performance_context(self.logger.name, f"paginated_data_collection_{endpoint}"):
            while True:
                # Sayfa parametrelerini hazırla
                page_params = {
                    page_param: page,
                    size_param: page_size
                }
                if params:
                    page_params.update(params)
                
                try:
                    # Sayfa verisini al
                    response_data = self.get_data(endpoint, params=page_params)
                    
                    # Veriyi çıkar
                    if data_key:
                        page_data = response_data.get(data_key, [])
                    else:
                        page_data = response_data if isinstance(response_data, list) else []
                    
                    if not page_data:
                        break
                    
                    all_data.extend(page_data)
                    self.logger.info(f"Sayfa {page} alındı: {len(page_data)} kayıt")
                    
                    # Maksimum sayfa kontrolü
                    if max_pages and page >= max_pages:
                        break
                    
                    page += 1
                    
                except Exception as e:
                    self.logger.error(f"Sayfa {page} alınırken hata: {e}")
                    break
        
        self.logger.info(f"Toplam {len(all_data)} kayıt toplandı ({page-1} sayfa)")
        return all_data
    
    def get_streaming_data(self, endpoint: str, 
                          chunk_size: int = 1024,
                          params: Optional[Dict[str, Any]] = None) -> bytes:
        """
        Streaming veri al (büyük dosyalar için)
        
        Args:
            endpoint: API endpoint
            chunk_size: Chunk boyutu
            params: Query parametreleri
            
        Returns:
            Streaming veri
        """
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.get(url, params=params, stream=True)
            
            if response.status_code >= 400:
                raise APIError(
                    f"Streaming API isteği başarısız: {response.status_code}",
                    endpoint=endpoint,
                    method='GET',
                    status_code=response.status_code
                )
            
            data = b''
            total_size = 0
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    data += chunk
                    total_size += len(chunk)
            
            self.logger.info(f"Streaming veri alındı: {endpoint} - {total_size} bytes")
            return data
            
        except Exception as e:
            raise DataCollectionError(
                f"Streaming veri toplama hatası: {e}",
                source="api",
                url=url
            )
    
    async def get_data_async(self, endpoint: str, 
                           params: Optional[Dict[str, Any]] = None,
                           headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Asenkron GET isteği
        
        Args:
            endpoint: API endpoint
            params: Query parametreleri
            headers: Özel headers
            
        Returns:
            API response data
        """
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        async with aiohttp.ClientSession() as session:
            try:
                # Headers'ı birleştir
                request_headers = {**self.session.headers}
                if headers:
                    request_headers.update(headers)
                
                start_time = time.time()
                
                async with session.get(url, params=params, headers=request_headers) as response:
                    response_time = time.time() - start_time
                    
                    # Response'u logla
                    log_api_request(
                        self.logger.name,
                        'GET',
                        url,
                        response.status,
                        response_time,
                        {'params': params}
                    )
                    
                    if response.status >= 400:
                        raise APIError(
                            f"Async API isteği başarısız: {response.status}",
                            endpoint=endpoint,
                            method='GET',
                            status_code=response.status
                        )
                    
                    data = await response.json()
                    self.logger.info(f"Async API verisi alındı: {endpoint}")
                    return data
                    
            except Exception as e:
                raise DataCollectionError(
                    f"Async API veri toplama hatası: {e}",
                    source="api",
                    url=url
                )
    
    async def get_multiple_data_async(self, endpoints: List[str],
                                    params_list: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Çoklu endpoint'ten asenkron veri al
        
        Args:
            endpoints: API endpoint listesi
            params_list: Her endpoint için parametre listesi
            
        Returns:
            Her endpoint için response data listesi
        """
        if params_list is None:
            params_list = [{}] * len(endpoints)
        
        tasks = []
        for endpoint, params in zip(endpoints, params_list):
            task = self.get_data_async(endpoint, params)
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Hataları kontrol et
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Endpoint {endpoints[i]} hatası: {result}")
                else:
                    valid_results.append(result)
            
            return valid_results
            
        except Exception as e:
            raise DataCollectionError(
                f"Çoklu async API veri toplama hatası: {e}",
                source="api"
            )
    
    def _handle_rate_limiting(self):
        """Rate limiting kontrolü"""
        if self._rate_limit_delay > 0:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            
            if time_since_last < self._rate_limit_delay:
                sleep_time = self._rate_limit_delay - time_since_last
                self.logger.debug(f"Rate limiting: {sleep_time:.2f} saniye bekle")
                time.sleep(sleep_time)
            
            self._last_request_time = time.time()
    
    def set_rate_limit(self, requests_per_second: float):
        """Rate limit ayarla"""
        if requests_per_second > 0:
            self._rate_limit_delay = 1.0 / requests_per_second
        else:
            self._rate_limit_delay = 0
    
    def set_auth_token(self, token: str, auth_type: str = 'Bearer'):
        """Authentication token ayarla"""
        self._auth_headers['Authorization'] = f'{auth_type} {token}'
        self.session.headers.update(self._auth_headers)
    
    def set_api_key(self, api_key: str, header_name: str = 'X-API-Key'):
        """API key ayarla"""
        self._auth_headers[header_name] = api_key
        self.session.headers.update(self._auth_headers)
    
    def clear_auth(self):
        """Authentication bilgilerini temizle"""
        self._auth_headers = {}
        # Default headers'ı geri yükle
        self._setup_session()
    
    def save_response_to_file(self, response_data: Any, filepath: str, 
                            format: str = 'json') -> str:
        """
        Response'u dosyaya kaydet
        
        Args:
            response_data: Kaydedilecek veri
            filepath: Dosya yolu
            format: Dosya formatı (json, csv, pickle)
            
        Returns:
            Kaydedilen dosya yolu
        """
        from ...core.utils import write_file
        
        try:
            if format == 'csv' and isinstance(response_data, list):
                # List of dicts'i DataFrame'e çevir
                df = pd.DataFrame(response_data)
                write_file(df, filepath, '.csv')
            else:
                write_file(response_data, filepath, f'.{format}')
            
            self.logger.info(f"Response kaydedildi: {filepath}")
            return filepath
            
        except Exception as e:
            raise DataCollectionError(
                f"Response kaydetme hatası: {e}",
                source="file_system"
            )
    
    def close(self):
        """Session'ı kapat"""
        if self.session:
            self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience functions
def collect_api_data(endpoint: str, base_url: str = None, 
                    api_key: str = None, **kwargs) -> Dict[str, Any]:
    """Hızlı API veri toplama"""
    config = get_api_config()
    if base_url:
        config.base_url = base_url
    if api_key:
        config.api_key = api_key
    
    with APICollector(config) as collector:
        return collector.get_data(endpoint, **kwargs)


def collect_paginated_api_data(endpoint: str, base_url: str = None,
                             api_key: str = None, **kwargs) -> List[Dict[str, Any]]:
    """Hızlı sayfalanmış API veri toplama"""
    config = get_api_config()
    if base_url:
        config.base_url = base_url
    if api_key:
        config.api_key = api_key
    
    with APICollector(config) as collector:
        return collector.get_paginated_data(endpoint, **kwargs)


if __name__ == "__main__":
    # Test
    logger = get_logger(__name__)
    
    # Örnek kullanım
    try:
        # JSONPlaceholder API test
        collector = APICollector()
        
        # Basit GET isteği
        posts = collector.get_data('posts', params={'_limit': 5})
        logger.info(f"5 post alındı: {len(posts)}")
        
        # Sayfalanmış veri
        all_posts = collector.get_paginated_data('posts', page_size=10, max_pages=3)
        logger.info(f"Toplam {len(all_posts)} post alındı")
        
        # POST isteği
        new_post = {
            'title': 'Test Post',
            'body': 'Test content',
            'userId': 1
        }
        response = collector.post_data('posts', new_post)
        logger.info(f"Yeni post oluşturuldu: {response.get('id')}")
        
        collector.close()
        
    except Exception as e:
        logger.error(f"Test hatası: {e}") 