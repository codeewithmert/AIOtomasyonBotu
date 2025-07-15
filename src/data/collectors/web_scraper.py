"""
Web Scraper
Web sayfalarından veri toplama modülü - statik ve dinamik içerik desteği
"""

import requests
import time
import random
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging
import json

# Web scraping libraries
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from ...core.logger import get_logger, performance_context
from ...core.exceptions import DataCollectionError, TimeoutError, ValidationError
from ...core.utils import retry_on_failure, validate_url, safe_filename
from ...core.config import get_config


class WebScraper:
    """
    Web scraping sınıfı
    Statik ve dinamik içerik desteği
    """
    
    def __init__(self, headless: bool = True, browser: str = 'chrome', 
                 user_agent: str = None, timeout: int = 30):
        self.logger = get_logger(__name__)
        self.headless = headless
        self.browser = browser.lower()
        self.timeout = timeout
        self.user_agent = user_agent or self._get_default_user_agent()
        
        self.driver = None
        self.session = requests.Session()
        self._setup_session()
    
    def _get_default_user_agent(self) -> str:
        """Varsayılan user agent"""
        return 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    
    def _setup_session(self):
        """HTTP session kur"""
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def _setup_driver(self):
        """Selenium driver kur"""
        if self.browser == 'chrome':
            options = Options()
            if self.headless:
                options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument(f'--user-agent={self.user_agent}')
            options.add_argument('--window-size=1920,1080')
            
            self.driver = webdriver.Chrome(options=options)
            
        elif self.browser == 'firefox':
            options = FirefoxOptions()
            if self.headless:
                options.add_argument('--headless')
            options.add_argument(f'--user-agent={self.user_agent}')
            
            self.driver = webdriver.Firefox(options=options)
            
        else:
            raise ValueError(f"Desteklenmeyen browser: {self.browser}")
        
        self.driver.set_page_load_timeout(self.timeout)
        self.driver.implicitly_wait(10)
    
    @retry_on_failure(max_retries=3, delay=2.0, backoff_factor=2.0)
    def get_static_content(self, url: str, params: Optional[Dict[str, Any]] = None,
                          headers: Optional[Dict[str, str]] = None) -> str:
        """
        Statik içerik al (requests + BeautifulSoup)
        
        Args:
            url: Web sayfası URL'i
            params: Query parametreleri
            headers: Özel headers
            
        Returns:
            HTML içeriği
        """
        validate_url(url)
        
        try:
            # Headers'ı birleştir
            request_headers = {**self.session.headers}
            if headers:
                request_headers.update(headers)
            
            response = self.session.get(url, params=params, headers=request_headers, timeout=self.timeout)
            response.raise_for_status()
            
            self.logger.info(f"Statik içerik alındı: {url} - {len(response.text)} karakter")
            return response.text
            
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Web sayfası zaman aşımı: {url}",
                operation="web_scraping",
                timeout_duration=self.timeout
            )
        except requests.exceptions.RequestException as e:
            raise DataCollectionError(
                f"Statik içerik alma hatası: {e}",
                source="web_scraping",
                url=url
            )
    
    @retry_on_failure(max_retries=3, delay=2.0, backoff_factor=2.0)
    def get_dynamic_content(self, url: str, wait_for: Optional[str] = None,
                           wait_time: int = 10) -> str:
        """
        Dinamik içerik al (Selenium)
        
        Args:
            url: Web sayfası URL'i
            wait_for: Beklenecek element (CSS selector)
            wait_time: Maksimum bekleme süresi
            
        Returns:
            HTML içeriği
        """
        validate_url(url)
        
        if self.driver is None:
            self._setup_driver()
        
        try:
            self.driver.get(url)
            
            # Belirli element için bekle
            if wait_for:
                WebDriverWait(self.driver, wait_time).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_for))
                )
            
            # Sayfa yüklenmesi için kısa bekle
            time.sleep(2)
            
            page_source = self.driver.page_source
            self.logger.info(f"Dinamik içerik alındı: {url} - {len(page_source)} karakter")
            return page_source
            
        except TimeoutException:
            raise TimeoutError(
                f"Sayfa yükleme zaman aşımı: {url}",
                operation="web_scraping",
                timeout_duration=wait_time
            )
        except Exception as e:
            raise DataCollectionError(
                f"Dinamik içerik alma hatası: {e}",
                source="web_scraping",
                url=url
            )
    
    def extract_text(self, html_content: str, selector: str = None) -> str:
        """
        HTML'den metin çıkar
        
        Args:
            html_content: HTML içeriği
            selector: CSS selector (None ise tüm metin)
            
        Returns:
            Çıkarılan metin
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        if selector:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
            return ""
        else:
            return soup.get_text(strip=True)
    
    def extract_links(self, html_content: str, base_url: str = None) -> List[Dict[str, str]]:
        """
        HTML'den linkleri çıkar
        
        Args:
            html_content: HTML içeriği
            base_url: Base URL (relative linkler için)
            
        Returns:
            Link listesi
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)
            
            # Relative URL'leri absolute yap
            if base_url and href.startswith('/'):
                href = f"{base_url.rstrip('/')}{href}"
            elif base_url and not href.startswith(('http://', 'https://')):
                href = f"{base_url.rstrip('/')}/{href.lstrip('/')}"
            
            links.append({
                'url': href,
                'text': text,
                'title': link.get('title', '')
            })
        
        return links
    
    def extract_images(self, html_content: str, base_url: str = None) -> List[Dict[str, str]]:
        """
        HTML'den resimleri çıkar
        
        Args:
            html_content: HTML içeriği
            base_url: Base URL (relative linkler için)
            
        Returns:
            Resim listesi
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        images = []
        
        for img in soup.find_all('img'):
            src = img.get('src', '')
            alt = img.get('alt', '')
            
            # Relative URL'leri absolute yap
            if base_url and src.startswith('/'):
                src = f"{base_url.rstrip('/')}{src}"
            elif base_url and not src.startswith(('http://', 'https://')):
                src = f"{base_url.rstrip('/')}/{src.lstrip('/')}"
            
            images.append({
                'src': src,
                'alt': alt,
                'title': img.get('title', ''),
                'width': img.get('width', ''),
                'height': img.get('height', '')
            })
        
        return images
    
    def extract_table_data(self, html_content: str, table_selector: str = None) -> List[Dict[str, str]]:
        """
        HTML tablosundan veri çıkar
        
        Args:
            html_content: HTML içeriği
            table_selector: Tablo CSS selector
            
        Returns:
            Tablo verisi
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        if table_selector:
            table = soup.select_one(table_selector)
        else:
            table = soup.find('table')
        
        if not table:
            return []
        
        # Header'ları al
        headers = []
        header_row = table.find('thead')
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
        else:
            # İlk satırı header olarak kullan
            first_row = table.find('tr')
            if first_row:
                headers = [th.get_text(strip=True) for th in first_row.find_all(['th', 'td'])]
        
        if not headers:
            return []
        
        # Veri satırlarını al
        data = []
        rows = table.find_all('tr')[1:] if header_row else table.find_all('tr')
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) == len(headers):
                row_data = {}
                for i, cell in enumerate(cells):
                    row_data[headers[i]] = cell.get_text(strip=True)
                data.append(row_data)
        
        return data
    
    def extract_json_data(self, html_content: str, script_selector: str = None) -> List[Dict[str, Any]]:
        """
        HTML'den JSON verisi çıkar
        
        Args:
            html_content: HTML içeriği
            script_selector: Script tag CSS selector
            
        Returns:
            JSON verisi listesi
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        json_data = []
        
        # Script tag'lerini bul
        scripts = soup.find_all('script')
        if script_selector:
            scripts = soup.select(script_selector)
        
        for script in scripts:
            script_content = script.string
            if not script_content:
                continue
            
            # JSON pattern'lerini ara
            json_patterns = [
                r'window\.__INITIAL_STATE__\s*=\s*({.*?});',
                r'window\.__PRELOADED_STATE__\s*=\s*({.*?});',
                r'var\s+data\s*=\s*({.*?});',
                r'const\s+data\s*=\s*({.*?});',
                r'let\s+data\s*=\s*({.*?});',
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, script_content, re.DOTALL)
                for match in matches:
                    try:
                        data = json.loads(match)
                        json_data.append(data)
                    except json.JSONDecodeError:
                        continue
        
        return json_data
    
    def scrape_page(self, url: str, selectors: Dict[str, str], 
                   use_selenium: bool = False, wait_for: str = None) -> Dict[str, Any]:
        """
        Sayfadan belirli elementleri çıkar
        
        Args:
            url: Web sayfası URL'i
            selectors: CSS selector'lar ve çıkarılacak veri türleri
            use_selenium: Selenium kullanılsın mı
            wait_for: Beklenecek element
            
        Returns:
            Çıkarılan veri
        """
        # HTML içeriğini al
        if use_selenium:
            html_content = self.get_dynamic_content(url, wait_for=wait_for)
        else:
            html_content = self.get_static_content(url)
        
        # Veriyi çıkar
        extracted_data = {}
        
        for key, selector in selectors.items():
            try:
                if selector.startswith('text:'):
                    # Metin çıkar
                    css_selector = selector[5:]
                    extracted_data[key] = self.extract_text(html_content, css_selector)
                
                elif selector.startswith('attr:'):
                    # Attribute çıkar
                    parts = selector[5:].split(':', 1)
                    if len(parts) == 2:
                        css_selector, attr_name = parts
                        soup = BeautifulSoup(html_content, 'html.parser')
                        element = soup.select_one(css_selector)
                        if element:
                            extracted_data[key] = element.get(attr_name, '')
                
                elif selector.startswith('links:'):
                    # Linkler çıkar
                    css_selector = selector[6:] if len(selector) > 6 else None
                    soup = BeautifulSoup(html_content, 'html.parser')
                    if css_selector:
                        container = soup.select_one(css_selector)
                        if container:
                            extracted_data[key] = self.extract_links(str(container), url)
                    else:
                        extracted_data[key] = self.extract_links(html_content, url)
                
                elif selector.startswith('table:'):
                    # Tablo verisi çıkar
                    css_selector = selector[6:] if len(selector) > 6 else None
                    extracted_data[key] = self.extract_table_data(html_content, css_selector)
                
                else:
                    # Varsayılan olarak metin çıkar
                    extracted_data[key] = self.extract_text(html_content, selector)
                    
            except Exception as e:
                self.logger.warning(f"Selector '{selector}' için veri çıkarılamadı: {e}")
                extracted_data[key] = None
        
        return extracted_data
    
    def scrape_multiple_pages(self, urls: List[str], selectors: Dict[str, str],
                            use_selenium: bool = False, delay: float = 1.0) -> List[Dict[str, Any]]:
        """
        Çoklu sayfadan veri çıkar
        
        Args:
            urls: URL listesi
            selectors: CSS selector'lar
            use_selenium: Selenium kullanılsın mı
            delay: Sayfalar arası bekleme süresi
            
        Returns:
            Çıkarılan veri listesi
        """
        results = []
        
        for i, url in enumerate(urls):
            try:
                self.logger.info(f"Sayfa {i+1}/{len(urls)} işleniyor: {url}")
                
                data = self.scrape_page(url, selectors, use_selenium)
                data['url'] = url
                data['scraped_at'] = datetime.now().isoformat()
                results.append(data)
                
                # Rate limiting
                if i < len(urls) - 1 and delay > 0:
                    time.sleep(delay + random.uniform(0, 1))
                    
            except Exception as e:
                self.logger.error(f"Sayfa {url} işlenirken hata: {e}")
                results.append({
                    'url': url,
                    'error': str(e),
                    'scraped_at': datetime.now().isoformat()
                })
        
        return results
    
    def save_scraped_data(self, data: List[Dict[str, Any]], filepath: str,
                         format: str = 'json') -> str:
        """
        Çıkarılan veriyi dosyaya kaydet
        
        Args:
            data: Kaydedilecek veri
            filepath: Dosya yolu
            format: Dosya formatı (json, csv, excel)
            
        Returns:
            Kaydedilen dosya yolu
        """
        from ...core.utils import write_file
        
        try:
            if format == 'csv':
                df = pd.DataFrame(data)
                write_file(df, filepath, '.csv')
            elif format == 'excel':
                df = pd.DataFrame(data)
                write_file(df, filepath, '.xlsx')
            else:
                write_file(data, filepath, '.json')
            
            self.logger.info(f"Çıkarılan veri kaydedildi: {filepath}")
            return filepath
            
        except Exception as e:
            raise DataCollectionError(
                f"Veri kaydetme hatası: {e}",
                source="file_system"
            )
    
    def take_screenshot(self, url: str, filepath: str, 
                       wait_for: str = None, full_page: bool = True) -> str:
        """
        Sayfa ekran görüntüsü al
        
        Args:
            url: Web sayfası URL'i
            filepath: Kaydedilecek dosya yolu
            wait_for: Beklenecek element
            full_page: Tam sayfa ekran görüntüsü alınsın mı
            
        Returns:
            Kaydedilen dosya yolu
        """
        if self.driver is None:
            self._setup_driver()
        
        try:
            self.driver.get(url)
            
            if wait_for:
                WebDriverWait(self.driver, self.timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_for))
                )
            
            time.sleep(2)
            
            if full_page:
                # Tam sayfa ekran görüntüsü
                self.driver.save_screenshot(filepath)
            else:
                # Görünür alan ekran görüntüsü
                self.driver.save_screenshot(filepath)
            
            self.logger.info(f"Ekran görüntüsü alındı: {filepath}")
            return filepath
            
        except Exception as e:
            raise DataCollectionError(
                f"Ekran görüntüsü alma hatası: {e}",
                source="web_scraping",
                url=url
            )
    
    def close(self):
        """Driver'ı kapat"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience functions
def scrape_webpage(url: str, selectors: Dict[str, str], 
                  use_selenium: bool = False, **kwargs) -> Dict[str, Any]:
    """Hızlı web scraping"""
    with WebScraper(**kwargs) as scraper:
        return scraper.scrape_page(url, selectors, use_selenium)


def scrape_multiple_webpages(urls: List[str], selectors: Dict[str, str],
                           use_selenium: bool = False, **kwargs) -> List[Dict[str, Any]]:
    """Hızlı çoklu web scraping"""
    with WebScraper(**kwargs) as scraper:
        return scraper.scrape_multiple_pages(urls, selectors, use_selenium)


if __name__ == "__main__":
    # Test
    logger = get_logger(__name__)
    
    # Örnek kullanım
    try:
        scraper = WebScraper(headless=True)
        
        # Basit scraping
        url = "https://httpbin.org/html"
        selectors = {
            'title': 'h1',
            'content': 'p',
            'links': 'links:'
        }
        
        data = scraper.scrape_page(url, selectors)
        logger.info(f"Çıkarılan veri: {data}")
        
        # Ekran görüntüsü
        screenshot_path = scraper.take_screenshot(url, "screenshot.png")
        logger.info(f"Ekran görüntüsü: {screenshot_path}")
        
        scraper.close()
        
    except Exception as e:
        logger.error(f"Test hatası: {e}") 