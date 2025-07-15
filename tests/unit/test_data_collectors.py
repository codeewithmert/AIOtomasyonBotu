"""
Unit Tests for Data Collectors

Tests for API and web scraping data collection functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.data.collectors.api_collector import APICollector
from src.data.collectors.web_scraper import WebScraper
from src.core.exceptions import DataProcessingError


class TestAPICollector(unittest.TestCase):
    """Test cases for APICollector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = APICollector()
        self.test_url = "https://api.example.com/data"
        self.test_headers = {"Authorization": "Bearer test-token"}
        self.test_params = {"limit": 10, "offset": 0}
    
    def test_init(self):
        """Test APICollector initialization."""
        self.assertIsNotNone(self.collector)
        self.assertEqual(self.collector.timeout, 30)
        self.assertEqual(self.collector.max_retries, 3)
    
    def test_get_data_success(self):
        """Test successful GET data collection."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": 1, "name": "test"}]}
        mock_response.raise_for_status.return_value = None
        
        with patch('requests.get', return_value=mock_response):
            result = self.collector.get_data(
                self.test_url, 
                headers=self.test_headers, 
                params=self.test_params
            )
        
        self.assertEqual(result, {"data": [{"id": 1, "name": "test"}]})
    
    def test_get_data_http_error(self):
        """Test GET data collection with HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.HTTPError("Not Found")
        
        with patch('requests.get', return_value=mock_response):
            with self.assertRaises(DataProcessingError):
                self.collector.get_data(self.test_url)
    
    def test_post_data_success(self):
        """Test successful POST data collection."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"status": "created", "id": 123}
        mock_response.raise_for_status.return_value = None
        
        test_data = {"name": "test", "value": 42}
        
        with patch('requests.post', return_value=mock_response):
            result = self.collector.post_data(
                self.test_url, 
                headers=self.test_headers, 
                data=test_data
            )
        
        self.assertEqual(result, {"status": "created", "id": 123})
    
    def test_post_data_connection_error(self):
        """Test POST data collection with connection error."""
        with patch('requests.post', side_effect=requests.ConnectionError("Connection failed")):
            with self.assertRaises(DataProcessingError):
                self.collector.post_data(self.test_url, data={})
    
    def test_retry_mechanism(self):
        """Test retry mechanism for failed requests."""
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"data": "success"}
        mock_response_success.raise_for_status.return_value = None
        
        mock_response_failure = Mock()
        mock_response_failure.status_code = 500
        mock_response_failure.raise_for_status.side_effect = requests.HTTPError("Server Error")
        
        with patch('requests.get', side_effect=[mock_response_failure, mock_response_failure, mock_response_success]):
            result = self.collector.get_data(self.test_url)
        
        self.assertEqual(result, {"data": "success"})
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status.return_value = None
        
        with patch('requests.get', return_value=mock_response):
            with patch('time.sleep') as mock_sleep:
                self.collector.get_data(self.test_url)
                self.collector.get_data(self.test_url)
                
                # Should have slept between requests
                mock_sleep.assert_called()


class TestWebScraper(unittest.TestCase):
    """Test cases for WebScraper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scraper = WebScraper()
        self.test_url = "https://example.com"
        self.test_html = """
        <html>
            <body>
                <h1>Test Title</h1>
                <div class="content">
                    <p>Test content</p>
                    <a href="/link">Test link</a>
                </div>
                <table>
                    <tr><td>Data 1</td><td>Data 2</td></tr>
                    <tr><td>Data 3</td><td>Data 4</td></tr>
                </table>
            </body>
        </html>
        """
    
    def test_init(self):
        """Test WebScraper initialization."""
        self.assertIsNotNone(self.scraper)
        self.assertEqual(self.scraper.timeout, 30)
        self.assertIsNone(self.scraper.driver)
    
    def test_scrape_page_static(self):
        """Test static page scraping."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = self.test_html
        mock_response.raise_for_status.return_value = None
        
        with patch('requests.get', return_value=mock_response):
            result = self.scraper.scrape_page(self.test_url)
        
        self.assertIsInstance(result, dict)
        self.assertIn('title', result)
        self.assertIn('content', result)
    
    def test_scrape_page_with_selectors(self):
        """Test page scraping with custom selectors."""
        selectors = {
            'title': 'h1',
            'content': '.content p',
            'links': '.content a',
            'table_data': 'table tr'
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = self.test_html
        mock_response.raise_for_status.return_value = None
        
        with patch('requests.get', return_value=mock_response):
            result = self.scraper.scrape_page(self.test_url, selectors=selectors)
        
        self.assertEqual(result['title'], 'Test Title')
        self.assertIn('Test content', result['content'])
        self.assertIn('Test link', result['links'])
    
    def test_scrape_page_http_error(self):
        """Test page scraping with HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.HTTPError("Not Found")
        
        with patch('requests.get', return_value=mock_response):
            with self.assertRaises(DataProcessingError):
                self.scraper.scrape_page(self.test_url)
    
    def test_extract_text(self):
        """Test text extraction from HTML."""
        soup = BeautifulSoup(self.test_html, 'html.parser')
        text = self.scraper._extract_text(soup.find('div', class_='content'))
        
        self.assertIn('Test content', text)
        self.assertIn('Test link', text)
    
    def test_extract_links(self):
        """Test link extraction from HTML."""
        soup = BeautifulSoup(self.test_html, 'html.parser')
        links = self.scraper._extract_links(soup)
        
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]['text'], 'Test link')
        self.assertEqual(links[0]['href'], '/link')
    
    def test_extract_table_data(self):
        """Test table data extraction from HTML."""
        soup = BeautifulSoup(self.test_html, 'html.parser')
        table_data = self.scraper._extract_table_data(soup.find('table'))
        
        self.assertEqual(len(table_data), 2)
        self.assertEqual(table_data[0], ['Data 1', 'Data 2'])
        self.assertEqual(table_data[1], ['Data 3', 'Data 4'])
    
    def test_take_screenshot(self):
        """Test screenshot functionality."""
        with patch('selenium.webdriver.Chrome') as mock_driver:
            mock_driver_instance = Mock()
            mock_driver.return_value = mock_driver_instance
            
            screenshot_path = self.scraper.take_screenshot(self.test_url)
            
            mock_driver_instance.get.assert_called_with(self.test_url)
            mock_driver_instance.save_screenshot.assert_called()
            mock_driver_instance.quit.assert_called()
            
            self.assertIsInstance(screenshot_path, str)
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = "  Test   text  with   extra   spaces  "
        clean_text = self.scraper._clean_text(dirty_text)
        
        self.assertEqual(clean_text, "Test text with extra spaces")
    
    def test_validate_url(self):
        """Test URL validation."""
        valid_url = "https://example.com"
        invalid_url = "not-a-url"
        
        self.assertTrue(self.scraper._validate_url(valid_url))
        self.assertFalse(self.scraper._validate_url(invalid_url))


class TestDataCollectorsIntegration(unittest.TestCase):
    """Integration tests for data collectors."""
    
    def test_api_collector_with_real_endpoint(self):
        """Test API collector with a real test endpoint."""
        collector = APICollector()
        
        # Using a public test API
        test_url = "https://jsonplaceholder.typicode.com/posts/1"
        
        try:
            result = collector.get_data(test_url)
            self.assertIsInstance(result, dict)
            self.assertIn('id', result)
            self.assertIn('title', result)
        except DataProcessingError:
            # Skip if network is not available
            self.skipTest("Network not available for integration test")
    
    def test_web_scraper_with_real_page(self):
        """Test web scraper with a real test page."""
        scraper = WebScraper()
        
        # Using a simple test page
        test_url = "https://httpbin.org/html"
        
        try:
            result = scraper.scrape_page(test_url)
            self.assertIsInstance(result, dict)
            self.assertIn('title', result)
        except DataProcessingError:
            # Skip if network is not available
            self.skipTest("Network not available for integration test")


if __name__ == '__main__':
    unittest.main() 