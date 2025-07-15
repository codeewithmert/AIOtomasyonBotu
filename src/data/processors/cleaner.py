"""
Data Cleaner

Provides data cleaning and preprocessing functionality for collected data
in the AI Automation Bot.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import unicodedata

from ...core.exceptions import DataProcessingError
from ...core.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """Data cleaner for cleaning and preprocessing collected data."""
    
    def __init__(self):
        """Initialize data cleaner."""
        self.cleaning_stats = {}
    
    def clean_dataframe(self, df: pd.DataFrame, cleaning_config: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Clean a pandas DataFrame.
        
        Args:
            df: Input DataFrame
            cleaning_config: Cleaning configuration
            
        Returns:
            Cleaned DataFrame
            
        Raises:
            DataProcessingError: If cleaning fails
        """
        try:
            if df.empty:
                logger.warning("Empty DataFrame provided for cleaning")
                return df
            
            # Reset cleaning stats
            self.cleaning_stats = {
                'original_rows': len(df),
                'original_columns': len(df.columns),
                'removed_rows': 0,
                'removed_columns': 0,
                'cleaned_columns': []
            }
            
            # Apply cleaning operations
            df_cleaned = df.copy()
            
            # Remove duplicates
            if cleaning_config.get('remove_duplicates', True):
                df_cleaned = self._remove_duplicates(df_cleaned)
            
            # Handle missing values
            if cleaning_config.get('handle_missing', True):
                df_cleaned = self._handle_missing_values(df_cleaned, cleaning_config)
            
            # Clean text columns
            if cleaning_config.get('clean_text', True):
                df_cleaned = self._clean_text_columns(df_cleaned, cleaning_config)
            
            # Remove outliers
            if cleaning_config.get('remove_outliers', False):
                df_cleaned = self._remove_outliers(df_cleaned, cleaning_config)
            
            # Standardize data types
            if cleaning_config.get('standardize_types', True):
                df_cleaned = self._standardize_data_types(df_cleaned, cleaning_config)
            
            # Update cleaning stats
            self.cleaning_stats['final_rows'] = len(df_cleaned)
            self.cleaning_stats['final_columns'] = len(df_cleaned.columns)
            self.cleaning_stats['removed_rows'] = self.cleaning_stats['original_rows'] - self.cleaning_stats['final_rows']
            
            logger.info(f"Data cleaning completed. Removed {self.cleaning_stats['removed_rows']} rows")
            
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Data cleaning error: {e}")
            raise DataProcessingError(f"Failed to clean data: {str(e)}")
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows from DataFrame."""
        try:
            original_count = len(df)
            df_cleaned = df.drop_duplicates()
            removed_count = original_count - len(df_cleaned)
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} duplicate rows")
            
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Duplicate removal error: {e}")
            raise DataProcessingError(f"Failed to remove duplicates: {str(e)}")
    
    def _handle_missing_values(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Handle missing values in DataFrame."""
        try:
            missing_strategy = config.get('missing_strategy', 'drop')
            fill_value = config.get('fill_value', None)
            threshold = config.get('missing_threshold', 0.5)
            
            if missing_strategy == 'drop':
                # Drop rows with too many missing values
                df_cleaned = df.dropna(thresh=len(df.columns) * (1 - threshold))
                
                # Drop columns with too many missing values
                df_cleaned = df_cleaned.dropna(axis=1, thresh=len(df_cleaned) * (1 - threshold))
                
            elif missing_strategy == 'fill':
                # Fill missing values
                if fill_value is not None:
                    df_cleaned = df.fillna(fill_value)
                else:
                    # Use appropriate fill methods for different data types
                    df_cleaned = df.copy()
                    
                    for column in df_cleaned.columns:
                        if df_cleaned[column].dtype in ['int64', 'float64']:
                            # Fill numeric columns with median
                            df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].median())
                        elif df_cleaned[column].dtype == 'object':
                            # Fill categorical columns with mode
                            mode_value = df_cleaned[column].mode()
                            if not mode_value.empty:
                                df_cleaned[column] = df_cleaned[column].fillna(mode_value[0])
                            else:
                                df_cleaned[column] = df_cleaned[column].fillna('Unknown')
            
            removed_count = len(df) - len(df_cleaned)
            if removed_count > 0:
                logger.info(f"Handled missing values: removed {removed_count} rows")
            
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Missing value handling error: {e}")
            raise DataProcessingError(f"Failed to handle missing values: {str(e)}")
    
    def _clean_text_columns(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Clean text columns in DataFrame."""
        try:
            df_cleaned = df.copy()
            text_columns = df_cleaned.select_dtypes(include=['object']).columns
            
            for column in text_columns:
                if df_cleaned[column].dtype == 'object':
                    df_cleaned[column] = df_cleaned[column].astype(str).apply(self._clean_text)
                    self.cleaning_stats['cleaned_columns'].append(column)
            
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Text cleaning error: {e}")
            raise DataProcessingError(f"Failed to clean text columns: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean individual text string."""
        try:
            if pd.isna(text) or text == 'nan':
                return ''
            
            # Convert to string
            text = str(text)
            
            # Normalize unicode characters
            text = unicodedata.normalize('NFKC', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            # Remove special characters (keep alphanumeric, spaces, and common punctuation)
            text = re.sub(r'[^\w\s\.\,\!\?\-\_\:\;\(\)\[\]\{\}]', '', text)
            
            # Convert to lowercase
            text = text.lower()
            
            return text
            
        except Exception as e:
            logger.warning(f"Text cleaning failed for '{text}': {e}")
            return str(text)
    
    def _remove_outliers(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Remove outliers from numeric columns."""
        try:
            df_cleaned = df.copy()
            numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
            outlier_method = config.get('outlier_method', 'iqr')
            outlier_threshold = config.get('outlier_threshold', 1.5)
            
            for column in numeric_columns:
                if outlier_method == 'iqr':
                    df_cleaned = self._remove_outliers_iqr(df_cleaned, column, outlier_threshold)
                elif outlier_method == 'zscore':
                    df_cleaned = self._remove_outliers_zscore(df_cleaned, column, outlier_threshold)
            
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Outlier removal error: {e}")
            raise DataProcessingError(f"Failed to remove outliers: {str(e)}")
    
    def _remove_outliers_iqr(self, df: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        try:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            original_count = len(df)
            df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            removed_count = original_count - len(df_cleaned)
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} outliers from column '{column}' using IQR method")
            
            return df_cleaned
            
        except Exception as e:
            logger.warning(f"IQR outlier removal failed for column '{column}': {e}")
            return df
    
    def _remove_outliers_zscore(self, df: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
        """Remove outliers using Z-score method."""
        try:
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            
            original_count = len(df)
            df_cleaned = df[z_scores < threshold]
            removed_count = original_count - len(df_cleaned)
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} outliers from column '{column}' using Z-score method")
            
            return df_cleaned
            
        except Exception as e:
            logger.warning(f"Z-score outlier removal failed for column '{column}': {e}")
            return df
    
    def _standardize_data_types(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Standardize data types in DataFrame."""
        try:
            df_cleaned = df.copy()
            
            # Convert date columns
            date_columns = config.get('date_columns', [])
            for column in date_columns:
                if column in df_cleaned.columns:
                    df_cleaned[column] = pd.to_datetime(df_cleaned[column], errors='coerce')
            
            # Convert numeric columns
            numeric_columns = config.get('numeric_columns', [])
            for column in numeric_columns:
                if column in df_cleaned.columns:
                    df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors='coerce')
            
            # Convert categorical columns
            categorical_columns = config.get('categorical_columns', [])
            for column in categorical_columns:
                if column in df_cleaned.columns:
                    df_cleaned[column] = df_cleaned[column].astype('category')
            
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Data type standardization error: {e}")
            raise DataProcessingError(f"Failed to standardize data types: {str(e)}")
    
    def clean_json_data(self, data: Union[Dict, List], cleaning_config: Dict[str, Any] = None) -> Union[Dict, List]:
        """
        Clean JSON data.
        
        Args:
            data: Input JSON data
            cleaning_config: Cleaning configuration
            
        Returns:
            Cleaned JSON data
        """
        try:
            if isinstance(data, list):
                return [self._clean_json_object(item, cleaning_config) for item in data]
            elif isinstance(data, dict):
                return self._clean_json_object(data, cleaning_config)
            else:
                return data
                
        except Exception as e:
            logger.error(f"JSON cleaning error: {e}")
            raise DataProcessingError(f"Failed to clean JSON data: {str(e)}")
    
    def _clean_json_object(self, obj: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Clean individual JSON object."""
        try:
            cleaned_obj = {}
            
            for key, value in obj.items():
                # Clean key
                cleaned_key = self._clean_text(key) if config.get('clean_keys', True) else key
                
                # Clean value
                if isinstance(value, str):
                    cleaned_value = self._clean_text(value) if config.get('clean_text', True) else value
                elif isinstance(value, dict):
                    cleaned_value = self._clean_json_object(value, config)
                elif isinstance(value, list):
                    cleaned_value = [self._clean_json_object(item, config) if isinstance(item, dict) else item for item in value]
                else:
                    cleaned_value = value
                
                cleaned_obj[cleaned_key] = cleaned_value
            
            return cleaned_obj
            
        except Exception as e:
            logger.warning(f"JSON object cleaning failed: {e}")
            return obj
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """
        Get cleaning statistics and report.
        
        Returns:
            Dictionary with cleaning statistics
        """
        return {
            'cleaning_stats': self.cleaning_stats,
            'summary': {
                'total_rows_removed': self.cleaning_stats.get('removed_rows', 0),
                'total_columns_cleaned': len(self.cleaning_stats.get('cleaned_columns', [])),
                'cleaning_efficiency': self._calculate_cleaning_efficiency()
            }
        }
    
    def _calculate_cleaning_efficiency(self) -> float:
        """Calculate cleaning efficiency percentage."""
        try:
            original_rows = self.cleaning_stats.get('original_rows', 0)
            removed_rows = self.cleaning_stats.get('removed_rows', 0)
            
            if original_rows == 0:
                return 0.0
            
            efficiency = ((original_rows - removed_rows) / original_rows) * 100
            return round(efficiency, 2)
            
        except Exception:
            return 0.0 