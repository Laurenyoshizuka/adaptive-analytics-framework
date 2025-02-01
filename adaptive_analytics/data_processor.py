import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from enum import Enum, auto
import streamlit as st

class DataType(Enum):
    TIME_SERIES = auto()
    CATEGORICAL = auto()
    NUMERICAL = auto()
    MIXED = auto()

class DataProcessor:
    def __init__(self, dataframe: pd.DataFrame):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._validate_input(dataframe)
        
        self.original_df = dataframe
        self.processed_df = None
        self.data_context = self._comprehensive_context_detection()
    
    def _validate_input(self, df: pd.DataFrame):
        """
        Comprehensive input validation
        """
        if df is None or df.empty:
            raise ValueError("Input dataframe cannot be None or empty")
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        # Check minimum data requirements
        if df.shape[0] < 5:
            self.logger.warning("Dataset is very small. Insights may be limited.")
        
        if df.shape[1] < 2:
            raise ValueError("Dataset must have at least 2 columns")
    
    def _comprehensive_context_detection(self) -> Dict[str, Any]:
        """
        Advanced context detection with multiple layers of analysis
        """
        df = self.original_df
        context = {
            'data_type': None,
            'time_column': None,
            'target_columns': [],
            'categorical_columns': [],
            'numerical_columns': [],
            'potential_analysis_types': []
        }

        # Detect column types
        context['numerical_columns'] = list(df.select_dtypes(include=['float64', 'int64']).columns)
        context['categorical_columns'] = list(df.select_dtypes(include=['object', 'category']).columns)
        
        # Time column detection
        time_detection_strategies = [
            df.select_dtypes(include=['datetime64']).columns,
            [col for col in df.columns if 'date' in col.lower()],
            [col for col in df.columns if 'time' in col.lower()]
        ]
        
        # Ensure time column is selected correctly
        for strategy in time_detection_strategies:
            if isinstance(strategy, pd.Index) and not strategy.empty:
                context['time_column'] = strategy[0]
                break
            elif isinstance(strategy, list) and strategy:
                context['time_column'] = strategy[0]
                break
        
        # Target column detection heuristics
        target_keywords = [
            'sales', 'revenue', 'price', 'count', 'value', 
            'amount', 'total', 'metric', 'score'
        ]
        
        context['target_columns'] = [
            col for col in context['numerical_columns']
            if any(keyword in col.lower() for keyword in target_keywords)
        ]
        
        # If no explicit target found, use primary numerical column
        if not context['target_columns'] and context['numerical_columns']:
            context['target_columns'] = [context['numerical_columns'][0]]
        
        # Determine overall data type
        if context['time_column']:
            context['data_type'] = DataType.TIME_SERIES
            context['potential_analysis_types'] = [
                'time_series_decomposition',
                'trend_analysis',
                'seasonality_detection'
            ]
        elif len(context['numerical_columns']) > len(context['categorical_columns']):
            context['data_type'] = DataType.NUMERICAL
            context['potential_analysis_types'] = [
                'correlation_analysis',
                'clustering',
                'regression_prediction'
            ]
        elif len(context['categorical_columns']) > len(context['numerical_columns']):
            context['data_type'] = DataType.CATEGORICAL
            context['potential_analysis_types'] = [
                'category_distribution',
            ]
        else:
            context['data_type'] = DataType.MIXED
            context['potential_analysis_types'] = [
                'multi_dimensional_analysis',
                'feature_interaction_study'
            ]
        
        return context

    def preprocess(self) -> pd.DataFrame:
        """
        Advanced preprocessing with multiple error handling mechanisms 
        for robust date column detection and parsing.
        """
        df = self.original_df.copy()
        context = self.data_context

        try:
            # Handle missing values dynamically
            for col in df.columns:
                missing_percentage = df[col].isnull().mean() * 100
                
                if missing_percentage > 50:
                    df.drop(columns=[col], inplace=True)
                    continue

                if df[col].dtype.kind in ['f', 'i']:  # Float or Integer
                    df[col] = df[col].fillna(df[col].median())

                elif df[col].dtype.kind in ['O', 'c']:  # Object or Category
                    if not df[col].mode().empty:
                        df[col] = df[col].fillna(df[col].mode().iloc[0])
            
            # Dynamic date parsing 
            def parse_dates(df, col):
                """
                Try parsing dates with predefined formats, but only convert if a significant portion are valid dates.
                """
                DATE_FORMATS = [
                    "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y",
                    "%Y/%m/%d", "%Y-%m-%d",
                    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
                    "%Y-%m-%d %H:%M:%S", "%Y%m%d %H:%M:%S"
                ]
                
                original_col = df[col].copy()  # Store original data

                for fmt in DATE_FORMATS:
                    try:
                        parsed_dates = pd.to_datetime(df[col], format=fmt, errors='coerce')
                        if parsed_dates.notnull().mean() > 0.5:  # Convert only if at least 50% are valid dates
                            df[col] = parsed_dates
                            return df[col]
                    except Exception:
                        continue

                return original_col  # If no valid date format is found, keep original values


            for col in df.select_dtypes(include=['object']).columns:
                if col in context['categorical_columns']:  
                    continue
                df[col] = parse_dates(df, col)


            # Validate and clean time column if found
            if context.get('time_column') and context['time_column'] in df.columns:
                df[context['time_column']] = pd.to_datetime(df[context['time_column']], errors='coerce')
                df = df.dropna(subset=[context['time_column']])
                
                if not df.empty:
                    df = df.sort_values(by=context['time_column'])

            # Handle extreme outliers (based on numerical columns)
            for col in context.get('numerical_columns', []):
                if col in df.columns:
                    mean = df[col].mean()
                    std = df[col].std()
                    df = df[df[col].between(mean - 3 * std, mean + 3 * std)]
            
            self.processed_df = df
            return df
        
        except Exception as e:
            raise ValueError(f"Preprocessing failed: {e}")
    
    def prepare_time_series(self) -> pd.DataFrame:
        """
        Prepare DataFrame for time series analysis with robust date handling.
        """
        df = self.original_df.copy()
        
        date_columns = df.select_dtypes(include=['object']).columns
        
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(
                    df[col].str.extract(r'(\d{4}-\d{2}-\d{2})')[0], 
                    errors='coerce'
                )
            except Exception:
                continue
        
        time_columns = df.select_dtypes(include=['datetime64']).columns
        
        if len(time_columns) == 0:
            raise ValueError("No valid time column found")
        
        # Use first datetime column as index
        df.set_index(time_columns[0], inplace=True)
        df.sort_index(inplace=True)
  
        df = df[~df.index.duplicated(keep='first')]
        df.dropna(subset=[df.index.name], inplace=True)
        
        return df

    def get_context_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive context report with tables
        """
        context = self.data_context

        # Data preview
        data_preview = self.original_df.head()

        # Data type
        data_type_info = pd.DataFrame({
            "Data Type": [context['data_type'].name],
            "Description": [f"Detected as {context['data_type'].name} data"]
        })

        # Columns summary
        columns_summary = pd.DataFrame({
            "Column Type": ["Total", "Numerical", "Categorical"],
            "Count": [len(self.original_df.columns), len(context['numerical_columns']), len(context['categorical_columns'])]
        })

        # Column Dtypes summary
        column_types = pd.DataFrame({
            "Column Name": self.original_df.columns,
            "Data Type": self.original_df.dtypes.values
        })

        # Potential analysis types
        potential_analyses = pd.DataFrame({
            "Potential Analysis Types": context['potential_analysis_types']
        })

        return {
            "data_preview": data_preview,
            "data_type_info": data_type_info,
            "columns_summary": columns_summary,
            "column_types": column_types,
            "potential_analyses": potential_analyses
        }