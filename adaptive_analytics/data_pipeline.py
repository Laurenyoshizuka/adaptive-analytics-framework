import pandas as pd
from adaptive_analytics.data_processor import DataProcessor
from adaptive_analytics.insights_generator import InsightsGenerator

class DataPipeline:
    def __init__(self, data):
        """
        Initialize DataPipeline with either a file (uploaded_file) or a DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            self.df = data
        else:
            self.uploaded_file = data
            self.df = None

        self.processed_df = None
        self.context_report = None
        self.insights = None

    def _load_data(self):
        """Load data only if it's an uploaded file."""
        if self.df is None and self.uploaded_file is not None:
            try:
                file_name = self.uploaded_file.name
                if file_name.endswith('.csv'):
                    self.df = pd.read_csv(self.uploaded_file)
                elif file_name.endswith('.xls'):
                    self.df = pd.read_excel(self.uploaded_file, engine='xlrd')
                elif file_name.endswith('.json'):
                    self.df = pd.read_json(self.uploaded_file)
                else:
                    raise ValueError("Unsupported file format.")
            except Exception as e:
                raise ValueError(f"Error loading data: {e}")

    def load_and_preprocess(self):
        """Load and preprocess the data (skip loading if DataFrame is provided)."""
        if self.df is None:
            self._load_data()

        if self.df is None or self.df.empty:
            raise ValueError("No data found in the uploaded file or DataFrame.")

        processor = DataProcessor(self.df)
        self.processed_df = processor.preprocess()
        self.context_report = processor.get_context_report()

        try:
            time_series_df = self.prepare_time_series(self.processed_df)
            self.context_report['analysis_availability'] = {
                "time_series_analysis": time_series_df is not None
            }
            self.time_series_df = time_series_df
        except Exception:
            self.context_report['analysis_availability'] = {
                "time_series_analysis": False
            }

    def generate_insights(self):
        """Generate insights based on processed data."""
        insights_generator = InsightsGenerator(self.processed_df)
        self.insights = insights_generator.generate_comprehensive_insights()

    def get_context(self):
        """Return context data for visualization."""
        return self.context_report
    
    def get_insights(self):
        """Return insights data for visualization."""
        return self.insights