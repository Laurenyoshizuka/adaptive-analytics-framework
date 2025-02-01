import pandas as pd
import numpy as np
import streamlit as st
import logging
from typing import Dict, Any
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from scipy.signal import periodogram
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy.stats import shapiro
from sklearn.preprocessing import LabelEncoder

class InsightsGenerator:
    def __init__(self, processed_dataframe: pd.DataFrame):
        self.df = processed_dataframe
        self.logger = logging.getLogger(__name__)
    
    def generate_comprehensive_insights(self):
        """Generate insights for the dataset."""
        insights = {}
        insights['numeric_columns'] = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        insights['descriptive_statistics'] = self._get_descriptive_statistics()
        insights['correlation_analysis'] = self._advanced_correlation_analysis()
        insights['clustering_analysis'] = self._perform_clustering()
        insights['distribution_analysis'] = self._analyze_distributions()

        # Categorical Insights
        insights['categorical_columns'] = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if insights['categorical_columns']:
            insights['category_distribution'] = self.category_distribution()

        # Check for time series compatibility
        if self._is_time_series():
            # Only check availability, don't generate analysis yet
            insights['time_series_analysis'] = {
                'available': True,
                'numeric_columns': self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            }
        else:
            insights['time_series_analysis'] = {'available': False}

        # Linear Regression Analysis
        insights['regression_analysis'] = {'available': False}
        if insights['numeric_columns']:
            insights['regression_analysis'] = {
                'available': True,
                'numeric_columns': insights['numeric_columns']
            }

        return insights

    def _get_descriptive_statistics(self) -> Dict[str, Any]:
        """Comprehensive descriptive statistics"""
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        return self.df[numeric_cols].describe().to_dict()
    
    def _advanced_correlation_analysis(self) -> Dict[str, Any]:
        """Advanced correlation matrix with significance testing and heatmap visualization"""
        
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        correlations = self.df[numeric_cols].corr()

        significant_correlations = {}
        p_values = pd.DataFrame(np.ones(correlations.shape), columns=correlations.columns, index=correlations.index)

        for col1 in correlations.columns:
            for col2 in correlations.columns:
                if col1 != col2:
                    correlation, p_value = stats.pearsonr(
                        self.df[col1].dropna(), 
                        self.df[col2].dropna()
                    )
                    p_values.loc[col1, col2] = p_value
                    if p_value < 0.05:
                        significant_correlations[(col1, col2)] = {
                            'correlation': correlation,
                            'p_value': p_value
                        }

        masked_correlations = correlations.copy()
        masked_correlations[p_values >= 0.05] = np.nan

        heatmap = go.Figure(
            data=go.Heatmap(
                z=masked_correlations.values,
                x=masked_correlations.columns,
                y=masked_correlations.index,
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Correlation")
            )
        )

        heatmap.update_layout(
            title="Significant Correlation Heatmap",
            xaxis=dict(title="Features"),
            yaxis=dict(title="Features", autorange="reversed")
        )

        return {
            "significant_correlations": significant_correlations,
            "heatmap_figure": heatmap
        }
    
    def _detect_datetime_column(self):
        """Detect and return the most appropriate datetime column."""
        datetime_cols = self.df.select_dtypes(include=['datetime64[ns]', 'datetime']).columns

        if len(datetime_cols) == 0:
            self.logger.error("No datetime columns found in dataset.")
            return None
        
        # Select the column with the fewest missing values
        best_col = sorted(datetime_cols, key=lambda col: self.df[col].isnull().sum())[0]
        return best_col

    def _is_time_series(self):
        """Check if the dataset contains a valid datetime column and set it as the index."""
        time_col = self._detect_datetime_column()
        
        if time_col is None:
            return False
        
        self.df[time_col] = pd.to_datetime(self.df[time_col], errors='coerce')
        self.df = self.df.dropna(subset=[time_col]).set_index(time_col)
        self.df.sort_index(inplace=True)
        return True

    def generate_time_series_analysis(self, target_col: str):
        """Generate time series analysis for a specific column"""
        if not self._is_time_series():
            return None

        prophet_df = self.df.reset_index().rename(columns={self.df.index.name: "ds", target_col: "y"})

        try:
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True
            )
            model.fit(prophet_df)

            future = model.make_future_dataframe(periods=12, freq='ME')
            forecast = model.predict(future)

            trend = forecast['trend']
            seasonal_weekly = forecast['weekly']
            seasonal_yearly = forecast['yearly']
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], 
                                   mode='lines', name='Forecast', 
                                   line=dict(color='royalblue')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=trend, 
                                   mode='lines', name='Trend', 
                                   line=dict(color='springgreen')))

            # Add seasonal components
            fig.add_trace(go.Scatter(x=forecast['ds'], y=seasonal_weekly, 
                                   mode='lines', name='Weekly Seasonality', 
                                   line=dict(color='darkolivegreen', dash='dash')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=seasonal_yearly, 
                                   mode='lines', name='Yearly Seasonality', 
                                   line=dict(color='darkorange', dash='dot')))    #violet

            # Add confidence intervals
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], 
                                   mode='lines', name='Upper Bound', 
                                   line=dict(dash='dot', color='slateblue')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], 
                                   mode='lines', name='Lower Bound', 
                                   line=dict(dash='dot', color='gray')))

            fig.update_layout(
                title=f"Time Series Decomposition and Forecast for {target_col}",
                xaxis_title="Date",
                yaxis_title="Value",
                template="plotly_dark",
                hovermode="x unified"
            )

            return {
                'forecast_plot': fig,
                'trend': trend.tolist(),
                'seasonal': {
                    'weekly': seasonal_weekly.tolist(),
                    'yearly': seasonal_yearly.tolist()
                },
                'forecast': forecast['yhat'].tolist()
            }

        except Exception as e:
            self.logger.error(f"Time series decomposition failed: {e}")
            return None
        
    def trend_analysis(self, target_col: str):
        """Detects trends in a time series using linear regression."""
        if target_col not in self.df.columns:
            self.logger.error(f"Column {target_col} not found in DataFrame.")
            return None

        self.df = self.df.dropna(subset=[target_col])
        self.df['time_index'] = np.arange(len(self.df))

        slope, intercept, r_value, p_value, std_err = linregress(
            self.df['time_index'], self.df[target_col]
        )

        trend_line = intercept + slope * self.df['time_index']

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df[target_col], 
                                 mode='lines', name='Actual Data'))
        fig.add_trace(go.Scatter(x=self.df.index, y=trend_line, 
                                 mode='lines', name='Trend Line', line=dict(dash='dot')))
        
        fig.update_layout(title=f"Trend Analysis for {target_col}",
                          xaxis_title="Time",
                          yaxis_title=target_col)

        return {
            'slope': slope,
            'r_squared': r_value ** 2,
            'trend_plot': fig
        }

    def seasonality_detection(self, target_col: str, freq: str = 'D'):
        """Detects seasonality using periodogram and autocorrelation."""
        if target_col not in self.df.columns:
            self.logger.error(f"Column {target_col} not found in DataFrame.")
            return None

        self.df = self.df.dropna(subset=[target_col])
        
        # Compute periodogram
        freqs, power = periodogram(self.df[target_col].values)
        
        # Identify peak frequency (most dominant seasonal pattern)
        peak_freq = freqs[np.argmax(power)]
        peak_period = 1 / peak_freq if peak_freq > 0 else None

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freqs, y=power, mode='lines', name='Power Spectrum'))
        fig.update_layout(title=f"Seasonality Detection for {target_col}",
                          xaxis_title="Frequency",
                          yaxis_title="Power")

        return {
            'peak_frequency': peak_freq,
            'estimated_period': peak_period,
            'seasonality_plot': fig
        }
    
    def _perform_clustering(self) -> Dict[str, Any]:
        """K-means clustering analysis with visualization"""
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        if len(numeric_cols) < 2:
            return {}
        
        # Scaling
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[numeric_cols])
        
        # Clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        self.df['Cluster'] = clusters
        
        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        self.df['PCA1'] = pca_result[:, 0]
        self.df['PCA2'] = pca_result[:, 1]
        
        # Cluster centers
        cluster_centers = kmeans.cluster_centers_
        cluster_centers_df = pd.DataFrame(cluster_centers, columns=[f'Feature {i+1}' for i in range(cluster_centers.shape[1])])
        cluster_centers_df['Cluster'] = range(len(cluster_centers))

        fig = px.scatter(self.df, x='PCA1', y='PCA2', color='Cluster', 
                         title="Cluster Visualization (PCA Reduced)",
                         labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'},
                         hover_data=['Cluster'])

        fig.add_scatter(x=cluster_centers_df['Feature 1'], 
                        y=cluster_centers_df['Feature 2'], 
                        mode='markers+text', 
                        marker=dict(size=12, color='black'),
                        text=cluster_centers_df['Cluster'], 
                        textposition='top center', name="Cluster Centers")

        clustering_insights = {
            'cluster_centers': cluster_centers.tolist(),
            'cluster_distribution': np.unique(clusters, return_counts=True)[1].tolist(),
            'clustered_data': self.df,
            'clustering_visualization': fig
        }
        
        return clustering_insights
    
    def _analyze_distributions(self) -> Dict[str, Any]:
        """Detailed distribution analysis"""
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        distributions = {}
        for col in numeric_cols:
            distributions[col] = {
                'skewness': stats.skew(self.df[col]),
                'kurtosis': stats.kurtosis(self.df[col]),
                'normality_test': stats.normaltest(self.df[col]).pvalue
            }
        
        return distributions
   

    def _is_linear_regression_applicable(self, target_col: str):
        """Check if linear regression is applicable for the dataset."""
        if target_col not in self.df.columns:
            self.logger.error(f"Column {target_col} not found in DataFrame.")
            return False

        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if target_col not in numeric_cols:
            self.logger.error(f"Target column {target_col} must be numeric.")
            return False

        if self.df[target_col].isnull().sum() > 0:
            self.logger.warning(f"Missing values detected in {target_col}. Consider imputing or dropping them.")
            return False

        if len(numeric_cols) < 2:
            self.logger.error("Not enough numeric columns for regression.")
            return False

        # Check for linear relationships using correlation
        correlations = self.df[numeric_cols].corr()[target_col].drop(target_col)
        if correlations.abs().max() < 0.3:
            self.logger.warning(f"Weak correlation detected. Linear regression may not be suitable.")
            return False

        # Check for multicollinearity using Variance Inflation Factor (VIF)
        X = self.df[numeric_cols].drop(columns=[target_col])
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        if vif_data["VIF"].max() > 10:
            self.logger.warning(f"High multicollinearity detected. Consider removing correlated features.")
            return False

        return True

    def linear_regression(self, target_col: str):
        """Perform linear regression after validating assumptions."""
        if not self._is_linear_regression_applicable(target_col):
            st.write(f"Linear regression not applicable for {target_col}")
            return None

        self.df = self.df.dropna(subset=[target_col])

        if self.df[target_col].isnull().sum() == len(self.df):
            st.write(f"No valid data for target column: {target_col}")
            return None 

        X = self.df.drop(columns=[target_col]).select_dtypes(include=['float64', 'int64'])
        y = self.df[target_col]

        if len(X) < 2 or len(y) < 2:
            st.write(f"Not enough data points for regression on {target_col}")
            return None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        
        regression_prediction = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })

        r_squared = model.score(X_test, y_test)
        residuals = y_test - y_pred
        
        # Residual plot for homoscedasticity
        fig_residuals = go.Figure()
        fig_residuals.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'))
        fig_residuals.update_layout(
            title="Residual Plot",
            xaxis_title="Predicted Values",
            yaxis_title="Residuals"
        )

        # Check normality of residuals using Shapiro-Wilk test
        shapiro_test = shapiro(residuals)
        residuals_are_normal = shapiro_test.pvalue > 0.05  # Normal if p > 0.05

        # Regression plot (Actual vs Predicted)
        fig_regression = go.Figure()
        fig_regression.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Actual vs Predicted'))
        fig_regression.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal Fit', line=dict(dash='dot')))
        fig_regression.update_layout(
            title=f"Linear Regression: {target_col}",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values"
        )

        return {
            'coefficients': model.coef_.tolist(),
            'intercept': model.intercept_,
            'r_squared': r_squared,
            'residual_plot': fig_residuals,
            'regression_plot': fig_regression,
            'residuals_normality_test': {
                'statistic': shapiro_test.statistic,
                'p_value': shapiro_test.pvalue,
                'normal': residuals_are_normal
            },
            'regression_prediction': regression_prediction
        }

    def category_distribution(self):
        """Visualize category distribution in categorical columns using Plotly."""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        category_distribution_figures = {}
        
        for col in categorical_cols:
            category_counts = self.df[col].value_counts().reset_index()
            category_counts.columns = [col, 'count']
            
            fig = px.bar(category_counts, x=col, y='count', 
                         title=f"Category Distribution for {col}",
                         labels={col: 'Category', 'count': 'Count'},
                         color=col)
            category_distribution_figures[col] = fig
            
        return category_distribution_figures
    