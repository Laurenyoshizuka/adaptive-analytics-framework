import streamlit as st
import logging
from adaptive_analytics.data_pipeline import DataPipeline
from adaptive_analytics.insights_generator import InsightsGenerator
from adaptive_analytics.ETL_pipeline_analysis import *
from prefect import flow


def setup_logging():
    """Configure logging for Streamlit app"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger

@flow
def process_sample_data():
    st.success("🤖 Triggered ETL pipeline for sample store data")

    sample_data_url = 'https://public.tableau.com/app/sample-data/sample_-_superstore.xls'
    
    try:
        data = download_file_from_url(sample_data_url)
        load_data_to_db(data)
        st.success(f"🤖 Data loaded into SQLite DB")
        run_dbt()
        table_data = load_transformed_data_from_sqlite()
        orders_df = table_data.get('Orders')
        returns_df = table_data.get('Returns')
        people_df = table_data.get('People')
        combined_df = table_data.get('combined_data_model')
        display_data(orders_df, returns_df, people_df, combined_df)
        delete_xls_files()
    except Exception as e:
        st.error(f"Error in process_sample_data: {e}")
        raise

def main():
    st.title("Advanced Analytics Framework")
    with st.expander("ℹ️  About"):
        st.markdown("""
                    This application is a demonstration of an automated analytics framework that can be used to analyze and generate insights from 2 data source options. The tools used in this framework include:
                    - dbt for data modeling
                    - Prefect for workflow automation
                    - Streamlit for the front-end
                    
                    If the user uploads custom data, the framework automatically detects the data type and suggests relevant analysis strategies. 
                    If the sample store data (Tableau's Superstore dataset) is selected, the framework triggers a pipeline that:

                    1. Downloads the dataset from the web
                    2. Creates a database and parses the data into tables
                    3. Builds a data model using dbt
                    4. Initializes a dashboard with key performance indicator (KPI) insights

                    📖 Author:

                    This project was developed by Lauren Yoshizuka (laurenyoshizuka@gmail.com)—an enthusiast in advanced analytics & data science. Feel free to reach out for collaboration, questions, or further inquiries!
                    """)
    
    data_option = st.radio(
        "Choose the data source", 
        options=["Use Sample Store Data", "Upload Your Own Data"]
        )

    if data_option == "Use Sample Store Data":
        try:
            with st.spinner('Processing sample store data...'):
                df = process_sample_data()

        except Exception as e:
            st.error(f"Error processing sample data: {e}")
    
    elif data_option == "Upload Your Own Data":
        uploaded_file = st.file_uploader("Upload Data File", type=['csv', 'xls', 'json'])
        
        if uploaded_file is not None:
            try:
                with st.spinner('Processing your uploaded file...'):
                    pipeline = DataPipeline(uploaded_file)
                    pipeline.load_and_preprocess()
                    pipeline.generate_insights()

                    insights_generator = InsightsGenerator(pipeline.processed_df)
                    insights = insights_generator.generate_comprehensive_insights()
                    context_report = pipeline.get_context()

                    display_insights(context_report, insights, insights_generator)

            except Exception as e:
                st.error(f"Error processing uploaded data: {e}")

@st.cache_data(ttl=3600)
def display_insights(context_report, insights, insights_generator):
    """Displays the insights and context report in Streamlit"""
    st.subheader("Data Detection")
    st.table(context_report["data_type_info"])

    st.subheader("Data Preview")
    st.dataframe(context_report["data_preview"], hide_index=True)

    st.subheader("Columns Summary")
    st.table(context_report["columns_summary"])

    st.subheader("Column Data Types")
    st.table(context_report["column_types"])

    st.subheader("Descriptive Statistics")
    st.dataframe(insights['descriptive_statistics'], use_container_width=True)

    st.subheader("Potential Analyses")
    st.table(context_report["potential_analyses"])

    analysis_options = context_report["potential_analyses"]["Potential Analysis Types"].tolist()
    selected_analysis = st.selectbox("Select Analysis Type to Run", options=analysis_options)

    if selected_analysis == "time_series_decomposition":
        if insights['time_series_analysis']['available']:
            numeric_cols = insights['time_series_analysis']['numeric_columns']
            if numeric_cols:
                target_col = st.selectbox(
                    'Select a numeric column for forecasting',
                    options=numeric_cols
                )
                ts_results = insights_generator.generate_time_series_analysis(target_col)
                if ts_results:
                    st.plotly_chart(ts_results['forecast_plot'], use_container_width=True)
            else:
                st.warning("No numeric columns available for time series analysis.")
        else:
            st.warning("Time series analysis is not available for this dataset.")

    elif selected_analysis == 'trend_analysis':
        if insights['time_series_analysis']['available']:
            numeric_cols = insights['time_series_analysis']['numeric_columns']
            if numeric_cols:
                target_col = st.selectbox(
                    'Select a target column for trend analysis',
                    options=numeric_cols
                )
                trend_results = insights_generator.trend_analysis(target_col)

                if trend_results:
                    st.plotly_chart(trend_results['trend_plot'], use_container_width=True)
                    st.write(f"Slope: {trend_results['slope']}")
                    st.write(f"R-squared: {trend_results['r_squared']}")

    elif selected_analysis == 'seasonality_detection':
        if insights['time_series_analysis']['available']:
            numeric_cols = insights['time_series_analysis']['numeric_columns']
            if numeric_cols:
                target_col = st.selectbox(
                    'Select a target column for seasonality detection',
                    options=numeric_cols
                )
                seasonality_results = insights_generator.seasonality_detection(target_col)

                if seasonality_results:
                    st.plotly_chart(seasonality_results['seasonality_plot'], use_container_width=True)
                    st.write(f"Peak Frequency: {seasonality_results['peak_frequency']}")
                    st.write(f"Estimated Period: {seasonality_results['estimated_period']}")
    
    elif selected_analysis == "correlation_analysis":
        st.subheader("Correlation Analysis")
        st.table(insights['correlation_analysis']["significant_correlations"])
        st.plotly_chart(insights['correlation_analysis']["heatmap_figure"], use_container_width=True)

    elif selected_analysis == "clustering":
        st.subheader("Clustering Analysis")
        st.table(insights['clustering_analysis']['cluster_centers'])
        st.plotly_chart(insights['clustering_analysis']['clustering_visualization'], 
                        use_container_width=True)

    elif selected_analysis == 'regression_prediction':

        insights = insights_generator.generate_comprehensive_insights()
        numeric_cols = insights['numeric_columns']

        if numeric_cols:
            target_col = st.selectbox(
                'Select a target column for regression prediction',
                options=numeric_cols
            )

            if target_col:
                regression_results = insights_generator.linear_regression(target_col)

                if regression_results:
                    st.write("Regression Summary:")
                    st.write(f"R-squared: {regression_results['r_squared']}")
                    st.write(f"Coefficients: {regression_results['coefficients']}")
                    st.write(f"Intercept: {regression_results['intercept']}")

                    if 'regression_plot' in regression_results:
                        st.write("Regression plot available")
                        st.plotly_chart(regression_results['regression_plot'], use_container_width=True)
                    else:
                        st.write("Regression plot not available")
            else:
                st.write("Please select a valid target column.")

    elif selected_analysis == "distribution_analysis":
        st.subheader("Distribution Analysis")
        st.table(insights['distribution_analysis'])

    elif selected_analysis == "category_distribution":
        st.subheader("Category Distribution")
        category_distribution_figures = insights['category_distribution']
        for col, fig in category_distribution_figures.items():
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
