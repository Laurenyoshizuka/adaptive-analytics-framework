# ETL_pipeline_analysis.py for sample_data
import os
import requests
import pandas as pd
import sqlite3
from prefect import get_client
from prefect import flow, task
from prefect_dbt.cli.commands import DbtCoreOperation
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import glob
import yaml 
import prefect

prefect_api_key = st.secrets["PREFECT"]["API_KEY"]
client = get_client(api_key=prefect_api_key)


# client = get_client()
BASE_DIR = st.secrets["BASE_DIR"]
DBT_PROJECT_PATH = st.secrets["DBT_PROJECT_PATH"]
DB_FILE = st.secrets["DB_FILE"]
DBT_PROFILES_PATH = st.secrets["DBT_PROFILES_PATH"]
USER_ID = st.secrets["USER_ID"]
PREFECT_API_URL = st.secrets["PREFECT_API_URL"]


os.makedirs(DBT_PROFILES_PATH, exist_ok=True)
profiles_path = os.path.join(DBT_PROFILES_PATH, "profiles.yml")
dbt_profile = {
    "sqlite_profile": {
        "target": "dev",
        "outputs": {
            "dev": {
                "type": "sqlite",
                "threads": 1,
                "database": "sample_superstore.db", 
                "schema": "main",
                "schemas_and_paths": {
                    "main": DB_FILE
                },
                "schema_directory": DB_FILE
            }
        }
    }
}
with open(profiles_path, "w") as file:
    yaml.dump(dbt_profile, file, default_flow_style=False)



@task
def download_file_from_url(url: str) -> dict:
    file_path = 'sample_superstore.xls'
    
    response = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(response.content)
        
    return pd.read_excel(file_path, sheet_name=None)

@task
def load_data_to_db(data: dict) -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE) 
    
    tables_created = False
    
    for sheet_name, df in data.items():
        clean_sheet_name = ''.join(c if c.isalnum() else '_' for c in sheet_name)
        
        if not df.empty:
            st.write(f"Loading sheet: {sheet_name} with shape {df.shape}")
            df.to_sql(clean_sheet_name, conn, index=False, if_exists='replace')
            tables_created = True
        else:
            st.write(f"Sheet {sheet_name} is empty and was not loaded")
    
    if not tables_created:
        raise ValueError("No tables could be created from the Excel file")
    
    return conn

@task
def run_dbt():
    dbt_run = DbtCoreOperation(
        commands=["dbt run"],
        project_dir=DBT_PROJECT_PATH,
        profiles_dir=DBT_PROFILES_PATH,
        retries=2,
        retry_delay=10
    )
    result = dbt_run.run()
    
    if result:
        print("dbt run completed successfully!")
        return result
    else:
        raise RuntimeError("dbt run failed!")

@task
def load_transformed_data_from_sqlite():
    conn = sqlite3.connect(DB_FILE)
    
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    if not tables:
        raise ValueError("No tables found in the database")
    
    st.write(f"Found {len(tables)} tables: {', '.join(table[0] for table in tables)}")
    
    table_data = {}
    for table_name in tables:
        table_name = table_name[0]
        
        query = f'SELECT * FROM "{table_name}";'
        df = pd.read_sql(query, conn)
        table_data[table_name] = df
        
        st.subheader(f"Preview of table: {table_name}")
        st.dataframe(df.head())

    return table_data


# Dashboard
def display_data(orders_df, returns_df, people_df, combined_df):
    st.title("Superstore KPI Dashboard")
    st.write("Global KPIs & Regional Insights")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # 1. Total Sales
        total_sales = orders_df["Sales"].sum()
        st.metric("Total Sales", f"${total_sales:,.0f}")
    with col2:
        # 2. Profit in Dollars and Percent
        total_profit = orders_df["Profit"].sum()
        profit_percentage = (total_profit / total_sales) * 100 if total_sales != 0 else 0
        st.metric("Total Profit", f"${total_profit:,.0f}", f"{profit_percentage:.1f}%")
    with col3:
        # 3. Total Customers
        total_customers = orders_df["Customer ID"].nunique()
        st.metric("Total Customers", total_customers)
    
    st.divider()
    # Group by Region
    region_sales = orders_df.groupby("Region")["Sales"].sum().reset_index()
    region_profit = orders_df.groupby("Region")["Profit"].sum().reset_index()
    region_customers = orders_df.groupby("Region")["Customer ID"].nunique().reset_index()

    region_sales = region_sales.sort_values("Region")
    region_profit = region_profit.sort_values("Region")
    region_customers = region_customers.sort_values("Region")

    # 3 Columns for the donut charts
    c1, c2, c3 = st.columns(3)
    
    with c1:
        # Sales by region
        sales_percent = region_sales.set_index("Region")["Sales"] / total_sales * 100
        fig_sales_donut = go.Figure(go.Pie(
            labels=region_sales["Region"], 
            values=sales_percent, 
            hole=0.3, 
            textinfo="label+percent",
            title="Sales by Region (%)",
            sort=False
        ))
        st.plotly_chart(fig_sales_donut)
    
    with c2:
        # Profit by region
        profit_percent = region_profit.set_index("Region")["Profit"] / total_profit * 100
        fig_profit_donut = go.Figure(go.Pie(
            labels=region_profit["Region"], 
            values=profit_percent, 
            hole=0.3, 
            textinfo="label+percent",
            title="Profit by Region (%)",
            sort=False
        ))
        st.plotly_chart(fig_profit_donut)
    
    with c3:
        # Customers by region
        customers_percent = region_customers.set_index("Region")["Customer ID"] / total_customers * 100
        fig_customers_donut = go.Figure(go.Pie(
            labels=region_customers["Region"], 
            values=customers_percent, 
            hole=0.3, 
            textinfo="label+percent",
            title="Customers by Region (%)",
            sort=False
        ))
        st.plotly_chart(fig_customers_donut)

    st.divider()

    # Region Selection Button
    selected_region = st.selectbox("Select a Region", options=region_sales["Region"].unique())
    region_data = combined_df[combined_df["Region"] == selected_region]
    region_returns = combined_df[combined_df["Region"] == selected_region]

    # Region-specific analytics
    st.header(f"Analytics for {selected_region} Region")
    
    # 1. Returns stats
    return_count = region_returns['Return_Order_ID'].nunique()  
    return_rate = (return_count / region_data.shape[0]) * 100 if region_data.shape[0] != 0 else 0
    st.subheader(f"Returns Overview")
    st.write(f"Number of Returns: {return_count}")
    st.write(f"Return Rate: {return_rate:.2f}%")
    
    # 2. Sales by Segment
    if 'Segment' in region_data.columns:
        sales_by_segment = region_data.groupby("Segment")["Sales"].sum().reset_index()
        fig_segment_sales = go.Figure(go.Bar(
            x=sales_by_segment["Segment"], 
            y=sales_by_segment["Sales"],
            marker=dict(color=sales_by_segment["Sales"], colorscale='Viridis')
        ))

        fig_segment_sales.update_layout(
            title="Sales by Segment",
            xaxis_title="Segment",
            yaxis_title="Sales ($)"
        )
        st.plotly_chart(fig_segment_sales)

    # 3. Sales and Profit by State
    if 'State' in region_data.columns:
        state_sales_profit = region_data.groupby("State")[["Sales", "Profit"]].sum().reset_index()
        state_sales_profit = state_sales_profit.sort_values("State")

        # Sales by state
        fig_state_sales = go.Figure(go.Bar(
            x=state_sales_profit["State"], 
            y=state_sales_profit["Sales"],
            marker=dict(color=state_sales_profit["Sales"], colorscale='Blues')
        ))

        fig_state_sales.update_layout(
            title="Sales by State",
            xaxis_title="State",
            yaxis_title="Sales ($)"
        )
        st.plotly_chart(fig_state_sales)

        
        # Profit by state
        fig_state_profit = go.Figure(go.Bar(
            x=state_sales_profit["State"], 
            y=state_sales_profit["Profit"],
            marker=dict(color=state_sales_profit["Profit"], colorscale='Greens')
        ))

        fig_state_profit.update_layout(
            title="Profit by State",
            xaxis_title="State",
            yaxis_title="Profit ($)"
        )
        st.plotly_chart(fig_state_profit)

    # 5. Profit Line Chart by Month
    region_data['Order Date'] = pd.to_datetime(region_data['Order Date'])
    region_data['Month'] = region_data['Order Date'].dt.to_period('M').astype(str)
    profit_by_month = region_data.groupby('Month')['Profit'].sum().reset_index()
    
    fig_profit_month = px.line(profit_by_month, x='Month', y='Profit', 
                               title="Profit by Month", 
                               labels={"Month": "Month", "Profit": "Profit ($)"})
    st.plotly_chart(fig_profit_month)
    
    # 6. Profit by Category and Sub-category (Drill-down)
    if 'Category' in region_data.columns and 'Sub-Category' in region_data.columns:
        profit_by_category_subcategory = region_data.groupby(['Category', 'Sub-Category'])['Profit'].sum().reset_index()

        view_option = st.selectbox("Select View", ["Profit by Category", "Profit by Sub-category"])

        if view_option == "Profit by Category":
            profit_by_category = profit_by_category_subcategory.groupby('Category')['Profit'].sum().reset_index()

            fig_profit_category = px.bar(
                profit_by_category, 
                x='Category', 
                y='Profit', 
                title="Profit by Category",
                labels={'Profit': 'Profit ($)', 'Category': 'Category'},
                hover_data={'Category': True, 'Profit': True},
                color='Profit',
                color_continuous_scale='Plasma'
            )
            st.plotly_chart(fig_profit_category)

        elif view_option == "Profit by Sub-category":
            fig_profit_subcategory = px.bar(
                profit_by_category_subcategory, 
                x='Sub-Category', 
                y='Profit', 
                color='Category',
                title="Profit by Sub-category (Drill-down)",
                labels={'Profit': 'Profit ($)', 'Sub-Category': 'Sub-category'},
                hover_data={'Sub-Category': True, 'Profit': True, 'Category': True}
            )
            st.plotly_chart(fig_profit_subcategory)
    
    # 7. Sales Heat Map by Segment and Sub-category
    sales_by_segment_subcategory = region_data.groupby(['Segment', 'Sub-Category'])['Sales'].sum().unstack(fill_value=0)

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=sales_by_segment_subcategory.values,
        x=sales_by_segment_subcategory.columns,
        y=sales_by_segment_subcategory.index,
        colorscale='YlGnBu'
    ))

    fig_heatmap.update_layout(
        title="Sales Heat Map by Segment and Sub-category",
        xaxis_title="Segment",
        yaxis_title="Sub-Category",
        coloraxis_colorbar_title="Sales ($)"
    )

    st.plotly_chart(fig_heatmap)

# Orchestration
@flow
def process_and_visualize_data(url: str):

    file_path = download_file_from_url(url)
    load_data_to_db(file_path)
    try:
        run_dbt()
    except RuntimeError as e:
        print(f"Error running dbt: {e}")
        return
    df = load_transformed_data_from_sqlite()
    delete_xls_files()
    
    return df

def delete_xls_files():
    """Deletes all .xls files in the working directory."""
    for file in glob.glob("*.xls"):
        os.remove(file)