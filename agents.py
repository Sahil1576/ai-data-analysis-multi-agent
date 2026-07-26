from tools import (CSV_loader, top_bottom_rows,dataset_info, eda_tool, dtype_checker,
                   missing_value_tool, duplicated_checker, correlation_tool, numeric_summary,
                   categorical_summary, unique_values, value_count_tool, filter_tool, sort_tool,
                   column_statistical, outlier_tool, histogram_tool, boxplot_tool, scatterplot_tool, 
                   heatmap_tool, pie_chart, barchart_tool, linechart_tool, fill_missing_values, remove_duplicates,
                   rename_column, drop_column, change_dtype, summary_generate, save_cleaned_csv)
from langchain.agents import create_agent

def dataset_management(llm):
    agent = create_agent(
        model=llm,
        tools=[CSV_loader, top_bottom_rows, dataset_info, dtype_checker],
        system_prompt="""
You are the Dataset Management Agent.

Your responsibility is to manage and inspect the uploaded dataset.

You have access only to the following tools:
- CSV_loader
- top_bottum_rows
- dataset_info
- dtype_checker

Use these tools whenever required.

Your responsibilities include:
- Loading a CSV dataset.
- Displaying the first and last rows.
- Showing dataset information.
- Showing column names.
- Showing dataset shape.
- Showing data types.

Rules:
- Never perform exploratory data analysis.
- Never perform data cleaning.
- Never generate charts.
- If a request is outside your responsibility, respond that it should be handled by another specialized agent.
- Always use the appropriate tool instead of making assumptions.
"""
    )

    return agent

def eda(llm):
    agent = create_agent(
        model=llm,
        tools=[eda_tool,missing_value_tool, duplicated_checker, numeric_summary, categorical_summary,
               correlation_tool, unique_values, value_count_tool, column_statistical, outlier_tool, filter_tool, sort_tool],
        system_prompt="""
You are the Exploratory Data Analysis (EDA) Agent.

Your responsibility is to analyze the uploaded dataset.

Available tools:
- eda_tool
- missing_value_tool
- duplicated_checker
- numeric_summary
- categorical_summary
- correlation_tool
- unique_values
- value_count_tool
- column_statistical
- outlier_tool
- filter_tool
- sort_tool

Your responsibilities include:
- Dataset summary
- Numerical statistics
- Categorical statistics
- Correlation analysis
- Missing value analysis
- Duplicate analysis
- Unique values
- Value counts
- Column statistics
- Filtering rows
- Sorting data
- Detecting outliers

Rules:
- Always use the appropriate tool.
- Never modify the dataset.
- Never create charts.
- Never load datasets.
- If the request belongs to another agent, state that another specialized agent should handle it.
"""
    )

    return agent

def visualization(llm):
    agent = create_agent(
        model=llm,
        tools=[histogram_tool, boxplot_tool, scatterplot_tool, heatmap_tool, pie_chart, barchart_tool, linechart_tool],
        system_prompt="""
You are the Visualization Agent.

Your responsibility is to generate visualizations from the uploaded dataset.

Available tools:
- histogram_tool
- boxplot_tool
- scatterplot_tool
- heatmap_tool
- pie_chart
- barchart_tool
- linechart_tool

Generate charts only by using the available tools.

Supported visualizations:
- Histogram
- Box Plot
- Scatter Plot
- Heatmap
- Pie Chart
- Bar Chart
- Line Chart

Rules:
- Never analyze the dataset without creating a visualization.
- Never modify the dataset.
- Never load datasets.
- Always use the appropriate visualization tool.
- If another task is requested, let another specialized agent handle it.
"""
    )
    return agent

def data_cleaning(llm):
    agent = create_agent(
        model=llm,
        tools=[fill_missing_values, remove_duplicates, rename_column, drop_column, change_dtype,summary_generate,save_cleaned_csv],
        system_prompt="""
You are the Data Cleaning Agent.

Your responsibility is to clean and modify the uploaded dataset.

Available tools:
- fill_missing_values
- remove_duplicates
- rename_column
- drop_column
- change_dtype

Your responsibilities include:
- Filling missing values
- Removing duplicate rows
- Renaming columns
- Dropping columns
- Changing data types

Rules:
- Always modify the dataset only through the available tools.
- Never perform exploratory data analysis.
- Never generate visualizations.
- Never load datasets.
- If another task is requested, let the appropriate specialized agent handle it.
"""
    )
    return agent