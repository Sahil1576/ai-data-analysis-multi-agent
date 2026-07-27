import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.tools import tool
import io

df = None

@tool
def CSV_loader(csv_file)->str:
    """
    Load a CSV file into a pandas DataFrame and store it globally.
    """
    global df

    df = pd.read_csv(csv_file)

    return "CSV Loaded successfully."

@tool
def top_bottom_rows()->str:
    """
    Returns the first 5 rows and last 5 rows of the dataset
    to provide a quick preview of the uploaded data.
    """
    global df

    head = f"Top 5 rows : {df.head().to_string()}"

    tail = f"Last 5 rows : {df.tail().to_string()}"

    return f"{head}, {tail}"

@tool
def dataset_info()->str:
    """
Retrieve basic information about the currently loaded dataset.

Returns:
- List of all column names
- Dataset shape (rows and columns)
- Datatype of each column
- General dataset information (non-null count and memory usage)
"""
    global df

    columns = f"Columns : {', '.join(df.columns)}"

    shape = f"Shape : {df.shape}"

    datatypes = f"Data Types : {(df.dtypes).to_string()}"

    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info = buffer.getvalue()

    return f"""
    {columns}
    {shape}
    {datatypes}
    {df_info}
    """

@tool
def eda_tool()->str:
    """
Perform basic Exploratory Data Analysis (EDA) on the currently loaded dataset.

Returns:
- Dataset shape
- Statistical summary of numeric columns
- Missing value count for each column
- Number of duplicate rows
- List of all column names
"""
    global df

    shape = f"Shape : {df.shape}"

    describe = f"Describe : {df.describe().to_string()}"

    null_values = f"Null values : {df.isnull().sum().to_string()}"

    duplicate_values = f"Duplicate Values : {df.duplicated().sum()}"

    columns = f"Columns : {', '.join(df.columns)}"

    return f"""
    {shape}
    {describe}
    {null_values}
    {duplicate_values}
    {columns}
    """

@tool
def dtype_checker()->str:
    """
Display the datatype of each column in the dataset.

Returns:
- Column names along with their corresponding datatypes
"""

    global df

    check = f"Data types checker : {df.dtypes.to_string()}"

    return f"{check}"

@tool
def missing_value_tool()->str:
    """
Calculate the percentage of missing values for each column.

Returns:
- Missing value percentage for every column in the dataset
"""

    global df

    missing = f"Missing values percentage in columns : {(df.isnull().mean() * 100).to_string()}"

    return f"{missing}"

@tool
def duplicated_checker()->str:
    """
Check the dataset for duplicate rows.

Returns:
- Total number of duplicate rows in the dataset
"""

    global df

    duplicate = f"Duplicated Rows : {df.duplicated().sum()}"

    return f"{duplicate}"

@tool
def correlation_tool()->str:
    """
Calculate the correlation matrix for all numeric columns.

Returns:
- Pairwise correlation coefficients between numeric columns
"""

    global df

    corr = f"Correlation : {df.corr(numeric_only=True).to_string()}"

    return f"{corr}"

@tool
def numeric_summary()->str:
    """
Generate descriptive statistics for all numeric columns.

Returns:
- Count
- Mean
- Standard deviation
- Minimum value
- 25th percentile
- Median (50th percentile)
- 75th percentile
- Maximum value
"""

    global df

    summary = f"Numerical Summary : {df.describe().to_string()}"

    return f"{summary}"

@tool
def categorical_summary()->str:
    """
Generate descriptive statistics for all categorical columns.

Returns:
- Count
- Number of unique values
- Most frequent value (top)
- Frequency of the most common value
"""

    global df

    summary = f"Categorical Summary : {df.describe(include='object').to_string()}"

    return f"{summary}"

@tool
def unique_values(column:str)->str:
    """
Retrieve all unique values from a specified column.

Parameters:
- column (str): Name of the column.

Returns:
- List of unique values
- Total number of unique values
"""

    global df

    unique = f"Unique Values of {column} : {', '.join(map(str, df[column].unique()))}"

    count = f"Total Unique : {df[column].nunique()}"

    return f"""
    {unique}
    {count}
    """

@tool
def value_count_tool(column:str)->str:
    """
Count the occurrences of each unique value in a specified column.

Parameters:
- column (str): Name of the column.

Returns:
- Frequency count of each unique value in the column
"""

    global df

    count_value = f"Values Count : {df[column].value_counts()}"

    return f"{count_value}"

@tool
def filter_tool(condition:str)->str:
    """
Filter rows from the currently loaded dataset using a pandas query expression.

Parameters:
- condition (str): A valid pandas query condition.

Examples:
- "Age > 30"
- "Gender == 'Male'"
- "Salary >= 50000"
- "City == 'Surat' and Age < 25"

Returns:
- Filtered rows matching the specified condition
- A message if no matching rows are found
- An error message if the query is invalid
"""

    global df

    try:
        filter_df = df.query(condition)

        if filter_df.empty:
            return "No rows matched the given condition."

        return f"Filtered Rows : {filter_df.to_string(index=False)}"
    except Exception as e:
        return f"Error : {e}"

@tool
def sort_tool(column:str, ascending:bool)->str:
    """
Sort the dataset based on a specified column.

Parameters:
- column (str): Name of the column to sort by.
- ascending (bool): Sort order. Use True for ascending and False for descending.

Returns:
- Dataset sorted according to the specified column and order
"""

    global df

    sorted_df = f"{column} wise sort by {ascending} : {df.sort_values(by=column, ascending=ascending).to_string()}"

    return f"{sorted_df}"

@tool
def column_statistical(column)->str:
    """
Generate statistical information for a specified column.

Parameters:
- column (str): Name of the column.

Returns:
For numeric columns:
- Mean
- Median
- Minimum value
- Maximum value
- Standard deviation

For categorical columns:
- Mode
- Number of unique values
- List of unique categories
- Frequency count of each category
"""

    global df

    numeric_column = df.select_dtypes(include='number').columns

    if column in numeric_column:
        mean = f"Mean : {df[column].mean()}"

        median = f"Median : {df[column].median()}"

        minimum = f"Minimum : {df[column].min()}"

        maximum = f"Maximum : {df[column].max()}"

        std = f"Standard Deviation : {df[column].std()}"

        return f"""
        {mean}
        {median}
        {minimum}
        {maximum}
        {std}
        """
    else:
        mode = f"Mode (Most Frequency): {df[column].mode().to_list()}"

        unique_count = f"Unique Values : {df[column].nunique()}"

        unique_value_list = f"Categories : {', '.join(map(str, df[column].unique()))}"

        counts = f"Values Counts : {df[column].value_counts().to_string()}"

        return f"""
            {mode}
            {unique_count}
            {unique_value_list}
            {counts}
        """

@tool
def outlier_tool(column)->str:
    """
Detect outliers in a numeric column using the Interquartile Range (IQR) method.

Parameters:
- column (str): Name of the numeric column.

Returns:
- Total number of detected outliers
- An error message if the selected column is not numeric
"""

    global df

    numeric_column = df.select_dtypes(include='number').columns

    if column in numeric_column:
        Q1 = df[column].quantile(0.25)

        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR

        outlier = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

        return f"Outliers : {outlier.shape[0]}"

    else:
        return "You need numeric column to see the total outlier."

@tool
def histogram_tool(column) -> object:
    """
    Generate a histogram for a numeric column.

    Parameters:
    - column (str): Name of the numeric column.

    Returns:
    - A matplotlib Figure object containing the histogram with a KDE curve
    - An error message if the selected column is not numeric
    """

    global df

    numeric_column = df.select_dtypes(include="number").columns

    try:
        if column not in numeric_column:
            return "You need numeric column to generate the histogram plot."

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.histplot(data=df, x=column, kde=True, ax=ax)

        ax.set_title(f"Histogram of {column}")
        fig.tight_layout()

        return fig

    except Exception as e:
        return f"Error : {e}"

@tool
def boxplot_tool(column) -> object:
    """
    Generate a box plot for a numeric column.

    Parameters:
    - column (str): Name of the numeric column.

    Returns:
    - A matplotlib Figure object containing the box plot
    - An error message if the selected column is not numeric
    """

    global df

    numeric_column = df.select_dtypes(include="number").columns

    try:
        if column not in numeric_column:
            return "You need numeric column to generate the box plot."

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.boxplot(x=df[column], ax=ax)

        ax.set_title(f"Boxplot of {column}")
        fig.tight_layout()

        return fig

    except Exception as e:
        return f"Error : {e}"

@tool
def scatterplot_tool(x, y) -> object:
    """
    Generate a scatter plot to visualize the relationship between two numeric columns.

    Parameters:
    - x (str): Name of the numeric column for the x-axis.
    - y (str): Name of the numeric column for the y-axis.

    Returns:
    - A matplotlib Figure object containing the scatter plot
    - An error message if either column is not numeric
    """

    global df

    numeric_column = df.select_dtypes(include="number").columns

    try:
        if x not in numeric_column or y not in numeric_column:
            return "You need both columns numerics to generate the scatterplot."

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.scatterplot(data=df, x=x, y=y, ax=ax)

        ax.set_title(f"{x} and {y} wise Scatterplot")
        fig.tight_layout()

        return fig

    except Exception as e:
        return f"Error : {e}"

@tool
def heatmap_tool() -> object:
    """
    Generate a correlation heatmap for all numeric columns in the dataset.

    Returns:
    - A matplotlib Figure object containing the correlation heatmap
    - An error message if the heatmap cannot be generated
    """

    global df

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            df.corr(numeric_only=True),
            annot=True,
            fmt=".2f",
            ax=ax
        )

        ax.set_title("Heatmap of numeric columns")
        fig.tight_layout()

        return fig

    except Exception as e:
        return f"Error : {e}"

@tool
def pie_chart(column) -> object:
    """
    Generate a pie chart showing the distribution of values in a categorical column.

    Parameters:
    - column (str): Name of the categorical column.

    Returns:
    - A matplotlib Figure object containing the pie chart
    - An error message if the selected column is not categorical
    """

    global df

    categorical_columns = df.select_dtypes(include="object").columns

    try:
        if column not in categorical_columns:
            return "You need a categorical column to generate the pie chart."

        counts = df[column].value_counts()

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.pie(
            counts,
            labels=counts.index,
            autopct="%1.1f%%"
        )

        ax.set_title(f"Distribution of {column}")
        ax.axis("equal")

        fig.tight_layout()

        return fig

    except Exception as e:
        return f"Error : {e}"

@tool
def barchart_tool(x, y) -> object:
    """
    Generate a bar chart using one categorical column and one numeric column.

    Parameters:
    - x (str): Name of the categorical column for the x-axis.
    - y (str): Name of the numeric column for the y-axis.

    Returns:
    - A matplotlib Figure object containing the bar chart
    - An error message if the selected columns are invalid
    """

    global df

    numeric_column = df.select_dtypes(exclude="object").columns
    categorical_column = df.select_dtypes(include="object").columns

    try:
        if x not in categorical_column or y not in numeric_column:
            return "You need one categorical column for x and one numeric column for y to generate the bar chart."

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.barplot(data=df, x=x, y=y, ax=ax)

        ax.set_title(f"{x} wise {y} Distribution")
        ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()

        return fig

    except Exception as e:
        return f"Error : {e}"

@tool
def linechart_tool(x, y) -> object:
    """
    Generate a line chart using two numeric columns.

    Parameters:
    - x (str): Name of the numeric column for the x-axis.
    - y (str): Name of the numeric column for the y-axis.

    Returns:
    - A matplotlib Figure object containing the line chart
    - An error message if either column is not numeric
    """

    global df

    numeric_column = df.select_dtypes(include="number").columns

    try:
        if x not in numeric_column or y not in numeric_column:
            return "You need numeric columns to generate the line chart."

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(df[x], df[y], marker=".")

        ax.set_title(f"{x} wise {y} Line Plot")

        fig.tight_layout()

        return fig

    except Exception as e:
        return f"Error : {e}"

@tool
def fill_missing_values(column)->str:
    """
Fill missing values in a specified column.

Parameters:
- column (str): Name of the column.

Returns:
For numeric columns:
- Missing values are filled using the column mean.

For categorical columns:
- Missing values are filled using the most frequent value (mode).

Also returns:
- Number of missing values before cleaning
- Number of missing values after cleaning
- Total missing values filled
"""

    global df

    numeric_column = df.select_dtypes(include='number').columns

    category_column = df.select_dtypes(include='object').columns

    before_clean = df[column].isnull().sum()

    if column in numeric_column:
        df[column] = df[column].fillna(df[column].mean())
    elif column in category_column:
        if df[column].mode().empty:
            return "Cannot fill missing values because the column has no valid mode."
        df[column] = df[column].fillna(df[column].mode()[0])
    else:
        return f"Column '{column}' not found."

    after_clean = df[column].isnull().sum()

    return f"""
    Column : {column}\n
    Missing Before : {before_clean}\n
    Missing After : {after_clean}\n
    Successfully Filled {before_clean} missing values.
    """

@tool
def remove_duplicates()->str:
    """
Remove duplicate rows from the currently loaded dataset.

Returns:
- Number of duplicate rows removed
- A message if no duplicate rows are found
"""

    global df

    duplicat_value = df.duplicated().sum()

    if duplicat_value == 0:
        return "No duplicated rows are available."
    else:
        df = df.drop_duplicates()

    return f"Remove {duplicat_value} Duplicates rows."

@tool
def rename_column(old,new)->str:
    """
Rename an existing column in the dataset.

Parameters:
- old (str): Current column name.
- new (str): New column name.

Returns:
- Success message after renaming the column
- An error message if the specified column does not exist
"""

    global df

    if old not in df.columns:
        return "Column not found."

    df = df.rename(columns={old: new})

    return f"{old} name rename to {new} successfully."

@tool
def drop_column(column:str)->str:
    """
Remove a column from the dataset.

Parameters:
- column (str): Name of the column to remove.

Returns:
- Success message after dropping the column
- An error message if the specified column does not exist
"""

    global df

    if column not in df.columns:
        return "Column not found."

    df = df.drop(columns=[column])

    return f"Column {column} Drop Successfully."

@tool
def change_dtype(column, datatype) -> str:
    """
Change the datatype of a specified column.

Parameters:
- column (str): Name of the column whose datatype should be changed.
- datatype (str): Target datatype. Supported values:
  int, float, str, datetime, category, bool.

Returns:
- Column name
- Previous datatype
- Updated datatype
- Success message after conversion
- An error message if the conversion fails or the column does not exist
"""

    global df

    try:
        if column not in df.columns:
            return f"Column '{column}' not found."

        before = str(df[column].dtype)

        if datatype == "int":
            df[column] = pd.to_numeric(df[column], errors='coerce')

        elif datatype == "float":
            df[column] = df[column].astype(float)

        elif datatype == "str":
            df[column] = df[column].astype(str)

        elif datatype == "category":
            df[column] = df[column].astype("category")

        elif datatype == "bool":
            df[column] = df[column].astype(bool)

        elif datatype == "datetime":
            df[column] = pd.to_datetime(df[column], errors='coerce')

        else:
            return "Supported datatypes: int, float, str, datetime, category, bool."

        after = str(df[column].dtype)

        return (
            f"Datatype changed successfully.\n"
            f"Column : {column}\n"
            f"Before : {before}\n"
            f"After  : {after}"
        )

    except Exception as e:
        return f"Error: {e}"

@tool
def summary_generate()->str:
    """
    Generate a comprehensive summary report of the dataset after data cleaning.

    This tool provides an overview of all preprocessing and data quality checks
    performed on the currently loaded dataset, including:

    - Dataset shape (rows and columns)
    - Total number of features
    - Column names
    - Data types of each column
    - Missing values count and percentage
    - Duplicate rows count
    - Memory usage
    - Numerical and categorical column summary
    - Outlier analysis (if performed)
    - Columns with high missing values
    - Columns removed during cleaning (if any)
    - Data type conversions (if any)
    - Missing value handling summary
      (filled, removed, or unchanged)
    - Duplicate handling summary
    - Final dataset shape after cleaning
    - Overall data quality observations
    - Recommendations for further preprocessing or modeling

    Returns:
        str: A well-formatted dataset summary report describing the current
        state of the cleaned dataset and all preprocessing insights.
    """

@tool
def save_cleaned_csv():
    """
    Save the currently cleaned dataset as a CSV file.

    Returns:
    - Saves the modified dataset to the current working directory
      with the filename 'Cleaned_dataset.csv'.
    - Returns a success message after the file is saved.
    - Returns an error message if no dataset has been loaded.
    """

    global df

    df.to_csv(f"Cleaned_dataset.csv", index=False)

    return "Cleaned dataset saved successfully as 'Cleaned_dataset.csv'."
