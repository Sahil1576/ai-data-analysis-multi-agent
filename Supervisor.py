from agents import dataset_management, eda, visualization, data_cleaning
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
load_dotenv()
from tools import summary_generate

llm = ChatMistralAI(model='mistral-small-2506')

class Supervisor():

    def __init__(self):
        self.dataset_management = dataset_management
        self.eda = eda
        self.visualization = visualization
        self.data_cleaning = data_cleaning
        self.summary_generate = summary_generate

    def create_decision(self, user_query):
        decision = llm.invoke(f"""
You are a Supervisor Agent responsible only for routing user requests.

Your job is to read the user's query and decide which specialized agent should handle it.

Available agents:

1. dataset_management
Use for:
- Loading CSV files
- Dataset preview
- First/last rows
- Dataset information
- Shape
- Columns
- Data types

2. eda
Use for:
- Dataset summary
- Statistics
- Missing values
- Duplicate checking
- Correlation
- Unique values
- Value counts
- Column statistics
- Filtering
- Sorting
- Outlier detection

3. visualization
Use for:
- Histogram
- Boxplot
- Scatterplot
- Heatmap
- Pie chart
- Bar chart
- Line chart
- Any graph or visualization

4. data_cleaning
Use for:
- Fill missing values
- Remove duplicates
- Rename columns
- Drop columns
- Change datatypes
- summary_generate
- Clean dataset
- save_cleaned_csv

If the query is unrelated to dataset operations, return:
general

User Query:
{user_query}

Instructions:
- Read the user query carefully.
- Decide which single agent is best suited.
- Return ONLY ONE of these exact values:

dataset_management
eda
visualization
data_cleaning
general

Do not explain.
Do not answer the user's question.
Do not return anything except the agent name.
""")

        return decision