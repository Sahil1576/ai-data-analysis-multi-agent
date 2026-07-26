# 🧠 AI Data Analysis Multi-Agent System

An intelligent **Multi-Agent AI Data Analysis Assistant** built with **LangChain**, **Mistral AI**, and **Streamlit**. The application automatically routes user queries to specialized AI agents for dataset management, exploratory data analysis (EDA), visualization, and data cleaning.

---

## 🎥 Demo

### 🌐 Live Application

**Streamlit Demo:**  
https://ai-data-analysis-multi-agent-zjnhjbztnw22bxswfasdyt.streamlit.app/

---

## 🚀 Features

- 🤖 Supervisor Agent for intelligent query routing
- 📂 Upload CSV datasets
- 📁 Dataset Management Agent
- 🔍 Exploratory Data Analysis (EDA) Agent
- 📈 Visualization Agent
- 🧹 Data Cleaning Agent
- 💬 Conversational AI interface
- 📊 Automatic chart generation
- 📋 Dataset information & preview
- 📉 Correlation analysis
- 🚨 Missing value analysis
- 🔍 Duplicate detection
- 📦 Outlier detection
- 📝 Automatic data cleaning summary
- 💾 Download cleaned dataset
- ⚡ Quick Action buttons
- 🎨 Professional Streamlit Dashboard

---

# 🏗️ System Architecture

```
                        User
                          │
                          ▼
                 Supervisor Agent
                          │
     ┌────────────┬──────────────┬──────────────┬──────────────┐
     │            │              │              │
     ▼            ▼              ▼              ▼
 Dataset      EDA Agent    Visualization    Data Cleaning
Management                     Agent            Agent
     │            │              │              │
     └────────────┴──────────────┴──────────────┘
                    LangChain Tools
```

---

# 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Mistral AI
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Python Dotenv

---

# 📊 Supported Operations

## 📁 Dataset Management

- Load CSV files
- Dataset Preview
- First & Last Rows
- Dataset Shape
- Dataset Information
- Column Names
- Data Types

---

## 🔍 Exploratory Data Analysis

- Dataset Summary
- Statistical Summary
- Numerical Analysis
- Categorical Analysis
- Missing Value Analysis
- Duplicate Detection
- Correlation Analysis
- Unique Values
- Value Counts
- Column Statistics
- Filtering
- Sorting
- Outlier Detection

---

## 📈 Visualization

- Histogram
- Scatter Plot
- Box Plot
- Correlation Heatmap
- Pie Chart
- Bar Chart
- Line Chart

---

## 🧹 Data Cleaning

- Fill Missing Values
- Remove Duplicate Rows
- Rename Columns
- Drop Columns
- Change Data Types
- Generate Cleaning Summary
- Save Cleaned Dataset

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/your-username/ai-data-analysis-multi-agent.git

cd ai-data-analysis-multi-agent
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file.

```env
MISTRAL_API_KEY=YOUR_API_KEY
```

---

## Run Application

```bash
streamlit run app.py
```

---

# 💬 Example Questions

### Dataset Management

- Show dataset information
- Show first five rows
- Display column names
- Show dataset shape

### EDA

- Give me a complete EDA summary
- Show missing values
- Show duplicate rows
- Find outliers in Age
- Show correlation matrix

### Visualization

- Create a histogram of Sales
- Generate a heatmap
- Plot a scatter plot between Age and Salary
- Create a pie chart of Gender
- Generate a bar chart

### Data Cleaning

- Fill missing values in Age
- Remove duplicate rows
- Rename CustomerID to Customer_ID
- Drop Email column
- Convert Date column to datetime
- Save cleaned dataset

---

# 📂 Project Structure

```
.
├── app.py
├── Supervisor.py
├── agents.py
├── tools.py
├── requirements.txt
├── .env
├── README.md
```

---

# 🚀 Future Improvements

- SQL Database Support
- Excel File Support
- PDF Analysis
- Machine Learning Integration
- Automated Feature Engineering
- PDF Report Generation
- Cloud Database Support
- Voice Commands
- Multi-file Dataset Analysis

---

# 📌 GitHub Topics

```
langchain
streamlit
multi-agent
ai
llm
mistral
python
data-analysis
eda
csv
visualization
machine-learning
data-cleaning
genai
```

---

# 👨‍💻 Author

**Sahil Katve**

If you found this project useful, consider giving it a ⭐ on GitHub.
