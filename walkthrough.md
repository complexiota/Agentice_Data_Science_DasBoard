# 🤖 Agentic Data Scientist Walkthrough

Welcome to your **Agentic Data Scientist** dashboard! This project has been transformed from a monolithic dashboard into a modular, chat-first autonomous machine learning orchestrator.

## 🌟 What it Can Do Currently

The application provides a **ChatGPT-like interface** where you can interact with your data using natural language. Behind the scenes, a **LangGraph ReAct Agent** coordinates the following tasks:

### 1. Data Loading & Understanding
- **Multi-format Support**: Upload CSV, XML, JSON, Excel, or Text files.
- **Auto-Analysis**: The agent can describe your dataset's shape, column types, and identify missing values using the `analyze_data` and `get_data_summary` tools.

### 2. Autonomous Data Cleaning
- **Intelligent Imputation**: The agent can clean your data using various methods:
    - Dropping rows/columns with nulls.
    - Statistical filling (Mean, Median, Mode).
    - Advanced techniques like **KNN Imputation** and **Forward/Backward Fill**.
- **Targeted Cleaning**: You can ask the agent to clean specific columns or the entire dataset.

### 3. Machine Learning Lifecycle
- **Problem Type Detection**: Automatically detects whether your task is **Classification** or **Regression** based on the target column's data type and distribution.
- **Algorithm Selection**: Supports a wide range of models:
    - **Classification**: KNN, Decision Tree, SVM, Random Forest.
    - **Regression**: Linear/Lasso Regression, GBR, MLP (Neural Networks).
- **Evaluation**: Performs k-fold cross-validation and reports key metrics (Accuracy, R², RMSE, etc.) directly in the chat.

### 4. Interactive Feedback
- **Thought Streaming**: Watch the agent "think" in real-time through an expandable log of its reasoning and tool usage.
- **Metric Visualizations**: Key model performance metrics are rendered as Streamlit components within the chat stream.

---

## 🚀 How to Run

### 1. Environment Setup
Ensure you have the dependencies installed:
```powershell
pip install -r requirements.txt
```

### 2. Configuration
Create or update your `.env` file with your Groq API Key:
```text
GROQ_API_KEY="your_api_key_here"
GROQ_MODEL="openai/gpt-oss-120b"
```

### 3. Launch the App
Run the Streamlit application:
```powershell
streamlit run app.py
```

---

## 🏗️ Architecture Overview

The project is structured modularly to separate business logic from the UI and Agent orchestration:

```text
agentic-dashboard/
├── app.py                 # Main Streamlit Chat Interface
├── src/
│   ├── core/              # Pure Business Logic (Non-Streamlit)
│   │   ├── data_loader.py    # File processing
│   │   ├── data_cleaner.py   # Data cleaning algorithms
│   │   ├── model_trainer.py  # ML training pipeline
│   │   └── data_store.py     # Shared memory for UI and Agent
│   └── agent/             # LangGraph Orchestration
│       ├── tools.py       # LangChain tool definitions
│       └── graph.py       # ReAct Agent graph definition
```

### Key Innovation: The Shared Data Store
Because Streamlit tools run in background threads, they cannot access `st.session_state` reliably. We implemented `src/core/data_store.py` as a singleton module to share the active DataFrame and model results between the UI thread and the Agent tools.

---

## 🛠️ How to Add More Functionalities

Adding new capabilities is a straightforward 3-step process:

### Step 1: Implement Core Logic
Add a new Python function in `src/core/` (e.g., in a new file `visualizer.py` or `feature_eng.py`).
- *Example*: A function that generates a Plotly chart from a DataFrame.

### Step 2: Create an Agent Tool
In `src/agent/tools.py`, wrap your core function with the `@tool` decorator.
```python
@tool
def my_new_feature(param1: str) -> str:
    """Description for the Agent to understand when to use this."""
    df = data_store.get_data()
    # Call your core logic...
    return "Result for the agent"
```

### Step 3: Register the Tool
Import and add your new tool to the list in `src/agent/graph.py`:
```python
from src.agent.tools import ..., my_new_feature

def get_agent():
    # ...
    tools = [..., my_new_feature]
    # ...
```

The agent will automatically "discover" the new capability and start using it based on its description!

---

## 📝 Current Development Progress
- [x] ChatGPT-style UI Overhaul
- [x] LangGraph ReAct Agent Integration
- [x] Robust Problem Type Detection (Fixed binary vs continuous bug)
- [x] Shared Data Store Implementation
- [x] Support for EDA, Cleaning, and Training via Chat
- [ ] Support for Visualizations via Chat (Next recommended feature)
- [ ] Exporting Cleaned Data via Chat (Next recommended feature)
