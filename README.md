# 🤖 Agentic Data Science Dashboard

An advanced, research-grade autonomous data science platform built with Streamlit and LangGraph. This dashboard automates the entire machine learning pipeline—from exploratory data analysis (EDA) to fairness audits and reproducible notebook exports—orchestrated by an intelligent agent.

## 🚀 Key Features

### 1. 🧠 Agentic Orchestration
*   **LangGraph Integration**: Uses a state-machine based agent to navigate complex data tasks.
*   **Autonomous Reasoning**: The agent can perform EDA, clean data, and train models based on high-level natural language instructions.

### 2. 🔬 Interpretability & "Glass-Box" Modeling
*   **EBM (Explainable Boosting Machines)**: Highly interpretable, additive models with feature interaction constraints.
*   **FIGS (Fast Interpretable Greedy Trees)**: Produces small, human-readable rule sets for complex problems.
*   **Feature Importance & Local Explanations**: Visualizes how every feature contributes to a specific prediction.

### 3. ⚖️ Fairness & Ethics
*   **Fairness Diagnostic**: Automatically computes parity gaps and accuracy disparities across sensitive attributes (Gender, Race, etc.).
*   **Bias Detection**: Flags potential biases in your data before they reach production.

### 4. 📐 Advanced ML & Tuning
*   **PSO Hyperparameter Search**: Particle Swarm Optimization for finding global optima in parameter spaces.
*   **Cascade Classification**: A structural approach to handling extreme class imbalance by decomposing problems into a hierarchy of binary classifiers.
*   **Domain Knowledge Constraints**: Allows experts to define "impossible" prediction boundaries to ensure logical consistency.

### 5. 📓 Reproducibility & Collaboration
*   **Notebook Exporter**: Generates fully executable `.ipynb` files capturing every pipeline step.
*   **Pipeline Sharing**: Export and import pipeline configurations as JSON for team collaboration and side-by-side comparison.

---

## 🦁 Model Zoo

The platform supports over **20+ model configurations** across 14 unique base algorithms.

### 🔬 Glass-Box (Interpretable) Models
*   **EBM (Explainable Boosting Machines)**: Full transparency via additive functions.
*   **FIGS (Fast Interpretable Greedy Trees)**: Human-readable logic flows and rules.

### 📊 Classification Models
*   **Standard**: KNN, Decision Tree, SVM, Random Forest, Gradient Boosting, MLP Neural Network.
*   **Specialized**: **Cascade Classifier** for extreme class imbalance (e.g., fraud detection).

### 📈 Regression Models
*   **Standard**: Linear, Lasso, Ridge, Decision Tree, Random Forest, Gradient Boosting, KNN Regressor, SVR, MLP Neural Network.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
*   Python 3.11+
*   A Groq API Key (for the agent logic)

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/complexiota/Agentice_Data_Science_DasBoard.git
cd Agentice_Data_Science_DasBoard
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=llama3-70b-8192
```

---

## 🏃 How to Run

Launch the dashboard using Streamlit:
```bash
streamlit run app.py
```

### Typical Workflow:
1.  **Upload**: Drop a CSV, Excel, XML, or JSON file.
2.  **Consult the Agent**: Use the chat interface to ask questions about your data.
3.  **Pipeline Navigation**: Move through EDA, Preprocessing, and ML Algorithm selection using the sidebar.
4.  **Interpret & Audit**: Check the "Glass-Box" explanations and "Fairness" diagnostics after training.
5.  **Export**: Download your trained model or export the entire workflow as a Jupyter Notebook.

---

## 📄 License
This project is intended for research and educational purposes.
