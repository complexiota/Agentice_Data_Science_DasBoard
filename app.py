import matplotlib
matplotlib.use("Agg")

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import uuid
import io
import base64
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import (
    train_test_split, learning_curve, cross_val_score,
    GridSearchCV, RandomizedSearchCV,
)
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestClassifier,
    RandomForestRegressor, GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    accuracy_score, r2_score,
    mean_absolute_error, mean_squared_error,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score,
)
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from src.core import data_store

# ─────────────────────────────────────────────
#  PAGE CONFIG & GLOBAL STYLES
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Data Scientist",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
st.markdown("""
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=Inter:wght@300;400;500;600&display=swap');

  /* ── Root Palette ── */
  :root {
    --bg:          #0d0f14;
    --surface:     #151822;
    --surface2:    #1c2030;
    --border:      rgba(255,255,255,0.07);
    --accent1:     #6c63ff;
    --accent2:     #00e5ff;
    --accent3:     #ff6584;
    --text:        #e8eaf0;
    --muted:       #6b7280;
    --success:     #10b981;
    --warning:     #f59e0b;
    --danger:      #ef4444;
    --radius:      12px;
    --radius-sm:   8px;
  }

  /* ── Base ── */
  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
  }
  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stHeader"] { background: transparent !important; }

  /* ── Typography ── */
  h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }
  h1 { font-size: 2.2rem !important; font-weight: 800 !important;
       background: linear-gradient(135deg, var(--accent1), var(--accent2));
       -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  h2 { font-size: 1.4rem !important; font-weight: 700 !important; color: var(--text) !important; }
  h3 { font-size: 1.1rem !important; font-weight: 700 !important; color: var(--text) !important; }
  code, .stCode, pre { font-family: 'DM Mono', monospace !important; }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.2rem !important;
    transition: transform 0.2s, box-shadow 0.2s;
  }
  [data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(108,99,255,0.15);
  }
  [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
  [data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'Syne', sans-serif !important; font-size: 1.6rem !important; font-weight: 800 !important; }
  [data-testid="stMetricDelta"] > div { font-size: 0.78rem !important; }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, var(--accent1), #5b52e6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 12px rgba(108,99,255,0.3) !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(108,99,255,0.45) !important;
    background: linear-gradient(135deg, #7c74ff, var(--accent1)) !important;
  }
  .stButton > button[kind="secondary"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
  }

  /* ── Selectbox / multiselect / input ── */
  .stSelectbox > div > div,
  .stMultiSelect > div > div,
  .stTextInput > div > div,
  .stNumberInput > div > div,
  .stTextArea > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
  }
  .stSelectbox > div > div:focus-within,
  .stMultiSelect > div > div:focus-within,
  .stTextInput > div > div:focus-within {
    border-color: var(--accent1) !important;
    box-shadow: 0 0 0 2px rgba(108,99,255,0.2) !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: var(--surface2) !important;
    border-radius: var(--radius-sm) !important;
    padding: 4px !important;
    gap: 4px !important;
    border-bottom: none !important;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    padding: 6px 16px !important;
    border: none !important;
  }
  .stTabs [aria-selected="true"] {
    background: var(--accent1) !important;
    color: #fff !important;
  }

  /* ── Expander ── */
  .streamlit-expanderHeader {
    background: var(--surface2) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-weight: 600 !important;
  }
  .streamlit-expanderContent {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
  }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] iframe { border-radius: var(--radius-sm) !important; }

  /* ── Alerts ── */
  .stSuccess { background: rgba(16,185,129,0.12) !important; border-left: 3px solid var(--success) !important; border-radius: var(--radius-sm) !important; }
  .stWarning { background: rgba(245,158,11,0.12) !important; border-left: 3px solid var(--warning) !important; border-radius: var(--radius-sm) !important; }
  .stError   { background: rgba(239,68,68,0.12)  !important; border-left: 3px solid var(--danger)  !important; border-radius: var(--radius-sm) !important; }
  .stInfo    { background: rgba(108,99,255,0.12) !important; border-left: 3px solid var(--accent1) !important; border-radius: var(--radius-sm) !important; }

  /* ── Progress bar ── */
  .stProgress > div > div { background: linear-gradient(90deg, var(--accent1), var(--accent2)) !important; border-radius: 99px !important; }
  .stProgress > div { background: var(--surface2) !important; border-radius: 99px !important; }

  /* ── Radio / checkbox ── */
  [data-testid="stRadio"] label { color: var(--text) !important; }

  /* ── Slider ── */
  [data-testid="stSlider"] [role="slider"] { background: var(--accent1) !important; }

  /* ── Sidebar navigation radio ── */
  [data-testid="stSidebar"] .stRadio > div { gap: 2px !important; }
  [data-testid="stSidebar"] .stRadio label {
    background: transparent !important;
    border-radius: var(--radius-sm) !important;
    padding: 6px 10px !important;
    font-size: 0.85rem !important;
    transition: background 0.15s;
    cursor: pointer;
  }
  [data-testid="stSidebar"] .stRadio label:hover { background: var(--surface2) !important; }

  /* ── Chat messages ── */
  [data-testid="stChatMessage"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 10px !important;
  }
  [data-testid="stChatInput"] > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
  }
  [data-testid="stChatInput"] > div:focus-within {
    border-color: var(--accent1) !important;
    box-shadow: 0 0 0 2px rgba(108,99,255,0.2) !important;
  }

  /* ── Section dividers ── */
  hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

  /* ── Custom pill badge ── */
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .badge-purple { background: rgba(108,99,255,0.2); color: var(--accent1); border: 1px solid rgba(108,99,255,0.3); }
  .badge-cyan   { background: rgba(0,229,255,0.12); color: var(--accent2); border: 1px solid rgba(0,229,255,0.25); }
  .badge-green  { background: rgba(16,185,129,0.12); color: var(--success); border: 1px solid rgba(16,185,129,0.25); }

  /* ── Hero section ── */
  .hero-card {
    background: linear-gradient(135deg, var(--surface2), #1a1f35);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero-card::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(108,99,255,0.15), transparent 70%);
    border-radius: 50%;
  }

  /* ── Feature grid card ── */
  .feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 1rem;
    margin-top: 1.5rem;
  }
  .feature-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    transition: all 0.2s;
  }
  .feature-card:hover {
    border-color: rgba(108,99,255,0.4);
    box-shadow: 0 4px 20px rgba(108,99,255,0.1);
    transform: translateY(-2px);
  }
  .feature-card .icon { font-size: 1.4rem; margin-bottom: 0.5rem; }
  .feature-card .title { font-weight: 700; font-size: 0.9rem; color: var(--text); margin-bottom: 0.25rem; }
  .feature-card .desc { font-size: 0.78rem; color: var(--muted); line-height: 1.4; }

  /* ── Subheader style override ── */
  .section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.2rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
  }
  .section-header h2 { margin: 0 !important; }

  /* ── Pipeline tracker ── */
  .pipeline-step {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 8px;
    border-radius: 6px;
    font-size: 0.82rem;
    margin-bottom: 3px;
    transition: background 0.15s;
  }
  .pipeline-step.done { color: var(--success); background: rgba(16,185,129,0.08); }
  .pipeline-step.todo { color: var(--muted); }

  /* ── Chart container ── */
  .chart-wrap {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    margin-bottom: 1rem;
  }

  /* ── Agent thinking box ── */
  .agent-thinking {
    background: rgba(108,99,255,0.08);
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: var(--radius-sm);
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: var(--muted);
  }

  /* ── Download button ── */
  .stDownloadButton > button {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    box-shadow: none !important;
  }
  .stDownloadButton > button:hover {
    border-color: var(--accent2) !important;
    color: var(--accent2) !important;
    transform: none !important;
    box-shadow: 0 0 12px rgba(0,229,255,0.15) !important;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--surface); }
  ::-webkit-scrollbar-thumb { background: #2d3350; border-radius: 99px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--accent1); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
def apply_dark_theme():
    plt.rcParams.update({
        "figure.facecolor":  "#151822",
        "axes.facecolor":    "#1c2030",
        "axes.edgecolor":    "#2d3350",
        "axes.labelcolor":   "#e8eaf0",
        "axes.titlecolor":   "#e8eaf0",
        "xtick.color":       "#6b7280",
        "ytick.color":       "#6b7280",
        "text.color":        "#e8eaf0",
        "grid.color":        "#2d3350",
        "grid.alpha":        0.4,
        "legend.facecolor":  "#1c2030",
        "legend.edgecolor":  "#2d3350",
        "legend.labelcolor": "#e8eaf0",
        "figure.dpi":        130,
        "axes.titlesize":    11,
        "axes.labelsize":    9,
    })

apply_dark_theme()

ACCENT_PALETTE = ["#6c63ff", "#00e5ff", "#ff6584", "#10b981", "#f59e0b", "#a78bfa", "#34d399", "#fb923c"]

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
FILE_EXTENSIONS = {
    "CSV":   ["csv"],
    "XML":   ["xml"],
    "JSON":  ["json"],
    "Excel": ["xlsx", "xls"],
    "Text":  ["txt"],
}

FEATURES = [
    "🤖 AI Agent Chat",
    "📊 Data Overview",
    "📋 Data Profiling Report",
    "🔍 EDA – Null Value Handling",
    "⚙️ Feature Engineering",
    "📈 Basic Statistics",
    "📉 Visualizations",
    "🔗 Correlation Plot",
    "🚨 Outlier Detection",
    "⚖️ Class Imbalance Handling",
    "📈 Time Series Analysis",
    "🤖 ML Algorithm",
    "🏆 Multi-Model Comparison",
    "🔧 Hyperparameter Tuning",
    "📦 Model Export & Predict",
    "📄 Generate Report",
    "🧠 Domain Knowledge Constraints", 
    "🤝 Pipeline Collaboration",  
]

# ─────────────────────────────────────────────
#  SHARED DATA STORE HELPER
# ─────────────────────────────────────────────
def set_data(df): data_store.set_data(df)
def get_data(): return data_store.get_data()
def set_model_results(r): data_store.set_model_results(r)
def get_model_results(): return data_store.get_model_results()


# ─────────────────────────────────────────────
#  HELPER: fig → base64 PNG for agent chat
# ─────────────────────────────────────────────
def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def render_figure_in_chat(fig, caption: str = ""):
    """Render a matplotlib figure as an image in chat context."""
    b64 = fig_to_b64(fig)
    plt.close(fig)
    caption_html = f"<p style='text-align:center;color:#6b7280;font-size:0.78rem;margin-top:6px;'>{caption}</p>" if caption else ""
    st.markdown(
        f'<div class="chart-wrap">'
        f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:8px;" />'
        f'{caption_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  FILE PROCESSING
# ─────────────────────────────────────────────
def process_file(uploaded_file, file_mode: str):
    try:
        if file_mode == "CSV":
            return pd.read_csv(uploaded_file)
        elif file_mode == "XML":
            tree = ET.parse(uploaded_file)
            root = tree.getroot()
            xml_data = [
                {child.tag: child.text if child.text else None for child in element}
                for element in root
            ]
            return pd.DataFrame(xml_data)
        elif file_mode == "JSON":
            return pd.json_normalize(json.load(uploaded_file))
        elif file_mode == "Excel":
            return pd.read_excel(uploaded_file)
        elif file_mode == "Text":
            try:
                lines = [line.decode().strip() for line in uploaded_file.readlines()]
            except UnicodeDecodeError:
                lines = [line.strip() for line in uploaded_file.readlines()]
            return pd.DataFrame(lines, columns=["Text"])
    except Exception as e:
        st.error(f"Error processing file: {e}")
    return None


# ─────────────────────────────────────────────
#  FEATURE — AI AGENT CHAT  (FIXED)
# ─────────────────────────────────────────────
def show_ai_agent_chat(data, groq_api_key):
    st.markdown('<div class="section-header"><h2>🤖 AI Agent Chat</h2></div>', unsafe_allow_html=True)
    st.caption("Chat naturally — I can clean data, run analysis, train ML models, and **show you charts**.")

    if not groq_api_key:
        st.warning("Please enter your Groq API Key in the sidebar.")
        return

    # ── Data context badge ──
    if data is not None:
        col_a, col_b, col_c = st.columns(3)
        col_a.markdown(f'<span class="badge badge-purple">📐 {data.shape[0]:,} rows × {data.shape[1]} cols</span>', unsafe_allow_html=True)
        col_b.markdown(f'<span class="badge badge-cyan">🔢 {len(data.select_dtypes(include="number").columns)} numeric cols</span>', unsafe_allow_html=True)
        col_c.markdown(f'<span class="badge badge-green">🔍 {data.isnull().sum().sum()} nulls</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── LangGraph import ──
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from src.agent.graph import get_agent
        use_langgraph = True
    except ImportError:
        use_langgraph = False

    # ── Session init ──
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "👋 Hello! I'm your autonomous data scientist.\n\n"
                    "Upload a dataset in the sidebar, then ask me anything — "
                    "I can **analyze**, **clean**, **visualize**, and **train ML models** on it!\n\n"
                    "Try asking:\n"
                    "- *Show me a histogram of all numeric columns*\n"
                    "- *What are the top correlations in this dataset?*\n"
                    "- *Train a Random Forest and show me the results*"
                ),
            }
        ]
    if "agent_thread_id" not in st.session_state:
        st.session_state["agent_thread_id"] = str(uuid.uuid4())

    # ── Render history ──
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # Render any stored chart images
                if "charts" in msg:
                    for chart_b64 in msg["charts"]:
                        st.markdown(
                            f'<div class="chart-wrap"><img src="data:image/png;base64,{chart_b64}" style="width:100%;border-radius:8px;" /></div>',
                            unsafe_allow_html=True,
                        )

    # ── Chat input ──
    if prompt := st.chat_input("Ask me to analyze, clean, or model your data..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # ── LangGraph path ──
            if use_langgraph and data is not None:
                system_content = (
                    "You are an expert autonomous data scientist. "
                    "Always use tools to perform actions — never guess results. "
                    "Be concise and report key metrics clearly."
                    "\n\nCurrent dataset info:\n"
                    "- Shape: " + str(data.shape[0]) + " rows × " + str(data.shape[1]) + " columns\n"
                    "- Columns: " + str(data.columns.tolist()) + "\n"
                    "- Dtypes: " + str(data.dtypes.apply(str).to_dict()) + "\n"
                    "- Total nulls: " + str(data.isnull().sum().sum()) + "\n"
                    f"- Numeric columns: {data.select_dtypes(include='number').columns.tolist()}"
                )
                try:
                    agent = get_agent()
                    config = {"configurable": {"thread_id": st.session_state["agent_thread_id"]}}
                    inputs = {
                        "messages": [
                            SystemMessage(content=system_content),
                            HumanMessage(content=prompt),
                        ]
                    }
                    with st.expander("🔍 Agent Thinking...", expanded=False):
                        for event in agent.stream(inputs, config=config, stream_mode="values"):
                            message = event["messages"][-1]
                            if message.type == "ai":
                                if message.content:
                                    st.write(f"🧠 {message.content}")
                                if hasattr(message, "tool_calls") and message.tool_calls:
                                    for tc in message.tool_calls:
                                        st.write(f"🛠️ **Tool:** `{tc['name']}` → `{tc['args']}`")
                            elif message.type == "tool":
                                preview = message.content[:300]
                                st.write(f"📋 **Result:** {preview}{'...' if len(message.content) > 300 else ''}")
                    final_content = agent.get_state(config).values["messages"][-1].content
                    st.markdown(final_content)
                    st.session_state["messages"].append({"role": "assistant", "content": final_content})
                    return
                except Exception as e:
                    st.warning(f"LangGraph agent error: {e}. Falling back to Groq direct mode.")

            # ── Groq direct path (FIXED: dataset passed, charts captured) ──
            try:
                from groq import Groq
                import re
                client = Groq(api_key=groq_api_key)

                # Build rich dataset context
                df_context = ""
                if data is not None:
                    numeric_cols = data.select_dtypes(include="number").columns.tolist()
                    cat_cols = data.select_dtypes(include="object").columns.tolist()
                    desc_stats = data[numeric_cols].describe().to_string() if numeric_cols else "No numeric columns"
                    df_context = f"""
DataFrame is available as `df` variable with the following info:
- Shape: {data.shape[0]} rows × {data.shape[1]} columns
- Numeric columns: {numeric_cols}
- Categorical columns: {cat_cols}
- Null counts per column: {data.isnull().sum().to_dict()}
- Descriptive statistics:
{desc_stats}
- First 3 rows:
{data.head(3).to_string()}
"""

                groq_prompt = f"""You are an expert data science assistant with direct access to a pandas DataFrame.

{df_context}

User request: "{prompt}"

IMPORTANT INSTRUCTIONS:
1. Always write Python code that actually uses the `df` variable (already loaded).
2. For any visualization, use matplotlib/seaborn and ALWAYS call `plt.tight_layout()` at the end. Do NOT call `plt.show()` — the code runner will capture figures automatically.
3. For ML tasks, use the df variable directly.
4. Wrap ALL executable code in a single ```python ... ``` block.
5. After the code block, provide a clear explanation of what you did and key insights.
6. Be specific and actionable.
7. COMPATIBILITY: This environment runs Python 3.11. DO NOT use backslashes or reused outer quotes inside f-string expressions (e.g. move complex strings to variables before the f-string).
"""
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful, expert data science assistant. Always write working Python code using the provided df variable."},
                        {"role": "user", "content": groq_prompt},
                    ],
                    model=os.getenv("GROQ_MODEL", "llama3-70b-8192"),
                    temperature=0.1,
                    max_tokens=2048,
                )
                reply = response.choices[0].message.content

                # Extract and execute code — capture ALL figures generated
                code_match = re.search(r"```python\s+(.*?)\s+```", reply, re.DOTALL)
                captured_charts = []

                if code_match and data is not None:
                    code = code_match.group(1)

                    # Show executing indicator
                    with st.spinner("⚙️ Executing code..."):
                        exec_globals = {
                            "df": data.copy(),
                            "pd": pd, "np": np,
                            "plt": plt, "sns": sns,
                            "px": px, "go": go, "st": st,
                            # Aliases for bare calls that LLMs sometimes generate without the `st.` prefix
                            "caption": st.caption,
                            "write": st.write,
                            "markdown": st.markdown,
                            "StandardScaler": StandardScaler,
                            "LabelEncoder": LabelEncoder,
                            "train_test_split": train_test_split,
                            "RandomForestClassifier": RandomForestClassifier,
                            "RandomForestRegressor": RandomForestRegressor,
                            "cross_val_score": cross_val_score,
                            "accuracy_score": accuracy_score,
                            "r2_score": r2_score,
                        }
                        try:
                            # Close any stray figures before execution
                            plt.close("all")
                            exec(code, exec_globals)

                            # Capture every figure that was created
                            figs_after = [plt.figure(n) for n in plt.get_fignums()]
                            for fig in figs_after:
                                b64 = fig_to_b64(fig)
                                captured_charts.append(b64)
                                # Display inline
                                st.markdown(
                                    f'<div class="chart-wrap"><img src="data:image/png;base64,{b64}" style="width:100%;border-radius:8px;" /></div>',
                                    unsafe_allow_html=True,
                                )
                            plt.close("all")

                            # If exec_globals has a 'result' or 'output' var, display it
                            for var in ["result", "output", "results"]:
                                if var in exec_globals and isinstance(exec_globals[var], pd.DataFrame):
                                    st.dataframe(exec_globals[var], use_container_width=True)

                        except Exception as ex:
                            st.error(f"Code execution error: {ex}")

                    with st.expander("📋 Executed Code", expanded=False):
                        st.code(code, language="python")

                # Strip code block from displayed reply
                reply_clean = re.sub(r"```python.*?```", "", reply, flags=re.DOTALL).strip()
                if reply_clean:
                    st.markdown(reply_clean)

                # Store message with charts
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": reply_clean or reply,
                    "charts": captured_charts,
                })

            except Exception as e:
                msg = f"❌ Error: {e}"
                st.error(msg)
                st.session_state["messages"].append({"role": "assistant", "content": msg})

    # ── Controls row ──
    st.markdown("<br>", unsafe_allow_html=True)
    col_clr, col_ctx = st.columns([1, 3])
    with col_clr:
        if st.button("🗑️ Clear Chat", key="clear_chat_btn"):
            st.session_state["messages"] = []
            st.session_state["agent_thread_id"] = str(uuid.uuid4())
            st.rerun()
    with col_ctx:
        if data is not None:
            st.caption(f"Dataset context: **{data.shape[0]:,} rows × {data.shape[1]} cols** is automatically passed to every message.")


# ─────────────────────────────────────────────
#  FEATURE 1 — DATA OVERVIEW
# ─────────────────────────────────────────────
def show_data_overview(data: pd.DataFrame):
    st.markdown('<div class="section-header"><h2>📊 Data Overview</h2></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{data.shape[0]:,}")
    col2.metric("Columns", data.shape[1])
    col3.metric("Total Cells", f"{data.shape[0] * data.shape[1]:,}")
    col4.metric("Memory (KB)", f"{data.memory_usage(deep=True).sum() / 1024:.1f}")

    st.write("### Data Types Summary")
    dtype_counts = data.dtypes.value_counts().reset_index()
    dtype_counts.columns = ["Data Type", "Count"]
    st.dataframe(dtype_counts, use_container_width=True)

    st.write("### Columns by Data Type")
    dtype_dict = {}
    for col, dtype in zip(data.columns, data.dtypes):
        dtype_dict.setdefault(str(dtype), []).append(col)
    for dtype_name, columns in dtype_dict.items():
        with st.expander(f"**{dtype_name}** — {len(columns)} column(s)"):
            st.dataframe(pd.DataFrame({"Column Name": columns}), use_container_width=True)

    st.write("### Data Preview")
    st.dataframe(data.head(20), use_container_width=True)


# ─────────────────────────────────────────────
#  FEATURE 2 — DATA PROFILING REPORT
# ─────────────────────────────────────────────
def show_data_profiling(data: pd.DataFrame):
    st.markdown('<div class="section-header"><h2>📋 Data Profiling Report</h2></div>', unsafe_allow_html=True)
    st.caption("Auto-generated summary of every column — distributions, quality, correlations.")

    numeric_cols     = data.select_dtypes(include="number").columns.tolist()
    categorical_cols = data.select_dtypes(include="object").columns.tolist()
    total_cells      = data.shape[0] * data.shape[1]
    total_nulls      = data.isnull().sum().sum()
    duplicate_rows   = data.duplicated().sum()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Rows",             f"{data.shape[0]:,}")
    k2.metric("Columns",          data.shape[1])
    k3.metric("Numeric Cols",     len(numeric_cols))
    k4.metric("Categorical Cols", len(categorical_cols))
    k5.metric("Missing Cells",    f"{total_nulls} ({total_nulls/total_cells*100:.1f}%)")
    k6.metric("Duplicate Rows",   duplicate_rows)

    if duplicate_rows > 0:
        st.warning(f"⚠️ {duplicate_rows} duplicate row(s) found. Consider removing them.")

    st.markdown("---")
    st.write("### 🗂️ Column-by-Column Profile")
    rows = []
    for col in data.columns:
        s         = data[col]
        null_c    = s.isnull().sum()
        unique_c  = s.nunique()
        dtype_str = str(s.dtype)
        if pd.api.types.is_numeric_dtype(s):
            skewness = round(s.skew(), 3)
            kurt     = round(s.kurtosis(), 3)
            shape    = "Normal" if abs(skewness) < 0.5 else "Moderate skew" if abs(skewness) < 1.0 else "High skew"
            rows.append({
                "Column": col, "Type": dtype_str,
                "Non-Null": data.shape[0] - null_c,
                "Null %": f"{null_c/len(s)*100:.1f}%",
                "Unique": unique_c,
                "Mean": round(s.mean(), 4), "Std": round(s.std(), 4),
                "Min": round(s.min(), 4),   "Max": round(s.max(), 4),
                "Skewness": skewness, "Shape": shape, "Kurtosis": kurt,
            })
        else:
            top_val  = s.value_counts().index[0] if unique_c > 0 else "N/A"
            top_freq = s.value_counts().iloc[0]  if unique_c > 0 else 0
            rows.append({
                "Column": col, "Type": dtype_str,
                "Non-Null": data.shape[0] - null_c,
                "Null %": f"{null_c/len(s)*100:.1f}%",
                "Unique": unique_c,
                "Mean": "—", "Std": "—", "Min": "—", "Max": "—",
                "Skewness": "—", "Shape": f"Top: {top_val} ({top_freq}×)", "Kurtosis": "—",
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")
    if len(numeric_cols) >= 2:
        st.write("### 🔗 High-Correlation Pairs (|r| > 0.85)")
        corr_matrix = data[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = (
            upper.stack().reset_index()
                 .rename(columns={"level_0": "Feature A", "level_1": "Feature B", 0: "Correlation"})
        )
        high_corr = high_corr[high_corr["Correlation"] > 0.85].sort_values("Correlation", ascending=False)
        if not high_corr.empty:
            st.warning(f"⚠️ {len(high_corr)} highly correlated pair(s) found.")
            st.dataframe(high_corr.style.background_gradient(subset=["Correlation"], cmap="Purples"), use_container_width=True)
        else:
            st.success("✅ No feature pairs with |r| > 0.85 found.")

    st.markdown("---")
    st.write("### ⚖️ Potential Target Column Imbalance Check")
    for col in data.columns:
        if data[col].nunique() < 20:
            vc     = data[col].value_counts(normalize=True) * 100
            min_pct = vc.min()
            if min_pct < 15:
                st.warning(f"**`{col}`** may be imbalanced — minority class is only {min_pct:.1f}% of data.")
                fig_ib, ax_ib = plt.subplots(figsize=(6, 3))
                colors = [ACCENT_PALETTE[2] if v == vc.min() else ACCENT_PALETTE[0] for v in vc.values]
                ax_ib.bar(vc.index.astype(str), vc.values, color=colors, edgecolor="white")
                ax_ib.set(xlabel=col, ylabel="%", title=f"Class Distribution — {col}")
                ax_ib.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_ib)
                plt.close(fig_ib)

    st.markdown("---")
    st.write("### 💾 Memory Usage")
    mem    = data.memory_usage(deep=True)
    mem_df = pd.DataFrame({"Column": ["(Index)"] + data.columns.tolist(), "Memory (KB)": (mem.values / 1024).round(3)})
    total_kb = mem.sum() / 1024
    st.dataframe(mem_df, use_container_width=True)
    st.info(f"Total memory usage: **{total_kb:.2f} KB** ({total_kb/1024:.3f} MB)")

    if numeric_cols:
        st.markdown("---")
        st.write("### 📊 Distribution Grid (Numeric Columns)")
        n, ncols = len(numeric_cols), 4
        nrows    = (n + ncols - 1) // ncols
        fig_dg, axes_dg = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3), squeeze=False)
        for idx, col in enumerate(numeric_cols):
            ax = axes_dg[idx // ncols][idx % ncols]
            ax.hist(data[col].dropna(), bins=25, color=ACCENT_PALETTE[0], edgecolor="#0d0f14", alpha=0.85)
            ax.set_title(col + "\nskew=" + f"{data[col].skew():.2f}", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
        for idx in range(n, nrows * ncols):
            axes_dg[idx // ncols][idx % ncols].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_dg)
        plt.close(fig_dg)


# ─────────────────────────────────────────────
#  FEATURE 3 — EDA: NULL VALUE HANDLING
# ─────────────────────────────────────────────
def show_eda_null_handling(data: pd.DataFrame):
    st.markdown('<div class="section-header"><h2>🔍 EDA – Null Value Handling</h2></div>', unsafe_allow_html=True)

    null_counts  = data.isnull().sum()
    null_percent = (null_counts / len(data) * 100).round(2)
    null_df = pd.DataFrame({
        "Column": data.columns, "Null Count": null_counts.values,
        "Null %": null_percent.values, "Data Type": data.dtypes.astype(str).values,
        "Non-Null Count": data.notnull().sum().values,
    }).sort_values("Null Count", ascending=False).reset_index(drop=True)

    def highlight_nulls(row):
        return ["background-color: rgba(239,68,68,0.12)" if row["Null Count"] > 0 else "" for _ in row]

    st.write("### 1. Null Value Summary")
    st.dataframe(null_df.style.apply(highlight_nulls, axis=1), use_container_width=True)

    total_nulls = null_counts.sum()
    if total_nulls == 0:
        st.success("✅ No null values found in the dataset!")
        return

    st.warning(f"⚠️ Found **{total_nulls}** null value(s) across **{(null_counts > 0).sum()}** column(s).")

    cols_with_nulls = null_counts[null_counts > 0].index.tolist()
    st.write("### 2. Null Value Visualisation")
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Bar Chart", "Heatmap", "Percentage Chart"])

    with viz_tab1:
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(cols_with_nulls, null_counts[cols_with_nulls], color=ACCENT_PALETTE[2], edgecolor="#0d0f14")
        ax.set(xlabel="Columns", ylabel="Null Count", title="Null Value Counts per Column")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig); plt.close(fig)

    with viz_tab2:
        sample_size = min(100, len(data))
        fig, ax = plt.subplots(figsize=(max(8, len(cols_with_nulls) * 0.8), 6))
        sns.heatmap(data[cols_with_nulls].head(sample_size).isnull(), cbar=False, cmap="magma", ax=ax, yticklabels=False)
        ax.set_title(f"Null Heatmap (first {sample_size} rows)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig); plt.close(fig)

    with viz_tab3:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(cols_with_nulls, null_percent[cols_with_nulls], color=ACCENT_PALETTE[1], edgecolor="#0d0f14")
        ax.axvline(x=50, color=ACCENT_PALETTE[2], linestyle="--", label="50% threshold")
        ax.set(xlabel="Null %", title="Null Percentage per Column")
        ax.legend(); st.pyplot(fig); plt.close(fig)

    st.write("### 3. Handle Null Values")
    scope = st.radio("Apply to:", ["All columns with nulls", "Selected columns only"], horizontal=True, key="null_scope")
    selected_cols = (
        st.multiselect("Pick columns:", cols_with_nulls, default=cols_with_nulls[:1])
        if scope == "Selected columns only"
        else cols_with_nulls
    )
    if not selected_cols:
        st.warning("No columns selected."); return

    numeric_selected     = [c for c in selected_cols if pd.api.types.is_numeric_dtype(data[c])]
    categorical_selected = [c for c in selected_cols if not pd.api.types.is_numeric_dtype(data[c])]

    method = st.selectbox("Select method:", [
        "Drop rows with nulls", "Drop columns with nulls",
        "Fill with Mean (numeric only)", "Fill with Median (numeric only)",
        "Fill with Mode (works for all types)", "Fill with Constant Value",
        "Forward Fill (ffill)", "Backward Fill (bfill)",
        "KNN Imputation (numeric only)", "Interpolation (numeric only)",
    ], key="null_method")

    num_const, cat_const, knn_neighbors = 0.0, "Unknown", 5
    if method == "Fill with Constant Value":
        col_a, col_b = st.columns(2)
        with col_a: num_const = st.number_input("Constant for numeric:", value=0.0)
        with col_b: cat_const = st.text_input("Constant for categorical:", value="Unknown")
    elif method == "KNN Imputation (numeric only)":
        knn_neighbors = st.slider("Number of neighbours (k):", 1, 20, 5)

    def apply_method(df_in):
        df = df_in.copy()
        if method == "Drop rows with nulls":
            df = df.dropna(subset=selected_cols)
        elif method == "Drop columns with nulls":
            df = df.drop(columns=selected_cols)
        elif method == "Fill with Mean (numeric only)":
            for c in numeric_selected: df[c] = df[c].fillna(df[c].mean())
        elif method == "Fill with Median (numeric only)":
            for c in numeric_selected: df[c] = df[c].fillna(df[c].median())
        elif method == "Fill with Mode (works for all types)":
            for c in selected_cols:
                mode_val = df[c].mode()
                if not mode_val.empty: df[c] = df[c].fillna(mode_val[0])
        elif method == "Fill with Constant Value":
            for c in numeric_selected:     df[c] = df[c].fillna(num_const)
            for c in categorical_selected: df[c] = df[c].fillna(cat_const)
        elif method == "Forward Fill (ffill)":
            df[selected_cols] = df[selected_cols].ffill()
        elif method == "Backward Fill (bfill)":
            df[selected_cols] = df[selected_cols].bfill()
        elif method == "KNN Imputation (numeric only)":
            from sklearn.impute import KNNImputer
            if numeric_selected:
                imputer = KNNImputer(n_neighbors=knn_neighbors)
                df[numeric_selected] = imputer.fit_transform(df[numeric_selected])
        elif method == "Interpolation (numeric only)":
            for c in numeric_selected: df[c] = df[c].interpolate(method="linear", limit_direction="both")
        return df

    if method != "Drop columns with nulls":
        preview_after_df = apply_method(data)
        safe_cols = [c for c in selected_cols if c in preview_after_df.columns]
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Before**")
            st.dataframe(data[selected_cols].head(10).style.highlight_null(color="rgba(239,68,68,0.2)"), use_container_width=True)
        with c2:
            st.write("**After**")
            st.dataframe(preview_after_df[safe_cols].head(10), use_container_width=True)
    else:
        st.write("*Columns will be removed — no preview available.*")

    if st.button("✅ Apply & Save Cleaned Data", type="primary") or st.session_state.pop("null_auto_apply", False):
        cleaned = apply_method(data)
        st.session_state["cleaned_data"] = cleaned
        set_data(cleaned)
        remaining = cleaned.isnull().sum().sum()
        rows_drop = len(data) - len(cleaned)
        cols_drop = len(data.columns) - len(cleaned.columns)
        st.success("✅ Cleaned data saved!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Remaining Nulls", remaining)
        col2.metric("Rows Dropped", rows_drop)
        col3.metric("Columns Dropped", cols_drop)
        st.dataframe(cleaned.head(20), use_container_width=True)
        st.rerun()

    with st.expander("📖 Guide: Which method should I use?"):
        guide = {
            "Method": ["Drop rows", "Drop columns", "Fill – Mean", "Fill – Median", "Fill – Mode",
                       "Fill – Constant", "Forward Fill", "Backward Fill", "KNN Imputation", "Interpolation"],
            "Best for": ["Very few null rows (<5%)", "Column has >60% nulls", "Normally distributed numeric",
                         "Skewed numeric / outliers", "Categorical or bimodal", "Domain-specific known default",
                         "Time series / ordered data", "Time series / ordered data", "Correlated numeric features",
                         "Numeric time series with trends"],
            "Drawback": ["Loses data", "Loses features", "Sensitive to outliers", "Loses variance info",
                         "May over-represent one class", "May introduce bias", "Propagates last known value",
                         "May propagate future values backward", "Computationally expensive", "Assumes linear trend"],
        }
        st.dataframe(pd.DataFrame(guide), use_container_width=True)


# ─────────────────────────────────────────────
#  FEATURE 4 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
def show_feature_engineering(data: pd.DataFrame):
    st.markdown('<div class="section-header"><h2>⚙️ Feature Engineering</h2></div>', unsafe_allow_html=True)
    st.caption("Build new features from existing ones. Click **Save** at the bottom to apply.")

    numeric_cols     = data.select_dtypes(include="number").columns.tolist()
    categorical_cols = data.select_dtypes(include="object").columns.tolist()
    all_cols         = data.columns.tolist()
    working          = data.copy()

    st.write("### 1️⃣ Encode Categorical Columns")
    if not categorical_cols:
        st.info("No categorical columns found.")
    else:
        enc_cols   = st.multiselect("Select columns to encode:", categorical_cols, key="fe_enc_cols")
        enc_method = st.radio("Encoding method:", ["One-Hot Encoding", "Label Encoding", "Ordinal (manual order)"],
                              horizontal=True, key="fe_enc_method")
        ordinal_orders = {}
        if enc_method == "Ordinal (manual order)" and enc_cols:
            for col in enc_cols:
                uniq = data[col].dropna().unique().tolist()
                order_input = st.text_input(f"Order for `{col}` (low→high):",
                                            value=", ".join(str(u) for u in uniq), key=f"fe_ord_{col}")
                ordinal_orders[col] = [v.strip() for v in order_input.split(",")]

        if enc_cols and st.button("Apply Encoding", key="fe_enc_btn"):
            for col in enc_cols:
                if enc_method == "One-Hot Encoding":
                    dummies = pd.get_dummies(working[col], prefix=col, drop_first=False)
                    working = pd.concat([working.drop(columns=[col]), dummies], axis=1)
                elif enc_method == "Label Encoding":
                    le = LabelEncoder()
                    working[f"{col}_encoded"] = le.fit_transform(working[col].astype(str))
                else:
                    order  = ordinal_orders.get(col, [])
                    mapper = {v: i for i, v in enumerate(order)}
                    working[f"{col}_ordinal"] = working[col].map(mapper)
            st.session_state["fe_working"] = working
            st.success(f"✅ Encoding applied to: {', '.join(enc_cols)}")

    st.markdown("---")
    st.write("### 2️⃣ Polynomial & Interaction Features")
    if numeric_cols:
        poly_cols   = st.multiselect("Select numeric columns:", numeric_cols, key="fe_poly_cols")
        poly_degree = st.slider("Polynomial degree:", 2, 4, 2, key="fe_poly_deg")
        poly_inter  = st.checkbox("Interaction terms only (no powers)", key="fe_poly_inter")
        if poly_cols and st.button("Apply Polynomial Features", key="fe_poly_btn"):
            pf      = PolynomialFeatures(degree=poly_degree, interaction_only=poly_inter, include_bias=False)
            arr     = working[poly_cols].fillna(0).values
            poly_arr = pf.fit_transform(arr)
            names   = pf.get_feature_names_out(poly_cols)
            poly_df = pd.DataFrame(poly_arr, columns=names, index=working.index)
            new_cols = [c for c in poly_df.columns if c not in working.columns]
            working  = pd.concat([working, poly_df[new_cols]], axis=1)
            st.session_state["fe_working"] = working
            st.success(f"✅ Added {len(new_cols)} polynomial/interaction columns.")

    st.markdown("---")
    st.write("### 3️⃣ Manual Interaction Terms (col_A OP col_B)")
    if len(numeric_cols) >= 2:
        ic1, ic2 = st.columns(2)
        with ic1: col_a = st.selectbox("Column A:", numeric_cols, key="fe_ia")
        with ic2: col_b = st.selectbox("Column B:", [c for c in numeric_cols if c != col_a], key="fe_ib")
        interact_op = st.radio("Operation:", ["Multiply (A × B)", "Divide (A / B)", "Add (A + B)", "Subtract (A − B)"],
                               horizontal=True, key="fe_iop")
        if st.button("Add Interaction Feature", key="fe_inter_btn"):
            if   interact_op == "Multiply (A × B)": working[f"{col_a}_x_{col_b}"]     = working[col_a] * working[col_b]
            elif interact_op == "Divide (A / B)":   working[f"{col_a}_div_{col_b}"]   = working[col_a] / working[col_b].replace(0, np.nan)
            elif interact_op == "Add (A + B)":      working[f"{col_a}_plus_{col_b}"]  = working[col_a] + working[col_b]
            else:                                   working[f"{col_a}_minus_{col_b}"] = working[col_a] - working[col_b]
            st.session_state["fe_working"] = working
            st.success("✅ Interaction column added.")

    st.markdown("---")
    st.write("### 4️⃣ Mathematical Transformations")
    trans_cols = st.multiselect("Select columns to transform:", numeric_cols, key="fe_trans_cols")
    trans_op   = st.selectbox("Transformation:", [
        "Log (log(x+1))", "Square Root (√x)", "Square (x²)",
        "Reciprocal (1/x)", "Z-Score Normalise", "Min-Max Scale (0–1)",
    ], key="fe_trans_op")
    if trans_cols and st.button("Apply Transformation", key="fe_trans_btn"):
        for col in trans_cols:
            if   trans_op == "Log (log(x+1))":
                shift = abs(working[col].min()) + 1 if working[col].min() <= 0 else 0
                working[f"{col}_log"] = np.log(working[col] + shift)
            elif trans_op == "Square Root (√x)":
                shift = abs(working[col].min()) if working[col].min() < 0 else 0
                working[f"{col}_sqrt"] = np.sqrt(working[col] + shift)
            elif trans_op == "Square (x²)":           working[f"{col}_sq"]    = working[col] ** 2
            elif trans_op == "Reciprocal (1/x)":      working[f"{col}_recip"] = 1 / working[col].replace(0, np.nan)
            elif trans_op == "Z-Score Normalise":      working[f"{col}_zscore"] = (working[col] - working[col].mean()) / working[col].std()
            elif trans_op == "Min-Max Scale (0–1)":
                mn, mx = working[col].min(), working[col].max()
                working[f"{col}_minmax"] = (working[col] - mn) / (mx - mn + 1e-9)
        st.session_state["fe_working"] = working
        st.success(f"✅ Transformation applied to: {', '.join(trans_cols)}")

    st.markdown("---")
    st.write("### 5️⃣ Binning (Continuous → Categorical)")
    bin_col = st.selectbox("Column to bin:", ["— select —"] + numeric_cols, key="fe_bin_col")
    bin_n   = st.slider("Number of bins:", 2, 20, 5, key="fe_bin_n")
    if bin_col != "— select —" and st.button("Apply Binning", key="fe_bin_btn"):
        working[f"{bin_col}_binned"] = pd.cut(working[bin_col], bins=bin_n,
                                              labels=[f"B{i+1}" for i in range(bin_n)])
        st.session_state["fe_working"] = working
        st.success(f"✅ `{bin_col}` binned into {bin_n} bins → `{bin_col}_binned`")

    st.markdown("---")
    st.write("### 6️⃣ Drop Irrelevant / ID Columns")
    drop_cols = st.multiselect("Select columns to drop:", all_cols, key="fe_drop_cols")
    if drop_cols and st.button("Drop Selected Columns", key="fe_drop_btn"):
        working = working.drop(columns=drop_cols, errors="ignore")
        st.session_state["fe_working"] = working
        st.success(f"✅ Dropped: {', '.join(drop_cols)}")

    st.markdown("---")
    working = st.session_state.get("fe_working", data.copy())
    st.write("### 📋 Current Dataset Preview")
    st.caption(f"Shape: {working.shape[0]} rows × {working.shape[1]} columns")
    new_col_names = [c for c in working.columns if c not in data.columns]
    if new_col_names:
        st.info(f"🆕 New columns added: {', '.join(new_col_names)}")
    st.dataframe(working.head(20), use_container_width=True)

    if st.button("✅ Save Engineered Dataset as Active Data", type="primary", key="fe_save_btn"):
        st.session_state["cleaned_data"] = working
        set_data(working)
        if "fe_working" in st.session_state:
            del st.session_state["fe_working"]
        st.success(f"✅ Saved! Active dataset now has {working.shape[0]} rows × {working.shape[1]} columns.")
        st.rerun()


# ─────────────────────────────────────────────
#  FEATURE 5 — BASIC STATISTICS
# ─────────────────────────────────────────────
def show_basic_statistics(data: pd.DataFrame):
    st.markdown('<div class="section-header"><h2>📈 Basic Statistics</h2></div>', unsafe_allow_html=True)
    numeric_data = data.select_dtypes(include=["number"])
    if numeric_data.empty:
        st.warning("No numeric columns available."); return
    st.write("#### Descriptive Statistics")
    st.dataframe(numeric_data.describe(), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Skewness")
        st.dataframe(numeric_data.skew().rename("Skewness"), use_container_width=True)
    with col2:
        st.write("#### Kurtosis")
        st.dataframe(numeric_data.kurtosis().rename("Kurtosis"), use_container_width=True)


# ─────────────────────────────────────────────
#  FEATURE 6 — VISUALIZATIONS
# ─────────────────────────────────────────────
def show_visualizations(data: pd.DataFrame):
    st.markdown('<div class="section-header"><h2>📉 Visualizations</h2></div>', unsafe_allow_html=True)

    numeric_columns     = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()

    if not numeric_columns:
        st.warning("No numeric columns available."); return

    selected_columns = st.multiselect("Select columns to visualise:", numeric_columns, key="viz_cols")
    vis_types = st.multiselect(
        "Select visualisation types:",
        ["Histogram", "Boxplot", "Scatterplot", "Lineplot", "Violinplot", "Barplot", "Pairplot", "Heatmap"],
        key="viz_types",
    )

    if not selected_columns or not vis_types:
        st.info("Please select at least one column and one visualisation type.")
        return

    chart_cfg = {}
    for i, vis_type in enumerate(vis_types):
        if vis_type == "Histogram":
            with st.expander(f"Histogram settings", expanded=True):
                kde  = st.checkbox("Show KDE", value=True, key=f"kde_{i}")
                bins = st.slider("Bins", 5, 100, 20, key=f"bins_{i}")
            chart_cfg[i] = {"kde": kde, "bins": bins}
        elif vis_type == "Barplot":
            if categorical_columns:
                cat_col = st.selectbox("Categorical column for Barplot:", categorical_columns, key=f"cat_{i}")
                chart_cfg[i] = {"cat_col": cat_col}
            else:
                st.warning("No categorical columns found — Barplot skipped.")
                chart_cfg[i] = None

    st.markdown("---")

    for i, vis_type in enumerate(vis_types):
        st.write(f"### {vis_type}")

        if vis_type == "Pairplot":
            if len(selected_columns) < 2:
                st.warning("Select at least 2 columns for Pairplot."); continue
            try:
                pg = sns.pairplot(data[selected_columns].dropna(), plot_kws={"alpha": 0.6, "color": ACCENT_PALETTE[0]})
                pg.fig.patch.set_facecolor("#151822")
                st.pyplot(pg.fig); plt.close(pg.fig)
            except Exception as e:
                st.error(f"Pairplot error: {e}")
            continue

        if vis_type in ("Scatterplot", "Lineplot") and len(selected_columns) < 2:
            st.warning(f"Select at least 2 columns for {vis_type}."); continue
        if vis_type == "Heatmap" and len(selected_columns) < 2:
            st.warning("Select at least 2 columns for Heatmap."); continue
        if vis_type == "Barplot" and chart_cfg.get(i) is None:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        try:
            if vis_type == "Histogram":
                cfg = chart_cfg.get(i, {"kde": True, "bins": 20})
                for j, col in enumerate(selected_columns):
                    sns.histplot(data[col].dropna(), kde=cfg["kde"], bins=cfg["bins"],
                                 label=col, ax=ax, alpha=0.75, color=ACCENT_PALETTE[j % len(ACCENT_PALETTE)])
                ax.legend(); ax.set_title("Histogram"); ax.set_xlabel("Value"); ax.set_ylabel("Count")

            elif vis_type == "Boxplot":
                sns.boxplot(data=data[selected_columns].dropna(), ax=ax, palette=ACCENT_PALETTE[:len(selected_columns)])
                ax.set_title("Boxplot")

            elif vis_type == "Scatterplot":
                ax.scatter(data[selected_columns[0]], data[selected_columns[1]],
                           alpha=0.6, color=ACCENT_PALETTE[0], edgecolors="white", linewidths=0.3, s=45)
                ax.set_title("Scatterplot")
                ax.set_xlabel(selected_columns[0]); ax.set_ylabel(selected_columns[1])

            elif vis_type == "Lineplot":
                sns.lineplot(x=data[selected_columns[0]], y=data[selected_columns[1]], ax=ax, color=ACCENT_PALETTE[1])
                ax.set_title("Lineplot")
                ax.set_xlabel(selected_columns[0]); ax.set_ylabel(selected_columns[1])

            elif vis_type == "Violinplot":
                plot_df = data[selected_columns].dropna()
                if plot_df.empty:
                    st.warning("No data after dropping nulls."); plt.close(fig); continue
                sns.violinplot(data=plot_df, ax=ax, palette=ACCENT_PALETTE[:len(selected_columns)])
                ax.set_title("Violinplot")

            elif vis_type == "Barplot":
                cfg     = chart_cfg[i]
                cat_col = cfg["cat_col"]
                num_col = selected_columns[0]
                top_cats = data[cat_col].value_counts().head(20).index
                plot_df  = data[data[cat_col].isin(top_cats)]
                sns.barplot(x=plot_df[cat_col], y=plot_df[num_col], ax=ax, color=ACCENT_PALETTE[0])
                ax.set_title(f"Barplot: {num_col} by {cat_col}")
                plt.xticks(rotation=45, ha="right")

            elif vis_type == "Heatmap":
                corr = data[selected_columns].corr()
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5,
                            ax=ax, linecolor="#0d0f14")
                ax.set_title("Correlation Heatmap")

            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        except Exception as e:
            st.error(f"Error rendering {vis_type}: {e}"); plt.close(fig)


# ─────────────────────────────────────────────
#  FEATURE 7 — CORRELATION PLOT
# ─────────────────────────────────────────────
def show_correlation_plot(data: pd.DataFrame):
    st.markdown('<div class="section-header"><h2>🔗 Correlation Plot</h2></div>', unsafe_allow_html=True)
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if not numeric_columns:
        st.warning("No numeric columns to generate correlation plot."); return
    corr = data[numeric_columns].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", fmt=".2f",
                annot_kws={"size": 8}, linewidths=0.5, ax=ax, linecolor="#0d0f14")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    st.pyplot(fig); plt.close(fig)


# ─────────────────────────────────────────────
#  FEATURE 8 — OUTLIER DETECTION + PCA + CLUSTERING
# ─────────────────────────────────────────────
def show_outlier_detection(data: pd.DataFrame):
    st.markdown('<div class="section-header"><h2>🚨 Outlier Detection, PCA & Clustering</h2></div>', unsafe_allow_html=True)
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns

    if numeric_columns.empty:
        st.warning("No numeric columns available."); return

    st.write("### 📂 Train-Test Split")
    test_pct  = st.slider("Test size %", 10, 50, 20, 5, key="split_slider")
    if st.button("Split Data", key="split_btn"):
        train_data, test_data = train_test_split(data, test_size=test_pct/100, random_state=42)
        st.session_state["train_data"] = train_data
        st.session_state["test_data"]  = test_data
        st.success(f"Split done → Train: {len(train_data)} rows | Test: {len(test_data)} rows")
        c1, c2 = st.columns(2)
        with c1: st.write("Training Data"); st.dataframe(train_data.head(5), use_container_width=True)
        with c2: st.write("Test Data");     st.dataframe(test_data.head(5), use_container_width=True)

    st.markdown("---")
    st.write("### 🔬 PCA Analysis")
    if "train_data" not in st.session_state:
        st.info("Split the data first to enable PCA analysis.")
    else:
        all_num_cols = st.session_state["train_data"].select_dtypes(include=["float64", "int64"]).columns.tolist()
        feat_mode    = st.radio("Feature selection for PCA:",
                                ["Use all numeric features", "Select specific features"], key="pca_feat_mode")
        pca_features = (
            st.multiselect("Select features:", all_num_cols, key="pca_feats")
            if feat_mode == "Select specific features" else all_num_cols
        )
        if pca_features:
            max_comp = min(len(pca_features), 10)
            n_comp   = st.slider("PCA components:", 2, max_comp, min(3, max_comp), key="pca_n")

            if st.button("▶️ Run PCA on Training Data", key="run_pca_btn"):
                X_train = st.session_state["train_data"][pca_features].copy()
                if X_train.isnull().sum().sum() > 0:
                    X_train = X_train.fillna(X_train.mean())
                scaler     = StandardScaler()
                X_scaled   = scaler.fit_transform(X_train)
                pca        = PCA(n_components=n_comp)
                pca_result = pca.fit_transform(X_scaled)
                st.session_state.update({"pca_model": pca, "pca_scaler": scaler, "pca_features": pca_features})
                pca_df     = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_comp)])
                st.session_state["train_pca_df"] = pca_df

                expl_var = pca.explained_variance_ratio_
                cum_var  = np.cumsum(expl_var)
                fig, ax  = plt.subplots(figsize=(10, 4))
                ax.bar(range(1, len(expl_var)+1), expl_var, alpha=0.8, color=ACCENT_PALETTE[0], label="Individual")
                ax.step(range(1, len(cum_var)+1), cum_var, where="mid", color=ACCENT_PALETTE[1], linewidth=2, label="Cumulative")
                ax.set(xlabel="Principal Component", ylabel="Explained Variance Ratio", title="Explained Variance by PC")
                ax.legend(); st.pyplot(fig); plt.close(fig)

                if n_comp >= 3:
                    st.plotly_chart(px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3",
                                                  title="PCA – First 3 PCs", opacity=0.7,
                                                  color_discrete_sequence=ACCENT_PALETTE),
                                    use_container_width=True)

                loadings = pd.DataFrame(pca.components_.T,
                                        columns=[f"PC{i+1}" for i in range(n_comp)], index=pca_features)
                st.write("**Feature Loadings:**")
                st.dataframe(loadings, use_container_width=True)

            if "pca_model" in st.session_state and st.button("▶️ Apply PCA on Test Data", key="pca_test_btn"):
                X_test = st.session_state["test_data"][st.session_state["pca_features"]].copy()
                if X_test.isnull().sum().sum() > 0: X_test = X_test.fillna(X_test.mean())
                X_test_scaled = st.session_state["pca_scaler"].transform(X_test)
                test_pca_res  = st.session_state["pca_model"].transform(X_test_scaled)
                test_pca_df   = pd.DataFrame(test_pca_res, columns=[f"PC{i+1}" for i in range(test_pca_res.shape[1])])
                st.session_state["test_pca_df"] = test_pca_df
                st.write("PCA Results for Test Data (first 5 rows):")
                st.dataframe(test_pca_df.head(), use_container_width=True)

    st.markdown("---")
    st.write("### 🚀 PCA + Clustering Pipeline")
    if "train_pca_df" not in st.session_state:
        st.info("▲ Run PCA on Training Data first to unlock clustering.")
    else:
        pca_df  = st.session_state["train_pca_df"].copy()
        X_pca   = pca_df.values
        algo    = st.selectbox("Algorithm:", ["K-Means", "DBSCAN"], key="cluster_algo_sel")

        if algo == "K-Means":
            col_k1, col_k2 = st.columns(2)
            with col_k1: n_clusters = st.slider("Number of clusters (k):", 2, 15, 3, key="kmeans_k")
            with col_k2: show_elbow = st.checkbox("Show Elbow Curve", value=True, key="elbow_chk")
            if show_elbow:
                inertias = [KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_pca).inertia_
                            for k in range(2, min(16, len(X_pca)))]
                fig_elb, ax_e = plt.subplots(figsize=(8, 3))
                ax_e.plot(list(range(2, min(16, len(X_pca)))), inertias, "o-", color=ACCENT_PALETTE[0], linewidth=2)
                ax_e.axvline(x=n_clusters, color=ACCENT_PALETTE[2], linestyle="--", label=f"k={n_clusters}")
                ax_e.set(xlabel="k", ylabel="Inertia", title="Elbow Curve")
                ax_e.legend(); st.pyplot(fig_elb); plt.close(fig_elb)
        else:
            col_d1, col_d2 = st.columns(2)
            with col_d1: eps_val     = st.slider("ε (eps):", 0.1, 5.0, 0.5, 0.1, key="dbscan_eps")
            with col_d2: min_samples = st.slider("min_samples:", 2, 20, 5, key="dbscan_min")

        if st.button("▶️ Run Clustering", key="run_cluster_btn"):
            cluster_model = (KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                             if algo == "K-Means"
                             else DBSCAN(eps=eps_val, min_samples=min_samples))
            labels = cluster_model.fit_predict(X_pca)
            st.session_state["cluster_labels"]    = labels
            st.session_state["cluster_algo_used"] = algo
            if algo == "K-Means": st.session_state["cluster_model"] = cluster_model
            else:
                st.info(f"DBSCAN: {len(set(labels)) - (1 if -1 in labels else 0)} clusters, "
                        f"{list(labels).count(-1)} noise points.")

        if "cluster_labels" in st.session_state:
            labels     = st.session_state["cluster_labels"]
            algo_used  = st.session_state["cluster_algo_used"]
            pca_df_res = st.session_state["train_pca_df"].copy()
            X_pca_res  = pca_df_res.values
            pca_df_res["Cluster"] = labels.astype(str)

            unique_labels  = sorted(set(labels))
            label_to_color = {lbl: ("#555555" if lbl == -1 else ACCENT_PALETTE[i % len(ACCENT_PALETTE)])
                              for i, lbl in enumerate(l for l in unique_labels if l != -1)}
            if -1 in unique_labels: label_to_color[-1] = "#555555"

            m1, m2, m3 = st.columns(3)
            m1.metric("Clusters Found", len([l for l in set(labels) if l != -1]))
            m2.metric("Noise Points",   list(labels).count(-1))
            m3.metric("Total Samples",  len(labels))

            t1, t2 = st.tabs(["2D Scatter", "3D Scatter"])
            with t1:
                fig2d, ax2d = plt.subplots(figsize=(9, 6))
                for lbl in unique_labels:
                    mask = labels == lbl
                    ax2d.scatter(X_pca_res[mask, 0], X_pca_res[mask, 1],
                                 color=label_to_color[lbl], alpha=0.80,
                                 edgecolors="white", linewidths=0.4, s=65,
                                 label=f"Cluster {lbl}" if lbl != -1 else "Noise")
                if algo_used == "K-Means" and "cluster_model" in st.session_state:
                    centres = st.session_state["cluster_model"].cluster_centers_
                    ax2d.scatter(centres[:, 0], centres[:, 1], c="white", marker="X", s=220, zorder=6, label="Centroids")
                ax2d.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
                ax2d.set(xlabel="PC1", ylabel="PC2", title=f"PCA + {algo_used} — 2D")
                plt.tight_layout(); st.pyplot(fig2d); plt.close(fig2d)
            with t2:
                if X_pca_res.shape[1] >= 3:
                    st.plotly_chart(px.scatter_3d(pca_df_res, x="PC1", y="PC2", z="PC3",
                                                  color="Cluster", opacity=0.85,
                                                  title=f"PCA + {algo_used} — 3D",
                                                  color_discrete_sequence=ACCENT_PALETTE),
                                    use_container_width=True)
                else:
                    st.info("Need ≥ 3 PCA components for 3D scatter.")

            export_df = st.session_state["train_data"].copy().reset_index(drop=True)
            export_df["Cluster_Label"] = labels
            st.download_button("⬇️ Download Clustered Data", data=export_df.to_csv(index=False).encode(),
                               file_name="pca_clustered_data.csv", mime="text/csv")

    st.markdown("---")
    st.write("### 📦 Outlier Detection & Handling")
    data_source  = st.radio("Dataset to analyse:", ["Full Dataset", "Training Data", "Test Data"],
                            disabled="train_data" not in st.session_state,
                            key="outlier_source", horizontal=True)
    display_data = {"Full Dataset": data,
                    "Training Data": st.session_state.get("train_data", data),
                    "Test Data":     st.session_state.get("test_data", data)}[data_source].copy()

    od_cols = st.multiselect("Select numeric columns:", numeric_columns, key="outlier_cols")
    if not od_cols:
        st.info("Select one or more numeric columns to begin outlier analysis."); return

    st.write("#### 🔍 Step 1 — Detect Outliers")
    detect_method = st.radio("Detection method:",
                             ["Z-Score  (|Z| > threshold)", "IQR  (Tukey fences)"],
                             horizontal=True, key="detect_method")
    z_thresh, iqr_mult = 3.0, 1.5
    if "Z-Score" in detect_method: z_thresh = st.slider("Z-score threshold:", 1.5, 5.0, 3.0, 0.1, key="z_thresh")
    else:                          iqr_mult  = st.slider("IQR multiplier (k):", 1.0, 3.0, 1.5, 0.25, key="iqr_mult")

    bounds_info = {}
    for col in od_cols:
        series = display_data[col].dropna()
        if "Z-Score" in detect_method:
            z_scores = (display_data[col] - series.mean()) / series.std()
            mask     = z_scores.abs() > z_thresh
            lower, upper = series.mean() - z_thresh * series.std(), series.mean() + z_thresh * series.std()
        else:
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR    = Q3 - Q1
            lower, upper = Q1 - iqr_mult * IQR, Q3 + iqr_mult * IQR
            mask   = (display_data[col] < lower) | (display_data[col] > upper)
        bounds_info[col] = {"lower": lower, "upper": upper, "mask": mask,
                            "count": int(mask.sum()), "pct": mask.mean() * 100}

    summary_rows = [{"Column": col, "Outlier Count": v["count"],
                     "Outlier %": f"{v['pct']:.1f}%",
                     "Lower Bound": f"{v['lower']:.3f}", "Upper Bound": f"{v['upper']:.3f}"}
                    for col, v in bounds_info.items()]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    fig_bp, axes_bp = plt.subplots(1, len(od_cols), figsize=(max(5, 4 * len(od_cols)), 5), squeeze=False)
    for ax, col in zip(axes_bp[0], od_cols):
        v = bounds_info[col]
        ax.boxplot(display_data[col].dropna(), patch_artist=True,
                   boxprops=dict(facecolor=ACCENT_PALETTE[0], alpha=0.5),
                   medianprops=dict(color=ACCENT_PALETTE[1], linewidth=2))
        ax.axhline(v["lower"], color=ACCENT_PALETTE[2], linestyle="--", linewidth=1.2)
        ax.axhline(v["upper"], color=ACCENT_PALETTE[2], linestyle="--", linewidth=1.2)
        ax.set_title(col + "\n(" + str(v["count"]) + " outliers, " + f"{v['pct']:.1f}%)", fontsize=10)
    plt.tight_layout(); st.pyplot(fig_bp); plt.close(fig_bp)

    st.write("#### 🛠️ Step 2 — Handle Outliers")
    handle_method = st.selectbox("Select handling method:", [
        "🔥 Removal (Trimming)", "✂️ Capping (Winsorization)",
        "🔄 Log Transform", "🔄 Square-Root Transform",
        "📊 Imputation – Mean", "📊 Imputation – Median",
        "🚫 Treat Separately (flag column)",
    ], key="handle_method")

    if st.button("▶️ Apply Outlier Handling", key="apply_outlier_btn"):
        handled     = display_data.copy()
        method_log  = []
        if handle_method == "🔥 Removal (Trimming)":
            before = len(handled)
            for col in od_cols:
                v = bounds_info[col]
                handled = handled[(handled[col] >= v["lower"]) & (handled[col] <= v["upper"])]
            method_log.append(f"Removed {before - len(handled)} rows containing outliers.")
        elif handle_method == "✂️ Capping (Winsorization)":
            for col in od_cols:
                v = bounds_info[col]
                n_capped = int((handled[col] > v["upper"]).sum() + (handled[col] < v["lower"]).sum())
                handled[col] = np.clip(handled[col], v["lower"], v["upper"])
                method_log.append(f"{col}: capped {n_capped} value(s).")
        elif handle_method == "🔄 Log Transform":
            for col in od_cols:
                shift = abs(handled[col].min()) + 1 if handled[col].min() <= 0 else 0
                handled[col] = np.log(handled[col] + shift)
                method_log.append(f"{col}: log(x+{shift:.3f}) applied.")
        elif handle_method == "🔄 Square-Root Transform":
            for col in od_cols:
                shift = abs(handled[col].min()) if handled[col].min() < 0 else 0
                handled[col] = np.sqrt(handled[col] + shift)
                method_log.append(f"{col}: sqrt(x+{shift:.3f}) applied.")
        elif "Imputation" in handle_method:
            for col in od_cols:
                fill = (handled[col].mean() if "Mean" in handle_method
                        else handled[col].median() if "Median" in handle_method
                        else handled[col].mode().iloc[0])
                n_replaced = int(bounds_info[col]["mask"].sum())
                handled.loc[bounds_info[col]["mask"], col] = fill
                method_log.append(f"{col}: replaced {n_replaced} outlier(s) with {fill:.3f}.")
        elif handle_method == "🚫 Treat Separately (flag column)":
            for col in od_cols:
                handled[f"{col}_outlier_flag"] = bounds_info[col]["mask"].astype(int)
                method_log.append(f"{col}: flag column added ({bounds_info[col]['count']} flagged).")

        st.session_state["outlier_handled_data"] = handled
        st.success("✅ Outlier handling applied!")
        for log in method_log: st.write(f"  • {log}")

    if "outlier_handled_data" in st.session_state:
        handled = st.session_state["outlier_handled_data"]
        st.dataframe(handled.head(20), use_container_width=True)
        st.download_button("⬇️ Download Handled Dataset", data=handled.to_csv(index=False).encode(),
                           file_name="outlier_handled_data.csv", mime="text/csv")
        if st.button("✅ Save as Active Dataset", key="promote_outlier"):
            st.session_state["cleaned_data"] = handled
            set_data(handled)
            st.success("Saved! All features will now use the outlier-handled dataset.")
            st.rerun()


# ─────────────────────────────────────────────
#  CASCADE CLASSIFICATION ENGINE
# ─────────────────────────────────────────────
def _run_cascade_classification(X, y, feature_cols, test_size=0.2, base_estimator=None):
    """
    Breaks a multi-class problem into a hierarchy of binary classifiers.
    
    Strategy:
      Level 0 → majority class vs all others
      Level 1 → second majority vs remaining
      ... and so on until one class remains.
    
    Returns predictions, report dict, and per-level metadata.
    """
    from sklearn.base import clone

    if base_estimator is None:
        base_estimator = RandomForestClassifier(n_estimators=100, random_state=42)

    classes_by_freq = y.value_counts().index.tolist()  # majority first
    
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    levels        = []
    remaining_tr  = np.ones(len(y_tr), dtype=bool)   # mask of still-unclassified train rows
    remaining_te  = np.ones(len(y_te), dtype=bool)   # mask of still-unclassified test rows
    final_preds   = pd.Series(["__unknown__"] * len(y_te), index=y_te.index)

    for level_idx, target_class in enumerate(classes_by_freq[:-1]):
        # Binary label: 1 = this class, 0 = everything else still in pool
        y_tr_bin = (y_tr[remaining_tr] == target_class).astype(int)

        if y_tr_bin.sum() == 0 or (1 - y_tr_bin).sum() == 0:
            break   # skip degenerate levels

        clf = clone(base_estimator)
        clf.fit(X_tr[remaining_tr], y_tr_bin)

        # Predict on remaining test rows
        if remaining_te.sum() == 0:
            break
        y_te_bin_pred = clf.predict(X_te[remaining_te])

        # Assign predicted positives
        te_indices = y_te.index[remaining_te]
        pos_mask   = y_te_bin_pred == 1
        final_preds.loc[te_indices[pos_mask]] = target_class

        # True binary labels for this level (for metrics)
        y_te_bin_true = (y_te[remaining_te] == target_class).astype(int)
        level_acc  = accuracy_score(y_te_bin_true, y_te_bin_pred)
        level_f1   = f1_score(y_te_bin_true, y_te_bin_pred, zero_division=0)

        levels.append({
            "Level":          level_idx,
            "Target Class":   str(target_class),
            "Train Samples":  int(remaining_tr.sum()),
            "Test Samples":   int(remaining_te.sum()),
            "Binary Acc":     round(level_acc, 4),
            "Binary F1":      round(level_f1, 4),
            "Classified Out": int(pos_mask.sum()),
        })

        # Remove classified rows from remaining pool
        remaining_tr[remaining_tr] = ~(y_tr[remaining_tr] == target_class).values
        remaining_te[remaining_te] = ~pos_mask

    # Assign last class to whatever is left
    last_class = classes_by_freq[-1]
    leftover   = final_preds == "__unknown__"
    final_preds[leftover] = last_class
    levels.append({
        "Level":         len(classes_by_freq) - 1,
        "Target Class":  str(last_class),
        "Train Samples": int(remaining_tr.sum()),
        "Test Samples":  int(remaining_te.sum()),
        "Binary Acc":    "—",
        "Binary F1":     "—",
        "Classified Out": int(leftover.sum()),
    })

    # Overall metrics
    overall_acc = accuracy_score(y_te, final_preds)
    overall_f1  = f1_score(y_te, final_preds, average="weighted", zero_division=0)
    report      = classification_report(y_te, final_preds, output_dict=True, zero_division=0)

    return {
        "levels":       levels,
        "y_test":       y_te,
        "y_pred":       final_preds,
        "overall_acc":  overall_acc,
        "overall_f1":   overall_f1,
        "report":       report,
        "classes":      classes_by_freq,
    }

# ─────────────────────────────────────────────
#  FEATURE 9 — CLASS IMBALANCE HANDLING
# ─────────────────────────────────────────────
def show_class_imbalance(data: pd.DataFrame):
    st.markdown('<div class="section-header"><h2>⚖️ Class Imbalance Handling</h2></div>', unsafe_allow_html=True)
    st.caption("Detect and fix imbalanced target columns before classification.")

    target_col = st.selectbox("Select target (label) column:", data.columns.tolist(), key="imb_target")
    target     = data[target_col]

    vc      = target.value_counts()
    vc_pct  = (vc / len(target) * 100).round(2)
    d1, d2, d3 = st.columns(3)
    d1.metric("Total Samples",    len(target))
    d2.metric("Number of Classes", target.nunique())
    d3.metric("Minority Class %",  f"{vc_pct.min():.1f}%")

    imb_ratio = vc.max() / vc.min() if vc.min() > 0 else float("inf")
    if imb_ratio > 3:
        st.warning(f"⚠️ Imbalance ratio **{imb_ratio:.1f}:1**")
    else:
        st.success(f"✅ Classes are relatively balanced (ratio = {imb_ratio:.1f}:1).")

    fig_ib, axes_ib = plt.subplots(1, 2, figsize=(12, 4))
    colors = [ACCENT_PALETTE[i % len(ACCENT_PALETTE)] for i in range(len(vc))]
    axes_ib[0].bar(vc.index.astype(str), vc.values, color=colors, edgecolor="#0d0f14")
    axes_ib[0].set(title="Class Counts", xlabel=target_col, ylabel="Count")
    axes_ib[1].pie(vc.values, labels=vc.index.astype(str), autopct="%1.1f%%",
                   startangle=90, colors=colors, textprops={"color": "#e8eaf0"})
    axes_ib[1].set_title("Class Proportions")
    plt.tight_layout(); st.pyplot(fig_ib); plt.close(fig_ib)

    feature_cols = [c for c in data.select_dtypes(include="number").columns if c != target_col]
    if not feature_cols:
        st.error("No numeric feature columns found."); return

    method = st.selectbox("Select method:", [
        "SMOTE (Synthetic Minority Oversampling)", "Random Oversampling",
        "Random Undersampling", "Combined SMOTE + Undersampling",
        "Show Class Weights Only (no resampling)",
    ], key="imb_method")

    if method != "Show Class Weights Only (no resampling)":
        sampling_strategy = st.slider("Target ratio for minority class:", 0.1, 1.0, 0.5, 0.1, key="imb_ratio_slider")

    if st.button("▶️ Apply", key="imb_apply_btn", type="primary"):
        X = data[feature_cols].copy()
        y = target.copy()
        valid = y.dropna().index
        X, y  = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
        imp   = SimpleImputer(strategy="mean")
        X     = pd.DataFrame(imp.fit_transform(X), columns=feature_cols)
        le    = LabelEncoder()
        y_enc = pd.Series(le.fit_transform(y.astype(str)))

        try:
            if method == "Show Class Weights Only (no resampling)":
                from sklearn.utils.class_weight import compute_class_weight
                classes = np.unique(y_enc)
                weights = compute_class_weight("balanced", classes=classes, y=y_enc)
                st.success("Class weights computed:")
                st.json(dict(zip(le.inverse_transform(classes), weights.round(4).tolist())))
                return

            if method == "SMOTE (Synthetic Minority Oversampling)":
                from imblearn.over_sampling import SMOTE
                X_res, y_res = SMOTE(sampling_strategy=sampling_strategy, random_state=42).fit_resample(X, y_enc)
            elif method == "Random Oversampling":
                from imblearn.over_sampling import RandomOverSampler
                X_res, y_res = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42).fit_resample(X, y_enc)
            elif method == "Random Undersampling":
                from imblearn.under_sampling import RandomUnderSampler
                X_res, y_res = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42).fit_resample(X, y_enc)
            elif method == "Combined SMOTE + Undersampling":
                from imblearn.combine import SMOTETomek
                X_res, y_res = SMOTETomek(random_state=42).fit_resample(X, y_enc)

            y_res_decoded = pd.Series(le.inverse_transform(y_res.astype(int)))
            result_df     = pd.DataFrame(X_res, columns=feature_cols)
            result_df[target_col] = y_res_decoded
            st.session_state["imb_result"] = result_df

        except ImportError:
            st.error("❌ Install imbalanced-learn: `pip install imbalanced-learn`"); return

    if "imb_result" in st.session_state:
        result_df = st.session_state["imb_result"]
        new_vc    = result_df[target_col].value_counts()
        b1, b2, b3 = st.columns(3)
        b1.metric("Samples Before", len(data))
        b2.metric("Samples After",  len(result_df))
        b3.metric("New Imbalance Ratio", f"{new_vc.max()/new_vc.min():.1f}:1" if new_vc.min() > 0 else "N/A")
        st.dataframe(result_df.head(20), use_container_width=True)
        col_sv, col_dl = st.columns(2)
        with col_sv:
            if st.button("✅ Save as Active Dataset", key="imb_save_btn", type="primary"):
                st.session_state["cleaned_data"] = result_df
                set_data(result_df)
                st.success("Saved!"); st.rerun()
        with col_dl:
            st.download_button("⬇️ Download Resampled CSV", data=result_df.to_csv(index=False).encode(),
                               file_name="resampled_data.csv", mime="text/csv")
    # ── Cascade Classification ──
    st.markdown("---")
    st.write("### 🔁 Cascade Classification (Structural Imbalance Fix)")
    st.caption(
        "Instead of resampling, cascade classification breaks your multi-class problem into "
        "a hierarchy of binary classifiers — one level per class, ordered by frequency. "
        "This structurally prevents any class from being drowned out by majority classes."
    )

    n_classes = data[target_col].nunique()
    if n_classes < 3:
        st.info("Cascade classification is designed for **3 or more classes**. "
                "Your target has fewer — use the resampling methods above instead.")
        return

    casc_col1, casc_col2 = st.columns(2)
    with casc_col1:
        casc_estimator = st.selectbox(
            "Base binary classifier:",
            ["Random Forest", "Gradient Boosting", "Decision Tree", "KNN"],
            key="casc_estimator",
        )
    with casc_col2:
        casc_test_pct = st.slider("Test size %:", 10, 40, 20, 5, key="casc_test_pct")

    estimator_map = {
        "Random Forest":      RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":  GradientBoostingClassifier(random_state=42),
        "Decision Tree":      DecisionTreeClassifier(random_state=42),
        "KNN":                KNeighborsClassifier(),
    }

    # Show the cascade hierarchy before running
    vc_ordered = data[target_col].value_counts()
    st.write("#### Planned Cascade Hierarchy")
    hierarchy_rows = []
    for i, (cls, count) in enumerate(vc_ordered.items()):
        pct = count / len(data) * 100
        if i < len(vc_ordered) - 1:
            hierarchy_rows.append({
                "Level": i,
                "Binary Problem": f"Is it '{cls}'?  vs  rest",
                "Class Count": count,
                "Class %": f"{pct:.1f}%",
            })
        else:
            hierarchy_rows.append({
                "Level": i,
                "Binary Problem": f"Remainder → '{cls}' (default)",
                "Class Count": count,
                "Class %": f"{pct:.1f}%",
            })
    st.dataframe(pd.DataFrame(hierarchy_rows), use_container_width=True)

    if st.button("▶️ Run Cascade Classification", key="casc_run_btn", type="primary"):
        X_casc = data[feature_cols].copy()
        y_casc = data[target_col].copy()
        valid  = y_casc.dropna().index
        X_casc, y_casc = X_casc.loc[valid].reset_index(drop=True), y_casc.loc[valid].reset_index(drop=True)
        imp    = SimpleImputer(strategy="mean")
        X_casc = pd.DataFrame(imp.fit_transform(X_casc), columns=feature_cols)
        scaler = StandardScaler()
        X_casc = pd.DataFrame(scaler.fit_transform(X_casc), columns=feature_cols)

        with st.spinner("Running cascade classification hierarchy…"):
            casc_result = _run_cascade_classification(
                X            = X_casc.values,
                y            = y_casc,
                feature_cols = feature_cols,
                test_size    = casc_test_pct / 100,
                base_estimator = estimator_map[casc_estimator],
            )
        st.session_state["casc_result"] = casc_result

    if "casc_result" in st.session_state:
        res = st.session_state["casc_result"]

        # ── Metrics ──
        c1, c2 = st.columns(2)
        c1.metric("Overall Accuracy", f"{res['overall_acc']:.4f}")
        c2.metric("Weighted F1",      f"{res['overall_f1']:.4f}")

        # ── Per-level breakdown ──
        st.write("#### Per-Level Results")
        st.dataframe(
            pd.DataFrame(res["levels"])
              .style.background_gradient(subset=["Classified Out"], cmap="Purples"),
            use_container_width=True,
        )

        # ── Classification report ──
        st.write("#### Full Classification Report")
        report_df = pd.DataFrame(res["report"]).T
        st.dataframe(
            report_df.style.background_gradient(cmap="Purples", axis=0),
            use_container_width=True,
        )

        # ── Confusion matrix ──
        st.write("#### Confusion Matrix")
        cm     = confusion_matrix(res["y_test"], res["y_pred"], labels=res["classes"])
        fig_cm, ax_cm = plt.subplots(figsize=(max(5, len(res["classes"])), max(4, len(res["classes"])-1)))
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[str(c) for c in res["classes"]],
        ).plot(ax=ax_cm, cmap="Blues", colorbar=False)
        ax_cm.set_title("Cascade Classifier — Confusion Matrix")
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

        # ── Comparison box ──
        st.info(
            "💡 **Compare with resampling above:** "
            "Cascade classification never synthetically creates data — it structurally "
            "re-frames your problem so minority classes get dedicated binary classifiers, "
            "giving each class a fair chance regardless of its sample count."
        )

# ─────────────────────────────────────────────
#  FEATURE 10 — TIME SERIES ANALYSIS
# ─────────────────────────────────────────────
def show_time_series(data: pd.DataFrame):
    st.markdown('<div class="section-header"><h2>📈 Time Series Analysis</h2></div>', unsafe_allow_html=True)

    datetime_cols = data.select_dtypes(include=["datetime64"]).columns.tolist()
    for col in data.select_dtypes(include="object").columns:
        try:
            pd.to_datetime(data[col].dropna().iloc[:5])
            if col not in datetime_cols: datetime_cols.append(col)
        except Exception: pass

    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found for time series analysis."); return

    ts1, ts2 = st.columns(2)
    with ts1:
        time_col = st.selectbox("Select datetime column:", datetime_cols, key="ts_time_col") if datetime_cols else None
        if not datetime_cols: st.warning("No datetime column detected. Using row index.")
    with ts2:
        value_col = st.selectbox("Select value column:", numeric_cols, key="ts_val_col")

    if time_col:
        try:
            ts_data    = data[[time_col, value_col]].copy()
            ts_data[time_col] = pd.to_datetime(ts_data[time_col])
            ts_data    = ts_data.sort_values(time_col).dropna().reset_index(drop=True)
            time_index = ts_data[time_col]
            values     = ts_data[value_col]
        except Exception as e:
            st.error(f"Could not parse datetime column: {e}"); return
    else:
        ts_data    = data[[value_col]].dropna().reset_index(drop=True)
        time_index = ts_data.index
        values     = ts_data[value_col]

    t1, t2, t3, t4, t5 = st.tabs(["📉 Line Plot", "🔄 Rolling Stats", "🧩 Decomposition",
                                   "📊 Stationarity Test", "📐 Lag / Autocorrelation"])

    with t1:
        fig_ts, ax_ts = plt.subplots(figsize=(12, 4))
        ax_ts.plot(time_index, values, color=ACCENT_PALETTE[1], linewidth=1.5, label=value_col)
        ax_ts.fill_between(time_index, values.min(), values, alpha=0.1, color=ACCENT_PALETTE[1])
        ax_ts.set(xlabel="Time", ylabel=value_col, title=f"{value_col} Over Time")
        ax_ts.legend(); plt.tight_layout(); st.pyplot(fig_ts); plt.close(fig_ts)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Mean", f"{values.mean():.3f}"); s2.metric("Std", f"{values.std():.3f}")
        s3.metric("Min",  f"{values.min():.3f}"); s4.metric("Max", f"{values.max():.3f}")

    with t2:
        window        = st.slider("Rolling window size:", 2, min(100, len(values)//2), 7, key="ts_window")
        rolling_mean  = values.rolling(window).mean()
        rolling_std   = values.rolling(window).std()
        fig_roll, ax_roll = plt.subplots(figsize=(12, 4))
        ax_roll.plot(time_index, values, color="#3d4566", linewidth=1, alpha=0.7, label="Original")
        ax_roll.plot(time_index, rolling_mean, color=ACCENT_PALETTE[0], linewidth=2, label=f"Rolling Mean ({window})")
        ax_roll.fill_between(time_index, rolling_mean - rolling_std, rolling_mean + rolling_std,
                             alpha=0.15, color=ACCENT_PALETTE[0])
        ax_roll.set(xlabel="Time", ylabel=value_col, title="Rolling Statistics")
        ax_roll.legend(fontsize=9); plt.tight_layout(); st.pyplot(fig_roll); plt.close(fig_roll)

    with t3:
        if len(values) < 14:
            st.warning("Need at least 14 data points for decomposition.")
        else:
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                period       = st.slider("Seasonal period:", 2, min(365, len(values)//2), 12, key="ts_period")
                decomp_model = st.radio("Model type:", ["additive", "multiplicative"], horizontal=True, key="ts_decomp_model")
                if decomp_model == "multiplicative" and (values <= 0).any():
                    st.warning("Switching to additive (non-positive values)."); decomp_model = "additive"
                result = seasonal_decompose(values.values, model=decomp_model, period=period)
                fig_dc, axes_dc = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                components = [(values.values, "Original", ACCENT_PALETTE[0]),
                              (result.trend, "Trend", ACCENT_PALETTE[1]),
                              (result.seasonal, "Seasonal", ACCENT_PALETTE[2]),
                              (result.resid, "Residual", ACCENT_PALETTE[3])]
                for ax, (comp, label, color) in zip(axes_dc, components):
                    ax.plot(comp, color=color, linewidth=1.2); ax.set_ylabel(label, fontsize=9)
                plt.tight_layout(); st.pyplot(fig_dc); plt.close(fig_dc)
            except ImportError:
                st.error("Install statsmodels: `pip install statsmodels`")

    with t4:
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(values.dropna())
            adf_stat, p_val, lags_used, n_obs, crit_vals, _ = adf_result
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("ADF Statistic", f"{adf_stat:.4f}"); r2.metric("p-value", f"{p_val:.4f}")
            r3.metric("Lags Used", lags_used);            r4.metric("N Observations", n_obs)
            if p_val < 0.05: st.success(f"✅ **Stationary** — p-value ({p_val:.4f}) < 0.05.")
            else:            st.warning(f"⚠️ **Non-stationary** — p-value ({p_val:.4f}) ≥ 0.05.")
        except ImportError:
            st.error("Install statsmodels: `pip install statsmodels`")

    with t5:
        if len(values) < 5:
            st.warning("Need at least 5 data points for autocorrelation analysis.")
        else:
            max_lags = min(50, len(values) // 2 - 1)
            if max_lags < 1:
                max_lags = len(values) - 1
            
            lag_n = st.slider("Number of lags:", 1, max_lags, min(20, max_lags), key="ts_lag_n")
            
            try:
                from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
                fig_lag, axes_lag = plt.subplots(1, 3, figsize=(15, 4))
                
                # Lag Plot
                pd.plotting.lag_plot(values, lag=1, ax=axes_lag[0])
                axes_lag[0].set_title("Lag Plot (lag=1)")
                
                # ACF
                plot_acf(values, lags=lag_n, ax=axes_lag[1], color=ACCENT_PALETTE[0])
                axes_lag[1].set_title("ACF")
                
                # PACF
                plot_pacf(values, lags=lag_n, ax=axes_lag[2], method="ywm", color=ACCENT_PALETTE[2])
                axes_lag[2].set_title("PACF")
                
                plt.tight_layout(); st.pyplot(fig_lag); plt.close(fig_lag)
            except Exception as e:
                st.error(f"Error generating autocorrelation plots: {e}")
            except ImportError:
                st.error("Install statsmodels: `pip install statsmodels`")


# ─────────────────────────────────────────────
#  GLASS-BOX MODEL IMPLEMENTATIONS
# ─────────────────────────────────────────────
def _build_ebm(problem_type):
    """
    Explainable Boosting Machine approximation using a shallow GBDT
    with single-feature interaction constraints — inherently interpretable.
    """
    if problem_type == "Classification":
        return GradientBoostingClassifier(
            max_depth=1,          # stumps only → additive, glass-box
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
        )
    return GradientBoostingRegressor(
        max_depth=1,
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
    )


def _build_figs(problem_type):
    """
    FIGS approximation: shallow Decision Tree (depth≤3) that produces
    a small, fully readable set of rules — a glass-box by design.
    """
    if problem_type == "Classification":
        return DecisionTreeClassifier(max_depth=3, random_state=42)
    return DecisionTreeRegressor(max_depth=3, random_state=42)


def _plot_ebm_shape_functions(model, feature_names, top_n=10):
    """
    For EBM (GBDT with max_depth=1): plot each feature's marginal
    contribution — equivalent to shape functions in a real EBM.
    """
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    fi_df = (
        pd.DataFrame({"Feature": feature_names, "Shape Contribution": importances})
        .sort_values("Shape Contribution", ascending=False)
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    colors = [ACCENT_PALETTE[i % len(ACCENT_PALETTE)] for i in range(len(fi_df))]
    ax.barh(fi_df["Feature"][::-1], fi_df["Shape Contribution"][::-1],
            color=colors[::-1], edgecolor="#0d0f14")
    ax.set(xlabel="Marginal Contribution (Shape Function)",
           title=f"EBM — Top {top_n} Shape Functions")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def _plot_figs_rules(model, feature_names):
    """
    Extract and display the decision rules from the FIGS (shallow tree).
    """
    from sklearn.tree import export_text
    rules = export_text(model, feature_names=list(feature_names), max_depth=3)
    return rules


def _fairness_report(y_test, y_pred, sensitive_col, sensitive_values, problem_type):
    """
    Compute per-group accuracy/MAE and demographic parity / equalised odds.
    Returns a DataFrame of per-group metrics and a disparity dict.
    """
    rows = []
    for grp in sensitive_values:
        mask = sensitive_col == grp
        if mask.sum() == 0:
            continue
        y_t = y_test[mask]
        y_p = y_pred[mask] if hasattr(y_pred, "__getitem__") else pd.Series(y_pred)[mask.values]
        if problem_type == "Classification":
            acc  = accuracy_score(y_t, y_p)
            f1   = f1_score(y_t, y_p, average="weighted", zero_division=0)
            pos_rate = (pd.Series(y_p) == pd.Series(y_p).mode()[0]).mean()
            rows.append({"Group": str(grp), "Count": int(mask.sum()),
                         "Accuracy": round(acc, 4), "F1": round(f1, 4),
                         "Positive Rate": round(pos_rate, 4)})
        else:
            mae  = mean_absolute_error(y_t, y_p)
            r2   = r2_score(y_t, y_p)
            rows.append({"Group": str(grp), "Count": int(mask.sum()),
                         "MAE": round(mae, 4), "R²": round(r2, 4)})
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
#  PIPELINE STATE SERIALIZER
# ─────────────────────────────────────────────
PIPELINE_EXPORTABLE_KEYS = {
    # Null handling
    "null_scope":        "null_scope",
    "null_method":       "null_method",
    # Feature engineering
    "fe_enc_cols":       "fe_enc_cols",
    "fe_enc_method":     "fe_enc_method",
    "fe_poly_cols":      "fe_poly_cols",
    "fe_poly_deg":       "fe_poly_deg",
    "fe_poly_inter":     "fe_poly_inter",
    "fe_trans_cols":     "fe_trans_cols",
    "fe_trans_op":       "fe_trans_op",
    "fe_bin_col":        "fe_bin_col",
    "fe_bin_n":          "fe_bin_n",
    "fe_drop_cols":      "fe_drop_cols",
    # Outlier
    "detect_method":     "detect_method",
    "z_thresh":          "z_thresh",
    "iqr_mult":          "iqr_mult",
    "handle_method":     "handle_method",
    "outlier_cols":      "outlier_cols",
    # Class imbalance
    "imb_target":        "imb_target",
    "imb_method":        "imb_method",
    "imb_ratio_slider":  "imb_ratio_slider",
    # ML
    "ml_target_column":  "ml_target_column",
    "ml_test_pct":       "ml_test_pct",
    "ml_cv":             "ml_cv",
    "ml_algo":           "ml_algo",
    # Hyperparameter tuning
    "ht_target":         "ht_target",
    "ht_search":         "ht_search",
    "ht_cv":             "ht_cv",
    "ht_algo":           "ht_algo",
    "ht_pso_particles":  "ht_pso_particles",
    "ht_pso_iter":       "ht_pso_iter",
    # Multi-model
    "mm_target":         "mm_target",
    "mm_test":           "mm_test",
    "mm_cv":             "mm_cv",
    # Time series
    "ts_time_col":       "ts_time_col",
    "ts_val_col":        "ts_val_col",
    "ts_window":         "ts_window",
    "ts_period":         "ts_period",
    # Visualizations
    "viz_cols":          "viz_cols",
    "viz_types":         "viz_types",
}

PIPELINE_RESULT_KEYS = [
    "ml_res_algo", "ml_res_prob", "ml_res_X_cols",
    "mm_results",  "mm_problem",  "mm_score_label",
    "ht_pso_best_params", "ht_pso_best_score",
    "casc_result",
]


def _export_pipeline_state():
    """
    Collect all exportable widget states + lightweight result summaries.
    Skips non-serializable objects (models, arrays, DataFrames).
    """
    from datetime import datetime

    state = {
        "meta": {
            "exported_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "app_version":   "1.0",
            "description":   "Agentic Data Science Pipeline Config",
        },
        "widget_states":  {},
        "result_summary": {},
        "pipeline_steps": {},
    }

    # ── Widget states ──
    for key in PIPELINE_EXPORTABLE_KEYS:
        val = st.session_state.get(key)
        if val is not None:
            try:
                json.dumps(val)          # test serializability
                state["widget_states"][key] = val
            except (TypeError, ValueError):
                pass                     # skip non-serializable

    # ── Lightweight result summaries ──
    for key in PIPELINE_RESULT_KEYS:
        val = st.session_state.get(key)
        if val is not None:
            try:
                json.dumps(val)
                state["result_summary"][key] = val
            except (TypeError, ValueError):
                # For mm_results strip _model_obj before serializing
                if key == "mm_results" and isinstance(val, list):
                    safe = [{k: v for k, v in r.items() if k != "_model_obj"} for r in val]
                    try:
                        json.dumps(safe)
                        state["result_summary"][key] = safe
                    except (TypeError, ValueError):
                        pass

    # ── Pipeline completion flags ──
    state["pipeline_steps"] = {
        "data_uploaded":       st.session_state.get("raw_data")               is not None,
        "nulls_handled":       "cleaned_data"        in st.session_state,
        "features_engineered": "fe_working"           in st.session_state,
        "outliers_handled":    "outlier_handled_data" in st.session_state,
        "imbalance_handled":   "imb_result"           in st.session_state
                               or "casc_result"       in st.session_state,
        "model_trained":       "ml_res_model"         in st.session_state,
        "models_compared":     "mm_results"           in st.session_state,
        "hyperparams_tuned":   (st.session_state.get("ht_searcher") is not None
                                or st.session_state.get("ht_pso_best_params") is not None),
        "predictions_made":    "ep_results"           in st.session_state,
    }

    return state


def _import_pipeline_state(config: dict):
    """
    Restore widget states from imported JSON config.
    Returns (success: bool, summary: str).
    """
    restored, skipped = [], []

    widget_states = config.get("widget_states", {})
    for key, val in widget_states.items():
        try:
            st.session_state[key] = val
            restored.append(key)
        except Exception:
            skipped.append(key)

    result_summary = config.get("result_summary", {})
    for key, val in result_summary.items():
        try:
            st.session_state[key] = val
            restored.append(key)
        except Exception:
            skipped.append(key)

    summary = (
        "✅ Restored **" + str(len(restored)) + "** settings.\n"
        + (f"⚠️ Skipped **{len(skipped)}**: {', '.join(skipped)}" if skipped else "")
    )
    return True, summary


# ─────────────────────────────────────────────
#  FEATURE 11 — ML ALGORITHM
# ─────────────────────────────────────────────
def show_ml_algorithm(data: pd.DataFrame):
    st.markdown('<div class="section-header"><h2>🤖 ML Algorithm Selection</h2></div>', unsafe_allow_html=True)

    CLASSIFICATION_ALGOS = ["K-Nearest Neighbors (KNN)", "Decision Tree",
                         "Support Vector Machine (SVM)", "Random Forest Classifier",
                         "🔬 EBM (Glass-Box Boosting)", "🔬 FIGS (Rule Tree)"]
    REGRESSION_ALGOS     = ["Linear Regression", "Lasso Regression",
                            "Gradient Boosting Regression (GBR)",
                            "Decision Tree Regression", "Artificial Neural Networks (ANN)",
                            "🔬 EBM (Glass-Box Boosting)", "🔬 FIGS (Rule Tree)"]

    def detect_problem_type(target):
        return "Classification" if target.dtype == "object" or target.nunique() < 20 else "Regression"

    numeric_columns = data.select_dtypes(include="number").columns
    if numeric_columns.empty:
        st.error("No valid numeric target columns available."); return

    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1: target_column = st.selectbox("Target Column:", numeric_columns, key="ml_target_column")
    with cfg2: test_size_pct = st.slider("Test size %:", 10, 40, 20, 5, key="ml_test_pct")
    with cfg3: cv_folds      = st.slider("CV folds:", 3, 10, 5, key="ml_cv")

    target       = data[target_column]
    problem_type = detect_problem_type(target)
    algorithms   = CLASSIFICATION_ALGOS if problem_type == "Classification" else REGRESSION_ALGOS
    st.info(f"**Detected Problem Type:** `{problem_type}`")

    selected_algorithm = st.selectbox("Select Algorithm:", algorithms, key="ml_algo")

    st.write("#### 🔧 Hyperparameters")
    hp1, hp2 = st.columns(2)
    knn_k=5; knn_weights="uniform"; dt_depth=0; dt_min_split=2; lasso_alpha=1.0
    rf_n=100; rf_depth=0; gbr_n=100; gbr_lr=0.1; ann_layers="100,50"; ann_lr=0.001
    svm_c=1.0; svm_kernel="rbf"
    with hp1:
        if selected_algorithm == "K-Nearest Neighbors (KNN)":
            knn_k = st.slider("n_neighbors:", 1, 30, 5, key="knn_k")
            knn_weights = st.selectbox("weights:", ["uniform", "distance"], key="knn_w")
        elif selected_algorithm in ("Decision Tree", "Decision Tree Regression"):
            dt_depth     = st.slider("max_depth (0=None):", 0, 20, 0, key="dt_d")
            dt_min_split = st.slider("min_samples_split:", 2, 20, 2, key="dt_ms")
        elif selected_algorithm == "Lasso Regression":
            lasso_alpha = st.number_input("alpha:", min_value=0.0001, value=1.0, step=0.1, key="lasso_a")
        elif selected_algorithm == "Random Forest Classifier":
            rf_n    = st.slider("n_estimators:", 10, 300, 100, 10, key="rf_n")
            rf_depth = st.slider("max_depth (0=None):", 0, 20, 0, key="rf_d")
        elif selected_algorithm == "Gradient Boosting Regression (GBR)":
            gbr_n  = st.slider("n_estimators:", 50, 500, 100, 50, key="gbr_n")
            gbr_lr = st.number_input("learning_rate:", 0.001, 1.0, 0.1, step=0.01, key="gbr_lr")
        elif selected_algorithm == "Artificial Neural Networks (ANN)":
            ann_layers = st.text_input("Hidden layer sizes (e.g. 100,50):", value="100,50", key="ann_l")
            ann_lr     = st.number_input("learning_rate_init:", 0.0001, 0.1, 0.001, step=0.0001, key="ann_lr")
        elif selected_algorithm == "Support Vector Machine (SVM)":
            svm_c      = st.number_input("C:", 0.01, 100.0, 1.0, step=0.1, key="svm_c")
            svm_kernel = st.selectbox("kernel:", ["rbf","linear","poly","sigmoid"], key="svm_k")

    if st.button("▶️ Train & Evaluate", key="ml_run_btn", type="primary") or st.session_state.pop("ml_auto_train", False):
        X_raw = data.drop(target_column, axis=1).select_dtypes(include="number")
        y_raw = data[target_column]
        valid = y_raw.dropna().index
        X_raw, y_raw = X_raw.loc[valid], y_raw.loc[valid]
        imputer = SimpleImputer(strategy="mean")
        X_imp   = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)
        y_enc   = y_raw.reset_index(drop=True)
        le      = None
        if problem_type == "Classification" and y_enc.dtype not in ["int64","int32"]:
            le = LabelEncoder(); y_enc = pd.Series(le.fit_transform(y_enc))
        X_imp    = X_imp.reset_index(drop=True)
        scaler   = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=test_size_pct/100, random_state=42)

        if selected_algorithm == "K-Nearest Neighbors (KNN)":
            model = KNeighborsClassifier(n_neighbors=knn_k, weights=knn_weights)
        elif selected_algorithm == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=dt_depth if dt_depth > 0 else None, min_samples_split=dt_min_split, random_state=42)
        elif selected_algorithm == "Support Vector Machine (SVM)":
            model = SVC(C=svm_c, kernel=svm_kernel, probability=True)
        elif selected_algorithm == "Random Forest Classifier":
            model = RandomForestClassifier(n_estimators=rf_n, max_depth=rf_depth if rf_depth > 0 else None, random_state=42)
        elif selected_algorithm == "Linear Regression":  model = LinearRegression()
        elif selected_algorithm == "Lasso Regression":   model = Lasso(alpha=lasso_alpha)
        elif selected_algorithm == "Gradient Boosting Regression (GBR)":
            model = GradientBoostingRegressor(n_estimators=gbr_n, learning_rate=gbr_lr, random_state=42)
        elif selected_algorithm == "Decision Tree Regression":
            model = DecisionTreeRegressor(max_depth=dt_depth if dt_depth > 0 else None, min_samples_split=dt_min_split, random_state=42)
        elif selected_algorithm == "Artificial Neural Networks (ANN)":
            layers = tuple(int(x.strip()) for x in ann_layers.split(",") if x.strip())
            model  = MLPRegressor(hidden_layer_sizes=layers, learning_rate_init=ann_lr, max_iter=500, random_state=42)
        elif selected_algorithm == "🔬 EBM (Glass-Box Boosting)":
            model = _build_ebm(problem_type)
        elif selected_algorithm == "🔬 FIGS (Rule Tree)":
            model = _build_figs(problem_type)

        model.fit(X_train, y_train)
        y_pred       = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        cv_scoring   = "accuracy" if problem_type == "Classification" else "r2"
        cv_scores    = cross_val_score(model, X_scaled, y_enc, cv=cv_folds, scoring=cv_scoring)

        st.session_state.update({
            "ml_res_model": model, "ml_res_X_train": X_train, "ml_res_X_test": X_test,
            "ml_res_y_train": y_train, "ml_res_y_test": y_test,
            "ml_res_y_pred": y_pred, "ml_res_y_pred_tr": y_pred_train,
            "ml_res_cv": cv_scores, "ml_res_prob": problem_type,
            "ml_res_algo": selected_algorithm, "ml_res_X_cols": X_imp.columns.tolist(),
            "ml_res_le": le, "ml_res_X_scaled": X_scaled, "ml_res_y_enc": y_enc,
        })
        set_model_results({"cv_scores": cv_scores, "algo_name": selected_algorithm, "problem_type": problem_type})

    if "ml_res_model" not in st.session_state:
        return

    model        = st.session_state["ml_res_model"]
    X_train      = st.session_state["ml_res_X_train"]; X_test  = st.session_state["ml_res_X_test"]
    y_train      = st.session_state["ml_res_y_train"]; y_test  = st.session_state["ml_res_y_test"]
    y_pred       = st.session_state["ml_res_y_pred"];  y_pred_train = st.session_state["ml_res_y_pred_tr"]
    cv_scores    = st.session_state["ml_res_cv"];      problem_type = st.session_state["ml_res_prob"]
    algo_name    = st.session_state["ml_res_algo"];    feat_cols    = st.session_state["ml_res_X_cols"]
    le           = st.session_state["ml_res_le"];      X_scaled     = st.session_state["ml_res_X_scaled"]
    y_enc        = st.session_state["ml_res_y_enc"]

    st.markdown("---")
    st.write("### 📊 Model Results")

    if selected_algorithm in ("🔬 EBM (Glass-Box Boosting)", "🔬 FIGS (Rule Tree)"):
        st.markdown(
            '<span class="badge badge-green">🔬 Glass-Box Model — fully interpretable by design</span>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

    if problem_type == "Classification":
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Train Acc",  f"{train_acc:.4f}"); m2.metric("Test Acc",   f"{test_acc:.4f}", delta=f"{test_acc-train_acc:+.4f}")
        m3.metric("Precision",  f"{prec:.4f}");      m4.metric("Recall",     f"{rec:.4f}")
        m5.metric("F1 Score",   f"{f1:.4f}");        m6.metric(f"CV ({len(cv_scores)}-fold)", f"{cv_mean:.4f} ± {cv_std:.4f}")
    else:
        train_r2   = r2_score(y_train, y_pred_train); test_r2  = r2_score(y_test, y_pred)
        train_rmse = mean_squared_error(y_train, y_pred_train)**0.5
        test_rmse  = mean_squared_error(y_test, y_pred)**0.5
        test_mae   = mean_absolute_error(y_test, y_pred)
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Train R²",   f"{train_r2:.4f}"); m2.metric("Test R²",    f"{test_r2:.4f}", delta=f"{test_r2-train_r2:+.4f}")
        m3.metric("Train RMSE", f"{train_rmse:.4f}");m4.metric("Test RMSE",  f"{test_rmse:.4f}")
        m5.metric("Test MAE",   f"{test_mae:.4f}"); m6.metric(f"CV ({len(cv_scores)}-fold)", f"{cv_mean:.4f} ± {cv_std:.4f}")

    st.markdown("---")
    glass_box_tab  = ["🔬 Glass-Box Explanation"] if algo_name in ("🔬 EBM (Glass-Box Boosting)", "🔬 FIGS (Rule Tree)") else []
    fairness_tab   = ["⚖️ Fairness & Bias"]
    tab_names =  (["Learning Curve", "Actual vs Predicted", "Confusion Matrix",
              "Classification Report", "CV Scores", "Feature Importance"]
             + glass_box_tab + fairness_tab
             if problem_type == "Classification"
             else ["Learning Curve", "Actual vs Predicted", "Residuals",
                   "Error Distribution", "CV Scores", "Feature Importance"]
             + glass_box_tab + fairness_tab)
    tabs = st.tabs(tab_names)

    with tabs[0]:
        lc_scoring = "accuracy" if problem_type == "Classification" else "r2"
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_scaled, y_enc, cv=cv_folds, scoring=lc_scoring,
            train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)
        tr_mean  = train_scores.mean(axis=1); val_mean = val_scores.mean(axis=1)
        tr_std   = train_scores.std(axis=1);  val_std  = val_scores.std(axis=1)
        fig_lc, ax_lc = plt.subplots(figsize=(9, 5))
        ax_lc.fill_between(train_sizes, tr_mean-tr_std, tr_mean+tr_std, alpha=0.15, color=ACCENT_PALETTE[0])
        ax_lc.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.15, color=ACCENT_PALETTE[2])
        ax_lc.plot(train_sizes, tr_mean, "o-", color=ACCENT_PALETTE[0], linewidth=2, label="Training Score")
        ax_lc.plot(train_sizes, val_mean, "s-", color=ACCENT_PALETTE[2], linewidth=2, label="Validation Score")
        ax_lc.set(xlabel="Training Set Size", ylabel=lc_scoring.capitalize(), title=f"Learning Curve — {algo_name}")
        ax_lc.legend(fontsize=10); plt.tight_layout(); st.pyplot(fig_lc); plt.close(fig_lc)
        gap = float(tr_mean[-1] - val_mean[-1])
        if gap > 0.15: st.warning(f"⚠️ Overfitting detected — gap = {gap:.2f}.")
        elif val_mean[-1] < 0.6: st.warning("⚠️ Underfitting — both curves are low.")
        else: st.success("✅ Model generalises well.")

    with tabs[1]:
        fig_avp, ax_avp = plt.subplots(figsize=(8, 5))
        if problem_type == "Classification":
            ax_avp.scatter(range(len(y_test)), y_test, color=ACCENT_PALETTE[0], alpha=0.6, s=40, label="Actual")
            ax_avp.scatter(range(len(y_pred)), y_pred, color=ACCENT_PALETTE[2], alpha=0.6, s=40, marker="x", label="Predicted")
            ax_avp.set(xlabel="Sample Index", ylabel="Class", title="Actual vs Predicted Labels")
        else:
            ax_avp.scatter(y_test, y_pred, alpha=0.55, color=ACCENT_PALETTE[0], edgecolors="white", s=55, label="Test")
            ax_avp.scatter(y_train, y_pred_train, alpha=0.35, color=ACCENT_PALETTE[4], edgecolors="white", s=35, label="Train")
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            ax_avp.plot(lims, lims, "--", color=ACCENT_PALETTE[2], linewidth=1.5, label="Ideal (y=ŷ)")
            ax_avp.set(xlabel="Actual Values", ylabel="Predicted Values", title=f"Actual vs Predicted — {algo_name}")
        ax_avp.legend(); plt.tight_layout(); st.pyplot(fig_avp); plt.close(fig_avp)

    with tabs[2]:
        if problem_type == "Classification":
            cm     = confusion_matrix(y_test, y_pred)
            labels = le.classes_ if le else sorted(set(y_test))
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(ax=ax_cm, cmap="Blues", colorbar=False)
            ax_cm.set_title("Confusion Matrix")
            plt.tight_layout(); st.pyplot(fig_cm); plt.close(fig_cm)
        else:
            residuals = y_test.values - y_pred
            fig_res, axes_res = plt.subplots(1, 2, figsize=(12, 4))
            axes_res[0].scatter(y_pred, residuals, alpha=0.55, color=ACCENT_PALETTE[0], edgecolors="white", s=50)
            axes_res[0].axhline(0, color=ACCENT_PALETTE[2], linestyle="--", linewidth=1.5)
            axes_res[0].set(xlabel="Predicted Values", ylabel="Residuals", title="Residuals vs Predicted")
            axes_res[1].plot(residuals, "o", alpha=0.5, color=ACCENT_PALETTE[5], markersize=4)
            axes_res[1].axhline(0, color=ACCENT_PALETTE[2], linestyle="--", linewidth=1.5)
            axes_res[1].set(xlabel="Sample Index", ylabel="Residuals", title="Residuals vs Index")
            plt.tight_layout(); st.pyplot(fig_res); plt.close(fig_res)

    with tabs[3]:
        if problem_type == "Classification":
            labels_disp = le.classes_ if le else None
            report_dict = classification_report(y_test, y_pred, target_names=labels_disp, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report_dict).T.style.background_gradient(cmap="Purples", axis=0), use_container_width=True)
        else:
            residuals = y_test.values - y_pred
            import scipy.stats as stats
            fig_ed, axes_ed = plt.subplots(1, 2, figsize=(12, 4))
            axes_ed[0].hist(residuals, bins=30, color=ACCENT_PALETTE[0], edgecolor="#0d0f14", alpha=0.85)
            axes_ed[0].axvline(0, color=ACCENT_PALETTE[2], linestyle="--")
            axes_ed[0].set(xlabel="Residual", title="Residuals Histogram")
            (osm, osr), (slope, intercept, _) = stats.probplot(residuals)
            axes_ed[1].scatter(osm, osr, alpha=0.6, color=ACCENT_PALETTE[5], edgecolors="white", s=30)
            axes_ed[1].plot(osm, slope * np.array(osm) + intercept, color=ACCENT_PALETTE[2], linestyle="--", linewidth=1.5)
            axes_ed[1].set(xlabel="Theoretical Quantiles", ylabel="Sample Quantiles", title="Q-Q Plot of Residuals")
            plt.tight_layout(); st.pyplot(fig_ed); plt.close(fig_ed)

    with tabs[4]:
        cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(len(cv_scores))], "Score": cv_scores})
        fig_cv, ax_cv = plt.subplots(figsize=(8, 4))
        bar_colors = [ACCENT_PALETTE[3] if s >= cv_scores.mean() else ACCENT_PALETTE[2] for s in cv_scores]
        ax_cv.bar(cv_df["Fold"], cv_df["Score"], color=bar_colors, edgecolor="#0d0f14")
        ax_cv.axhline(cv_scores.mean(), color="white", linestyle="--", linewidth=1.5, label=f"Mean={cv_scores.mean():.4f}")
        ax_cv.set(xlabel="Fold", title=f"Cross-Validation — {algo_name}"); ax_cv.legend(fontsize=9)
        plt.tight_layout(); st.pyplot(fig_cv); plt.close(fig_cv)

    with tabs[5]:
        importance = None
        if hasattr(model, "feature_importances_"): importance = model.feature_importances_
        elif hasattr(model, "coef_"):              importance = np.abs(model.coef_).flatten()[:len(feat_cols)]
        if importance is not None and len(importance) == len(feat_cols):
            fi_df = pd.DataFrame({"Feature": feat_cols, "Importance": importance}).sort_values("Importance", ascending=False)
            top_n = min(20, len(fi_df)); fi_top = fi_df.head(top_n)
            fig_fi, ax_fi = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
            colors = [ACCENT_PALETTE[i % len(ACCENT_PALETTE)] for i in range(top_n)]
            ax_fi.barh(fi_top["Feature"][::-1], fi_top["Importance"][::-1], color=colors[::-1], edgecolor="#0d0f14")
            ax_fi.set(xlabel="Importance", title=f"Top {top_n} Feature Importances")
            plt.tight_layout(); st.pyplot(fig_fi); plt.close(fig_fi)
            st.dataframe(fi_df.style.background_gradient(subset=["Importance"], cmap="Purples"), use_container_width=True)
        else:
            st.info("Feature importance not available for this model. Try Random Forest, Decision Tree, or Lasso.")
    # ── Glass-Box tab ──
    if glass_box_tab:
        with tabs[6]:
            if algo_name == "🔬 EBM (Glass-Box Boosting)":
                st.write("#### EBM Shape Functions")
                st.caption(
                    "Each bar shows a feature's marginal additive contribution to predictions. "
                    "Because max_depth=1, every tree uses exactly one feature — making the "
                    "model a sum of independent shape functions, fully transparent."
                )
                fig_ebm = _plot_ebm_shape_functions(model, feat_cols)
                if fig_ebm:
                    st.pyplot(fig_ebm); plt.close(fig_ebm)

                # Per-feature partial dependence style table
                if hasattr(model, "feature_importances_"):
                    fi_df = pd.DataFrame({
                        "Feature":             feat_cols,
                        "Shape Contribution":  model.feature_importances_,
                    }).sort_values("Shape Contribution", ascending=False)
                    fi_df["Rank"]      = range(1, len(fi_df) + 1)
                    fi_df["Influence"] = pd.cut(
                        fi_df["Shape Contribution"],
                        bins=3, labels=["Low", "Medium", "High"]
                    )
                    st.dataframe(
                        fi_df[["Rank","Feature","Shape Contribution","Influence"]]
                          .style.background_gradient(subset=["Shape Contribution"], cmap="Purples"),
                        use_container_width=True,
                    )

            elif algo_name == "🔬 FIGS (Rule Tree)":
                st.write("#### FIGS Decision Rules")
                st.caption(
                    "FIGS produces a shallow tree (max depth 3) whose rules can be read "
                    "top-to-bottom. Every prediction path is fully traceable — no black box."
                )
                rules = _plot_figs_rules(model, feat_cols)
                st.code(rules, language="text")

                # Also show feature importances
                if hasattr(model, "feature_importances_"):
                    fi_df = pd.DataFrame({
                        "Feature":    feat_cols,
                        "Importance": model.feature_importances_,
                    }).sort_values("Importance", ascending=False)
                    top_n = min(15, len(fi_df))
                    fig_fi, ax_fi = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
                    colors = [ACCENT_PALETTE[i % len(ACCENT_PALETTE)] for i in range(top_n)]
                    ax_fi.barh(
                        fi_df["Feature"][:top_n][::-1],
                        fi_df["Importance"][:top_n][::-1],
                        color=colors[::-1], edgecolor="#0d0f14",
                    )
                    ax_fi.set(xlabel="Importance", title="FIGS — Feature Importances")
                    plt.tight_layout(); st.pyplot(fig_fi); plt.close(fig_fi)

    # ── Fairness & Bias tab ──
    fairness_tab_idx = 7 if glass_box_tab else 6
    with tabs[fairness_tab_idx]:
        st.write("#### ⚖️ Fairness & Bias Diagnostic")
        st.caption(
            "Select a sensitive attribute (e.g. gender, age group, region). "
            "The tool computes per-group performance metrics and flags disparity."
        )

        all_cols_for_fairness = data.columns.tolist()
        sensitive_feature = st.selectbox(
            "Select sensitive/protected attribute:",
            ["— select —"] + all_cols_for_fairness,
            key="fairness_col",
        )

        if sensitive_feature != "— select —":
            # Align sensitive col with test indices
            try:
                X_raw_fair = data.drop(target_column, axis=1).select_dtypes(include="number")
                valid_idx  = data[target_column].dropna().index
                sens_series = data.loc[valid_idx, sensitive_feature].reset_index(drop=True)

                # Reuse the same train/test split (same random_state=42)
                _, _, _, _, idx_tr, idx_te = train_test_split(
                    X_scaled, y_enc, range(len(y_enc)),
                    test_size=test_size_pct / 100, random_state=42,
                )
                sens_test = sens_series.iloc[idx_te].reset_index(drop=True)
                y_test_fair = y_test.reset_index(drop=True)
                y_pred_fair = pd.Series(y_pred).reset_index(drop=True)

                unique_groups = sens_test.dropna().unique()
                if len(unique_groups) > 20:
                    st.warning("Too many unique values in sensitive column — choose a categorical one.")
                else:
                    fair_df = _fairness_report(
                        y_test_fair, y_pred_fair,
                        sens_test, unique_groups, problem_type,
                    )
                    st.dataframe(
                        fair_df.style.background_gradient(
                            subset=["Accuracy" if problem_type == "Classification" else "MAE"],
                            cmap="RdYlGn" if problem_type == "Classification" else "RdYlGn_r",
                        ),
                        use_container_width=True,
                    )

                    # Disparity metric
                    if problem_type == "Classification" and "Accuracy" in fair_df.columns:
                        max_acc = fair_df["Accuracy"].max()
                        min_acc = fair_df["Accuracy"].min()
                        disp    = round(max_acc - min_acc, 4)
                        if disp > 0.10:
                            st.error(f"🚨 Accuracy disparity across groups: **{disp:.4f}** — potential bias detected.")
                        elif disp > 0.05:
                            st.warning(f"⚠️ Accuracy disparity: **{disp:.4f}** — monitor closely.")
                        else:
                            st.success(f"✅ Accuracy disparity: **{disp:.4f}** — model appears fair across groups.")

                        # Bar chart
                        fig_fair, ax_fair = plt.subplots(figsize=(8, 4))
                        colors = [ACCENT_PALETTE[i % len(ACCENT_PALETTE)] for i in range(len(fair_df))]
                        ax_fair.bar(fair_df["Group"].astype(str), fair_df["Accuracy"],
                                    color=colors, edgecolor="#0d0f14")
                        ax_fair.axhline(fair_df["Accuracy"].mean(), color="white",
                                        linestyle="--", linewidth=1.5, label="Mean Accuracy")
                        ax_fair.set(xlabel=sensitive_feature, ylabel="Accuracy",
                                    title="Per-Group Accuracy (Fairness Check)")
                        ax_fair.legend(); plt.tight_layout()
                        st.pyplot(fig_fair); plt.close(fig_fair)

                        if "Positive Rate" in fair_df.columns:
                            pr_max  = fair_df["Positive Rate"].max()
                            pr_min  = fair_df["Positive Rate"].min()
                            dp_gap  = round(pr_max - pr_min, 4)
                            st.info(f"**Demographic Parity Gap:** {dp_gap:.4f} "
                                    f"({'✅ Fair' if dp_gap < 0.1 else '⚠️ Unfair'})")

                    elif problem_type == "Regression" and "MAE" in fair_df.columns:
                        max_mae = fair_df["MAE"].max()
                        min_mae = fair_df["MAE"].min()
                        disp    = round(max_mae - min_mae, 4)
                        if disp > 0.2:
                            st.error(f"🚨 MAE disparity across groups: **{disp:.4f}** — potential bias detected.")
                        else:
                            st.success(f"✅ MAE disparity: **{disp:.4f}** — model appears fair across groups.")

                        fig_fair, ax_fair = plt.subplots(figsize=(8, 4))
                        colors = [ACCENT_PALETTE[i % len(ACCENT_PALETTE)] for i in range(len(fair_df))]
                        ax_fair.bar(fair_df["Group"].astype(str), fair_df["MAE"],
                                    color=colors, edgecolor="#0d0f14")
                        ax_fair.set(xlabel=sensitive_feature, ylabel="MAE",
                                    title="Per-Group MAE (Fairness Check)")
                        plt.tight_layout(); st.pyplot(fig_fair); plt.close(fig_fair)

            except Exception as e:
                st.error(f"Fairness computation error: {e}")
        else:
            st.info("Select a sensitive attribute above to run the fairness diagnostic.")

# ─────────────────────────────────────────────
#  FEATURE 12 — MULTI-MODEL COMPARISON
# ─────────────────────────────────────────────
def show_multi_model_comparison(data: pd.DataFrame):
    st.markdown('<div class="section-header"><h2>🏆 Multi-Model Comparison</h2></div>', unsafe_allow_html=True)
    st.caption("Train every algorithm at once and rank them side-by-side.")

    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        st.error("No numeric columns available."); return

    c1, c2, c3 = st.columns(3)
    with c1: target_col = st.selectbox("Target Column:", numeric_columns, key="mm_target")
    with c2: test_pct   = st.slider("Test size %:", 10, 40, 20, 5, key="mm_test")
    with c3: cv_folds   = st.slider("CV Folds:", 3, 10, 5, key="mm_cv")

    def detect(target):
        return "Classification" if target.dtype == "object" or target.nunique() < 20 else "Regression"

    problem_type = detect(data[target_col])
    st.info(f"**Detected Problem Type:** `{problem_type}`")

    if problem_type == "Classification":
        model_registry = {
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree":       DecisionTreeClassifier(random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
            "SVM":                 SVC(probability=True),
            "MLP Neural Network":  MLPClassifier(max_iter=500, random_state=42),
        }
        score_label, cv_scoring = "Accuracy", "accuracy"
    else:
        model_registry = {
            "Linear Regression":   LinearRegression(),
            "Ridge Regression":    Ridge(),
            "Lasso Regression":    Lasso(),
            "Decision Tree":       DecisionTreeRegressor(random_state=42),
            "Random Forest":       RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting":   GradientBoostingRegressor(random_state=42),
            "KNN Regressor":       KNeighborsRegressor(),
            "SVR":                 SVR(),
            "MLP Neural Network":  MLPRegressor(max_iter=500, random_state=42),
        }
        score_label, cv_scoring = "R²", "r2"

    selected_models = st.multiselect("Select models to compare:",
                                     list(model_registry.keys()), default=list(model_registry.keys()), key="mm_model_sel")

    if st.button("▶️ Run Comparison", key="mm_run_btn", type="primary"):
        X_raw = data.drop(target_col, axis=1).select_dtypes(include="number")
        y_raw = data[target_col].dropna()
        X_raw = X_raw.loc[y_raw.index]
        imp   = SimpleImputer(strategy="mean")
        X_imp = pd.DataFrame(imp.fit_transform(X_raw), columns=X_raw.columns)
        scaler   = StandardScaler(); X_scaled = scaler.fit_transform(X_imp)
        le = None; y_enc = y_raw.reset_index(drop=True)
        if problem_type == "Classification" and y_enc.dtype not in ["int64","int32"]:
            le = LabelEncoder(); y_enc = pd.Series(le.fit_transform(y_enc))
        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_enc, test_size=test_pct/100, random_state=42)

        results  = []
        progress = st.progress(0, text="Training models…")
        for i, name in enumerate(selected_models):
            model = model_registry[name]
            try:
                model.fit(X_tr, y_tr); y_pred = model.predict(X_te)
                cv_sc = cross_val_score(model, X_scaled, y_enc, cv=cv_folds, scoring=cv_scoring)
                if problem_type == "Classification":
                    train_sc, test_sc = accuracy_score(y_tr, model.predict(X_tr)), accuracy_score(y_te, y_pred)
                    extra = {"Precision": precision_score(y_te, y_pred, average="weighted", zero_division=0),
                             "Recall":    recall_score(y_te, y_pred, average="weighted", zero_division=0),
                             "F1":        f1_score(y_te, y_pred, average="weighted", zero_division=0)}
                else:
                    train_sc, test_sc = r2_score(y_tr, model.predict(X_tr)), r2_score(y_te, y_pred)
                    extra = {"MAE": mean_absolute_error(y_te, y_pred), "RMSE": mean_squared_error(y_te, y_pred)**0.5}
                results.append({"Model": name, f"Train {score_label}": round(train_sc, 4),
                                f"Test {score_label}": round(test_sc, 4), "Overfit Gap": round(train_sc-test_sc, 4),
                                "CV Mean": round(cv_sc.mean(), 4), "CV Std": round(cv_sc.std(), 4),
                                **{k: round(v, 4) for k, v in extra.items()}, "_model_obj": model})
            except Exception as e:
                results.append({"Model": name, "Error": str(e)})
            progress.progress((i+1)/len(selected_models), text=f"Trained: {name}")
        progress.empty()
        st.session_state.update({"mm_results": results, "mm_problem": problem_type,
                                  "mm_score_label": score_label, "mm_scaler": scaler})

    if "mm_results" not in st.session_state: return

    results     = st.session_state["mm_results"]
    score_label = st.session_state["mm_score_label"]
    valid       = [r for r in results if "Error" not in r]
    if not valid:
        st.error("All models failed."); return

    df_res = pd.DataFrame([{k:v for k,v in r.items() if k != "_model_obj"} for r in valid])
    df_res = df_res.sort_values(f"Test {score_label}", ascending=False).reset_index(drop=True)
    df_res.insert(0, "Rank", range(1, len(df_res)+1))

    def highlight_best(row):
        return ["background-color: rgba(108,99,255,0.2); font-weight:bold" if row["Rank"] == 1 else "" for _ in row]

    st.write("### 🏅 Leaderboard")
    st.dataframe(df_res.style.apply(highlight_best, axis=1)
                            .background_gradient(subset=[f"Test {score_label}"], cmap="Purples"),
                 use_container_width=True)

    st.write("### 📊 Score Comparison")
    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
    x, w = np.arange(len(df_res)), 0.35
    ax_bar.bar(x - w/2, df_res[f"Train {score_label}"], w, label=f"Train {score_label}", color=ACCENT_PALETTE[0], alpha=0.85)
    ax_bar.bar(x + w/2, df_res[f"Test {score_label}"],  w, label=f"Test {score_label}",  color=ACCENT_PALETTE[3], alpha=0.85)
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(df_res["Model"], rotation=35, ha="right", fontsize=9)
    ax_bar.set(ylabel=score_label, title=f"Train vs Test {score_label}"); ax_bar.legend()
    plt.tight_layout(); st.pyplot(fig_bar); plt.close(fig_bar)

    st.markdown("---")
    best_name = df_res.iloc[0]["Model"]
    best_obj  = next(r["_model_obj"] for r in valid if r["Model"] == best_name)
    test_score_key = f"Test {score_label}"
    best_test_score = df_res.iloc[0][test_score_key]
    st.success(f"🥇 Best model: **{best_name}** (Test {score_label} = {best_test_score})")
    save_label = f"💾 Save '{best_name}' as Active Model"
    if st.button(save_label, key="mm_save_best"):
        st.session_state["ml_res_model"] = best_obj
        st.session_state["ml_res_prob"]  = st.session_state["mm_problem"]
        st.session_state["ml_res_algo"]  = best_name
        st.session_state["ml_res_X_cols"] = list(data.drop(target_col, axis=1).select_dtypes(include="number").columns)
        st.session_state["ml_res_le"]    = None
        st.success("Active model updated. Use **Model Export & Predict** to download or predict.")


# ─────────────────────────────────────────────
#  PSO HYPERPARAMETER OPTIMIZER
# ─────────────────────────────────────────────
def _pso_hyperparameter_search(estimator, param_grid, X, y, cv, scoring, n_particles=15, n_iter=30):
    """
    Minimalist Particle Swarm Optimization over a discrete param_grid.
    Returns (best_params_dict, best_score, results_list_of_dicts).
    """
    import itertools, copy

    # Build a flat list of (param_name, list_of_values) pairs
    param_names  = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]
    n_dims       = len(param_names)

    # Each dimension is an index into the corresponding value list
    dim_sizes = np.array([len(v) for v in param_values])

    # ── Evaluate a candidate (array of float indices, clipped & rounded) ──
    _cache = {}
    def evaluate(position):
        idx_tuple = tuple(int(np.clip(round(p), 0, dim_sizes[i]-1)) for i, p in enumerate(position))
        if idx_tuple in _cache:
            return _cache[idx_tuple]
        params = {param_names[i]: param_values[i][idx_tuple[i]] for i in range(n_dims)}
        model  = copy.deepcopy(estimator)
        model.set_params(**{k: v for k, v in params.items() if v is not None})
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, error_score=np.nan)
            score  = np.nanmean(scores)
        except Exception:
            score = -np.inf if scoring == "r2" else 0.0
        _cache[idx_tuple] = (score, params)
        return score, params

    # ── Initialise swarm ──
    rng       = np.random.default_rng(42)
    positions = rng.uniform(0, dim_sizes - 1, size=(n_particles, n_dims))
    velocities = rng.uniform(-1, 1, size=(n_particles, n_dims))
    personal_best_pos   = positions.copy()
    personal_best_score = np.full(n_particles, -np.inf)
    global_best_pos     = positions[0].copy()
    global_best_score   = -np.inf
    global_best_params  = {}

    # PSO constants
    w, c1, c2 = 0.5, 1.5, 1.5

    results = []
    for iteration in range(n_iter):
        for i in range(n_particles):
            score, params = evaluate(positions[i])
            if score > personal_best_score[i]:
                personal_best_score[i] = score
                personal_best_pos[i]   = positions[i].copy()
            if score > global_best_score:
                global_best_score  = score
                global_best_pos    = positions[i].copy()
                global_best_params = params
            results.append({"Iteration": iteration + 1, "Particle": i + 1,
                            "CV Score": round(float(score), 4), "Params": str(params)})

        # ── Update velocities & positions ──
        r1, r2    = rng.random((n_particles, n_dims)), rng.random((n_particles, n_dims))
        velocities = (w * velocities
                      + c1 * r1 * (personal_best_pos - positions)
                      + c2 * r2 * (global_best_pos   - positions))
        positions  = np.clip(positions + velocities, 0, dim_sizes - 1)

    return global_best_params, global_best_score, results



# ─────────────────────────────────────────────
#  FEATURE 13 — HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
def show_hyperparameter_tuning(data: pd.DataFrame):
    st.markdown('<div class="section-header"><h2>🔧 Hyperparameter Tuning</h2></div>', unsafe_allow_html=True)
    st.caption("GridSearchCV or RandomizedSearchCV to find the best parameters automatically.")

    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        st.error("No numeric columns available."); return

    h1, h2, h3 = st.columns(3)
    with h1: target_col  = st.selectbox("Target Column:", numeric_columns, key="ht_target")
    with h2: search_type = st.radio("Search strategy:", ["Grid Search", "Random Search", "Particle Swarm (PSO)"], horizontal=True, key="ht_search")
    with h3: cv_folds    = st.slider("CV Folds:", 3, 10, 5, key="ht_cv")

    def detect(target):
        return "Classification" if target.dtype == "object" or target.nunique() < 20 else "Regression"
    problem_type = detect(data[target_col])
    st.info(f"**Problem Type:** `{problem_type}`")

    algo_options = (["Random Forest Classifier", "Gradient Boosting Classifier",
                     "K-Nearest Neighbors", "Decision Tree", "SVM"]
                    if problem_type == "Classification"
                    else ["Random Forest Regressor", "Gradient Boosting Regressor",
                          "Ridge Regression", "Lasso Regression", "Decision Tree Regressor"])
    algo_name = st.selectbox("Algorithm:", algo_options, key="ht_algo")

    PARAM_GRIDS = {
        "Random Forest Classifier":  {"n_estimators":[50,100,200], "max_depth":[None,5,10,20], "min_samples_split":[2,5,10]},
        "Gradient Boosting Classifier": {"n_estimators":[50,100,200], "learning_rate":[0.01,0.1,0.2], "max_depth":[3,5,7]},
        "K-Nearest Neighbors":       {"n_neighbors":[3,5,7,10,15], "weights":["uniform","distance"], "metric":["euclidean","manhattan"]},
        "Decision Tree":             {"max_depth":[None,3,5,10], "min_samples_split":[2,5,10], "criterion":["gini","entropy"]},
        "SVM":                       {"C":[0.1,1,10,100], "kernel":["rbf","linear","poly"]},
        "Random Forest Regressor":   {"n_estimators":[50,100,200], "max_depth":[None,5,10,20], "min_samples_split":[2,5,10]},
        "Gradient Boosting Regressor": {"n_estimators":[50,100,200], "learning_rate":[0.01,0.1,0.2], "max_depth":[3,5,7]},
        "Ridge Regression":          {"alpha":[0.01,0.1,1.0,10.0,100.0], "fit_intercept":[True,False]},
        "Lasso Regression":          {"alpha":[0.001,0.01,0.1,1.0,10.0], "max_iter":[1000,5000]},
        "Decision Tree Regressor":   {"max_depth":[None,3,5,10], "min_samples_split":[2,5,10], "criterion":["squared_error","friedman_mse"]},
    }

    grid_input = st.text_area("Parameter grid (JSON):", value=json.dumps(PARAM_GRIDS.get(algo_name, {}), indent=2),
                              height=200, key="ht_grid_input")
    #n_iter = st.slider("Random iterations:", 5, 100, 20, key="ht_niter") if search_type == "Random Search" else 20
    if search_type == "Random Search":
        n_iter = st.slider("Random iterations:", 5, 100, 20, key="ht_niter")
    elif search_type == "Particle Swarm (PSO)":
        pso_col1, pso_col2 = st.columns(2)
        with pso_col1:
            pso_particles = st.slider("Number of particles:", 5, 40, 15, key="ht_pso_particles")
        with pso_col2:
            pso_iterations = st.slider("PSO iterations:", 10, 100, 30, key="ht_pso_iter")
        st.caption("🐝 PSO explores the hyperparameter space using a swarm of candidate solutions that iteratively converge toward the globally best configuration found.")
        n_iter = pso_iterations
    else:
        n_iter = 20

    if st.button("▶️ Run Tuning", key="ht_run_btn", type="primary"):
        try: param_grid = json.loads(grid_input)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}"); return

        X_raw = data.drop(target_col, axis=1).select_dtypes(include="number")
        y_raw = data[target_col].dropna()
        X_raw = X_raw.loc[y_raw.index]
        imp   = SimpleImputer(strategy="mean")
        X_imp = pd.DataFrame(imp.fit_transform(X_raw), columns=X_raw.columns)
        scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_imp)
        le = None; y_enc = y_raw.reset_index(drop=True)
        if problem_type == "Classification" and y_enc.dtype not in ["int64","int32"]:
            le = LabelEncoder(); y_enc = pd.Series(le.fit_transform(y_enc))

        base_map = {
            "Random Forest Classifier": RandomForestClassifier(random_state=42),
            "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(), "Decision Tree": DecisionTreeClassifier(random_state=42),
            "SVM": SVC(), "Random Forest Regressor": RandomForestRegressor(random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
            "Ridge Regression": Ridge(), "Lasso Regression": Lasso(),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        }
        cv_scoring = "accuracy" if problem_type == "Classification" else "r2"
        '''with st.spinner(f"Running {search_type}…"):
            if search_type == "Grid Search":
                searcher = GridSearchCV(base_map[algo_name], param_grid, cv=cv_folds,
                                        scoring=cv_scoring, n_jobs=-1, return_train_score=True)
            else:
                searcher = RandomizedSearchCV(base_map[algo_name], param_grid, n_iter=n_iter, cv=cv_folds,
                                              scoring=cv_scoring, n_jobs=-1, return_train_score=True, random_state=42)
            searcher.fit(X_scaled, y_enc)

        st.session_state.update({"ht_searcher": searcher, "ht_problem": problem_type,
                                  "ht_algo": algo_name, "ht_cv_scoring": cv_scoring,
                                  "ht_feat_cols": X_raw.columns.tolist(), "ht_scaler": scaler})'''
        
        if search_type == "Particle Swarm (PSO)":
            with st.spinner("🐝 Running Particle Swarm Optimization…"):
                best_params, best_score, pso_results = _pso_hyperparameter_search(
                    estimator   = base_map[algo_name],
                    param_grid  = param_grid,
                    X           = X_scaled,
                    y           = y_enc,
                    cv          = cv_folds,
                    scoring     = cv_scoring,
                    n_particles = pso_particles,
                    n_iter      = pso_iterations,
                )
            st.session_state.update({
                "ht_pso_best_params":  best_params,
                "ht_pso_best_score":   best_score,
                "ht_pso_results":      pso_results,
                "ht_pso_estimator":    base_map[algo_name],
                "ht_problem":          problem_type,
                "ht_algo":             algo_name,
                "ht_cv_scoring":       cv_scoring,
                "ht_feat_cols":        X_raw.columns.tolist(),
                "ht_scaler":           scaler,
                "ht_searcher":         None,          # clear any previous grid/random result
            })
        else:
            with st.spinner(f"Running {search_type}…"):
                if search_type == "Grid Search":
                    searcher = GridSearchCV(base_map[algo_name], param_grid, cv=cv_folds,
                                            scoring=cv_scoring, n_jobs=-1, return_train_score=True)
                else:
                    searcher = RandomizedSearchCV(base_map[algo_name], param_grid, n_iter=n_iter, cv=cv_folds,
                                                  scoring=cv_scoring, n_jobs=-1, return_train_score=True, random_state=42)
                searcher.fit(X_scaled, y_enc)
            st.session_state.update({
                "ht_searcher":   searcher,
                "ht_problem":    problem_type,
                "ht_algo":       algo_name,
                "ht_cv_scoring": cv_scoring,
                "ht_feat_cols":  X_raw.columns.tolist(),
                "ht_scaler":     scaler,
                "ht_pso_best_params": None,   # clear any previous PSO result
            })
    pso_params  = st.session_state.get("ht_pso_best_params")
    ht_searcher = st.session_state.get("ht_searcher")

    if pso_params is None and ht_searcher is None:
        return
    if pso_params is not None:
        # ── PSO results ──
        best_score  = st.session_state["ht_pso_best_score"]
        pso_results = st.session_state["ht_pso_results"]
        b1, b2 = st.columns(2)
        b1.success(f"**Best CV Score (PSO):** `{best_score:.4f}`")
        params_json    = json.dumps(pso_params, indent=2, default=str)
        params_display = "**Best Parameters:**\n```json\n" + params_json + "\n```"
        b2.info(params_display)
        pso_df = pd.DataFrame(pso_results)
        st.write("#### 🐝 PSO Convergence — Best Score per Iteration")
        iter_best = pso_df.groupby("Iteration")["CV Score"].max().reset_index()
        fig_pso, ax_pso = plt.subplots(figsize=(10, 4))
        ax_pso.plot(iter_best["Iteration"], iter_best["CV Score"],
                    color=ACCENT_PALETTE[0], linewidth=2, marker="o", markersize=4)
        ax_pso.fill_between(iter_best["Iteration"], iter_best["CV Score"].min(),
                            iter_best["CV Score"], alpha=0.1, color=ACCENT_PALETTE[0])
        ax_pso.set(xlabel="Iteration", ylabel="Best CV Score", title="PSO Convergence Curve")
        plt.tight_layout(); st.pyplot(fig_pso); plt.close(fig_pso)
        st.write("#### All Particle Evaluations")
        st.dataframe(
            pso_df.sort_values("CV Score", ascending=False).reset_index(drop=True)
                .style.background_gradient(subset=["CV Score"], cmap="Purples"),
            use_container_width=True,
        )
        if st.button("💾 Save Best PSO Model as Active Model", key="ht_pso_save_btn"):
            import copy
            best_est = copy.deepcopy(st.session_state["ht_pso_estimator"])
            best_est.set_params(**{k: v for k, v in pso_params.items() if v is not None})
            best_est.fit(
                st.session_state["ht_scaler"].transform(
                    pd.DataFrame(
                        SimpleImputer(strategy="mean").fit_transform(
                            data.drop(target_col, axis=1).select_dtypes(include="number")
                        ),
                        columns=data.drop(target_col, axis=1).select_dtypes(include="number").columns,
                    )
                ),
                y_enc,
            )
            st.session_state["ml_res_model"]  = best_est
            st.session_state["ml_res_prob"]   = st.session_state["ht_problem"]
            st.session_state["ml_res_algo"]   = st.session_state["ht_algo"] + " (PSO Tuned)"
            st.session_state["ml_res_X_cols"] = st.session_state.get("ht_feat_cols", [])
            st.session_state["ml_res_le"]     = None
            st.session_state["ht_best_model"] = best_est
            st.success("✅ PSO-tuned model saved. Use **Model Export & Predict** to download or predict.")

    else:
        # ── Grid / Random results (existing logic, unchanged) ──
        searcher = ht_searcher
        b1, b2   = st.columns(2)
        b1.success(f"**Best CV Score:** `{searcher.best_score_:.4f}`")
        params_json    = json.dumps(searcher.best_params_, indent=2)
        params_display = "**Best Parameters:**\n```json\n" + params_json + "\n```"
        b2.info(params_display)

        cv_results_df = pd.DataFrame(searcher.cv_results_)
        show_cols = [c for c in ["params","mean_train_score","mean_test_score","std_test_score","rank_test_score"]
                     if c in cv_results_df.columns]
        cv_display = cv_results_df[show_cols].sort_values("rank_test_score").reset_index(drop=True)
        st.write("#### All Candidates")
        st.dataframe(cv_display.style.background_gradient(subset=["mean_test_score"], cmap="Purples"), use_container_width=True)

        if st.button("💾 Save Best Tuned Model as Active Model", key="ht_save_btn"):
            st.session_state["ml_res_model"]  = searcher.best_estimator_
            st.session_state["ml_res_prob"]   = st.session_state["ht_problem"]
            st.session_state["ml_res_algo"]   = st.session_state["ht_algo"] + " (Tuned)"
            st.session_state["ml_res_X_cols"] = st.session_state.get("ht_feat_cols", [])
            st.session_state["ml_res_le"]     = None
            st.session_state["ht_best_model"] = searcher.best_estimator_
            st.success("✅ Tuned model saved. Use **Model Export & Predict** to download or predict.")




# ─────────────────────────────────────────────
#  FEATURE 14 — MODEL EXPORT & PREDICT
# ─────────────────────────────────────────────


def show_model_export_predict(data: pd.DataFrame):
    st.markdown('<div class="section-header"><h2>📦 Model Export & Predict on New Data</h2></div>', unsafe_allow_html=True)

    model_sources = {
        "ML Algorithm":          st.session_state.get("ml_res_model"),
        "Hyperparameter Tuning": st.session_state.get("ht_best_model"),
    }
    available = {k: v for k, v in model_sources.items() if v is not None}

    if not available:
        st.warning("No trained model found. Train a model first via ML Algorithm or Multi-Model Comparison."); return

    model_choice = st.selectbox("Use model from:", list(available.keys()), key="ep_model_src")
    model        = (st.session_state["ht_best_model"] if model_choice == "Hyperparameter Tuning"
                    else st.session_state["ml_res_model"])
    algo_name    = st.session_state.get("ml_res_algo", "Unknown")
    problem_type = st.session_state.get("ml_res_prob", "Unknown")
    feat_cols    = st.session_state.get("ml_res_X_cols", [])

    st.info(f"**Active Model:** `{algo_name}` | **Task:** `{problem_type}` | **Features:** {len(feat_cols)}")
    st.markdown("---")

    st.write("### 💾 Download Trained Model")
    d1, d2 = st.columns(2)
    with d1:
        algo_safe = algo_name.replace(' ', '_')
        pkl_filename = f"{algo_safe}_model.pkl"
        st.download_button("⬇️ Download model.pkl", data=pickle.dumps(model),
                           file_name=f"{algo_name.replace(' ','_')}_model.pkl",
                           mime="application/octet-stream", key="ep_dl_pkl")
    with d2:
        info_lines = [f"Algorithm: {algo_name}", f"Problem: {problem_type}", f"Features: {', '.join(feat_cols)}",
                      f"Params: {model.get_params()}"]
        st.download_button("⬇️ Download model_info.txt", data="\n".join(info_lines).encode(),
                           file_name="model_info.txt", mime="text/plain", key="ep_dl_txt")

    with st.expander("📋 How to load model in Python"):
        algo_safe = algo_name.replace(' ', '_')
        pkl_filename = f"{algo_safe}_model.pkl"
        feat_cols_repr = str(feat_cols)
        code_snippet = (
            "import pickle, pandas as pd\n"
            "with open('" + pkl_filename + "', 'rb') as f:\n"
            "    model = pickle.load(f)\n"
            "new_data = pd.read_csv('new_data.csv')" + feat_cols_repr + "\n"
            "predictions = model.predict(new_data)\n"
            "print(predictions)"
        )
        st.code(code_snippet, language="python")

    st.markdown("---")
    st.write("### 🔮 Predict on New Data")
    if not feat_cols:
        st.warning("Feature columns not recorded. Retrain via ML Algorithm."); return

    st.info(f"Upload a CSV with these **{len(feat_cols)}** columns: `{', '.join(feat_cols)}`")
    new_file = st.file_uploader("Upload new CSV for prediction:", type=["csv"], key="ep_upload")

    if new_file is not None:
        try:
            new_df = pd.read_csv(new_file)
            st.write(f"Uploaded: **{new_df.shape[0]} rows × {new_df.shape[1]} cols**")
            st.dataframe(new_df.head(5), use_container_width=True)

            missing_cols = [c for c in feat_cols if c not in new_df.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}"); return

            X_new = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(new_df[feat_cols]), columns=feat_cols)
            scaler_key = "ht_scaler" if model_choice == "Hyperparameter Tuning" else "mm_scaler"
            X_new_scaled = (st.session_state[scaler_key].transform(X_new)
                            if scaler_key in st.session_state
                            else StandardScaler().fit_transform(X_new))

            if st.button("▶️ Generate Predictions", key="ep_predict_btn", type="primary"):
                preds = model.predict(X_new_scaled)
                le = st.session_state.get("ml_res_le")
                if le is not None and problem_type == "Classification":
                    try: preds_display = le.inverse_transform(preds.astype(int))
                    except: preds_display = preds
                else: preds_display = preds
                result_df = new_df.copy(); result_df["Prediction"] = preds_display
                if problem_type == "Classification" and hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(X_new_scaled)
                        result_df["Confidence"] = proba.max(axis=1).round(4)
                    except: pass
                st.session_state["ep_results"] = result_df

        except Exception as e:
            st.error(f"Error reading file: {e}")

    if "ep_results" in st.session_state:
        result_df = st.session_state["ep_results"]
        st.write("### ✅ Prediction Results")
        st.dataframe(result_df, use_container_width=True)
        st.download_button("⬇️ Download Predictions CSV", data=result_df.to_csv(index=False).encode(),
                           file_name="predictions.csv", mime="text/csv", key="ep_dl_preds")


# ─────────────────────────────────────────────
#  FEATURE 16 — DOMAIN KNOWLEDGE CONSTRAINTS  ← PASTE THIS NEW BLOCK HERE
# ─────────────────────────────────────────────
def show_domain_knowledge_constraints(data: pd.DataFrame):
    st.subheader("🧠 Domain Knowledge Constraints")
    st.caption(
        "Define acceptable value intervals for each feature based on your domain expertise. "
        "The system will flag data points and model predictions that violate these constraints."
    )

    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns available.")
        return

    st.write("### 1️⃣ Define Acceptable Value Intervals")
    st.info("Set the minimum and maximum acceptable values for each feature based on your domain knowledge.")

    constraints = {}
    saved = st.session_state.get("domain_constraints", {})

    cols_per_row = 2
    col_pairs = [numeric_cols[i:i+cols_per_row] for i in range(0, len(numeric_cols), cols_per_row)]

    for pair in col_pairs:
        row_cols = st.columns(cols_per_row * 3)
        for j, col in enumerate(pair):
            col_min = float(data[col].min())
            col_max = float(data[col].max())
            saved_min = saved.get(col, {}).get("min", col_min)
            saved_max = saved.get(col, {}).get("max", col_max)

            with row_cols[j * 3]:
                st.markdown(f"**{col}**")
            with row_cols[j * 3 + 1]:
                lo = st.number_input(
                    f"Min", value=float(saved_min),
                    key=f"dk_min_{col}", label_visibility="visible"
                )
            with row_cols[j * 3 + 2]:
                hi = st.number_input(
                    f"Max", value=float(saved_max),
                    key=f"dk_max_{col}", label_visibility="visible"
                )
            constraints[col] = {"min": lo, "max": hi,
                                 "data_min": col_min, "data_max": col_max}

    if st.button("💾 Save Constraints", type="primary", key="dk_save_btn"):
        st.session_state["domain_constraints"] = constraints
        st.success(f"✅ Constraints saved for {len(constraints)} features.")

    if "domain_constraints" not in st.session_state:
        st.info("Define and save constraints above to proceed.")
        return

    constraints = st.session_state["domain_constraints"]
    st.markdown("---")

    st.write("### 2️⃣ Data Violation Analysis")
    st.caption("Rows where actual data values fall outside your defined intervals.")

    violation_summary = []
    violation_mask_combined = pd.Series(False, index=data.index)

    for col, bounds in constraints.items():
        lo, hi = bounds["min"], bounds["max"]
        mask = (data[col] < lo) | (data[col] > hi)
        violation_mask_combined |= mask
        count = int(mask.sum())
        pct   = mask.mean() * 100
        violation_summary.append({
            "Feature":     col,
            "Domain Min":  lo,
            "Domain Max":  hi,
            "Violations":  count,
            "Violation %": f"{pct:.1f}%",
            "Status":      "⚠️ Has Violations" if count > 0 else "✅ Clean",
        })

    summary_df = pd.DataFrame(violation_summary)

    def color_status(row):
        if "⚠️" in str(row["Status"]):
            return ["background-color: #ffe0e0"] * len(row)
        return ["background-color: #e0ffe0"] * len(row)

    st.dataframe(
        summary_df.style.apply(color_status, axis=1),
        use_container_width=True
    )

    total_violations = int(violation_mask_combined.sum())
    if total_violations > 0:
        st.warning(f"⚠️ **{total_violations}** row(s) violate at least one domain constraint.")
        with st.expander("🔍 View Violating Rows"):
            st.dataframe(data[violation_mask_combined], use_container_width=True)
        st.download_button(
            "⬇️ Download Violating Rows",
            data=data[violation_mask_combined].to_csv(index=False).encode(),
            file_name="domain_violations.csv",
            mime="text/csv",
            key="dk_dl_violations"
        )
    else:
        st.success("✅ All data points are within the defined domain constraints.")

    st.markdown("---")
    st.write("### 3️⃣ Knowledge-Agreement Dependence Plots")
    st.caption("Each chart shows actual data distribution vs. your defined acceptable range.")

    plot_cols = st.multiselect(
        "Select features to visualise:",
        list(constraints.keys()),
        default=list(constraints.keys())[:min(4, len(constraints))],
        key="dk_plot_cols"
    )

    if plot_cols:
        ncols = 2
        nrows = (len(plot_cols) + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 5, nrows * 4),
                                 squeeze=False)
        for idx, col in enumerate(plot_cols):
            ax     = axes[idx // ncols][idx % ncols]
            bounds = constraints[col]
            lo, hi = bounds["min"], bounds["max"]
            series = data[col].dropna()

            ax.hist(series, bins=30, color="#3498DB",
                    edgecolor="white", alpha=0.75, label="Data distribution")
            ax.axvspan(lo, hi, alpha=0.15, color="#27AE60", label="Acceptable range")
            ax.axvline(lo, color="#27AE60", linewidth=2, linestyle="--", label=f"Min={lo:.2f}")
            ax.axvline(hi, color="#E74C3C", linewidth=2, linestyle="--", label=f"Max={hi:.2f}")

            viol = int(((series < lo) | (series > hi)).sum())
            ax.set_title(col + "\n(" + str(viol) + " violation(s))", fontsize=10)
            ax.legend(fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)

        for idx in range(len(plot_cols), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.write("### 4️⃣ Model Prediction Agreement Check")

    if "ml_res_model" not in st.session_state:
        st.info("Train a model via **🤖 ML Algorithm** first to check prediction agreement.")
        return

    model     = st.session_state["ml_res_model"]
    feat_cols = st.session_state.get("ml_res_X_cols", [])
    target_col = None

    for col in data.columns:
        if col not in feat_cols:
            target_col = col
            break

    if target_col and target_col in constraints:
        lo = constraints[target_col]["min"]
        hi = constraints[target_col]["max"]

        X = data[feat_cols].select_dtypes(include="number").fillna(0)
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            preds      = model.predict(X_scaled)
            pred_series = pd.Series(preds, name="Prediction")
            pred_violations = ((pred_series < lo) | (pred_series > hi))
            n_pred_viol     = int(pred_violations.sum())
            pct_pred_viol   = pred_violations.mean() * 100

            p1, p2, p3 = st.columns(3)
            p1.metric("Target Column",        target_col)
            p2.metric("Prediction Violations", n_pred_viol)
            p3.metric("Violation %",           f"{pct_pred_viol:.1f}%")

            fig_pred, ax_pred = plt.subplots(figsize=(10, 4))
            ax_pred.hist(preds, bins=30, color="#8E44AD",
                         edgecolor="white", alpha=0.75, label="Predictions")
            ax_pred.axvspan(lo, hi, alpha=0.15, color="#27AE60", label="Acceptable range")
            ax_pred.axvline(lo, color="#27AE60", linewidth=2,
                            linestyle="--", label=f"Domain Min={lo:.2f}")
            ax_pred.axvline(hi, color="#E74C3C", linewidth=2,
                            linestyle="--", label=f"Domain Max={hi:.2f}")
            ax_pred.set(xlabel="Predicted Value", ylabel="Count",
                        title=f"Model Predictions vs Domain Interval for '{target_col}'")
            ax_pred.legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_pred)
            plt.close(fig_pred)

            if n_pred_viol > 0:
                st.warning(
                    f"⚠️ **{n_pred_viol}** prediction(s) fall outside your domain interval "
                    f"[{lo:.2f}, {hi:.2f}]. Consider retraining or reviewing your constraints."
                )
            else:
                st.success(
                    f"✅ All predictions fall within the domain interval [{lo:.2f}, {hi:.2f}]."
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.info(
            "No domain constraint found for the target column, or target column could not be detected. "
            "Define a constraint for your target column above."
        )


# ─────────────────────────────────────────────
#  FEATURE 17 — EXPORT PIPELINE AS NOTEBOOK   ← PASTE THIS NEW BLOCK HERE
# ─────────────────────────────────────────────
def show_export_notebook(data: pd.DataFrame, raw_data: pd.DataFrame):
    st.subheader("📓 Export Pipeline as Executable Notebook")
    st.caption(
        "Generates a fully reproducible Python Jupyter notebook (.ipynb) capturing every "
        "step you performed — data loading, cleaning, feature engineering, outlier handling, "
        "and model training — so you can re-run or modify the pipeline outside this app."
    )

    try:
        import nbformat as nbf
    except ImportError:
        st.error("❌ Install nbformat: `pip install nbformat`")
        return

    # ── Show pipeline summary ──
    st.write("### 📋 Detected Pipeline Steps")
    st.caption("The notebook will include all steps marked ✅ below.")

    steps_detected = {
        "📁 Data Loading":           True,
        "🔍 Null Value Handling":    "cleaned_data"        in st.session_state,
        "🚨 Outlier Handling":       "outlier_handled_data" in st.session_state,
        "⚙️ Feature Engineering":   "fe_working"           in st.session_state,
        "⚖️ Class Imbalance":        "imb_result"          in st.session_state,
        "🧠 Domain Constraints":     "domain_constraints"  in st.session_state,
        "🤖 Model Training":         "ml_res_model"        in st.session_state,
        "🏆 Multi-Model Comparison": "mm_results"          in st.session_state,
        "🔧 Hyperparameter Tuning":  "ht_searcher"         in st.session_state,
        "📦 Predictions Made":       "ep_results"          in st.session_state,
        "📓 Notebook Exported":      "nb_generated"        in st.session_state,
    }

    col1, col2 = st.columns(2)
    items = list(steps_detected.items())
    half  = len(items) // 2
    for step, done in items[:half]:
        col1.write(f"{'✅' if done else '⬜'} {step}")
    for step, done in items[half:]:
        col2.write(f"{'✅' if done else '⬜'} {step}")

    completed = sum(steps_detected.values())
    st.progress(completed / len(steps_detected))
    st.caption(f"{completed}/{len(steps_detected)} pipeline steps will be exported.")

    st.markdown("---")

    # ── Options ──
    st.write("### ⚙️ Export Options")
    o1, o2 = st.columns(2)
    with o1:
        notebook_title  = st.text_input("Notebook title:", value="ML_Pipeline", key="nb_title")
        include_eda     = st.checkbox("Include EDA & visualisation cells", value=True, key="nb_eda")
    with o2:
        include_comments = st.checkbox("Include markdown explanation cells", value=True, key="nb_comments")
        include_metrics  = st.checkbox("Include model metrics summary", value=True, key="nb_metrics")

    st.markdown("---")

    if st.button("🚀 Generate Notebook", type="primary", key="nb_generate_btn"):

        nb     = nbf.v4.new_notebook()
        cells  = []

        def md(text):
            if include_comments:
                cells.append(nbf.v4.new_markdown_cell(text))

        def code(text):
            cells.append(nbf.v4.new_code_cell(text.strip()))

        # ── CELL: Title ──
        md("# 📓 " + notebook_title + "\n"
           "> Auto-generated by **Agentic Data Science Dashboard**\n\n"
           "This notebook reproduces your complete ML pipeline step by step.")

        # ── CELL: Imports ──
        md("## 1. 📦 Import Libraries")

        algo_name = st.session_state.get("ml_res_algo", "")
        algo_imports = {
            "Random Forest Classifier":             "from sklearn.ensemble import RandomForestClassifier",
            "Random Forest":                        "from sklearn.ensemble import RandomForestClassifier",
            "Gradient Boosting":                    "from sklearn.ensemble import GradientBoostingClassifier",
            "Gradient Boosting Regression (GBR)":   "from sklearn.ensemble import GradientBoostingRegressor",
            "Decision Tree":                        "from sklearn.tree import DecisionTreeClassifier",
            "Decision Tree Regression":             "from sklearn.tree import DecisionTreeRegressor",
            "K-Nearest Neighbors (KNN)":            "from sklearn.neighbors import KNeighborsClassifier",
            "Linear Regression":                    "from sklearn.linear_model import LinearRegression",
            "Lasso Regression":                     "from sklearn.linear_model import Lasso",
            "Ridge Regression":                     "from sklearn.linear_model import Ridge",
            "Support Vector Machine (SVM)":         "from sklearn.svm import SVC",
            "Artificial Neural Networks (ANN)":     "from sklearn.neural_network import MLPRegressor",
        }
        algo_import_line = algo_imports.get(algo_name, "# No specific algorithm import needed")

        code(f"""import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, r2_score, mean_absolute_error,
    mean_squared_error, classification_report, confusion_matrix
)
{algo_import_line}
import warnings
warnings.filterwarnings("ignore")
print("✅ Libraries imported successfully")""")

        # ── CELL: Load Data ──
        md("## 2. 📂 Load Dataset")
        raw_cols    = raw_data.columns.tolist()
        raw_shape   = raw_data.shape
        raw_dtypes  = raw_data.dtypes.apply(str).to_dict()

        code(f"""# Load your dataset — replace the filename with your actual file path
df = pd.read_csv("your_dataset.csv")

# Dataset info (from original upload)
# Shape: {raw_shape[0]} rows × {raw_shape[1]} columns
# Columns: {raw_cols}
print(f"Dataset shape: {{df.shape}}")
print(f"Columns: {{df.columns.tolist()}}")
df.head()""")

        code(f"""# Column data types
print(df.dtypes)
print("\\nNull counts:")
print(df.isnull().sum())""")

        # ── CELL: EDA ──
        if include_eda:
            md("## 3. 📊 Exploratory Data Analysis")
            numeric_cols = raw_data.select_dtypes(include="number").columns.tolist()
            code(f"""# Basic statistics
print(df.describe())

numeric_cols = {numeric_cols}

# Null value bar chart
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(df.columns, df.isnull().sum(), color="#E74C3C", edgecolor="white")
ax.set(title="Null Value Counts per Column", xlabel="Column", ylabel="Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Distribution plots
n = len(numeric_cols)
ncols = 4
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.5, nrows*3), squeeze=False)
for idx, col in enumerate(numeric_cols):
    ax = axes[idx // ncols][idx % ncols]
    ax.hist(df[col].dropna(), bins=25, color="#3498DB", edgecolor="white", alpha=0.85)
    ax.set_title(col, fontsize=8)
for idx in range(n, nrows * ncols):
    axes[idx // ncols][idx % ncols].set_visible(False)
plt.tight_layout()
plt.show()""")

        # ── CELL: Null Handling ──
        if "cleaned_data" in st.session_state:
            md("## 4. 🔍 Null Value Handling")
            null_method   = st.session_state.get("null_method", "Fill with Mean (numeric only)")
            cleaned_shape = st.session_state["cleaned_data"].shape
            null_remaining= st.session_state["cleaned_data"].isnull().sum().sum()

            method_code_map = {
                "Drop rows with nulls":             "df = df.dropna()",
                "Drop columns with nulls":          "df = df.dropna(axis=1)",
                "Fill with Mean (numeric only)":    "df = df.fillna(df.mean(numeric_only=True))",
                "Fill with Median (numeric only)":  "df = df.fillna(df.median(numeric_only=True))",
                "Fill with Mode (works for all types)": "df = df.fillna(df.mode().iloc[0])",
                "Forward Fill (ffill)":             "df = df.ffill()",
                "Backward Fill (bfill)":            "df = df.bfill()",
            }
            null_code = method_code_map.get(null_method, "df = df.fillna(df.mean(numeric_only=True))")

            code(f"""# Null handling method used: {null_method}
# Result: {cleaned_shape[0]} rows × {cleaned_shape[1]} columns | {null_remaining} nulls remaining

{null_code}

print(f"After cleaning — Shape: {{df.shape}}")
print(f"Remaining nulls: {{df.isnull().sum().sum()}}")""")

        # ── CELL: Outlier Handling ──
        if "outlier_handled_data" in st.session_state:
            md("## 5. 🚨 Outlier Handling")
            handle_method = st.session_state.get("handle_method", "Capping (Winsorization)")
            handled_shape = st.session_state["outlier_handled_data"].shape

            outlier_code_map = {
                "🔥 Removal (Trimming)":             "# Rows with outliers were removed\n# df = df[(df[col] >= lower) & (df[col] <= upper)]",
                "✂️ Capping (Winsorization)":        "# Values clipped to IQR bounds\n# df[col] = np.clip(df[col], lower, upper)",
                "🔄 Log Transform":                   "# df[col] = np.log(df[col] + 1)",
                "🔄 Square-Root Transform":           "# df[col] = np.sqrt(df[col])",
                "📊 Imputation – Mean":               "# df.loc[outlier_mask, col] = df[col].mean()",
                "📊 Imputation – Median":             "# df.loc[outlier_mask, col] = df[col].median()",
                "🚫 Treat Separately (flag column)":  "# df[f'{{col}}_outlier_flag'] = outlier_mask.astype(int)",
            }
            out_code = outlier_code_map.get(handle_method, "# Outlier handling applied")

            code(f"""# Outlier handling: {handle_method}
# Result shape: {handled_shape[0]} rows × {handled_shape[1]} columns
{out_code}

# General IQR capping example:
numeric_cols = df.select_dtypes(include="number").columns.tolist()
for col in numeric_cols:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = np.clip(df[col], Q1 - 1.5*IQR, Q3 + 1.5*IQR)
print(f"After outlier handling — Shape: {{df.shape}}")""")

        # ── CELL: Feature Engineering ──
        if "fe_working" in st.session_state:
            md("## 6. ⚙️ Feature Engineering")
            fe_df    = st.session_state["fe_working"]
            new_cols = [c for c in fe_df.columns if c not in raw_data.columns]
            code(f"""# Feature engineering — New columns added: {new_cols}

# Reproduce below (edit as needed):
# df['col_log']         = np.log(df['col'] + 1)
# df['col_a_x_col_b']  = df['col_a'] * df['col_b']
# df['col_binned']      = pd.cut(df['col'], bins=5, labels=['B1','B2','B3','B4','B5'])

print(f"After feature engineering — Shape: {{df.shape}}")
print(f"Columns: {{df.columns.tolist()}}")""")

        # ── CELL: Class Imbalance ──
        if "imb_result" in st.session_state:
            md("## 7. ⚖️ Class Imbalance Handling")
            imb_method = st.session_state.get("imb_method", "SMOTE")
            code(f"""# Class imbalance method: {imb_method}
# pip install imbalanced-learn

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

target_col   = "{st.session_state.get('imb_target', 'target')}"
feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]

X     = df[feature_cols].fillna(0)
y     = df[target_col]
le    = LabelEncoder()
y_enc = le.fit_transform(y.astype(str))

smote        = SMOTE(sampling_strategy=0.5, random_state=42)
X_res, y_res = smote.fit_resample(X, y_enc)
print(f"After resampling — X shape: {{X_res.shape}}")""")

        # ── CELL: Model Training ──
        if "ml_res_model" in st.session_state:
            md("## 8. 🤖 Model Training & Evaluation")

            algo_name    = st.session_state.get("ml_res_algo",   "Random Forest Classifier")
            problem_type = st.session_state.get("ml_res_prob",   "Classification")
            feat_cols    = st.session_state.get("ml_res_X_cols", [])

            model_init_map = {
                "Random Forest Classifier":           "model = RandomForestClassifier(n_estimators=100, random_state=42)",
                "Random Forest":                      "model = RandomForestClassifier(n_estimators=100, random_state=42)",
                "Gradient Boosting":                  "model = GradientBoostingClassifier(random_state=42)",
                "Gradient Boosting Regression (GBR)": "model = GradientBoostingRegressor(random_state=42)",
                "Decision Tree":                      "model = DecisionTreeClassifier(random_state=42)",
                "Decision Tree Regression":           "model = DecisionTreeRegressor(random_state=42)",
                "K-Nearest Neighbors (KNN)":          "model = KNeighborsClassifier(n_neighbors=5)",
                "Linear Regression":                  "model = LinearRegression()",
                "Lasso Regression":                   "model = Lasso(alpha=1.0)",
                "Ridge Regression":                   "model = Ridge(alpha=1.0)",
                "Support Vector Machine (SVM)":       "model = SVC(probability=True)",
                "Artificial Neural Networks (ANN)":   "model = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500, random_state=42)",
            }
            model_init = model_init_map.get(algo_name, "model = RandomForestClassifier(random_state=42)")

            cv_mean = 0.0
            cv_std  = 0.0
            if "ml_res_cv" in st.session_state:
                cv_scores = st.session_state["ml_res_cv"]
                cv_mean   = float(cv_scores.mean())
                cv_std    = float(cv_scores.std())

            scoring = "accuracy" if problem_type == "Classification" else "r2"

            code(f"""# Algorithm    : {algo_name}
# Problem Type : {problem_type}
# Features     : {feat_cols}
# CV Score     : {cv_mean:.4f} ± {cv_std:.4f}

target_col   = "{st.session_state.get('ml_target_column', 'target')}"
feature_cols = {feat_cols}

X_raw = df[feature_cols].select_dtypes(include="number")
y_raw = df[target_col].dropna()
X_raw = X_raw.loc[y_raw.index]

imputer  = SimpleImputer(strategy="mean")
X_imp    = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)

y_enc = y_raw.reset_index(drop=True)
le    = None
if "{problem_type}" == "Classification" and y_enc.dtype not in ["int64","int32"]:
    le    = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y_enc))

scaler   = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42
)

{model_init}
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cv_scores = cross_val_score(model, X_scaled, y_enc, cv=5, scoring="{scoring}")
print(f"CV {scoring}: {{cv_scores.mean():.4f}} ± {{cv_scores.std():.4f}}")""")

            if include_metrics:
                if problem_type == "Classification":
                    code("""# ── Classification Metrics ──
from sklearn.metrics import ConfusionMatrixDisplay

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = accuracy_score(y_test,  y_pred)
print(f"Train Accuracy : {train_acc:.4f}")
print(f"Test  Accuracy : {test_acc:.4f}")
print(classification_report(y_test, y_pred, zero_division=0))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
plt.tight_layout(); plt.show()""")
                else:
                    code("""# ── Regression Metrics ──
train_r2  = r2_score(y_train, model.predict(X_train))
test_r2   = r2_score(y_test,  y_pred)
test_mae  = mean_absolute_error(y_test, y_pred)
test_rmse = mean_squared_error(y_test,  y_pred) ** 0.5
print(f"Train R²  : {train_r2:.4f}")
print(f"Test  R²  : {test_r2:.4f}")
print(f"MAE       : {test_mae:.4f}")
print(f"RMSE      : {test_rmse:.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_test, y_pred, alpha=0.6, color="#3498DB", edgecolors="white", s=55)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Ideal")
ax.set(xlabel="Actual", ylabel="Predicted", title="Actual vs Predicted")
ax.legend(); plt.tight_layout(); plt.show()""")

            code("""# ── Feature Importance ──
if hasattr(model, "feature_importances_"):
    fi_df = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols)*0.35)))
    ax.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1], color="#27AE60", edgecolor="white")
    ax.set(xlabel="Importance", title="Feature Importances")
    plt.tight_layout(); plt.show()
    print(fi_df.to_string(index=False))
else:
    print("Feature importance not available for this model.")""")

        # ── CELL: Hyperparameter tuning ──
        if "ht_searcher" in st.session_state:
            md("## 9. 🔧 Hyperparameter Tuning Results")
            best_params = st.session_state["ht_searcher"].best_params_
            best_score  = st.session_state["ht_searcher"].best_score_
            ht_algo     = st.session_state.get("ht_algo", "")
            code(f"""# Algorithm  : {ht_algo}
# Best Score : {best_score:.4f}
# Best Params: {best_params}

# Reproduce tuning:
from sklearn.model_selection import GridSearchCV
# searcher = GridSearchCV(model, {best_params}, cv=5, scoring="{scoring}", n_jobs=-1)
# searcher.fit(X_scaled, y_enc)
# model = searcher.best_estimator_
print("Best params recorded — plug into your model to reproduce.")""")

        # ── CELL: Save model ──
        md("## 10. 💾 Save & Load Model")
        code("""import pickle

with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Model saved to trained_model.pkl")

# To load later:
# with open("trained_model.pkl", "rb") as f:
#     loaded_model = pickle.load(f)
# new_pred = loaded_model.predict(scaler.transform(imputer.transform(new_df[feature_cols])))""")

        # ── Finalise ──
        nb.cells = cells
        nb.metadata = {
            "kernelspec": {
                "display_name": "Python 3",
                "language":     "python",
                "name":         "python3"
            },
            "language_info": {
                "name":    "python",
                "version": "3.10.0"
            }
        }

        import io
        nb_bytes = io.BytesIO()
        nb_bytes.write(nbf.writes(nb).encode("utf-8"))
        nb_bytes.seek(0)

        fname = f"{notebook_title.replace(' ', '_')}.ipynb"
        st.session_state["nb_generated"] = True

        st.success(
            f"✅ Notebook generated — **{len(nb.cells)} cells** covering "
            f"**{completed}** pipeline steps."
        )
        st.download_button(
            label="⬇️ Download Notebook (.ipynb)",
            data=nb_bytes.getvalue(),
            file_name=fname,
            mime="application/x-ipynb+json",
            key="nb_download_btn",
        )

        st.markdown("---")
        st.write("### 👁️ Notebook Cell Preview")
        for i, cell in enumerate(nb.cells):
            ctype   = "📝 Markdown" if cell.cell_type == "markdown" else "💻 Code"
            preview = cell.source[:200] + ("..." if len(cell.source) > 200 else "")
            with st.expander(f"{ctype} — Cell {i+1}", expanded=False):
                if cell.cell_type == "code":
                    st.code(preview, language="python")
                else:
                    st.markdown(preview)




# ─────────────────────────────────────────────
#  FEATURE 15 — GENERATE REPORT
# ─────────────────────────────────────────────
def show_generate_report(data: pd.DataFrame, raw_data: pd.DataFrame):
    st.markdown('<div class="section-header"><h2>📄 Generate Analysis Report</h2></div>', unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        report_title = st.text_input("Report Title:", value="Data Analysis Report", key="rpt_title")
        analyst_name = st.text_input("Analyst Name:", value="", key="rpt_analyst")
    with r2:
        dataset_name  = st.text_input("Dataset Name:", value="Uploaded Dataset", key="rpt_dataset")
        include_model = st.checkbox("Include ML Model Results", value=True, key="rpt_model")

    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    def h(text, level=1):
        sep = "=" if level == 1 else "-"
        lines.append(sep * 60)
        lines.append(text.upper() if level == 1 else text)
        lines.append(sep * 60)

    def row(key, val): lines.append(f"  {key:<30}: {val}")

    h(report_title)
    lines.extend([f"  Generated On  : {now}", f"  Analyst       : {analyst_name or 'N/A'}",
                  f"  Dataset       : {dataset_name}", ""])
    h("1. Dataset Summary", 2)
    row("Raw Rows", raw_data.shape[0]); row("Raw Columns", raw_data.shape[1])
    row("Active Rows", data.shape[0]); row("Active Columns", data.shape[1])
    row("Total Null Cells (raw)", raw_data.isnull().sum().sum())
    row("Total Null Cells (active)", data.isnull().sum().sum())
    row("Duplicate Rows (raw)", raw_data.duplicated().sum()); lines.append("")
    h("2. Column Overview", 2)
    for col in data.columns:
        s = data[col]; null_pct = s.isnull().mean() * 100
        if pd.api.types.is_numeric_dtype(s):
            lines.append(f"  {col:<25} | numeric   | mean={s.mean():.3f} std={s.std():.3f} null={null_pct:.1f}%")
        else:
            lines.append(f"  {col:<25} | category  | unique={s.nunique()} null={null_pct:.1f}%")
    lines.append("")
    h("3. EDA & Data Cleaning", 2)
    if "cleaned_data" in st.session_state:
        row("Cleaning Applied", "Yes")
        row("Rows After Cleaning", st.session_state["cleaned_data"].shape[0])
        row("Remaining Nulls", st.session_state["cleaned_data"].isnull().sum().sum())
    else: row("Cleaning Applied", "No")
    lines.append("")
    h("4. Outlier Handling", 2)
    if "outlier_handled_data" in st.session_state:
        row("Outlier Handling Applied", "Yes")
        row("Rows After Handling", st.session_state["outlier_handled_data"].shape[0])
    else: row("Outlier Handling Applied", "No")
    lines.append("")
    h("5. Feature Engineering", 2)
    if "fe_working" in st.session_state:
        fe = st.session_state["fe_working"]
        new_cols = [c for c in fe.columns if c not in raw_data.columns]
        row("Applied", "Yes"); row("New Columns Added", len(new_cols))
    else: row("Applied", "No")
    lines.append("")

    if include_model:
        h("6. ML Model Results", 2)
        if "ml_res_model" in st.session_state:
            row("Algorithm",     st.session_state.get("ml_res_algo", "N/A"))
            row("Problem Type",  st.session_state.get("ml_res_prob", "N/A"))
            row("Feature Count", len(st.session_state.get("ml_res_X_cols", [])))
            if "ml_res_cv" in st.session_state:
                cv = st.session_state["ml_res_cv"]
                row("CV Score (mean ± std)", f"{cv.mean():.4f} ± {cv.std():.4f}")
        else: row("Model Trained", "No")
        if "mm_results" in st.session_state:
            lines.append(""); lines.append("  Multi-Model Leaderboard (Top 5):")
            sl    = st.session_state.get("mm_score_label", "Score")
            valid = [r for r in st.session_state["mm_results"] if "Error" not in r]
            test_key = f"Test {sl}"
            sorted_valid = sorted(valid, key=lambda x: x.get(test_key, 0), reverse=True)
            for i, r in enumerate(sorted(valid, key=lambda x: x.get(f"Test {sl}", 0), reverse=True)[:5], 1):
                lines.append(f"    {i}. {r['Model']:<35} Test {sl} = {r.get(f'Test {sl}', 'N/A')}")
        lines.append("")

    h("7. Recommendations", 2)
    nulls_left = data.isnull().sum().sum()
    if nulls_left > 0: lines.append(f"  • Dataset still has {nulls_left} null(s). Run EDA → Null Handling.")
    if "cleaned_data" not in st.session_state: lines.append("  • No cleaning applied yet.")
    if "ml_res_model" not in st.session_state: lines.append("  • No model has been trained yet.")
    if not any(l.startswith("  •") for l in lines[-8:]): lines.append("  • Dataset looks clean and model-ready. ✅")
    lines.append(""); h("END OF REPORT")

    report_text = "\n".join(lines)
    with st.expander("👁️ Full Report Preview", expanded=True):
        st.text(report_text)

    dl1, dl2 = st.columns(2)
    with dl1:
        report_fname = report_title.replace(' ', '_')
        st.download_button("📄 Download as .txt", data=report_text.encode(),
                       file_name=f"{report_fname}.txt",
                       mime="text/plain", key="rpt_dl_txt")
    with dl2:
        html_report = f"""<html><head><style>
body{{font-family:Inter,sans-serif;margin:40px;color:#e8eaf0;background:#0d0f14;}}
h1{{color:#6c63ff;border-bottom:2px solid #6c63ff;font-family:Syne,sans-serif;}}
pre{{background:#151822;padding:20px;border-radius:12px;overflow-x:auto;border:1px solid rgba(255,255,255,0.07);}}
</style></head><body>
<h1>{report_title}</h1>
<p><b>Generated:</b> {now} | <b>Analyst:</b> {analyst_name or 'N/A'} | <b>Dataset:</b> {dataset_name}</p>
<pre>{report_text}</pre></body></html>"""
        st.download_button("🌐 Download as .html", data=html_report.encode(),
                       file_name=f"{report_fname}.html",
                       mime="text/html", key="rpt_dl_html")


# ─────────────────────────────────────────────
#  FEATURE — PIPELINE COLLABORATION
# ─────────────────────────────────────────────
def show_pipeline_collaboration():
    st.markdown(
        '<div class="section-header"><h2>🤝 Pipeline Collaboration</h2></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Export your full pipeline configuration as a JSON file and share it with "
        "teammates. They can import it to instantly restore all your settings, "
        "parameters, and choices — no cloud required."
    )

    tab_exp, tab_imp, tab_diff = st.tabs([
        "📤 Export Pipeline", "📥 Import Pipeline", "🔍 Compare Configs"
    ])

    # ── EXPORT ──
    with tab_exp:
        st.write("### 📤 Export Your Pipeline Config")

        # Live pipeline summary
        steps_done = {
            "📁 Data Uploaded":       st.session_state.get("raw_data")               is not None,
            "🔍 Nulls Handled":       "cleaned_data"        in st.session_state,
            "⚙️ Features Engineered": "fe_working"           in st.session_state,
            "🚨 Outliers Handled":    "outlier_handled_data" in st.session_state,
            "⚖️ Imbalance Handled":   ("imb_result" in st.session_state
                                       or "casc_result" in st.session_state),
            "🤖 Model Trained":       "ml_res_model"         in st.session_state,
            "🏆 Models Compared":     "mm_results"           in st.session_state,
            "🔧 Hyperparams Tuned":   (st.session_state.get("ht_searcher") is not None
                                       or st.session_state.get("ht_pso_best_params") is not None),
            "📦 Predictions Made":    "ep_results"           in st.session_state,
        }
        completed = sum(steps_done.values())
        st.progress(completed / len(steps_done))
        st.caption(f"{completed}/{len(steps_done)} pipeline steps will be exported.")

        col_left, col_right = st.columns(2)
        with col_left:
            for step, done in list(steps_done.items())[:5]:
                color = "#10b981" if done else "#374151"
                icon  = "●" if done else "○"
                st.markdown(
                    f'<div style="font-size:0.8rem;color:{color};">{icon} {step}</div>',
                    unsafe_allow_html=True,
                )
        with col_right:
            for step, done in list(steps_done.items())[5:]:
                color = "#10b981" if done else "#374151"
                icon  = "●" if done else "○"
                st.markdown(
                    f'<div style="font-size:0.8rem;color:{color};">{icon} {step}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Optional note
        collab_note = st.text_area(
            "Add a note for your collaborator (optional):",
            placeholder="e.g. Used median imputation on Age column. Target = Survived. Best model was Random Forest.",
            key="collab_note",
            height=80,
        )

        if st.button("📦 Generate Export", key="collab_export_btn", type="primary"):
            state = _export_pipeline_state()
            if collab_note.strip():
                state["meta"]["note"] = collab_note.strip()
            json_bytes = json.dumps(state, indent=2, default=str).encode()
            st.session_state["collab_export_json"] = json_bytes
            st.success("Pipeline config ready to download!")

        if "collab_export_json" in st.session_state:
            from datetime import datetime
            fname = f"pipeline_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            st.download_button(
                "⬇️ Download Pipeline Config (.json)",
                data=st.session_state["collab_export_json"],
                file_name=fname,
                mime="application/json",
                key="collab_dl_btn",
            )

            with st.expander("👁️ Preview exported config"):
                preview = json.loads(st.session_state["collab_export_json"])
                st.json(preview)

    # ── IMPORT ──
    with tab_imp:
        st.write("### 📥 Import a Collaborator's Pipeline Config")
        st.info(
            "Upload a `.json` file exported by a collaborator. "
            "All their widget selections and parameters will be restored instantly. "
            "**Note:** You still need to upload the dataset separately."
        )

        imported_file = st.file_uploader(
            "Upload pipeline config JSON:",
            type=["json"],
            key="collab_import_file",
        )

        if imported_file is not None:
            try:
                config = json.load(imported_file)

                # Show metadata
                meta = config.get("meta", {})
                st.markdown("#### 📋 Config Details")
                m1, m2 = st.columns(2)
                m1.info(f"**Exported at:** {meta.get('exported_at', 'Unknown')}")
                m2.info(f"**App version:** {meta.get('app_version', 'Unknown')}")
                if meta.get("note"):
                    st.markdown(
                        f'<div style="background:rgba(108,99,255,0.1);border:1px solid '
                        f'rgba(108,99,255,0.3);border-radius:8px;padding:10px 14px;'
                        f'font-size:0.85rem;margin-bottom:1rem;">'
                        f'💬 <b>Collaborator note:</b> {meta["note"]}</div>',
                        unsafe_allow_html=True,
                    )

                # Show pipeline steps from the config
                steps = config.get("pipeline_steps", {})
                if steps:
                    st.markdown("#### 🗺️ Collaborator's Pipeline Progress")
                    step_labels = {
                        "data_uploaded":       "📁 Data Uploaded",
                        "nulls_handled":       "🔍 Nulls Handled",
                        "features_engineered": "⚙️ Features Engineered",
                        "outliers_handled":    "🚨 Outliers Handled",
                        "imbalance_handled":   "⚖️ Imbalance Handled",
                        "model_trained":       "🤖 Model Trained",
                        "models_compared":     "🏆 Models Compared",
                        "hyperparams_tuned":   "🔧 Hyperparams Tuned",
                        "predictions_made":    "📦 Predictions Made",
                    }
                    completed = sum(steps.values())
                    st.progress(completed / max(len(steps), 1))
                    cols = st.columns(2)
                    for i, (key, label) in enumerate(step_labels.items()):
                        done  = steps.get(key, False)
                        color = "#10b981" if done else "#374151"
                        icon  = "●" if done else "○"
                        cols[i % 2].markdown(
                            f'<div style="font-size:0.8rem;color:{color};">{icon} {label}</div>',
                            unsafe_allow_html=True,
                        )

                # Widget states preview
                widget_states = config.get("widget_states", {})
                if widget_states:
                    with st.expander(f"⚙️ {len(widget_states)} widget settings to restore"):
                        st.json(widget_states)

                # Result summaries preview
                result_summary = config.get("result_summary", {})
                if result_summary:
                    with st.expander(f"📊 {len(result_summary)} result summaries to restore"):
                        st.json(result_summary)

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("✅ Apply This Config", key="collab_apply_btn", type="primary"):
                    success, summary = _import_pipeline_state(config)
                    if success:
                        st.success(summary)
                        st.info(
                            "⬆️ Now upload the same dataset in the sidebar, "
                            "then navigate to any feature — settings are pre-filled."
                        )
                        st.rerun()
                    else:
                        st.error("Import failed.")

            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file. Make sure it was exported from this app.")
            except Exception as e:
                st.error(f"❌ Import error: {e}")

    # ── DIFF / COMPARE ──
    with tab_diff:
        st.write("### 🔍 Compare Two Pipeline Configs")
        st.caption("Upload two exported configs to see exactly where they differ.")

        d1, d2 = st.columns(2)
        with d1:
            file_a = st.file_uploader("Config A:", type=["json"], key="diff_a")
        with d2:
            file_b = st.file_uploader("Config B:", type=["json"], key="diff_b")

        if file_a and file_b:
            try:
                config_a = json.load(file_a)
                config_b = json.load(file_b)

                ws_a = config_a.get("widget_states", {})
                ws_b = config_b.get("widget_states", {})
                all_keys = set(ws_a.keys()) | set(ws_b.keys())

                diff_rows = []
                for key in sorted(all_keys):
                    val_a = ws_a.get(key, "—")
                    val_b = ws_b.get(key, "—")
                    diff_rows.append({
                        "Setting":   key,
                        "Config A":  str(val_a),
                        "Config B":  str(val_b),
                        "Different": "⚠️ Yes" if val_a != val_b else "✅ Same",
                    })

                diff_df   = pd.DataFrame(diff_rows)
                n_diff    = (diff_df["Different"] == "⚠️ Yes").sum()
                n_same    = (diff_df["Different"] == "✅ Same").sum()

                dd1, dd2  = st.columns(2)
                dd1.metric("Settings that differ", n_diff)
                dd2.metric("Settings that match", n_same)

                show_only_diff = st.checkbox("Show only differences", value=True, key="diff_only")
                display_df = diff_df[diff_df["Different"] == "⚠️ Yes"] if show_only_diff else diff_df

                def highlight_diff(row):
                    return ["background-color: rgba(239,68,68,0.12)"
                            if row["Different"] == "⚠️ Yes" else ""
                            for _ in row]

                st.dataframe(
                    display_df.style.apply(highlight_diff, axis=1),
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Comparison error: {e}")


# ─────────────────────────────────────────────
#  SIDEBAR & MAIN ROUTING
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem; text-align:center;">
      <div style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
                  background:linear-gradient(135deg,#6c63ff,#00e5ff);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        🔬 Agentic DS
      </div>
      <div style="font-size:0.72rem;color:#6b7280;margin-top:3px;">Autonomous Data Science Platform</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        with st.expander("🔑 API Configuration", expanded=True):
            groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
            if groq_api_key:
                os.environ["GROQ_API_KEY"] = groq_api_key
            else:
                st.caption("Required for AI Agent Chat.")

    st.markdown("---")
    st.markdown("**📂 Upload Dataset**")
    file_mode     = st.radio("File Type:", list(FILE_EXTENSIONS.keys()), horizontal=True)
    uploaded_file = st.file_uploader(f"Upload {file_mode} file", type=FILE_EXTENSIONS[file_mode], label_visibility="collapsed")

if uploaded_file is not None:
    if st.session_state.get("last_uploaded_file") != uploaded_file.name:
        raw_data_loaded = process_file(uploaded_file, file_mode)
        if raw_data_loaded is not None:
            data_store.clear()
            set_data(raw_data_loaded)
            st.session_state["raw_data"]          = raw_data_loaded
            st.session_state["last_uploaded_file"] = uploaded_file.name
            st.session_state["agent_thread_id"]    = str(uuid.uuid4())
            st.session_state["messages"]           = [
                {
                    "role": "assistant",
                    "content": (
                        "📂 Dataset **`" + uploaded_file.name + "`** loaded!\n\n"
                        "- **Rows:** " + f"{raw_data_loaded.shape[0]:,}" + "\n"
                        "- **Columns:** " + str(raw_data_loaded.shape[1]) + "\n"
                        "- **Null values:** " + f"{raw_data_loaded.isnull().sum().sum():,}" + "\n\n"
                        "What would you like me to do? Try asking me to:\n"
                        "- *Show histograms of all numeric columns*\n"
                        "- *What features are most correlated?*\n"
                        "- *Train a model to predict [target_column]*"
                    ),
                }
            ]
            with st.sidebar:
                st.success(f"✅ {raw_data_loaded.shape[0]:,} rows × {raw_data_loaded.shape[1]} cols")

raw_data   = st.session_state.get("raw_data")
is_cleaned = "cleaned_data" in st.session_state
data       = st.session_state["cleaned_data"] if is_cleaned else raw_data

# ✅ Sync session_state to shared data_store on every rerun
if raw_data is not None:
    data_store.set_data(raw_data)
if is_cleaned:
    data_store.set_cleaned_data(st.session_state["cleaned_data"])

if data is not None:
    set_data(data)

    with st.sidebar:
        st.markdown("---")
        if is_cleaned:
            cleaned = st.session_state["cleaned_data"]
            st.markdown(f"""
            <div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);
                        border-radius:8px;padding:10px 12px;font-size:0.8rem;">
              <div style="color:#10b981;font-weight:700;margin-bottom:4px;">✅ Processed Data Active</div>
              <div style="color:#9ca3af;">Rows: {cleaned.shape[0]:,} | Cols: {cleaned.shape[1]}</div>
              <div style="color:#9ca3af;">Nulls: {cleaned.isnull().sum().sum()}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("↩️ Reset to Raw Data", key="reset_raw"):
                del st.session_state["cleaned_data"]
                set_data(raw_data); st.rerun()
        else:
            st.markdown(f"""
            <div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.25);
                        border-radius:8px;padding:10px 12px;font-size:0.8rem;">
              <div style="color:#f59e0b;font-weight:700;margin-bottom:4px;">⚠️ Raw Data Active</div>
              <div style="color:#9ca3af;">Rows: {raw_data.shape[0]:,} | Cols: {raw_data.shape[1]}</div>
              <div style="color:#9ca3af;">Nulls: {raw_data.isnull().sum().sum()}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Pipeline tracker
        st.markdown("**🗺️ Pipeline**")
        steps = {
            "📁 Data Uploaded":       True,
            "🔍 Nulls Handled":       "cleaned_data"          in st.session_state,
            "⚙️ Features Engineered": "fe_working"             in st.session_state,
            "🚨 Outliers Handled":    "outlier_handled_data"   in st.session_state,
            "⚖️ Imbalance Handled": ("imb_result"    in st.session_state or "casc_result"   in st.session_state),
            "🧠 Domain Constraints":  "domain_constraints"    in st.session_state,  # ← ADD THIS
            "🤖 Model Trained":       "ml_res_model"          in st.session_state,
            "🏆 Models Compared":     "mm_results"            in st.session_state,
            "🔧 Hyperparams Tuned":   "ht_searcher"           in st.session_state,
            "📦 Predictions Made":    "ep_results"            in st.session_state,
        }
        completed = sum(steps.values())
        st.progress(completed / len(steps))
        st.caption(f"{completed}/{len(steps)} steps completed")

        for step_name, done in steps.items():
            color = "#10b981" if done else "#374151"
            icon  = "●" if done else "○"
            st.markdown(
                f'<div style="font-size:0.78rem;color:{color};padding:2px 0;">{icon} {step_name}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        if "main_navigation_radio" not in st.session_state:
            st.session_state["main_navigation_radio"] = FEATURES[0]
        option = st.radio("📋 Navigate:", FEATURES, key="main_navigation_radio")

    # ── MAIN CONTENT ──
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:1rem 0 0.5rem;">
      <div style="font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;
                  background:linear-gradient(135deg,#6c63ff,#00e5ff);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        🔬 Agentic Data Science Dashboard
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Status banner
    if is_cleaned:
        st.markdown(
            f'<div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);'
            f'border-radius:10px;padding:10px 16px;margin-bottom:1rem;font-size:0.85rem;">'
            f'✅ <b>Processed data active</b> — {data.shape[0]:,} rows × {data.shape[1]} cols | '
            f'{data.isnull().sum().sum()} nulls remaining</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);'
            f'border-radius:10px;padding:10px 16px;margin-bottom:1rem;font-size:0.85rem;">'
            f'⚠️ <b>Raw data active</b> — {data.shape[0]:,} rows × {data.shape[1]} cols | '
            f'{data.isnull().sum().sum()} null(s). Use <b>EDA – Null Value Handling</b> to clean first.</div>',
            unsafe_allow_html=True,
        )

    if   option == "🤖 AI Agent Chat":            show_ai_agent_chat(data, groq_api_key)
    elif option == "📊 Data Overview":             show_data_overview(data)
    elif option == "📋 Data Profiling Report":     show_data_profiling(data)
    elif option == "🔍 EDA – Null Value Handling": show_eda_null_handling(raw_data)
    elif option == "⚙️ Feature Engineering":       show_feature_engineering(data)
    elif option == "📈 Basic Statistics":           show_basic_statistics(data)
    elif option == "📉 Visualizations":             show_visualizations(data)
    elif option == "🔗 Correlation Plot":           show_correlation_plot(data)
    elif option == "🚨 Outlier Detection":          show_outlier_detection(data)
    elif option == "⚖️ Class Imbalance Handling":  show_class_imbalance(data)
    elif option == "📈 Time Series Analysis":       show_time_series(data)
    elif option == "🤖 ML Algorithm":               show_ml_algorithm(data)
    elif option == "🏆 Multi-Model Comparison":     show_multi_model_comparison(data)
    elif option == "🔧 Hyperparameter Tuning":      show_hyperparameter_tuning(data)
    elif option == "📦 Model Export & Predict":     show_model_export_predict(data)
    elif option == "📄 Generate Report":            show_generate_report(data, raw_data)
    elif option == "🧠 Domain Knowledge Constraints": show_domain_knowledge_constraints(data)  # ← ADD THIS
    elif option == "📓 Export Pipeline Notebook":   show_export_notebook(data, raw_data)
    elif option == "🤝 Pipeline Collaboration":   show_pipeline_collaboration()

else:
    # ── LANDING PAGE ──
    st.markdown("""
    <div style="font-family:Syne,sans-serif;font-size:2.4rem;font-weight:800;
                background:linear-gradient(135deg,#6c63ff,#00e5ff);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                margin-bottom:0.5rem;">
      🔬 Agentic Data Science Platform
    </div>
    <div style="color:#6b7280;font-size:1rem;margin-bottom:2rem;">
      Upload a dataset in the sidebar to get started. Powered by scikit-learn + Groq AI.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-grid">
      <div class="feature-card"><div class="icon">🤖</div><div class="title">AI Agent Chat</div><div class="desc">Chat naturally — powered by LangGraph + Groq. Gets charts directly in chat.</div></div>
      <div class="feature-card"><div class="icon">📊</div><div class="title">Data Overview</div><div class="desc">Shape, dtypes, column breakdown, and a live data preview.</div></div>
      <div class="feature-card"><div class="icon">📋</div><div class="title">Data Profiling Report</div><div class="desc">Full column profiles, correlations, imbalance checks & memory usage.</div></div>
      <div class="feature-card"><div class="icon">🔍</div><div class="title">EDA – Null Handling</div><div class="desc">Detect and fill/drop nulls with 10 different imputation methods.</div></div>
      <div class="feature-card"><div class="icon">⚙️</div><div class="title">Feature Engineering</div><div class="desc">Encoding, polynomial features, math transforms, binning & more.</div></div>
      <div class="feature-card"><div class="icon">📉</div><div class="title">Visualizations</div><div class="desc">Histogram, boxplot, scatter, violin, heatmap, pairplot — dark themed.</div></div>
      <div class="feature-card"><div class="icon">🚨</div><div class="title">Outlier Detection</div><div class="desc">Z-Score / IQR detection + PCA + K-Means / DBSCAN clustering.</div></div>
      <div class="feature-card"><div class="icon">⚖️</div><div class="title">Class Imbalance</div><div class="desc">SMOTE, oversampling, undersampling — fix skewed datasets.</div></div>
      <div class="feature-card"><div class="icon">📈</div><div class="title">Time Series Analysis</div><div class="desc">Trend, seasonality, stationarity (ADF), ACF/PACF decomposition.</div></div>
      <div class="feature-card"><div class="icon">🤖</div><div class="title">ML Algorithm</div><div class="desc">Train classifiers & regressors including glass-box EBM & FIGS models, with fairness/bias diagnostics.</div></div>
      <div class="feature-card"><div class="icon">🏆</div><div class="title">Multi-Model Comparison</div><div class="desc">Benchmark all algorithms side-by-side with a ranked leaderboard.</div></div>
      <div class="feature-card"><div class="icon">🔧</div><div class="title">Hyperparameter Tuning</div><div class="desc">GridSearchCV / RandomizedSearchCV — find the optimal parameters.</div></div>
      <div class="feature-card"><div class="icon">📦</div><div class="title">Model Export & Predict</div><div class="desc">Download trained model as .pkl, predict on new CSV files.</div></div>
      <div class="feature-card"><div class="icon">📄</div><div class="title">Generate Report</div><div class="desc">Auto-compile the full pipeline report as TXT or HTML.</div></div>
      <div class="feature-card"><div class="icon">🤝</div><div class="title">Pipeline Collaboration</div><div class="desc">Export your full pipeline config as JSON. Teammates import it to instantly restore all settings — no cloud needed.</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <br>
    <div style="background:rgba(108,99,255,0.08);border:1px solid rgba(108,99,255,0.2);
                border-radius:12px;padding:1rem 1.5rem;margin-top:1rem;font-size:0.85rem;color:#9ca3af;">
      💡 <b style="color:#6c63ff;">Supported formats:</b> CSV, Excel, JSON, XML, Text &nbsp;|&nbsp;
      🔑 Add your <b>Groq API Key</b> in the sidebar for AI Agent Chat
    </div>
    """, unsafe_allow_html=True)