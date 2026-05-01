"""
Shared in-memory data store accessible by both the Streamlit UI and the LangGraph agent tools.
Session state cannot be reliably accessed from within tool execution contexts,
so we use this module-level store as the single source of truth for the active dataset.
"""

_store: dict = {}

def set_data(df):
    """Store the active DataFrame."""
    _store["data"] = df

def get_data():
    """Retrieve the active DataFrame. Returns None if not set."""
    return _store.get("data", None)

def set_cleaned_data(df):
    """Store the cleaned DataFrame."""
    _store["cleaned_data"] = df

def get_cleaned_data():
    """Retrieve the cleaned DataFrame, falling back to raw data."""
    return _store.get("cleaned_data", _store.get("data", None))

def set_model_results(results: dict):
    """Store training results."""
    _store["model_results"] = results

def get_model_results():
    """Retrieve training results."""
    return _store.get("model_results", None)

def clear():
    """Clear all stored data."""
    _store.clear()
