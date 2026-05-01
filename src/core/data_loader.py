import pandas as pd
import xml.etree.ElementTree as ET
import json

def process_file(uploaded_file, file_mode: str) -> pd.DataFrame:
    """Processes an uploaded file and returns a pandas DataFrame."""
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
        raise ValueError(f"Error processing file: {e}")
