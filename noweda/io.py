import pandas as pd

def read(file_path, **kwargs):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path, **kwargs)
    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path, **kwargs)
    elif file_path.endswith(".json"):
        return pd.read_json(file_path, **kwargs)
    elif file_path.endswith(".xml"):
        return pd.read_xml(file_path, **kwargs)
    elif file_path.endswith(".html"):
        return pd.read_html(file_path, **kwargs)[0]
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
