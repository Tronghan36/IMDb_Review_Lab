import os
import pickle
import pandas as pd

def save_csv(df: pd.DataFrame, path: str):
    """Lưu DataFrame ra CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[✓] Đã lưu CSV ra '{path}'")

def save_pickle(obj, path: str):
    """Lưu đối tượng ra file Pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[✓] Đã lưu pickle ra '{path}'")
