import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

def load_imdb(path: str) -> pd.DataFrame:
    """Đọc dữ liệu IMDB CSV an toàn, tự động xử lý lỗi ngoặc/dòng và regex dự phòng."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file '{path}'. Vui lòng tải 'IMDB Dataset.csv' lên Google Colab!")
    
    try:
        d = pd.read_csv(path, on_bad_lines="skip")
    except pd.errors.ParserError:
        d = pd.read_csv(path, on_bad_lines="skip", engine="python")
        
    if len(d) >= 45000:
        df = d.drop_duplicates(subset=["review"]).reset_index(drop=True)
        df["label"] = (df["sentiment"] == "positive").astype(int)
        return df

    # Đọc dự phòng bằng Regex nếu pandas làm rơi mất dòng
    text = open(path, encoding="utf-8", errors="replace").read().replace("\r\n", "\n")
    body = text.split("\n", 1)[1] if "\n" in text else text
    rows = []
    for review, label in re.findall(r"(.*?),(positive|negative)\n", body, flags=re.S):
        review = review.strip()
        if len(review) >= 2 and review[0] == '"' and review[-1] == '"':
            review = review[1:-1].replace('""', '"')
        rows.append((review, label))
    
    df = pd.DataFrame(rows, columns=["review", "sentiment"])
    df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)
    df["label"] = (df["sentiment"] == "positive").astype(int)
    return df

def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Chia tập dữ liệu Train/Test phân tầng (Stratified Split)."""
    return train_test_split(
        df["clean"],
        df["label"],
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state
    )
