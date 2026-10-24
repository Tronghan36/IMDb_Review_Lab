import os
import pandas as pd

# Import đúng các hàm/lớp từ các file trong src/
from src.dataset import load_imdb, get_train_test_split
from src.preprocess import clean_text, build_tfidf_vectorizer
from src.models import (
    get_baseline_models,
    build_xgb_model,
    build_stacking_model,
    NBSVMClassifier
)
from src.evaluate import evaluate_predictions, plot_comparison_grid
from src.utils import save_csv

def run_pipeline():
    print("=" * 60)
    print("🚀 BẮT ĐẦU CHẠY PIPELINE PHÂN TÍCH CẢM XÚC IMDB")
    print("=" * 60)

    # 1. Xác định đường dẫn dữ liệu
    data_path = os.path.join("data", "raw", "IMDB Dataset.csv")
    if not os.path.exists(data_path):
        data_path = "IMDB Dataset.csv"  # Tìm ở thư mục gốc nếu không có trong data/raw/

    # 2. Nạp dữ liệu bằng load_imdb
    print("\n[1/4] Đang nạp dữ liệu...")
    df = load_imdb(data_path)
    print(f"   -> Đã tải thành công {len(df)} dòng dữ liệu.")

    # Làm sạch văn bản để tạo cột 'clean' phục vụ split
    print("   -> Đang làm sạch văn bản (loại bỏ HTML, stopwords, chuẩn hóa)...")
    df["clean"] = df["review"].apply(clean_text)

    # Chia tập Train / Test phân tầng
    X_train_raw, X_test_raw, y_train, y_test = get_train_test_split(df, test_size=0.2, random_state=42)

    # 3. Trích xuất đặc trưng TF-IDF
    print("\n[2/4] Đang trích xuất ma trận đặc trưng TF-IDF...")
    vectorizer = build_tfidf_vectorizer(ngram_range=(1, 2), max_features=20000)
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    print(f"   -> Kích thước ma trận Train TF-IDF: {X_train_tfidf.shape}")

    # 4. Khởi tạo danh sách các mô hình thực nghiệm
    print("\n[3/4] Khởi tạo các mô hình...")
    models = {}
    
    # Nạp các mô hình Baseline
    try:
        models.update(get_baseline_models())
    except Exception as e:
        print(f"   ⚠️ Lỗi Baseline: {e}")

    # Nạp XGBoost
    try:
        models["XGBoost (hist)"] = build_xgb_model()
    except Exception as e:
        print(f"   ⚠️ Lỗi XGBoost: {e}")

    # Nạp Stacking
    try:
        models["Stacking Meta-Learner"] = build_stacking_model()
    except Exception as e:
        print(f"   ⚠️ Lỗi Stacking: {e}")

    # Nạp NB-SVM
    try:
        models["NB-SVM (Custom)"] = NBSVMClassifier(C=2.0)
    except Exception as e:
        print(f"   ⚠️ Lỗi NB-SVM: {e}")

    # 5. Huấn luyện & Đánh giá
    print("\n[4/4] Bắt đầu huấn luyện và tính toán chỉ số...")
    results = []
    
    for model_name, model in models.items():
        print(f"   -> Đang chạy: {model_name}...")
        try:
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            
            # Tính xác suất hoặc khoảng cách phân loại để tính AUC (ROC)
            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_tfidf)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test_tfidf)
                
            metrics = evaluate_predictions(y_test, y_pred, y_prob=y_prob, model_name=model_name)
            results.append(metrics)
        except Exception as e:
            print(f"      ❌ Lỗi khi chạy mô hình {model_name}: {e}")

    if not results:
        print("\n❌ Không có mô hình nào thực thi thành công!")
        return

    # 6. Tổng hợp kết quả và xuất báo cáo
    df_results = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("📊 BẢNG KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 60)
    print(df_results.to_string(index=False))

    # Lưu file kết quả CSV
    report_path = os.path.join("outputs", "reports", "bang_so_sanh_mo_hinh.csv")
    save_csv(df_results, report_path)

    # Lưu biểu đồ so sánh
    chart_path = os.path.join("outputs", "figures", "so_sanh_hieu_nang.png")
    try:
        plot_comparison_grid(df_results, title="So sánh hiệu năng các mô hình phân tích cảm xúc IMDb", save_path=chart_path)
        print(f"[✓] Đã lưu biểu đồ so sánh ra '{chart_path}'")
    except Exception as e:
        print(f"   ⚠️ Lỗi vẽ biểu đồ: {e}")

    print("\n✅ HOÀN THÀNH HOÀN TOÀN PIPELINE! Báo cáo đã lưu vào thư mục outputs/")

if __name__ == "__main__":
    run_pipeline()