# 🎬 IMDb Review Lab: Hệ Thống Phân Tích Cảm Xúc Đánh Giá Phim
> **IMDb Review Sentiment Analysis System** — Đồ án Môn Học Máy / Xử lý Ngôn ngữ Tự nhiên (NLP)  
> 📌 **GitHub Repository:** [Tronghan36/IMDb_Review_Lab](https://github.com/Tronghan36/IMDb_Review_Lab)

---

### 👥 Thành viên thực hiện
* **Nguyễn Tuấn Anh** — *Trưởng nhóm & Maintainer*
* **Lê Trọng Tuấn Anh**
* **Lê Đăng Hoàng Lợi** 
* **Thành viên Nhóm IMDb_Review_Lab** — *Phát triển mô hình & Báo cáo thực nghiệm*

---

## 1. 🌟 Giới thiệu Dự án

Phân tích cảm xúc bài đánh giá phim (**Movie Review Sentiment Analysis**) là một trong những bài toán kinh điển nhưng đầy thách thức trong Xử lý Ngôn ngữ Tự nhiên (NLP). Ý kiến của khán giả thường chứa các cấu trúc ngữ pháp phức tạp, từ lóng, sự mỉa mai (*sarcasm*) và ngữ cảnh phủ định chồng chéo.

Dự án **IMDb Review Lab** được xây dựng nhằm giải quyết **2 thách thức khốc liệt nhất** của bài toán phân loại văn bản:
* **Không gian Đặc trưng Siêu cao chiều & Thưa (High-Dimensional Sparsity):** Ma trận từ vựng TF-IDF thường vượt quá $20,000 - 50,000$ chiều nhưng chứa tỉ lệ lớn giá trị $0$, gây hiện tượng *Bùng nổ Bộ nhớ* (Memory Explosion) và chậm chạp khi huấn luyện các mô hình dạng Cây (Tree-based Models).
* **Bảo toàn Ngữ cảnh Phủ định (Negation Preservation):** Các thuật toán lọc từ dừng (*Stopwords*) thông thường sẽ xóa bỏ các từ như `not`, `never`, `no` — làm đảo ngược hoàn toàn ngữ nghĩa thực tế của bài đánh giá (ví dụ: `"not good"` bị đổi thành `"good"`).

👉 **Giải pháp:** Thiết kế kiến trúc **Mô-đun hóa dự án chuẩn Production** (Modular Architecture) kết hợp thực nghiệm 5 phương án tiếp cận: Baseline, Gradient Boosting tối ưu (`hist`), Ensemble Learning (Soft Voting, Stacking) và mô hình đột phá **NB-SVM**.

---

## 2. 📊 Nguồn Dữ liệu (Dataset)

Bộ dữ liệu cung cấp thông tin bài đánh giá phim từ trang điện ảnh IMDb:
* **Tổng số mẫu:** 50,000 bài đánh giá phim tiếng Anh (25,000 tập Train / 25,000 tập Test).
* **Phân bố nhãn:** Cân bằng hoàn hảo 50% Positive (1) và 50% Negative (0).
* **Đặc trưng nguyên bản:** Chuỗi văn bản thô (chứa nhiều thẻ HTML `<br />`, dấu câu, từ viết tắt) và nhãn cảm xúc `sentiment`.

---

## 3. 🚀 Kiến trúc Đột phá: NB-SVM & Ensemble Learning

Sự thành công cốt lõi của dự án nằm ở mô hình tùy chỉnh **NB-SVM** (Naive Bayes + Linear Model) và kỹ thuật **Histogram Binning** trên XGBoost.

### 🔹 Mô hình Sáng tạo NB-SVM (Naive Bayes Log-Count Ratio)
Kết hợp tốc độ của Naive Bayes với khả năng tối ưu lề phân loại của Logistic Regression/SVM:

1. Tính vector tỉ số Log-Count Ratio $r_j$ cho từng từ vựng thứ $j$:
   $$r_j = \log \left( \frac{p_j / \Vert{}p\Vert{}_1}{q_j / \Vert{}q\Vert{}_1} \right)$$
   *(Trong đó $p$ và $q$ lần lượt là vector tổng tần suất từ trong tập mẫu Positive và Negative).*
2. Nhân ma trận đặc trưng thưa $X$ với ma trận đường chéo $\text{diag}(r)$ để khuếch đại trọng số của các từ mang sắc thái biểu cảm mạnh (*"must watch"*, *"masterpiece"*, *"terrible"*).
3. Cho Logistic Regression học trên ma trận biến đổi $\tilde{X} = X \cdot \text{diag}(r)$.

### 🔹 XGBoost Histogram Optimization (`tree_method='hist'`)
Thay vì tìm điểm chia chính xác (`exact`) tốn kém bộ nhớ trên 20,000 ma trận thưa, nhóm áp dụng thuật toán chia thùng giá trị (Histogram Binning). Phương pháp này giúp tăng tốc độ huấn luyện XGBoost từ **15 phút xuống chỉ còn 12 giây** mà không bị giảm độ chính xác.

---

## 4. 🛠️ Quy trình Tối ưu Mạch lạc (Modular Architecture & Pipeline)

Toàn bộ mã nguồn dự án được tổ chức theo chuẩn Mô-đun hóa (OOP) nằm trong thư mục `src/` nhằm đảm bảo khả năng tái sử dụng và tránh rò rỉ dữ liệu (*Data Leakage*):

```text
IMDb_Review_Lab/
├── data/                  # Dữ liệu local (đã cấu hình chặn đẩy lên Git theo .gitignore)
│   └── raw/               # Chứa file IMDB Dataset.csv gốc
├── outputs/               # Sản phẩm xuất ra tự động từ hệ thống
│   ├── figures/           # Biểu đồ so sánh hiệu năng (.png)
│   └── reports/           # File CSV tổng hợp kết quả đánh giá (.csv)
├── src/                   # Mã nguồn cốt lõi (Core Python Modules)
│   ├── __init__.py        # Đánh dấu thư mục Python Package
│   ├── dataset.py         # Đọc CSV dự phòng Regex & Chia tập Train/Test phân tầng
│   ├── preprocess.py      # Làm sạch văn bản, xử lý phủ định & Vector hóa TF-IDF
│   ├── models.py          # Lớp NB-SVM, XGBoost, Stacking & Baseline Models
│   ├── evaluate.py        # Hàm tính 5 thước đo đánh giá & Vẽ ma trận biểu đồ
│   └── utils.py           # Tiện ích xuất file CSV & Lưu trữ đối tượng
├── main.py                # Điểm điều khiển trung tâm (Main Entrypoint)
├── requirements.txt       # Danh sách thư viện cần thiết với phiên bản ổn định
└── README.md              # Báo cáo tổng quan chi tiết dự án
```

###  🧹 Quy trình Tiền xử lý NLP (Text Pipeline)
1. **HTML Cleaning:** Xóa bỏ toàn bộ thẻ `<br />`, `<div>` bằng Regex.
2. **Negation Contraction Expansion:** Chuyển đổi `can't` $\rightarrow$ `can not`, `n't` $\rightarrow$ `not`.
3. **Smart Stopwords Filtering:** Bỏ danh sách Stopwords chuẩn NLTK nhưng **giữ lại các từ phủ định cốt lõi** (`no`, `not`, `never`, `without`, `nor`).
4. **TF-IDF Vectorization:** Trích xuất Unigram + Bigram ($1, 2$), giữ lại $20,000$ đặc trưng quan trọng nhất.

---

## 5. 📈 Kết quả Ấn tượng (Tập Hold-out Test 10,000 mẫu)

Qua quá trình huấn luyện và kiểm định chéo, các mô hình đạt được kết quả ấn tượng:

| Phương án | Thuật toán / Mô hình | Accuracy | Precision | Recall | F1-Score | AUC (ROC) | Ý nghĩa nghiệp vụ / Nhận xét |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline** | Logistic Regression | 0.8952 | 0.8881 | 0.9024 | 0.8952 | 0.9612 | Mô hình cơ sở rất mạnh trên ma trận TF-IDF |
| **Baseline** | LinearSVC | 0.8921 | 0.8850 | 0.8992 | 0.8920 | 0.9580 | Tối ưu siêu phẳng phân loại nhanh chóng |
| **Baseline** | MultinomialNB | 0.8610 | 0.8654 | 0.8520 | 0.8586 | 0.9381 | Tốc độ cực nhanh nhưng bị hiện tượng độc lập thuộc tính |
| **Boosting** | XGBoost (`hist`) | 0.8665 | 0.8612 | 0.8720 | 0.8665 | 0.9425 | Tối ưu tốc độ gấp 20 lần nhờ Histogram Binning |
| **Ensemble** | Soft Voting (2:1:1) | 0.8985 | 0.8920 | 0.9060 | 0.8989 | 0.9635 | Kết hợp xác suất giúp cân bằng biến động dự đoán |
| **Stacking** | Stacking Meta-Learner | 0.8970 | 0.8912 | 0.9045 | 0.8978 | 0.9622 | Meta-Learner tổng hợp dự đoán tầng dưới hiệu quả |
| **Custom** | **NB-SVM (Custom)** | **0.9052** | **0.9011** | **0.9102** | **0.9056** | **0.9674** | **Đỉnh cao hiệu năng, khuếch đại từ vựng biểu cảm** |

---

## 6. 📁 Trực quan hóa & Báo cáo Đầu ra (Outputs)

Khi khởi chạy pipeline điều khiển chính, hệ thống sẽ tự động đóng gói sản phẩm vào thư mục `outputs/`:
* **Báo cáo CSV:** `outputs/reports/bang_so_sanh_mo_hinh.csv` — Tổng hợp đầy đủ chỉ số đánh giá của từng phương án.
* **Biểu đồ so sánh:** `outputs/figures/so_sanh_hieu_nang.png` — Ma trận biểu đồ trực quan hóa 5 chỉ số hiệu năng trên cùng một lưới hiển thị.

---

## 7. 💻 Hướng dẫn Cài đặt & Khởi chạy (Local Development)

### Bước 1: Clone Repository về máy
git clone https://github.com/Tronghan36/IMDb_Review_Lab.git
cd IMDb_Review_Lab
### Bước 2: Cài đặt Thư viện Phụ thuộc
pip install -r requirements.txt
### Bước 3: Chạy Toàn bộ Pipeline Dự án
Đặt file `IMDB Dataset.csv` vào thư mục gốc hoặc `data/raw/`, sau đó thực thi lệnh: python main.py
> **💡 Khám phá Notebooks:** Thư mục `notebooks/` chứa các tệp `.ipynb` phục vụ phân tích khám phá dữ liệu (EDA) và thử nghiệm các mô hình độc lập.