
# 🎬 Movie Recommender System

Hệ thống gợi ý phim đơn giản sử dụng phương pháp **lọc theo nội dung (Content-Based Filtering)**. Dự án không sử dụng thư viện recommender chuyên biệt như `scikit-learn`, mà tự xây dựng thuật toán từ đầu, chỉ dùng các thư viện hỗ trợ tính toán như `NumPy`, `Pandas`, và `Streamlit` cho phần giao diện.

---

## 🚀 Mục tiêu
Xây dựng một hệ thống gợi ý phim tương tự dựa trên:
- **Nội dung mô tả phim (`overview`)**
- **Thể loại phim (`genre`)**

Phương pháp sử dụng:
- **Tiền xử lý văn bản** (lọc stopwords, lowercase, tách từ)
- **Vector hóa bằng TF-IDF**
- **Tính toán độ tương đồng bằng cosine similarity**
- **Hiển thị kết quả tương tự với phim đã chọn**

---

## 🧠 Mô hình hoạt động

1. **Tiền xử lý văn bản:**
   - Ghép `overview` + `genre` thành trường `tags`
   - Loại bỏ ký tự đặc biệt, chuyển thường, bỏ stopwords

2. **Vector hóa nội dung:**
   - Xây dựng từ điển
   - Tính TF, IDF, TF-IDF vector cho mỗi phim

3. **Tính độ tương đồng:**
   - Dùng công thức cosine similarity để đo mức tương quan giữa các phim

4. **Giao diện Streamlit:**
   - Nhập tên phim bạn yêu thích
   - Hệ thống hiển thị top 5 phim tương tự

---

## 📁 Cấu trúc dự án

   - Cấu trúc dự án cơ bản sẽ như sau:

```
movie_recommender_system/
├── dataset.csv                # Dữ liệu phim gồm title, overview, genre
├── streamlit_app.py           # File giao diện người dùng chạy bằng Streamlit
├── model.py                   # File model
└── README.md                  # Tài liệu hướng dẫn
```

---

## ▶️ Hướng dẫn chạy demo

### 1. Cài các thư viện cần thiết
```bash
pip install pandas numpy streamlit
```

### 2. Chạy ứng dụng Streamlit
```bash
streamlit run streamlit_app.py
```

### 3. Giao diện sẽ mở tự động trên trình duyệt tại:
```
http://localhost:8501
```

---

## 📌 Lưu ý
- Dữ liệu `dataset.csv` phải chứa các cột: `title`, `overview`, `genre`
- Tốc độ xử lý có thể chậm nếu dữ liệu lớn, --> lưu sẵn các vector để tốc độ truy cập nhanh.
- Tối ưu hiệu năng: --> lưu sẵn ma trận similarity

---

## 📚 Áp dụng trong học tập
Dự án này phù hợp để làm **Bài tập lớn môn Trí tuệ nhân tạo**, giúp sinh viên hiểu rõ:
- Cách vector hóa văn bản (TF-IDF)
- Tính toán độ tương đồng
- Ứng dụng AI đơn giản mà hiệu quả trong thực tế

---

## 💡 Định hướng mở rộng
- Bổ sung lựa chọn dựa trên diễn viên, đạo diễn,...
- Kết hợp với giao diện người dùng nâng cao hơn

---

## 👨‍💻 Liên hệ
- Nếu còn thắc mắc cần được giải đáp liên hệ Xuanuio.
