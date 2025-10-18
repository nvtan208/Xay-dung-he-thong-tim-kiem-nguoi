# 🔍 Hệ thống tìm kiếm người bằng mô tả CLIP (Contrastive Language-Image Pre-Training)

Hệ thống này cho phép tìm kiếm ảnh người dựa trên mô tả bằng văn bản (text-to-image retrieval) sử dụng **CLIP**. Người dùng nhập mô tả (có thể bằng tiếng Việt), hệ thống sẽ dịch sang tiếng Anh, tạo embedding, và tìm các ảnh tương tự từ dataset [ICFG-PEDES](https://www.kaggle.com/datasets/meetrathi97/icfg-pedes).

---

## 📁 Cấu trúc thư mục

```
ICFG-PEDES/
├── imgs/                  # Thư mục chứa ảnh
│   ├── test/
│   └── ...
├── captions.csv
├── captions_cleaned.csv
├── ICFG-PEDES.json        # Thông tin file ảnh và captions
└── invalid_paths.csv
preprocess.py              # Tiền xử lý tạo embeddings
app.py                     # Flask API backend
index.html                 # Frontend
image_embeddings.npy       # Embeddings ảnh (sau khi chạy preprocess.py)
image_paths.npy            # Đường dẫn ảnh tương ứng với embeddings
```

---

## ⚙️ Cài đặt

1. **Clone repository**:

```bash
git clone https://github.com/nvtan208/Xay-dung-he-thong-tim-kiem-nguoi
cd Xay-dung-he-thong-tim-kiem-nguoi
```

2. **Tải dataset ICFG-PEDES** (GitHub không lưu dữ liệu lớn):

* Truy cập [Kaggle ICFG-PEDES dataset](https://www.kaggle.com/datasets/meetrathi97/icfg-pedes)
* Tải file zip về và giải nén vào thư mục dự án, đảm bảo cấu trúc thư mục giống như trên.

3. **Tạo virtual environment** (khuyến nghị):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

4. **Cài đặt các package cần thiết**:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pillow pandas numpy tqdm flask flask-cors scikit-learn deep-translator
```

> Lưu ý: Chọn phiên bản `torch` tương thích với GPU nếu có.

---

## 🛠 Tiền xử lý dữ liệu ( KHÔNG CẦN CHẠY LẠI VÌ ĐÃ CÓ Ở CODE )

Trước khi chạy hệ thống, cần tạo embeddings từ ảnh:

```bash
python preprocess.py
```

* Sẽ tạo ra:

  * `image_embeddings.npy`: embeddings của tất cả ảnh
  * `image_paths.npy`: đường dẫn ảnh tương ứng

---

## 🚀 Chạy ứng dụng

```bash
python app.py
```

* Server Flask sẽ chạy tại `http://127.0.0.1:5000/`
* Frontend truy cập tại: `http://127.0.0.1:5000/`

---

## 🖥 Giao diện frontend

* Nhập mô tả người (có thể bằng tiếng Việt) vào ô tìm kiếm.
* Hệ thống tự động dịch sang tiếng Anh và tìm ảnh tương ứng.
* Kết quả hiển thị dưới dạng lưới ảnh, click vào ảnh để xem lớn.

---

## 🔍 Công nghệ sử dụng
* **Backend**: Python, Flask, CLIP (Contrastive Language-Image Pre-Training)
* **Frontend**: HTML, CSS, JavaScript
* **Dịch tiếng Việt → Anh**: Deep-translator (Google Translate)
* **Tìm kiếm ảnh**: Cosine Similarity trên Embeddings
* **Dataset**: [ICFG-PEDES](https://www.kaggle.com/datasets/meetrathi97/icfg-pedes)

---

## ⚡ Lưu ý

* Dataset không được lưu trên GitHub, cần tải riêng từ Kaggle.
* Nếu dataset quá lớn, việc tạo embeddings có thể mất thời gian.
* Chạy trên GPU sẽ nhanh hơn nhiều.
* Hệ thống hiện chưa triển khai caching hoặc batch inference tối ưu, thích hợp cho demo và thử nghiệm.

---
