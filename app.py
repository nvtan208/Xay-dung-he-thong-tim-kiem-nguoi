from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator  # ✅ Thêm phần dịch
import os

app = Flask(__name__, static_folder='ICFG-PEDES')
CORS(app)  # Cho phép request từ domain khác (frontend)

# --- Tải mô hình và dữ liệu đã tiền xử lý ---
print("Loading model and data...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# Tải embeddings và đường dẫn đã được lưu
image_embeddings = np.load("image_embeddings.npy")
image_paths = np.load("image_paths.npy")
translator = GoogleTranslator(source='vi', target='en')  # ✅ Khởi tạo translator
print("Model and data loaded successfully.")

# --- API Endpoint ---
@app.route('/search', methods=['POST'])
def search():
    data = request.json
    text_query = data.get('query', '')
    top_k = data.get('top_k', 10)  # Số lượng kết quả trả về

    if not text_query:
        return jsonify({"error": "Query is empty"}), 400

    print(f"Received query (VI): '{text_query}'")

    # 🔁 1. Dịch tiếng Việt sang tiếng Anh
    try:
        text_query_en = translator.translate(text_query)
        print(f"Translated query (EN): '{text_query_en}'")
    except Exception as e:
        print("Translation failed, fallback to original:", e)
        text_query_en = text_query

    # 2. Tạo text embedding cho câu query tiếng Anh
    inputs = processor(text=text_query_en, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_embedding = text_features.cpu().numpy()

    # 3. Tính độ tương đồng cosine
    similarities = cosine_similarity(text_embedding, image_embeddings).flatten()

    # 4. Lấy top K chỉ số có độ tương đồng cao nhất
    top_k_indices = np.argsort(similarities)[::-1][:top_k]

    # 5. Lấy ra các đường dẫn ảnh tương ứng và thêm prefix để Flask truy cập được
    results = []
    for p in image_paths[top_k_indices]:
        if not p.startswith("ICFG-PEDES/"):
            full_path = f"ICFG-PEDES/imgs/{p}"
        else:
            full_path = p
        results.append(full_path)

    print(f"Found {len(results)} results.")
    return jsonify({"results": results})

# Route để phục vụ file HTML chính
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Route để phục vụ các file ảnh từ thư mục ICFG-PEDES/imgs
@app.route('/ICFG-PEDES/<path:filename>')
def serve_images(filename):
    return send_from_directory('ICFG-PEDES', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
