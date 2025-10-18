import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# --- Cấu hình ---
DATA_DIR = "ICFG-PEDES"
IMAGE_DIR = os.path.join(DATA_DIR, "imgs")
CAPTIONS_FILE = os.path.join(DATA_DIR, "ICFG-PEDES.json")
OUTPUT_EMBEDDINGS_FILE = "image_embeddings.npy"
OUTPUT_PATHS_FILE = "image_paths.npy"

# Chọn thiết bị (GPU nếu có)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Tải mô hình CLIP đã được huấn luyện sẵn
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# Đọc file json để lấy danh sách ảnh và captions
# File json có cấu trúc tốt hơn để lấy cả id và path
df = pd.read_json(CAPTIONS_FILE)

# --- Xử lý ---
all_embeddings = []
all_image_paths = []

print("Starting image feature extraction...")

# Sử dụng tqdm để theo dõi tiến trình
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    img_path = os.path.join(DATA_DIR, "imgs", row['file_path'])

    # Kiểm tra xem ảnh có tồn tại không
    if not os.path.exists(img_path):
        continue

    try:
        # Mở và xử lý ảnh
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Lấy image features (embedding)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        # Chuyển về CPU, numpy và thêm vào danh sách
        all_embeddings.append(image_features.cpu().numpy())
        all_image_paths.append(row['file_path'])

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# Chuyển danh sách embeddings thành một ma trận numpy lớn
image_embeddings = np.vstack(all_embeddings)

# Lưu embeddings và đường dẫn ảnh
np.save(OUTPUT_EMBEDDINGS_FILE, image_embeddings)
np.save(OUTPUT_PATHS_FILE, np.array(all_image_paths))

print(f"\nPreprocessing complete!")
print(f"Saved {len(image_embeddings)} embeddings to {OUTPUT_EMBEDDINGS_FILE}")
print(f"Saved {len(all_image_paths)} paths to {OUTPUT_PATHS_FILE}")