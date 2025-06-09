import os
import cv2

# Klasör yolları
base_dir = "."
ppm_dir = os.path.join(base_dir, "Train")
gt_path = os.path.join(base_dir, "gt.txt")
img_out_dir = os.path.join(base_dir, "images", "train")
label_out_dir = os.path.join(base_dir, "labels", "train")

# Klasörleri oluştur
os.makedirs(img_out_dir, exist_ok=True)
os.makedirs(label_out_dir, exist_ok=True)

# Etiketleri grupla
labels = {}
with open(gt_path, "r") as f:
    for line in f:
        parts = line.strip().split(";")
        ppm_name = parts[0]
        class_id = int(parts[5])
        x1, y1, x2, y2 = map(int, parts[1:5])
        jpg_name = ppm_name.replace(".ppm", ".jpg")

        if jpg_name not in labels:
            labels[jpg_name] = []
        labels[jpg_name].append((class_id, x1, y1, x2, y2))

# Dönüştür ve etiketle
for file in os.listdir(ppm_dir):
    if not file.endswith(".ppm"):
        continue

    img_path = os.path.join(ppm_dir, file)
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]
    jpg_name = file.replace(".ppm", ".jpg")
    out_img_path = os.path.join(img_out_dir, jpg_name)
    out_txt_path = os.path.join(label_out_dir, jpg_name.replace(".jpg", ".txt"))

    # Görseli kaydet
    cv2.imwrite(out_img_path, img)

    # Etiket varsa yaz, yoksa boş bırak
    with open(out_txt_path, "w") as f:
        for item in labels.get(jpg_name, []):
            class_id, x1, y1, x2, y2 = item
            x_center = (x1 + x2) / 2.0 / w
            y_center = (y1 + y2) / 2.0 / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
