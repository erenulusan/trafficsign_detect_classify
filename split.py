import os
import shutil
import random


base_dir = "."
image_train = os.path.join(base_dir, "images", "train")
label_train = os.path.join(base_dir, "labels", "train")
image_val = os.path.join(base_dir, "images", "val")
label_val = os.path.join(base_dir, "labels", "val")


os.makedirs(image_val, exist_ok=True)
os.makedirs(label_val, exist_ok=True)


all_images = [f for f in os.listdir(image_train) if f.endswith(".jpg")]
random.seed(42) 

val_split = int(0.2 * len(all_images))
val_images = random.sample(all_images, val_split) 

for img_name in val_images:
    label_name = img_name.replace(".jpg", ".txt")

    shutil.move(os.path.join(image_train, img_name), os.path.join(image_val, img_name))

    src_label_path = os.path.join(label_train, label_name)
    dst_label_path = os.path.join(label_val, label_name)

    if os.path.exists(src_label_path):
        shutil.move(src_label_path, dst_label_path)
    else:
        with open(dst_label_path, "w") as f:
            pass # boş .txt dosyası oluştur