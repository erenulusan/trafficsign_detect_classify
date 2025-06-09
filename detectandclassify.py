import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

gtsrb_labels = ["20 km/s", "30 km/s", "50 km/s", "60 km/s", "70 km/s", "80 km/s", "80 km/s sinirlama sonu", "100 km/s", "120 km/s", "Sollama yasak",
                "Kamyon sollama yasak", "Siradaki kavsakta oncelik", "Ana yol", "Yol ver", "Dur", "Her iki yonde trafik yasak", "Kamyon giremez", "Girilmez", "Dikkat",
                "Sola viraj", "Saga viraj", "S viraj", "Duzensiz yol", "Kaygan yol", "Yol daralir", "Yol calismasi", "Trafik isigi", "Yaya gecidi",
                "Okul gecidi", "Bisiklet gecidi", "Kar", "Hayvan gecidi", "Kisitlama sonu", "Saga don", "Sola don", "Duz git", "Saga veya duz", "Sola veya duz",
                "Sagdan git", "Soldan git", "Gobekli kavsak", "Sollama yasagi sonu", "Kamyon sollama yasagi sonu"]

                

    
detector = YOLO("best (1).pt")
classifier = load_model("gtsrb_finetuned.keras", compile=False)

img = cv2.imread("00285.jpg")
img_copy = img.copy()


# YOLO ile tespit
results = detector(img)[0]

# Tüm kutular için sınıflandırma
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    crop_resized = cv2.resize(crop, (224, 224))
    crop_input = preprocess_input(crop_resized.astype(np.float32))
    crop_input = np.expand_dims(crop_input, 0)

    pred = classifier.predict(crop_input, verbose=0)[0]
    class_id = int(np.argmax(pred))
    confidence = float(np.max(pred))
    label = gtsrb_labels[class_id] if class_id < len(gtsrb_labels) else f"Class {class_id}"

    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_copy, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# Her iki resmi de 640x640 boyutuna getir
img_resized = cv2.resize(img, (640, 640))
img_copy_resized = cv2.resize(img_copy, (640, 640))

# Ayrı ayrı pencerelerde göster
cv2.imshow("Orijinal Foto", img_resized)
cv2.imshow("Tespit ve siniflandirma", img_copy_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

