# predict_rebuilt.py
import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# --- Bangun ulang arsitektur model yang SAMA persis ---
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def predict_image(image_path):
    print("Membangun ulang model...")
    model = create_model()
    
    # Muat BOBOT saja (bukan arsitektur)
    print("Memuat bobot model...")
    model.load_weights("model/banana_cnn_model.h5")
    
    # Proses gambar
    print(f"Membaca gambar: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((128, 128))
    image_array = np.array(image_resized) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    # Prediksi
    print("Melakukan prediksi...")
    pred = model.predict(image_batch, verbose=0)[0][0]
    confidence = max(pred, 1 - pred)

    if pred > 0.5:
        result = "PISANG BUSUK"
        color = "merah"
    else:
        result = "PISANG SEGAR"
        color = "hijau"

    print("\n" + "="*50)
    print(f"🎯 HASIL: {result}")
    print(f"✅ Confidence: {confidence:.2%}")
    print("="*50)

    # Tampilkan gambar
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f"{result} (Confidence: {confidence:.2%})")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Penggunaan: python predict_rebuilt.py <path_ke_gambar>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"❌ File tidak ditemukan: {image_path}")
        sys.exit(1)

    predict_image(image_path)