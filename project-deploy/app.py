import os
import tkinter as tk
from tkinter import filedialog, messagebox, Label
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

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

# Muat model sekali saat aplikasi dibuka (lebih efisien)
print("Memuat model...")
model = create_model()
model.load_weights("model/banana_cnn_model.h5")
print("Model siap digunakan.")

def predict_image(image_path):
    # Proses gambar
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((128, 128))
    image_array = np.array(image_resized) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    # Prediksi
    pred = model.predict(image_batch, verbose=0)[0][0]
    confidence = max(pred, 1 - pred)

    if pred > 0.5:
        result = "PISANG BUSUK"
        color = "red"
    else:
        result = "PISANG SEGAR"
        color = "green"

    return result, confidence, color

def upload_image():
    file_path = filedialog.askopenfilename(
        title="Pilih Gambar Pisang",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return

    if not os.path.exists(file_path):
        messagebox.showerror("Error", "File tidak ditemukan!")
        return

    try:
        # Prediksi
        result, conf, color = predict_image(file_path)

        # Tampilkan gambar (resize agar muat di GUI)
        img = Image.open(file_path)
        img.thumbnail((300, 300))  # Resize untuk tampilan GUI
        img_tk = ImageTk.PhotoImage(img)

        # Update tampilan
        panel_image.config(image=img_tk)
        panel_image.image = img_tk  # Simpan referensi agar tidak di-GC

        label_result.config(text=f"Hasil: {result}", fg=color)
        label_conf.config(text=f"Confidence: {conf:.2%}")

    except Exception as e:
        messagebox.showerror("Error Prediksi", f"Terjadi kesalahan:\n{str(e)}")

# --- GUI dengan tkinter ---
root = tk.Tk()
root.title("Klasifikasi Pisang - Segar vs Busuk")
root.geometry("500x600")
root.resizable(False, False)

# Judul
title = tk.Label(root, text="📸 Deteksi Kondisi Pisang", font=("Arial", 16, "bold"))
title.pack(pady=10)

# Tombol Upload
btn_upload = tk.Button(root, text="📁 Upload Gambar", command=upload_image, font=("Arial", 12), bg="#4CAF50", fg="white", padx=20, pady=5)
btn_upload.pack(pady=10)

# Panel untuk menampilkan gambar
panel_image = Label(root, bg="white", relief="solid", width=300, height=300)
panel_image.pack(pady=10)

# Label hasil
label_result = tk.Label(root, text="Hasil: -", font=("Arial", 14, "bold"))
label_result.pack(pady=5)

label_conf = tk.Label(root, text="Confidence: -", font=("Arial", 12))
label_conf.pack(pady=5)

# Catatan
note = tk.Label(root, text="Model: CNN (Banana Fresh vs Rotten)", font=("Arial", 9), fg="gray")
note.pack(side="bottom", pady=10)

# Jalankan GUI
if __name__ == "__main__":
    root.mainloop()