# 🌾 AgroVA: Solusi Cerdas untuk Ketahanan Pangan di Indonesia

## 📝 Ringkasan Eksekutif

**AgroVA** adalah sistem cerdas berbasis AI yang dirancang untuk membantu petani Indonesia meningkatkan ketahanan pangan. Dengan tiga fitur utama:
- **🌦️ Prediksi Kegagalan Panen**: Mengantisipasi risiko berdasarkan data cuaca.
- **🌱 Klasifikasi Penyakit Tanaman**: Identifikasi otomatis penyakit pada padi, gandum, singkong, dan kentang.
- **🤖 Chatbot Interaktif**: Rekomendasi penanganan penyakit secara cepat dan akurat.
-----

## 🧩 Modul yang Tersedia

- 🤖 [chatbot/](./chatbot/): Chatbot AI untuk konsultasi pertanian
- 🌿 [klasifikasi-gambar/](./klasifikasi-gambar/): beberapa source code yang digunakan untuk membuat model deteksi penyakit tanaman dari gambar
- ☁️ [prediksi-cuaca/](./prediksi-cuaca/): Prediksi cuaca lokal untuk pertanian
- 📱 [AgroVA-app/](./AgroVA-app/): Aplikasi utama sebagai penghubung semua modul


## 🚀 Cara Menjalankan Proyek

1. Clone repositori:
   ```bash
   git clone https://github.com/Sintasitinuriah/AgroVa-LAI25-SM060.git
   cd AgroVa-LAI25-SM060

2. Masuk Ke setiap folder untuk menjalankan masing-masing modul.

3. Install depedensi umum:
    ```bash
    pip install -r requirements.txt

---- 

# Dataset yang digunakan:
|---------------|------------------|
|Keterangan|Source|
|1. Prediksi Cuaca|[Weather Prediction](https://www.kaggle.com/datasets/thedevastator/weather-prediction)|
|2. Klasifikasi penyakit tanaman|[Gandum](https://www.kaggle.com/datasets/kushagra3204/wheat-plant-diseases)|
|                 |[Padi](https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image)|
|                 |[Singkong](https://www.kaggle.com/datasets/nirmalsankalana/cassava-leaf-disease-classification)|
|                 |[Kentang](https://www.kaggle.com/datasets/rizwan123456789/potato-disease-leaf-datasetpld)|
|3. Chatbot       | Dataset yang digunakan hasil observasi pada buku buku penyakit tanaman padi, kentang, gandum dan singkong|
|-----------------|-----------|

Dataset diatas dijadikan refresnsi utama dalam proses pembuatan AgroVA.

----
# Teknologi yang digunakan
- python
- flask
- tensorflow
- scikit-learn
- gdwon
- numpy
- Heroku
- Pillow
- gunicorn

# 🚀 Cara Menjalankan Aplikasi AgroVA-app
1. Masuk ke folder path AgroVA-app
2. jalankan file `app.py` untuk melihat aplikasi yang digunakan. ketikan `python app.py`
4. Masuk ke `localhost:5000` atau `127.0.0.1:5000`
3. Login menggunakan akun admin:
    ```bash
        username: admin
        password: admin123

