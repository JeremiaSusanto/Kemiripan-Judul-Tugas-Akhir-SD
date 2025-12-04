# Aplikasi Deteksi Kemiripan Judul Tugas Akhir

Aplikasi berbasis Streamlit untuk membandingkan metode SVM dan Random Forest dalam deteksi kemiripan judul menggunakan Machine Learning.

## 🚀 Quick Start

### 1. Instalasi Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Training Model

**PENTING:** Sebelum menjalankan aplikasi, Anda harus melatih model terlebih dahulu!

#### Opsi A: Lightweight Model (Recommended untuk GitHub)

```powershell
python train_model_lightweight.py
```

Model akan berukuran kecil (<25MB) dan sudah ter-commit di repository. **Cocok untuk deployment GitHub/Cloud.**

#### Opsi B: Full Model (Akurasi Maksimal)

```powershell
python train_model.py
```

Model berukuran lebih besar (~50-100MB), tidak disertakan di repo. **Cocok untuk development/production lokal.**

Script akan:

- Memuat dataset dari `data/dataset_TA.csv`
- Melakukan preprocessing dengan Sastrawi (stemming Bahasa Indonesia)
- Membuat pasangan berlabel secara otomatis berdasarkan cosine similarity
- Ekstraksi 9 fitur similarity untuk setiap pasangan
- Training model SVM dan Random Forest dengan GridSearchCV
- Menyimpan model dan artefak ke folder `model_outputs/`

Output training:

- `model_outputs/tfidf.joblib` - TF-IDF vectorizer
- `model_outputs/scaler.joblib` - Feature scaler
- `model_outputs/best_svm.joblib` - Trained SVM model
- `model_outputs/best_rf.joblib` - Trained Random Forest model
- `model_outputs/titles_preprocessed.csv` - Preprocessed corpus

### 3. Menjalankan Aplikasi Streamlit

```powershell
streamlit run main.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 📁 Struktur Folder

```
psdrb/
├── data/
│   └── dataset_TA.csv          # Dataset judul tugas akhir (123 judul)
├── model_outputs/              # Folder model hasil training (dibuat otomatis)
│   ├── tfidf.joblib
│   ├── scaler.joblib
│   ├── best_svm.joblib
│   ├── best_rf.joblib
│   └── titles_preprocessed.csv
├── preprocessing.py            # Modul preprocessing dengan Sastrawi
├── train_model.py              # Script training model
├── main.py                     # Aplikasi Streamlit
├── requirements.txt            # Dependencies Python
└── README.md                   # Dokumentasi
```

## 🧠 Fitur Aplikasi

- ✅ **Preprocessing Teks**: Sastrawi stemmer untuk Bahasa Indonesia
- ✅ **Ekstraksi 9 Fitur**: Cosine similarity, L1/L2 distance, Jaccard, Levenshtein, statistik TF-IDF
- ✅ **Dual Model**: SVM dan Random Forest dengan hyperparameter tuning
- ✅ **Visualisasi**: Plotly bar charts untuk perbandingan probabilitas
- ✅ **Export**: Download hasil ke CSV

## 📊 Dataset

Dataset `dataset_TA.csv` berisi 123 judul tugas akhir dalam Bahasa Indonesia dengan topik beragam:

- Machine Learning & Deep Learning
- Sentiment Analysis
- Forecasting (SARIMA, LSTM, ARFIMA)
- Object Detection (YOLO, CNN)
- Classification & Prediction

## 👥 Tim Penyusun

- Eksanty F Sugma (122450001)
- Dhea Amelia Putri (122450004)
- Jeremia Susanto (122450022)

**Tugas Besar Projek Sains Data**

---

## 🌐 Deployment ke GitHub

### Push ke GitHub Repository

```powershell
# 1. Initialize git (jika belum)
git init

# 2. Add semua file
git add .

# 3. Commit dengan pesan
git commit -m "Initial commit: Deteksi Kemiripan Judul TA"

# 4. Add remote repository
git branch -M main
git remote add origin https://github.com/USERNAME/REPO_NAME.git

# 5. Push ke GitHub
git push -u origin main
```

### Clone dan Setup (User Baru)

**Untuk Lightweight Model (sudah ada di repo):**

```powershell
git clone https://github.com/USERNAME/REPO_NAME.git
cd REPO_NAME
pip install -r requirements.txt
streamlit run main.py  # Model sudah siap!
```

**Untuk Full Model (training ulang):**

```powershell
git clone https://github.com/USERNAME/REPO_NAME.git
cd REPO_NAME
pip install -r requirements.txt
python train_model.py  # Training dengan akurasi maksimal
streamlit run main.py
```

### ☁️ Deployment ke Streamlit Cloud

1. Push repository ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Login dengan GitHub
4. Pilih repository Anda
5. Klik "Deploy"!

**Catatan:** Lightweight model sudah ter-commit, jadi aplikasi langsung bisa jalan di Streamlit Cloud tanpa perlu training ulang.

---

## 🔧 Troubleshooting

**Q: Model tidak ditemukan saat run aplikasi?**
A: Jalankan `python train_model_lightweight.py` untuk generate model.

**Q: Aplikasi error di Streamlit Cloud?**
A: Pastikan folder `model_outputs/` ada di repository dan berisi semua file model.

**Q: Ingin akurasi lebih tinggi?**
A: Gunakan `python train_model.py` untuk full model dengan hyperparameter lebih kompleks.
