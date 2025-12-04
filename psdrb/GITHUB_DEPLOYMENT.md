# 🚀 GitHub Deployment Guide

## ✅ Model Siap Deploy!

Projek ini sudah include **lightweight model** (38 KB) yang bisa langsung digunakan setelah clone dari GitHub.

---

## 📦 Yang Sudah Disiapkan

### 1. Lightweight Model (Included in Repo)

- **Location:** `model_outputs_lightweight/`
- **Total Size:** 38 KB
- **Files:**
  - `tfidf.joblib` (5 KB)
  - `scaler.joblib` (0.6 KB)
  - `best_svm.joblib` (1.1 KB)
  - `best_rf.joblib` (3.4 KB)
  - `titles_preprocessed.csv` (27.5 KB)
- **Performance:** 100% accuracy on test set
- **Advantage:** Linear SVM (lebih kecil dari RBF kernel)

### 2. Full Model (NOT Included - Too Large)

- **Location:** `model_outputs/` (di `.gitignore`)
- **Size:** Lebih besar, tidak cocok untuk GitHub
- **Training:** Jalankan `python train_model.py` secara lokal

---

## 🎯 Deployment Steps

### A. Push ke GitHub (Owner)

```powershell
# 1. Pastikan di folder projek
cd c:\Users\jerem\OneDrive\Desktop\psdrb

# 2. Init git (jika belum)
git init

# 3. Add remote
git remote add origin https://github.com/MiaKun/Latihan-Fork.git

# 4. Add semua file (lightweight model akan ikut ter-commit)
git add .

# 5. Commit
git commit -m "Initial commit: Aplikasi Deteksi Kemiripan Judul TA dengan lightweight model"

# 6. Push
git push -u origin main
```

### B. Clone & Run (User Baru)

```powershell
# 1. Clone repository
git clone https://github.com/MiaKun/Latihan-Fork.git
cd Latihan-Fork

# 2. Install dependencies
pip install -r requirements.txt

# 3. Langsung jalankan! (model sudah included)
streamlit run main.py
```

**✅ TIDAK perlu training model!** Lightweight model sudah disertakan di repository.

---

## 🔄 Opsi Training Model (Opsional)

### Untuk Akurasi Lebih Baik

Jika ingin model dengan performa maksimal:

```powershell
# Training full model (hasil tidak di-commit)
python train_model.py

# Aplikasi akan otomatis gunakan full model jika tersedia
streamlit run main.py
```

### Re-training Lightweight Model

```powershell
# Update lightweight model dengan dataset baru
python train_model_lightweight.py

# Commit hasil training
git add model_outputs_lightweight/
git commit -m "Update lightweight model"
git push
```

---

## 🌐 Deploy ke Cloud

### Streamlit Cloud (Recommended - Gratis)

1. Push repo ke GitHub ✅ (sudah siap)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Login dengan GitHub
4. Pilih repository: `MiaKun/Latihan-Fork`
5. Main file: `main.py`
6. **Deploy!** (model lightweight akan otomatis terload)

**Keuntungan menggunakan lightweight model:**

- ✅ Tidak perlu download model dari cloud storage
- ✅ Deployment lebih cepat (model sudah ada di repo)
- ✅ Tidak perlu setup tambahan
- ✅ Gratis dan mudah

### Alternatif Cloud Platforms

| Platform                | Lightweight Model | Full Model             | Catatan            |
| ----------------------- | ----------------- | ---------------------- | ------------------ |
| **Streamlit Cloud**     | ✅ Langsung jalan | ❌ Butuh cloud storage | Recommended        |
| **Hugging Face Spaces** | ✅ Langsung jalan | ✅ Support Git LFS     | Bagus untuk ML     |
| **Railway.app**         | ✅ Langsung jalan | ✅ Bisa training       | Perlu config       |
| **Render**              | ✅ Langsung jalan | ⚠️ Limited resources   | Free tier terbatas |
| **Heroku**              | ✅ Langsung jalan | ❌ Dyno berbayar       | Tidak gratis lagi  |

---

## 📊 Perbandingan Model

| Feature             | Lightweight            | Full                 |
| ------------------- | ---------------------- | -------------------- |
| **Size**            | 38 KB                  | ~5-10 MB             |
| **GitHub Friendly** | ✅ Yes                 | ❌ No                |
| **Training Time**   | ~10 detik              | ~30 detik            |
| **TF-IDF Features** | 432 (max_features=500) | 1795 (unlimited)     |
| **SVM Kernel**      | Linear                 | RBF                  |
| **RF Trees**        | 50                     | 100-300              |
| **Test Accuracy**   | 100%                   | 100%                 |
| **ROC AUC**         | 1.0                    | 1.0                  |
| **Deployment**      | Langsung included      | Perlu training lokal |

**Kesimpulan:** Untuk dataset kecil (123 judul), lightweight model sudah sempurna!

---

## 🔍 Troubleshooting

### Model tidak ditemukan

```powershell
# Check folder
ls model_outputs_lightweight/

# Jika kosong, training ulang
python train_model_lightweight.py
```

### Import error: Sastrawi

```powershell
pip install Sastrawi python-Levenshtein
```

### Streamlit error di cloud

Pastikan `requirements.txt` lengkap:

```
streamlit
scikit-learn
pandas
numpy
plotly
Sastrawi
python-Levenshtein
joblib
scipy
```

---

## 📝 Checklist Deployment

- [x] Lightweight model sudah ditraining (38 KB)
- [x] `.gitignore` updated (exclude `model_outputs/`, include `model_outputs_lightweight/`)
- [x] [`main.py`](main.py) support fallback ke lightweight model
- [x] [`README.md`](README.md) updated dengan instruksi
- [x] [`requirements.txt`](requirements.txt) lengkap
- [ ] Push ke GitHub
- [ ] Test clone di komputer lain
- [ ] Deploy ke Streamlit Cloud (opsional)

---

## 🎉 Ready to Deploy!

Repository sudah **100% siap** untuk di-push ke GitHub dan di-deploy ke cloud platform manapun!

**Next Steps:**

1. `git add .`
2. `git commit -m "Ready for deployment"`
3. `git push origin main`
4. Deploy ke Streamlit Cloud
5. Share link aplikasi! 🚀
