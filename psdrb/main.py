"""
Aplikasi Streamlit untuk Deteksi Kemiripan Judul Tugas Akhir
Menggunakan model ML (SVM dan Random Forest) yang sudah di-training.
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from Levenshtein import ratio as levenshtein_ratio
from preprocessing import preprocess_text


# === Cache Loading Model & Data ===
@st.cache_resource
def load_models():
    """
    Load trained models dan artefak.
    Prioritas: model_outputs/ (full), fallback ke model_outputs_lightweight/ (GitHub-friendly)
    """
    # Possible paths (local vs Streamlit Cloud)
    possible_paths = [
        "model_outputs",
        "model_outputs_lightweight",
        "psdrb/model_outputs",
        "psdrb/model_outputs_lightweight"
    ]
    
    model_dir = None
    model_type = None
    
    # Cek semua kemungkinan path
    for path in possible_paths:
        if os.path.exists(f"{path}/tfidf.joblib"):
            model_dir = path
            if "lightweight" in path:
                model_type = "Lightweight Model (GitHub)"
            else:
                model_type = "Full Model"
            break
    
    if model_dir is None:
        # Debug: tampilkan current directory dan file yang ada
        import glob
        current_dir = os.getcwd()
        files = glob.glob("**/*.joblib", recursive=True)
        
        st.error(f"""
        ⚠️ Model belum tersedia!
        
        **Debug Info:**
        - Current directory: `{current_dir}`
        - Found .joblib files: {files[:5] if files else 'None'}
        
        **Pilihan:**
        1. Training full model (akurasi terbaik):
           ```
           python train_model.py
           ```
        
        2. Training lightweight model (untuk GitHub, lebih cepat):
           ```
           python train_model_lightweight.py
           ```
        """)
        st.stop()
    
    # Load model dari folder yang dipilih
    tfidf = joblib.load(f"{model_dir}/tfidf.joblib")
    scaler = joblib.load(f"{model_dir}/scaler.joblib")
    svm_model = joblib.load(f"{model_dir}/best_svm.joblib")
    rf_model = joblib.load(f"{model_dir}/best_rf.joblib")
    df_corpus = pd.read_csv(f"{model_dir}/titles_preprocessed.csv")
    
    # Info model yang digunakan
    st.sidebar.info(f"🤖 Menggunakan: **{model_type}**")
    
    return tfidf, scaler, svm_model, rf_model, df_corpus


# === Ekstraksi Fitur Pasangan ===
def jaccard_sim(a_text, b_text):
    """Jaccard similarity pada token set"""
    a = set(a_text.split())
    b = set(b_text.split())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def extract_pair_features(query_vec, ref_vec, query_proc, ref_proc, query_raw, ref_raw):
    """
    Ekstraksi 9 fitur untuk pasangan (query, reference):
    1. Cosine similarity
    2. L1 distance
    3. L2 distance
    4. Jaccard token overlap
    5. Levenshtein ratio
    6-9. Statistik perbedaan TF-IDF
    """
    vi = query_vec.toarray().flatten()
    vj = ref_vec.toarray().flatten()
    
    # 1. Cosine
    cos = float(cosine_similarity([vi], [vj])[0, 0])
    
    # 2. L1 distance
    l1 = float(np.sum(np.abs(vi - vj)))
    
    # 3. L2 distance
    l2 = float(np.sqrt(np.sum((vi - vj) ** 2)))
    
    # 4. Jaccard
    jacc = jaccard_sim(query_proc, ref_proc)
    
    # 5. Levenshtein
    lev = levenshtein_ratio(query_raw, ref_raw)
    
    # 6-9. Diff stats
    diff = np.abs(vi - vj)
    diff_stats = [diff.mean(), diff.std(), diff.max(), diff.min()]
    
    return [cos, l1, l2, jacc, lev] + diff_stats


# === Compute Similarities with ML Models ===
def compute_similarities(query: str, tfidf, scaler, svm_model, rf_model, df_corpus) -> pd.DataFrame:
    """
    Menghitung kemiripan query dengan setiap judul di korpus menggunakan model ML.
    Returns: DataFrame dengan kolom judul_ref, svm_probability, rf_probability, svm_label, rf_label
    """
    # Preprocessing query
    query_proc = preprocess_text(query)
    if not query_proc.strip():
        st.warning("⚠️ Query tidak valid setelah preprocessing. Coba gunakan judul yang lebih deskriptif.")
        return pd.DataFrame()
    
    # TF-IDF transform
    query_tfidf = tfidf.transform([query_proc])
    
    # Ekstraksi fitur untuk setiap pasangan
    results = []
    for idx, row in df_corpus.iterrows():
        ref_raw = row['Judul']
        ref_proc = row['judul_proc']
        ref_tfidf = tfidf.transform([ref_proc])
        
        # Ekstrak 9 fitur
        feats = extract_pair_features(
            query_tfidf, ref_tfidf, 
            query_proc, ref_proc,
            query, ref_raw
        )
        
        # Scale features
        feats_scaled = scaler.transform([feats])
        
        # Predict with both models
        svm_proba = svm_model.predict_proba(feats_scaled)[0, 1]  # probability class 1 (mirip)
        rf_proba = rf_model.predict_proba(feats_scaled)[0, 1]
        
        svm_label = "MIRIP" if svm_proba >= 0.5 else "TIDAK MIRIP"
        rf_label = "MIRIP" if rf_proba >= 0.5 else "TIDAK MIRIP"
        
        results.append({
            "judul_ref": ref_raw,
            "svm_probability": round(svm_proba, 3),
            "rf_probability": round(rf_proba, 3),
            "svm_label": svm_label,
            "rf_label": rf_label
        })
    
    df = pd.DataFrame(results)
    return df


def main():
    st.set_page_config(page_title="Deteksi Kemiripan Judul Tugas Akhir", layout="wide")

    # Custom CSS
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #fbfdff 0%, #ffffff 100%); padding-top:18px; }
        .header-banner { background: linear-gradient(90deg,#ffd89b 0%,#f6a6c1 50%,#ff7eb3 100%); padding:20px; border-radius:12px; color:#222; position:relative; z-index:9999; box-shadow: 0 6px 18px rgba(0,0,0,0.08); }
        .card { background: rgba(255,255,255,0.95); padding:12px; border-radius:10px; box-shadow: 0 4px 14px rgba(0,0,0,0.06); margin-bottom:10px }
        .small-note { color:#444; font-size:0.95rem }
        header[data-testid="stAppHeader"] { position:relative; z-index:10000 }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar menu
    menu = st.sidebar.radio(
        "Menu",
        ["🏠 Home", "ℹ️ About", "👥 Team"],
        index=0,
    )

    if menu == "🏠 Home":
        # Load models
        tfidf, scaler, svm_model, rf_model, df_corpus = load_models()
        
        # Header
        st.markdown(
            """
            <div class="header-banner">
            <h1 style="margin:0; font-size:1.9rem; font-weight:700">
            Perbandingan Metode SVM dan Random Forest dalam Deteksi Kemiripan Judul Tugas Akhir Menggunakan Multi-Fitur Similarity
            </h1>
            <p style="margin-top:6px; font-size:0.95rem">
            Masukkan judul skripsi untuk mengecek kemiripan dengan dataset yang tersedia.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # Input section
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📝 Input Judul")
        query = st.text_area(
            "Masukkan judul tugas akhir yang ingin dicek:",
            placeholder="Contoh: Implementasi Algoritma Machine Learning untuk Prediksi Harga Saham Menggunakan LSTM",
            height=100,
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"<p class='small-note'>Dataset: {len(df_corpus)} judul dari dataset_TA.csv</p>", unsafe_allow_html=True)
        with col2:
            run = st.button("🔍 Cek Kemiripan", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Hasil
        if run:
            if not query.strip():
                st.warning("⚠️ Mohon masukkan judul terlebih dahulu!")
            else:
                with st.spinner("🔄 Memproses..."):
                    df_results = compute_similarities(query, tfidf, scaler, svm_model, rf_model, df_corpus)
                
                if df_results.empty:
                    st.stop()
                
                st.success("✅ Analisis selesai!")
                
                # Sort by SVM probability (descending)
                df_results = df_results.sort_values("svm_probability", ascending=False).reset_index(drop=True)
                df_results.index = df_results.index + 1  # Mulai dari 1, bukan 0
                
                # Results table
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("📊 Hasil Prediksi")
                
                # Display table with styling
                st.dataframe(
                    df_results.style.background_gradient(
                        subset=["svm_probability", "rf_probability"],
                        cmap="YlOrRd",
                        vmin=0,
                        vmax=1,
                    ),
                    use_container_width=True,
                    height=400,
                )
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Top similar titles
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("🏆 Top 5 Judul Paling Mirip")
                
                col_svm, col_rf = st.columns(2)
                
                with col_svm:
                    st.markdown("**🔹 SVM**")
                    top_svm = df_results.nlargest(5, "svm_probability")[["judul_ref", "svm_probability", "svm_label"]]
                    st.dataframe(top_svm, use_container_width=True, hide_index=True)
                
                with col_rf:
                    st.markdown("**🔸 Random Forest**")
                    top_rf = df_results.nlargest(5, "rf_probability")[["judul_ref", "rf_probability", "rf_label"]]
                    st.dataframe(top_rf, use_container_width=True, hide_index=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Visualization
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("📈 Visualisasi Perbandingan")
                
                # Get top 10 for visualization
                top10 = df_results.head(10).copy()
                top10['Judul (singkat)'] = top10['judul_ref'].str[:50] + "..."
                
                fig_data = []
                for _, row in top10.iterrows():
                    fig_data.append({'Judul': row['Judul (singkat)'], 'Model': 'SVM', 'Probabilitas': row['svm_probability']})
                    fig_data.append({'Judul': row['Judul (singkat)'], 'Model': 'Random Forest', 'Probabilitas': row['rf_probability']})
                
                fig_df = pd.DataFrame(fig_data)
                fig = px.bar(
                    fig_df,
                    y='Judul',
                    x='Probabilitas',
                    color='Model',
                    orientation='h',
                    barmode='group',
                    title="Top 10 Probabilitas Kemiripan (SVM vs Random Forest)",
                    color_discrete_map={'SVM': '#ff7eb3', 'Random Forest': '#ffd89b'}
                )
                fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Download button
                csv = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Hasil (CSV)",
                    data=csv,
                    file_name="hasil_kemiripan.csv",
                    mime="text/csv",
                )

    elif menu == "ℹ️ About":
        st.title("ℹ️ Tentang Aplikasi")
        st.markdown(
            """
            ### Perbandingan Metode SVM dan Random Forest dalam Deteksi Kemiripan Judul Tugas Akhir
            
            Aplikasi ini menggunakan **Machine Learning** untuk mendeteksi kemiripan antara judul tugas akhir.
            
            **Fitur Utama:**
            - ✅ Preprocessing teks dengan **Sastrawi** (stemming Bahasa Indonesia)
            - ✅ Ekstraksi 9 fitur similarity (cosine, L1, L2, Jaccard, Levenshtein, statistik TF-IDF)
            - ✅ Model **SVM** dan **Random Forest** yang sudah di-training dengan GridSearchCV
            - ✅ Visualisasi interaktif perbandingan kedua metode
            - ✅ Dataset: 123 judul tugas akhir dari dataset_TA.csv
            
            **Teknologi:**
            - Python, Streamlit, scikit-learn, Sastrawi, Plotly
            
            ---
            
            **🎓 Projek ini dibuat guna menyelesaikan Tugas Besar Projek Sains Data**
            
            Aplikasi ini membantu mendeteksi kemiripan judul untuk mencegah duplikasi topik penelitian 
            dan memberikan wawasan tentang performa model SVM vs Random Forest dalam klasifikasi teks.
            """
        )

    elif menu == "👥 Team":
        st.title("👥 Tim Penyusun")
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                """
                <div class='card' style='text-align:center'>
                <img src="path_to_eksanty_f_sugma_image.jpg" alt="Eksanty F Sugma" style="width:100px;height:100px;border-radius:50%;">
                <h3>👤 Eksanty F Sugma</h3>
                <p><strong>NIM:</strong> 122450001</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with col2:
            st.markdown(
                """
                <div class='card' style='text-align:center'>
                <img src="path_to_dhea_amelia_putri_image.jpg" alt="Dhea Amelia Putri" style="width:100px;height:100px;border-radius:50%;">
                <h3>👤 Dhea Amelia Putri</h3>
                <p><strong>NIM:</strong> 122450004</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with col3:
            st.markdown(
                """
                <div class='card' style='text-align:center'>
                <img src="path_to_jeremia_susanto_image.jpg" alt="Jeremia Susanto" style="width:100px;height:100px;border-radius:50%;">
                <h3>👤 Jeremia Susanto</h3>
                <p><strong>NIM:</strong> 122450022</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
