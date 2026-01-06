import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ====================================================
# Page Config
# ====================================================
st.set_page_config(
    page_title="Prediksi Stunting Balita",
    layout="wide"
)

st.title("📊 Prediksi Persentase Balita Stunting")
st.caption("Metode Random Forest Regression")

# ====================================================
# Upload Dataset
# ====================================================
uploaded_file = st.file_uploader(
    "📂 Upload Dataset CSV Stunting",
    type=["csv"]
)

if uploaded_file is not None:

    # ====================================================
    # Load & Preprocessing
    # ====================================================
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]

    df['persentase_balita_stunting'] = pd.to_numeric(
        df['persentase_balita_stunting'], errors='coerce'
    )

    df = df.dropna(subset=[
        'persentase_balita_stunting',
        'nama_kabupaten_kota',
        'tahun'
    ])

    df['tahun'] = df['tahun'].astype(int)

    st.subheader("📄 Preview Data")
    st.dataframe(df.head())

    # ====================================================
    # Distribusi Data
    # ====================================================
    st.subheader("📈 Distribusi Persentase Stunting")

    fig, ax = plt.subplots()
    ax.hist(df['persentase_balita_stunting'], bins=30)
    ax.set_xlabel("Persentase Stunting (%)")
    ax.set_ylabel("Frekuensi")
    st.pyplot(fig)

    # ====================================================
    # Feature Engineering (SAMA DENGAN KODE AWAL)
    # ====================================================
    rows = []

    for (_, _, _), g in df.groupby(
        ['nama_provinsi', 'kode_kabupaten_kota', 'nama_kabupaten_kota']
    ):
        g = g.sort_values('tahun')
        years = g['tahun'].values
        vals = g['persentase_balita_stunting'].values

        for i in range(1, len(vals)):
            v1 = vals[i-1]
            v2 = vals[i-2] if i-2 >= 0 else np.nan
            v3 = vals[i-3] if i-3 >= 0 else np.nan

            mean_prev = np.nanmean([v1, v2, v3])

            if i >= 2:
                xi = years[max(0, i-3):i]
                yi = vals[max(0, i-3):i]
                slope = np.polyfit(xi, yi, 1)[0]
            else:
                slope = 0

            rows.append([
                v1, v2, v3,
                mean_prev, slope,
                vals[i]
            ])

    data_rf = pd.DataFrame(rows, columns=[
        'lag1', 'lag2', 'lag3',
        'mean_prev', 'slope_prev',
        'target'
    ])

    features = ['lag1', 'lag2', 'lag3', 'mean_prev', 'slope_prev']
    X = data_rf[features].fillna(data_rf[features].mean())
    y = data_rf['target']

    # ====================================================
    # Sidebar Parameter
    # ====================================================
    st.sidebar.header("⚙️ Parameter Model")

    n_estimators = st.sidebar.slider(
        "Jumlah Pohon (n_estimators)",
        min_value=50,
        max_value=500,
        value=200,
        step=50
    )

    # ====================================================
    # Training Model
    # ====================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    pred_test = rf.predict(X_test)

    # ====================================================
    # Evaluasi Model
    # ====================================================
    st.subheader("📊 Evaluasi Model")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("MAE", round(mean_absolute_error(y_test, pred_test), 2))
    c2.metric("RMSE", round(sqrt(mean_squared_error(y_test, pred_test)), 2))
    c3.metric("R²", round(r2_score(y_test, pred_test), 3))
    c4.metric(
        "MAPE (%)",
        round(np.mean(np.abs((y_test - pred_test) / y_test)) * 100, 2)
    )

    # ====================================================
    # Feature Importance
    # ====================================================
    st.subheader("⭐ Feature Importance")

    imp = pd.DataFrame({
        'Fitur': features,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots()
    ax.barh(imp['Fitur'], imp['Importance'])
    ax.set_xlabel("Tingkat Kepentingan")
    ax.invert_yaxis()
    st.pyplot(fig)

    # ====================================================
    # Input Manual (DIPERJELAS)
    # ====================================================
    st.subheader("🧮 Prediksi Stunting Tahun Berikutnya")

    st.markdown("""
    **Keterangan Input:**
    - **Lag-1** : Persentase stunting **1 tahun terakhir**
    - **Lag-2** : Persentase stunting **2 tahun sebelumnya**
    - **Lag-3** : Persentase stunting **3 tahun sebelumnya**
    - **Slope / Tren** : Arah perubahan stunting per tahun  
      (positif = meningkat, negatif = menurun)
    """)

    col1, col2, col3 = st.columns(3)

    lag1 = col1.number_input(
        "Lag-1 (1 Tahun Terakhir, %)", 0.0, 100.0, 15.0
    )
    lag2 = col2.number_input(
        "Lag-2 (2 Tahun Terakhir, %)", 0.0, 100.0, 16.0
    )
    lag3 = col3.number_input(
        "Lag-3 (3 Tahun Terakhir, %)", 0.0, 100.0, 17.0
    )

    mean_prev = np.mean([lag1, lag2, lag3])

    slope_prev = st.number_input(
        "Slope / Tren Perubahan (%)",
        -10.0, 10.0, 0.0,
        help="Contoh: -1 berarti rata-rata turun 1% per tahun"
    )

    if st.button("🔮 Prediksi"):
        feat = np.array([[lag1, lag2, lag3, mean_prev, slope_prev]])
        pred_val = rf.predict(feat)[0]

        if pred_val < 10:
            kategori = "Rendah"
        elif pred_val <= 20:
            kategori = "Sedang"
        else:
            kategori = "Tinggi"

        st.success(f"""
        ### ✅ Hasil Prediksi
        - **Prediksi Stunting** : **{pred_val:.2f}%**
        - **Kategori Prioritas** : **{kategori}**
        """)

