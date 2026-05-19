import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Prediksi Harga Mobil",
    page_icon="🚗",
    layout="wide"
)

# =========================
# STYLE CSS
# =========================
st.markdown("""
<style>
    .main-title {
        font-size: 60px;
        font-weight: 900;
        color: #f2f3f4;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 20px;
    }
    .result-card {
        background-color: #fff2a8;
        padding: 28px;
        border-radius: 18px;
        border: 2px solid #1f2937;
        text-align: center;
        margin-bottom: 20px;
    }

    .estimate-title {
        color: #8a7f05;
        font-weight: 800;
        font-size: 22px;
    }

    .price-text {
        font-size: 44px;
        font-weight: 900;
        color: #111827;
    }

    .info-card {
        background-color: #d7ecff;
        padding: 22px;
        border-radius: 18px;
        border: 2px solid #1f2937;
        text-align: center;
        margin-top: 25px;
        color: #0f4c56;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("Car_sales.csv")
    return df

df = load_data()

# =========================
# PREPROCESSING & MODEL
# =========================
@st.cache_resource
def train_model(data):
    data = data.copy()

    # Target prediksi: Price_in_thousands
    data = data.dropna(subset=["Price_in_thousands"])

    # Kolom yang tidak dipakai sebagai fitur utama
    drop_cols = ["Price_in_thousands", "Latest_Launch", "Model"]
    X = data.drop(columns=drop_cols, errors="ignore")
    y = data["Price_in_thousands"]

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=None
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "R2 Score": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }

    return pipeline, X, y, metrics, numeric_features, categorical_features

model, X_all, y_all, metrics, numeric_features, categorical_features = train_model(df)

# =========================
# HEADER
# =========================
st.markdown('<div class="main-title">Aplikasi Prediksi Harga Mobil</div>', unsafe_allow_html=True)

# =========================
# LAYOUT UTAMA
# =========================
left, right = st.columns([1, 1.25])

with left:
    st.subheader("Input Data Mobil")

    input_data = {}

    # Input kategori
    if "Manufacturer" in X_all.columns:
        input_data["Manufacturer"] = st.selectbox(
            "Manufacturer",
            sorted(X_all["Manufacturer"].dropna().unique())
        )

    if "Vehicle_type" in X_all.columns:
        input_data["Vehicle_type"] = st.selectbox(
            "Vehicle Type",
            sorted(X_all["Vehicle_type"].dropna().unique())
        )

    st.markdown("---")

    # Input numerik
    for col in numeric_features:
        min_val = float(X_all[col].min())
        max_val = float(X_all[col].max())
        median_val = float(X_all[col].median())

        input_data[col] = st.number_input(
            label=col.replace("_", " "),
            min_value=min_val,
            max_value=max_val,
            value=median_val,
            step=0.1
        )

    predict_button = st.button("Hitung Harga Mobil", use_container_width=True)

with right:
    st.subheader("Perkiraan Harga Mobil")

    if predict_button:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        # Dataset menggunakan satuan ribuan dolar
        price_usd = prediction * 1000

        st.markdown(f"""
        <div class="result-card">
            <div class="estimate-title">Estimasi Harga Mobil</div>
            <div class="price-text">${price_usd:,.2f}</div>
            <div style="color:#374151;">atau {prediction:.3f} ribu USD</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("**Data input yang digunakan:**")
        st.dataframe(input_df, use_container_width=True)
    else:
        st.markdown("""
        <div class="result-card">
            <div class="estimate-title">Estimasi Harga Mobil</div>
            <div class="price-text">$0.00</div>
            <div style="color:#374151;">Masukkan data lalu klik tombol prediksi</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Evaluasi Model")
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{metrics['R2 Score']:.3f}")
    c2.metric("MAE", f"{metrics['MAE']:.3f}")
    c3.metric("RMSE", f"{metrics['RMSE']:.3f}")

    st.markdown("""
    <div class="info-card">
        <b>Sistem ini dibuat oleh:</b><br>
        Nama: Salma Khairunnisa Azzahra<br>
        NPM: 237006097
    </div>
    """, unsafe_allow_html=True)

# =========================
# DATASET PREVIEW
# =========================
with st.expander("Lihat Dataset"):
    st.dataframe(df, use_container_width=True)

with st.expander("Informasi Dataset"):
    st.write("Jumlah data:", df.shape[0])
    st.write("Jumlah kolom:", df.shape[1])
    st.write("Kolom dataset:")
    st.write(df.columns.tolist())