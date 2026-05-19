# Aplikasi Prediksi Harga Mobil Streamlit

## Cara menjalankan

1. Pastikan file berikut berada dalam satu folder:
   - app.py
   - Car_sales.csv
   - requirements.txt

2. Install library:
```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi:
```bash
streamlit run app.py
```

Target prediksi adalah kolom `Price_in_thousands`, sehingga hasil dikonversi menjadi USD dengan mengalikan 1000.
