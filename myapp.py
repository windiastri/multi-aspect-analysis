import streamlit as st
import pandas as pd
import numpy as np

# Libraries for Text Preprocessing
import re
# import neattext.functions as nfx

# libraries for model predictions
from keras.models import load_model
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# libraries visualize data
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

model_mapping = {
        "harga": "aspek_harga.h5",
        "makanan": "aspek_makanan.h5",
        "kamar": "aspek_kamar.h5",
        "pelayanan": "aspek_pelayanan.h5",
        "lokasi": "aspek_lokasi.h5",
        "fasilitas": "aspek_fasilitas.h5",
    }

aspect_options = ["lokasi", "makanan", "kamar", "pelayanan", "harga", "fasilitas"]

def cleaning(text):
    # text = nfx.remove_numbers(text) # Hapus number
    text = re.sub('[^0-9a-zA-Z]+', ' ', text) # Hapus karakter selain alfabet dan angka
    return text

def casefolding(text):
    return text.lower()

def get_selected_model(selected_aspect):
    model_filename = model_mapping.get(selected_aspect)
    models_directory = "aspect_models"

    if model_filename is not None:
        model_path = os.path.join(models_directory, model_filename)
        model = load_model(model_path)
        print(model_path)
        return model
    else:
        st.info("Model not available for the selected aspect.")
        return None
    
def get_models(selected_aspects):
    model_names = []

    for selected_aspect in selected_aspects:
        model_name = model_mapping.get(selected_aspect)
        if model_name is not None:
            model_names.append(model_name)
        else:
            st.error(f"Model not available for the selected aspect: {selected_aspect}")
    
    return model_names

def two_aspect():
    review_test = st.text_input('Write a Review', value=None)
    st.markdown("---")
    uploaded_file_single_aspect = st.file_uploader("Unggah file review kalimat kombinasi aspect", type=["xlsx"])
    st.markdown("---")
    selected_aspects = st.multiselect("Pilih 2 aspek:", aspect_options, max_selections=2)

    if len(selected_aspects) == 2:
        model_filenames = get_models(selected_aspects)
        models = []
        for filename in model_filenames:
            model = load_model(filename)
            models.append(model)
        
        if uploaded_file_single_aspect is not None:
            df = pd.read_excel(uploaded_file_single_aspect)
        else:
            df = pd.DataFrame({'text': [review_test]})
        review_cleaned = df['text'].apply(cleaning).apply(casefolding)

        # tokenize text
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(review_cleaned)
        X=tokenizer.texts_to_sequences(review_cleaned)
        X=pad_sequences(X, maxlen=31, padding='post')

        # # Lakukan prediksi pada seluruh data test menggunakan setiap model
        predictions = np.zeros((X.shape[0], len(models)))
        for i, model in enumerate(models):
            preds = model.predict(X)
            preds_binary = np.where(preds > 0.5, 1, 0)
            for j, pred in enumerate(preds_binary):
                predictions[j][i] = pred[0]
                
        prediction_aspect_result = predictions.astype(int)
        # Tampilkan data yang sudah diunggah
        st.write("Data yang diunggah:")
        st.write(df)

        # Tampilkan hasil prediksi
        st.write("Hasil Prediksi:", str(selected_aspects))
        st.write(prediction_aspect_result)

def three_aspect():
    review_test = st.text_input('Write a Review', value=None)
    st.markdown("---")
    uploaded_file_single_aspect = st.file_uploader("Unggah file review kalimat kombinasi aspect", type=["xlsx"])
    st.markdown("---")
    selected_aspects = st.multiselect("Pilih 3 aspek:", aspect_options, max_selections=3)

    if len(selected_aspects) == 2:
        model_filenames = get_models(selected_aspects)
        models = []
        for filename in model_filenames:
            model = load_model(filename)
            models.append(model)
        
        if uploaded_file_single_aspect is not None:
            df = pd.read_excel(uploaded_file_single_aspect)
        else:
            df = pd.DataFrame({'text': [review_test]})
        review_cleaned = df['text'].apply(cleaning).apply(casefolding)

        # tokenize text
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(review_cleaned)
        X=tokenizer.texts_to_sequences(review_cleaned)
        X=pad_sequences(X, maxlen=31, padding='post')
        
        # # Lakukan prediksi pada seluruh data test menggunakan setiap model
        predictions = np.zeros((X.shape[0], len(models)))
        for i, model in enumerate(models):
            preds = model.predict(X)
            preds_binary = np.where(preds > 0.5, 1, 0)
            for j, pred in enumerate(preds_binary):
                predictions[j][i] = pred[0]
                
        prediction_aspect_result = predictions.astype(int)
        # Tampilkan data yang sudah diunggah
        st.write("Data yang diunggah:")
        st.write(df)

        # Tampilkan hasil prediksi
        st.write("Hasil Prediksi:", str(selected_aspects))
        st.write(prediction_aspect_result)


def single_aspect():
    review_test = st.text_input('Write a Review', value=None)
    st.markdown("---")
    uploaded_file_single_aspect = st.file_uploader("Unggah file review kalimat single aspect", type=["xlsx"])
    st.markdown("---")
    single_aspect_options = ["none","lokasi", "makanan", "kamar", "pelayanan", "harga", "fasilitas"]
    selected_aspect = st.selectbox("Pilih aspek:", single_aspect_options, index=0)
    st.markdown("---")

    if (review_test is not None or uploaded_file_single_aspect is not None) and selected_aspect != "none":
        model = get_selected_model(selected_aspect)

        if uploaded_file_single_aspect is not None:
            df = pd.read_excel(uploaded_file_single_aspect)  # Ganti dengan metode yang sesuai untuk membaca file yang diupload
        else:
            df = pd.DataFrame({'text': [review_test]})
        review_cleaned = df['text'].apply(cleaning).apply(casefolding)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(review_cleaned)
        X=tokenizer.texts_to_sequences(review_cleaned)
        X=pad_sequences(X, maxlen=31, padding='post')

        threshold = 0.5
        predicted_result = []
        predictions_label = model.predict(X)
        predictions_label = tf.where(predictions_label < threshold, 0, 1)

        for result in predictions_label:
            predicted_result.append(result[0].numpy())
        
        st.write("aspect :", selected_aspect)
        st.write("predicted aspect :", predicted_result)

# Control display data
def show_content(menu):
    if menu == "Aspect Based":
        st.subheader("Aspect Based")
        single_aspect()
    
    elif menu == "Kombinasi 2 Aspect":
        st.subheader("Kombinasi kalimat 2 aspek")
        two_aspect()

    elif menu == "Kombinasi 3 Aspect":
        st.subheader("Kombinasi kalimat 3 aspek")
        three_aspect()

st.title('Multi Aspect Sentiment Analysis')

# Daftar menu navbar
menu_list = ["Aspect Based", "Kombinasi 2 Aspect", "Kombinasi 3 Aspect"]

# Pilihan menu dari user
selected_menu = st.sidebar.radio("Menu", menu_list)

# Tampilkan konten berdasarkan menu yang dipilih
show_content(selected_menu)


