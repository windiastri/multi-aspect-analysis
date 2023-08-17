import streamlit as st
import pandas as pd
import numpy as np

# Libraries for Text Preprocessing
import re
import neattext.functions as nfx

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
    text = nfx.remove_numbers(text)
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

def three_aspect():
    review_test = st.text_input('Write a Review', value=None)
    st.markdown("---")
    uploaded_file_single_aspect = st.file_uploader("Unggah file review kalimat kombinasi aspect", type=["xlsx"])
    st.markdown("---")
    selected_aspects = st.multiselect("Pilih 3 aspek:", aspect_options, max_selections=3)

    if (review_test is not None or uploaded_file_single_aspect is not None) and len(selected_aspects) == 3:
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
                
        prediction_aspect_result = predictions.astype(int).tolist()
        actual_aspect = (df[selected_aspects[0]].astype(str) + ',' + df[selected_aspects[1]].astype(str) + ',' + df[selected_aspects[2]].astype(str)) if uploaded_file_single_aspect is not None else None
        
        # Membuat DataFrame hasil prediksi
        results = pd.DataFrame({'Text': df['text'], 'Actual Aspect': actual_aspect, 'Predicted Aspect': prediction_aspect_result})
        table_data = [results.columns.tolist()] + results.values.tolist()

        # Menampilkan tabel hasil prediksi
        st.subheader('Predicted Result :')
        st.write("selected aspects :", selected_aspects)
        st.table(table_data)

def two_aspect():
    review_test = st.text_input('Write a Review', value=None)
    st.markdown("---")
    uploaded_file_single_aspect = st.file_uploader("Unggah file review kalimat kombinasi aspect", type=["xlsx"])
    st.markdown("---")
    selected_aspects = st.multiselect("Pilih 2 aspek:", aspect_options, max_selections=2)
    st.markdown("---")

    if (review_test is not None or uploaded_file_single_aspect is not None) and len(selected_aspects) == 2:
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
                
        prediction_aspect_result = predictions.astype(int).tolist()
        actual_aspect = (df[selected_aspects[0]].astype(str) + ',' + df[selected_aspects[1]].astype(str)) if uploaded_file_single_aspect is not None else None

        # Membuat DataFrame hasil prediksi
        results = pd.DataFrame({'Text': df['text'], 'Actual Aspect': actual_aspect, 'Predicted Aspect': prediction_aspect_result})
        table_data = [results.columns.tolist()] + results.values.tolist()

        # Menampilkan tabel hasil prediksi
        st.subheader('Predicted Result :')
        st.write("selected aspects :", selected_aspects)
        st.table(table_data)

def single_aspect():
    review_test = st.text_input('Write a Review', value=None)
    st.markdown("---")
    uploaded_file_single_aspect = st.file_uploader("Unggah file review kalimat single aspect", type=["xlsx"])
    st.markdown("---")
    single_aspect_options = ["none","lokasi", "makanan", "kamar", "pelayanan", "harga", "fasilitas"]
    selected_aspect = st.selectbox("Pilih aspek:", single_aspect_options, index=0)
    st.markdown("---")

    if (review_test is not None or uploaded_file_single_aspect is not None) and selected_aspect != "none":
        if uploaded_file_single_aspect is not None:
            df = pd.read_excel(uploaded_file_single_aspect)
        else:
            df = pd.DataFrame({'text': [review_test]})

        review_cleaned = df['text'].apply(cleaning).apply(casefolding)
        model = get_selected_model(selected_aspect)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(review_cleaned)
        X=tokenizer.texts_to_sequences(review_cleaned)
        X=pad_sequences(X, maxlen=31, padding='post')

        threshold = 0.5
        predicted_result = []
        predictions_label = model.predict(X)
        predictions_label = tf.where(predictions_label < threshold, 0, 1)

        for result in predictions_label:
            if result[0] == 1:
                predicted_result.append(selected_aspect)
            else:
                predicted_result.append("lainnya")

        # Membuat DataFrame hasil prediksi
        results = pd.DataFrame({'Text': df['text'], 'Predicted Result': predicted_result})
        table_data = [results.columns.tolist()] + results.values.tolist()

        # Menampilkan tabel hasil prediksi
        st.subheader('Predicted Result :')
        st.table(table_data)

def sentiment():
    review_test = st.text_input('Write a Review', value=None)
    st.markdown("---")
    uploaded_file_single_aspect = st.file_uploader("Unggah file review kalimat", type=["xlsx"])
    st.markdown("---")

    if review_test is not None or uploaded_file_single_aspect is not None:
        if uploaded_file_single_aspect is not None:
            df = pd.read_excel(uploaded_file_single_aspect)
        else:
            df = pd.DataFrame({'Text': [review_test]})

        review_cleaned = df['Text'].apply(cleaning).apply(casefolding)
        model = load_model('sentiment_aspek.h5')

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(review_cleaned)
        X=tokenizer.texts_to_sequences(review_cleaned)
        X=pad_sequences(X, maxlen=31, padding='post')

        threshold = 0.5
        predicted_result = []
        predictions_label = model.predict(X)
        predictions_label = tf.where(predictions_label < threshold, 0, 1)

        for result in predictions_label:
            if result[0] == 1:
                predicted_result.append("positive")
            else:
                predicted_result.append("negative")

        # Membuat DataFrame hasil prediksi
        results = pd.DataFrame({'Text': df['Text'], 'Predicted Result': predicted_result})
        table_data = [results.columns.tolist()] + results.values.tolist()

        # Menampilkan tabel hasil prediksi
        st.subheader('Predicted Result :')
        st.table(table_data)

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

    elif menu == "Sentimen Kalimat":
        st.subheader("Sentimen Kalimat")
        sentiment()

st.title('Multi Aspect Sentiment Analysis')

# display menu
menu_list = ["Aspect Based", "Kombinasi 2 Aspect", "Kombinasi 3 Aspect", "Sentimen Kalimat"]
selected_menu = st.sidebar.radio("Menu", menu_list)

# Tampilkan konten berdasarkan menu yang dipilih
show_content(selected_menu)


