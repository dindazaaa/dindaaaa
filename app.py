import streamlit as st
import pandas as pd
import pickle
from sklearn.datasets import load_iris

# Load model dan data info
iris = load_iris()
model = pickle.load(open('iris_model.pkl', 'rb'))

st.title("Iris Flower Prediction App")
st.write("Aplikasi ini memprediksi jenis bunga Iris berdasarkan input pengguna.")

# Sidebar untuk input
st.sidebar.header("Input Parameter")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Parameter Input Pengguna')
st.write(df)

# Prediksi
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader('Prediksi')
st.write(iris.target_names[prediction][0])

st.subheader('Probabilitas Prediksi')
st.write(prediction_proba)
