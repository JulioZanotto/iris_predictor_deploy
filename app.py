import streamlit as st
import numpy as np
import joblib

# Carrega o modelo
model = joblib.load('svm_grid.joblib')
flower_dict = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# Cria a interface
# Título
st.title("Iris type model classifier")

# Criar os boxes para entrada de dados
f1 = st.number_input("First feature")
f2 = st.number_input("Second feature")
f3 = st.number_input("Third feature")
f4 = st.number_input("Fourth feature")

final_features = [np.array([f1, f2, f3, f4])]
prediction = model.predict(final_features)
output = flower_dict[prediction[0]]
if st.button('Predict'):
    st.write("## It is the Iris {}".format(output))
