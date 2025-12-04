import numpy as np
import streamlit as st
import pandas as pd


st.write("# Predicción de temperatura")
st.image("Temperatura.jpg", caption="Predicción de temperatura.")

st.header("Datos de la actividad")

def user_input_features():


    City=st.number_input("City",min_value=0,max_value=2,value=1,step=1, )

    Year=st.number_input("Year",min_value=0,max_value=2025,value=0,step=1,)

    Month=st.number_input("Month",min_value=0,max_value=12,value=1,step=1,)



    user_input_data = {
        "City": City,
        "Year": Year,
        "Month": Month,
    }

    features = pd.DataFrame(user_input_data, index=[0])
    return features

df = user_input_features()

datos = pd.read_csv("temp_mexico.csv", encoding="latin-1")
X= datos.drop("AverageTemperature", axis=1)
y= datos["AverageTemperature"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1613629)

LR=LinearRegression()
LR.fit(X_train,y_train)
modelo = LinearRegression()
modelo.fit(X_train, y_train)


b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df["City"] + b1[1]*df["Year"] + b1[2]*df["Month"]

st.subheader("Calculo de temperatura")
st.write(f"La temperatura estimada en tu ciudad es: **${prediccion:,.2f}** pesos")
