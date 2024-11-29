# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 15:59:49 2024

@author: rondo
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import base64
import plotly.express as px
import pickle

#Charger le model et le scaler
@st.cache_resource
def charger_model_et_scaler():
    model = pickle.load(open("classification_model.pkl", "rb"))
    scaler = pickle.load(open("scaler_2.pkl", "rb"))
    return model, scaler

#rappel du modele
model, scaler = charger_model_et_scaler()

#charger le fichier csv
uploaded_file = st.file_uploader("Charger le fichier csv contenant les données à prédire", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Aperçu des données: ")
    st.dataframe(data, use_container_width=True)
    
    #verifier que les colonnes nécessaires sont présentes
    collonnes_attendues = ["diagonal", "height_left", "height_right", "margin_low", "margin_up", "length"]
    if not all(col in data.columns for col in collonnes_attendues):
        st.error(f"Le fichier doit contenir les colonnes suivantes : {', '.join(collonnes_attendues)}")
    else:
        X=data[collonnes_attendues]
        X_scaled = scaler.transform(X)
        with st.spinner("prédictions..."):
            predictions = model.predict(X_scaled)
            probabilites= model.predict_proba(X_scaled)
            
        #Créer un dataframe pour les résultats
        results_df = pd.DataFrame(
            {
                "Prédictions":predictions,
                "Probabilités Faux":probabilites[:,0],
                "Probabilités Vrai":probabilites[:,1]
                })
        
        #Afficher les resultats à deux chiffres après la virgule
        results_df["Probabilités Faux"] = results_df["Probabilités Faux"].map(lambda x: f"{x:2f}")
        results_df["Probabilités Vrai"] = results_df["Probabilités Vrai"].map(lambda x: f"{x:2f}")
        
        #Afficher les résultats
        st.write("Résultts des Prédictions : ")
        st.dataframe(results_df, use_container_width=True)
        
        #Histogramme
        prediction_counts = results_df["Prédictions"].replace({0: "Faux billet", 1: "Vrai billet"}).value_counts().reset_index()
        prediction_counts.columns = ["Type de billet",  "Nombre"]
        
        fig = px.bar(prediction_counts,
                     x = "Type de billet", y ="Nombre",
                     title = "Distribution des Prédictions")
        st.plotly_chart(fig, use_container_width= True)
        
        