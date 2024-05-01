import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from train_model import train_model
    

def display_result(prediction):
    proba = prediction[0][0]
    result = "Pneumonia" if proba > 0.5 else "Normal"
    color = "red" if result == "Pneumonia" else "green"
    st.markdown(f"<h1 style='color: {color};'>Résultat: {result}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color: {color};'>Probabilité de Pneumonia: {proba * 100:.2f}%</h2>", unsafe_allow_html=True)

def preprocess_image(image):
        image = image.resize((150, 150))
        image_array = np.array(image) / 255.0  # Normaliser l'image
        image_array = np.expand_dims(image_array, axis=0)  # Ajouter une dimension batch
        image_array = np.expand_dims(image_array, axis=1)  # Ajouter une dimension temporelle
        return image_array



# Chargement du modèle
model_path = 'C:\\Users\\nabil\\Desktop\\YNOV M2\\Deep learning\\Projet\\projet_partie1_DL_ynov.keras'
model = tf.keras.models.load_model(model_path)

st.title('Application de Classification des Radiographies : Pneumonie / Normale')

st.markdown(
    """
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">Réentrainement du modèle avec nouveaux hyperparamètres</p>', unsafe_allow_html=True)

with st.form(key='model_form'):
    epochs = st.slider('Nombre d\'époch', min_value=1, max_value=100, value=10)
    steps_per_epoch = st.slider('Steps par époque', min_value=10, max_value=500, value=100)
    validation_steps = st.slider('Steps de validation', min_value=5, max_value=200, value=50)
    
    submit_button = st.form_submit_button(label='Entraîner le modèle')

if submit_button:
    with st.spinner('Entraînement en cours...'):
        # Appel de la fonction d'entraînement
        history, report, cm = train_model(epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
        
        st.success('Entraînement terminé !')
        
        # Affichage des métriques
        st.write("Rapport de classification:")
        st.json(report)
        
        st.write("Matrice de confusion:")
        st.write(cm)


st.markdown('<p class="big-font">Statistiques de Performance du Modèle</p>', unsafe_allow_html=True)

# Charger le rapport de classification
with open('classification_report.json', 'r') as f:
    report = json.load(f)

# Afficher le rapport de classification

st.markdown('<p class="big-font">1/ Rapport de classification</p>', unsafe_allow_html=True)
for label, metrics in report.items():
    if label not in ['accuracy', 'macro avg', 'weighted avg']:  
        st.markdown(f"#### Classe: {label}")
        st.write(f"Précision: {metrics['precision']:.2f}")
        st.write(f"Rappel: {metrics['recall']:.2f}")
        st.write(f"Score F1: {metrics['f1-score']:.2f}")

# Afficher également les métriques globales 
st.markdown('<p class="big-font">2/ Taux de précision</p>', unsafe_allow_html=True)
st.write(f"Précision Globale: {report['accuracy']:.2f}")

cm = np.load('confusion_matrix.npy')
st.markdown('<p class="big-font">3/ Matrice de confusion</p>', unsafe_allow_html=True)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
plt.title('Matrice de Confusion')
plt.xlabel('Prédictions')
plt.ylabel('Véritables Classes')
st.pyplot(plt)



history_df = pd.read_csv('training_history.csv')

# Extraire les colonnes nécessaires
epochs = history_df['epoch']
acc = history_df['accuracy'] 
val_acc = history_df['val_accuracy']
loss = history_df['loss']
val_loss = history_df['val_loss']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
# Graphique précision
ax1.plot(epochs, acc, 'bo-', label='Précision Entraînement')
ax1.plot(epochs, val_acc, 'ro-', label='Précision Validation')
ax1.set_title('Précision d\'entraînement et de validation')
ax1.set_xlabel('Époques')
ax1.set_ylabel('Précision')
ax1.legend()

# Graphique perte
ax2.plot(epochs, loss, 'bo-', label='Perte Entraînement')
ax2.plot(epochs, val_loss, 'ro-', label='Perte Validation')
ax2.set_title('Perte d\'entraînement et de validation')
ax2.set_xlabel('Époques')
ax2.set_ylabel('Perte')
ax2.legend()

# Afficher la figure dans Streamlit
st.pyplot(fig)

# Section pour le téléchargement et la prédiction
st.header('Téléchargement d\'Image et Prédiction')

uploaded_file = st.file_uploader("Chargez une image de radiographie", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Image chargée', use_column_width=True)
    

    # Prétraitement et prédiction
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    display_result(prediction)

    predicted_class = 'Pneumonia' if prediction[0] > 0.5 else 'Normal'
    probability = prediction[0][0]


    if 'history' not in st.session_state:
        st.session_state['history'] = []
    st.session_state['history'].append((uploaded_file.name, predicted_class, probability))


    with st.expander("Voir l'historique des prédictions"):
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        for item in st.session_state['history']:
            st.write(f"Image: {item[0]}, Prédiction: {item[1]}, Probabilité de Pneumonia: {item[2] * 100:.2f}%")
