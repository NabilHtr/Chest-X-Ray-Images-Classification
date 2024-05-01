# train_model.py
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LSTM, TimeDistributed, Reshape, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import json

def train_model(epochs, steps_per_epoch, validation_steps):
    
    # Chemin vers les dossiers de données
    base_dir = r'C:\Users\nabil\Desktop\YNOV M2\Deep learning\Projet\Data\chest_xray'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

    # Générateurs de données
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
    validation_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=(150, 150), batch_size=1, class_mode='binary', shuffle=False)
    
    
    model = Sequential([
        TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(1, 150, 150, 3)),
        TimeDistributed(MaxPooling2D(2, 2)),
        TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
        TimeDistributed(MaxPooling2D(2, 2)),
        TimeDistributed(Conv2D(128, (3, 3), activation='relu')),
        TimeDistributed(MaxPooling2D(2, 2)),
        TimeDistributed(Flatten()),

        LSTM(64),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

    # Adapter les données d'entrée pour ajouter une dimension temporelle
    def add_time_dimension(generator):
        for batch_x, batch_y in generator:
            batch_x_with_time = np.expand_dims(batch_x, 1)  # Ajoute une dimension temporelle
            yield batch_x_with_time, batch_y

    # Entraînement du modèle
    history = model.fit(
        add_time_dimension(train_generator),
        steps_per_epoch=steps_per_epoch,  
        epochs=epochs,
        validation_data=add_time_dimension(validation_generator),
        validation_steps=validation_steps,  

    )

    # Générer la matrice de confusion et le rapport de classification
    test_generator.reset()
    predictions = model.predict(add_time_dimension(test_generator), steps=len(test_generator))
    predicted_classes = (predictions > 0.5).astype(int)
    true_classes = test_generator.classes
    cm = confusion_matrix(true_classes, predicted_classes)
    report = classification_report(true_classes, predicted_classes, output_dict=True)

    # Sauvegarde de la matrice et du rapport
    np.save('confusion_matrix.npy', cm)
    with open('classification_report.json', 'w') as f:
        json.dump(report, f)
    
    return history.history, report, cm
if __name__ == "__main__":
    train_model()    