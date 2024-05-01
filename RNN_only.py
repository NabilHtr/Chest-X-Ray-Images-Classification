import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Chemin vers le dossier contenant les sous-dossiers train, val et test
base_dir = r'C:\Users\HichemStinson13\Desktop\PROJET\ml_projet_final\medical_image_classification\chest_xray'

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Configuration du générateur d'images
train_datagen = ImageDataGenerator(rescale=1./255)
test_val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary',
    shuffle=True
)

validation_generator = test_val_datagen.flow_from_directory(
        val_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

# Construction du modèle RNN
model = Sequential([
    TimeDistributed(Flatten(input_shape=(None, 150, 150, 3))),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compilation du modèle
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(
      train_generator,
      steps_per_epoch=100,  # Ajustez ce nombre en fonction de la taille de votre ensemble d'entraînement
      epochs=20,
      validation_data=validation_generator,
      validation_steps=50)  # Ajustez ce nombre en fonction de la taille de votre ensemble de validation

# Évaluation du modèle
test_generator = test_val_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=10,
        class_mode='binary')

test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')
