# CRPC_ID_NLP.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf

def train_crpc_model():
    # Load CRPC dataset
    with open('data/crpc.json', encoding='utf-8') as f:
        crpc_data = pd.read_json(f)

    # Handle missing values (replace NaN with a placeholder)
    crpc_data = crpc_data.fillna('UNKNOWN')

    # Encode categorical labels
    label_encoder_crpc = LabelEncoder()
    crpc_data['label'] = label_encoder_crpc.fit_transform(crpc_data['section_desc'])

    # Split the data into training and testing sets
    train_data_crpc, test_data_crpc = train_test_split(crpc_data, test_size=0.2, random_state=42)

    vectorizer_crpc= CountVectorizer(max_features=1000)
    X_train_crpc = vectorizer_crpc.fit_transform(train_data_crpc['section_desc']).toarray()
    y_train_crpc = train_data_crpc['label'].values

    # Define the model for CRPC
    model_crpc = Sequential([
        Dense(32, activation='relu', input_dim=1000),
        Dense(len(label_encoder_crpc.classes_), activation='softmax')
    ])

    # Compile the model
    model_crpc.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model_crpc.fit(X_train_crpc, y_train_crpc, epochs=100, batch_size=64, 
              validation_split=0.2, callbacks=[early_stopping])


    # Save the model
   # model_crpc.save('model_crpc.h5')

    return model_crpc, vectorizer_crpc, label_encoder_crpc

def predict_crpc_section(model_crpc, vectorizer_crpc, label_encoder_crpc, new_description_crpc):
    # Tokenize and predict CRPC sections
    new_description_vectorized_crpc = vectorizer_crpc.transform([new_description_crpc]).toarray()
    predicted_crpc_probs = model_crpc.predict(new_description_vectorized_crpc)

    # Decode the predicted CRPC section
    predicted_crpc_label = label_encoder_crpc.inverse_transform([predicted_crpc_probs.argmax()])[0]

    return predicted_crpc_label
