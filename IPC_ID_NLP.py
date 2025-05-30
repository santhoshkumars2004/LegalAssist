import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf


#def train_ipc_model():
    # Load IPC dataset
   # with open('data/ipc.json', encoding='utf-8') as f:
    #    ipc_data = pd.read_json(f)

    # Handle missing values (replace NaN with a placeholder)
    #ipc_data = ipc_data.fillna('UNKNOWN')

    # Encode categorical labels
    #label_encoder_ipc = LabelEncoder()
    #ipc_data['label'] = label_encoder_ipc.fit_transform(ipc_data['section_desc'])

    # Split the data into training and testing sets
    #train_data_ipc, test_data_ipc = train_test_split(ipc_data, test_size=0.2, random_state=42)

    # Train Word2Vec model on your corpus
    #word2vec_model = Word2Vec(sentences=train_data_ipc['section_desc'], vector_size=100, window=5, min_count=1, workers=4)

    # Tokenize and pad sequences
    #tokenizer = Tokenizer()
    #tokenizer.fit_on_texts(train_data_ipc['section_desc'])
    #X_train_ipc = pad_sequences(tokenizer.texts_to_sequences(train_data_ipc['section_desc']))
    #y_train_ipc = train_data_ipc['label'].values

    # Embedding layer using the pre-trained Word2Vec model
    #embedding_matrix = create_embedding_matrix(word2vec_model, tokenizer.word_index)
    #vocab_size = len(tokenizer.word_index) + 1

    #model_ipc = Sequential([
    #    Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix], input_length=X_train_ipc.shape[1], trainable=False),
    #    Flatten(),
    #    Dense(128, activation='relu'),
     #   Dropout(0.5),  # Add dropout layer
      #  Dense(len(label_encoder_ipc.classes_), activation='softmax')
    #])

    # Compile the model
    #model_ipc.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    #model_ipc.fit(X_train_ipc, y_train_ipc, epochs=100, batch_size=64, 
     #              validation_split=0.2, callbacks=[early_stopping])

    # Save the model
    # model_ipc.save('model_ipc.h5')

#    return model_ipc, tokenizer, label_encoder_ipc, embedding_matrix

#def create_embedding_matrix(model, word_index):
   # embedding_matrix = np.zeros((len(word_index) + 1, model.vector_size))
  #  for word, i in word_index.items():
 #       if word in model.wv:
    #        embedding_matrix[i] = model.wv[word]
    #return embedding_matrix

#def predict_ipc_section(model_ipc, tokenizer, label_encoder_ipc, embedding_matrix, new_description):
    # Tokenize and get embedding for new IPC section
 #   new_description_seq = pad_sequences(tokenizer.texts_to_sequences([new_description]))
  #  new_embedding = create_embedding_matrix(word2vec_model, tokenizer.word_index)

    # Predict IPC sections using the existing model
#    predicted_ipc_probs = model_ipc.predict(new_description_seq)

    # Decode the predicted IPC section
 #   predicted_ipc_label = label_encoder_ipc.inverse_transform([predicted_ipc_probs.argmax()])[0]

  #  return predicted_ipc_label
