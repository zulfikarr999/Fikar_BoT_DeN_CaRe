import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
import json
import nltk
import string
import pickle
import matplotlib.pyplot as plt
import streamlit as st
import random

from nltk.stem import WordNetLemmatizer
from nltk import download

# Unduh data NLTK jika belum ada
download('punkt')
download('wordnet')

# Inisialisasi variabel
lemmatizer = WordNetLemmatizer()

# Tentukan jalur file JSON lokal
json_file_path = 'intents.json'

# Muat data JSON
try:
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data1 = json.load(file)
except FileNotFoundError:
    st.error(f"File not found: {json_file_path}")
    st.stop()
except json.JSONDecodeError:
    st.error("Error decoding JSON")
    st.stop()

# Proses data JSON dan inisialisasi variabel
tags = []
inputs = []
responses = {}
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Proses data JSON
for intent in data1.get('intents', []):
    responses[intent['tag']] = intent['responses']
    for lines in intent['patterns']:
        inputs.append(lines)
        tags.append(intent['tag'])
        w = nltk.word_tokenize(lines)
        words.extend(w)
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Buat DataFrame dari data
data = pd.DataFrame({"patterns": inputs, "tags": tags})

# Membersihkan dan memproses teks
data['patterns'] = data['patterns'].apply(lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['patterns'] = data['patterns'].apply(lambda wrd: ''.join(wrd))

# Lemmatization
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Tokenisasi dan padding
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['patterns'])
train = tokenizer.texts_to_sequences(data['patterns'])
x_train = pad_sequences(train, maxlen=100)  # Sesuaikan panjang urutan sesuai kebutuhan

# Encoding tag
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

input_shape = x_train.shape[1]
vocabulary = len(tokenizer.word_index)
output_length = len(le.classes_)

# Simpan tokenizer dan label encoder untuk digunakan saat prediksi
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('label_encoder.pkl', 'wb') as handle:
    pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Definisikan model
i = Input(shape=(input_shape,))
x = Embedding(vocabulary + 1, 10)(i)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(i, x)

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

# Proses pelatihan model
history = model.fit(x_train, y_train, epochs=400)  # Sesuaikan jumlah epoch sesuai kebutuhan

# Menyimpan model setelah pelatihan
model.save("chatbot_model.keras")

# Menyimpan dan menampilkan grafik pelatihan
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Set Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Set Loss')
plt.legend(loc='upper right')
plt.title('Loss')
plt.savefig('training_plot.png')
plt.close()

# Streamlit app
st.markdown("<h1 style='text-align: center; color: blue;'>BoT DeN CaRe</h1>", unsafe_allow_html=True)
st.markdown("### Berbincang dengan **BoT DeN CaRe** dan dapatkan solusi cepat!", unsafe_allow_html=True)

# Tambahkan gambar atau logo
st.image("image/gigi.png", width=150)  # Ganti dengan path gambar yang sesuai

# Inisialisasi state session
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Preprocess input
def preprocess_input(user_input):
    # Tokenisasi dan lemmatization
    prediction_input = nltk.word_tokenize(user_input)
    prediction_input = [lemmatizer.lemmatize(word.lower()) for word in prediction_input if word not in string.punctuation]

    # Ubah ke dalam bentuk sequence dan pad
    prediction_input = tokenizer.texts_to_sequences([prediction_input])
    prediction_input = pad_sequences(prediction_input, maxlen=input_shape)  # Sesuaikan panjang urutan sesuai input_shape

    return prediction_input

# Fungsi menampilkan pesan
def display_message(sender, message, is_user=True):
    # Menampilkan pesan dengan gaya chat
    if is_user:
        st.markdown(f"<div class='chat-box user'>{message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-box bot'>{message}</div>", unsafe_allow_html=True)

def chat():
    # Muat tokenizer, label encoder, dan model
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open('label_encoder.pkl', 'rb') as handle:
        le = pickle.load(handle)

    model = tf.keras.models.load_model('chatbot_model.keras')

    # Display chat history
    if st.session_state.messages:
        for message in st.session_state.messages:
            display_message('user', message['user'], is_user=True)
            display_message('bot', message['bot'], is_user=False)
    
    # Get user input
    user_input = st.text_input('Ketik pesan:', '')
    if user_input:
        # Process input
        prediction_input = preprocess_input(user_input)

        with st.spinner('Dentalcarebot sedang berpikir...'):
            output = model.predict(prediction_input)
            output = output.argmax()

            response_tag = le.inverse_transform([output])[0]
            response = random.choice(responses[response_tag])
        
        # Store messages
        st.session_state.messages.append({'user': user_input, 'bot': response})
        
        # Display messages
        display_message('user', user_input, is_user=True)
        display_message('bot', response, is_user=False)

        if response_tag == "goodbye":
            st.session_state.messages.append({'user': 'goodbye', 'bot': 'Goodbye!'})
            st.write("=" * 60 + "\n")

# Menampilkan chat
chat()

# Feedback
st.markdown("### Bagaimana pengalaman Anda?")
feedback = st.slider("Rating", 1, 5)
st.write(f"Anda memberikan rating {feedback} 🌟")

# CSS untuk mempercantik chatbox
st.markdown("""
<style>
.chat-box { 
    padding: 10px; 
    margin-bottom: 15px;
    border-radius: 10px;
    max-width: 60%;
    clear: both;
}
.user {
    background-color: #d4f0fc;
    text-align: right;
    margin-left: auto;
}
.bot {
    background-color: #e8f5e9;
    text-align: left;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)
