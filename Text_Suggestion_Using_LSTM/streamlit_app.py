import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the trained LSTM model
model = load_model('text_suggestion_model.h5')

# Set the max length (same used during training)
max_len = 15

st.title("📝 Text Suggestion Using LSTM")

input_text = st.text_input("Enter your sentence:", "")

if st.button("Suggest Next Word"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        seq = tokenizer.texts_to_sequences([input_text])[0]
        seq = pad_sequences([seq], maxlen=max_len, padding='pre')
        prediction = model.predict(seq, verbose=0)
        predicted_word_index = np.argmax(prediction)

        # Map index back to word
        word = ""
        for w, idx in tokenizer.word_index.items():
            if idx == predicted_word_index:
                word = w
                break

        st.success(f"Predicted next word: **{word}**")
