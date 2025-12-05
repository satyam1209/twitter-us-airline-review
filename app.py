import streamlit as st
import numpy as np
st.title("Movie sentiment analysis")
st.write('Enter a movie review to classfiy it as positive or negative')
user_input = st.text_area('Movie Review')
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


with open('tokenizer1.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

def get_padded_text(review):
    sequences = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(sequences, maxlen=50)
    return padded

from tensorflow.keras.models import load_model
model = load_model('twitter_airline.h5')
st.write(model.summary())
if st.button("Classify"):
    padded_review = get_padded_text(user_input)
    prediction = model.predict(padded_review)
    pred_class = np.argmax(prediction)
    sentiment = le.inverse_transform([pred_class])[0]
    # print(prediction)
    # if pred_class == 1:
    #     sentiment = 'Positive'
    # elif pred_class ==2:
    #     sentiment = 'Neutral'
    # else:
    #     sentiment = 'Negative'
    st.write(f"The sentiment is {sentiment}")
else:
    st.write("Pls enter review")

