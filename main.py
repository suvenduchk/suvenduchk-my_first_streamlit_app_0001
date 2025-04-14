# Step 1: Import Libraries
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# For reproducibility
tf.random.set_seed(42)

# Step 2: Define a Custom SimpleRNN to Remove Unsupported Keyword Args
class CustomSimpleRNN(tf.keras.layers.SimpleRNN):
    def __init__(self, *args, **kwargs):
        # Remove the 'time_major' keyword if it exists in the config
        if 'time_major' in kwargs:
            kwargs.pop('time_major')
        super(CustomSimpleRNN, self).__init__(*args, **kwargs)

# Step 3: Load the IMDB Dataset Word Index and Create Reverse Mapping
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Step 4: Load the Pre-trained Model With a Custom Object for SimpleRNN
try:
    # This uses our CustomSimpleRNN in place of the regular SimpleRNN
    model = load_model('simple_rnn_imdb.h5', custom_objects={'SimpleRNN': CustomSimpleRNN})
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.write("Loading fallback model...")
    # Clear any lingering session state to avoid name_scope issues.
    tf.keras.backend.clear_session()
    # Build a fallback model:
    # Using an Embedding layer to handle integer token sequences
    model = Sequential([
        Embedding(input_dim=10000, output_dim=32, input_length=500),
        # The SimpleRNN here does not need the 'time_major' parameter.
        SimpleRNN(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    st.warning("Fallback model loaded. Note: It is untrained so predictions may not be accurate.")

# Step 5: Helper Functions

def decode_review(encoded_review):
    """Decode a review from integer tokens to words."""
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    """Convert a text review into a padded sequence of integers."""
    words = text.lower().split()
    # Use word_index to convert words into integers; use '2' (UNK token) if not found.
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 6: Build the Streamlit App UI

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

# Text area for user input
user_input = st.text_area("Movie Review")

if st.button("Classify"):
    preprocessed_input = preprocess_text(user_input)
    try:
        prediction = model.predict(preprocessed_input)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score: {prediction[0][0]:.4f}")
    except Exception as pred_e:
        st.error(f"Error during prediction: {pred_e}")
else:
    st.write("Please enter a movie review.")