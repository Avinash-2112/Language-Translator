# Import necessary modules from Flask, Streamlit, and other libraries.
from flask import Flask, request, jsonify
from transformers import MarianMTModel, MarianTokenizer
import streamlit as st
from concurrent.futures import ThreadPoolExecutor


# Create an instance of Flask application.
app = Flask(__name__)

# Define pre-trained model names for translation between different languages.
model_name_en_de = "Helsinki-NLP/opus-mt-en-de"
model_name_en_fr = "Helsinki-NLP/opus-mt-en-fr"
model_name_en_es = "Helsinki-NLP/opus-mt-en-es"

# Load pre-trained models for translation.
model_de = MarianMTModel.from_pretrained(model_name_en_de)
tokenizer_de = MarianTokenizer.from_pretrained(model_name_en_de)

model_fr = MarianMTModel.from_pretrained(model_name_en_fr)
tokenizer_fr = MarianTokenizer.from_pretrained(model_name_en_fr)

model_es = MarianMTModel.from_pretrained(model_name_en_es)
tokenizer_es = MarianTokenizer.from_pretrained(model_name_en_es)

# Create a ThreadPoolExecutor for concurrent processing.
executor = ThreadPoolExecutor()


# Define the translation logic function.
def translate_text(input_text, target_language):
    if target_language == "German":
        model = model_de
        tokenizer = tokenizer_de
    elif target_language == "French":
        model = model_fr
        tokenizer = tokenizer_fr
    elif target_language == "Spanish":
        model = model_es
        tokenizer = tokenizer_es
    else:
        return "Translation to this language is not supported."

    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    translation = model.generate(input_ids)
    translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)

    return translated_text


# Define the main function that will run the Streamlit app.
def main():
    st.title("Language Translation Tool")
    input_text = st.text_area("Enter Text to Translate", "")

    target_languages = ["German", "French", "Spanish"]
    target_language = st.selectbox("Select Target Language", target_languages)

    if st.button("Translate"):
        translated_text = translate_text(input_text, target_language)
        st.text("Translated Text:")
        st.write(translated_text)


# Run the Flask app and Streamlit app when the script is executed directly.
if __name__ == '__main__':  # streamlit run app_frontend.py
    main()
