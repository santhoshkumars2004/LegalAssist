import streamlit as st
from pdf_extractor import process_uploaded_file
import speech_recognition as sr
from langdetect import detect
#from googletrans import Translator
from deep_translator import GoogleTranslator, MyMemoryTranslator
import pandas as pd
import os
import tempfile
import nltk
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.feature_extraction.text import TfidfVectorizer
import pytesseract
from PIL import Image
import google.generativeai as genai
import textwrap
from IPython.display import display, Markdown
import fitz

# from IPC_ID_NLP import train_ipc_model, predict_ipc_section
# from CRPC_ID_NLP import train_crpc_model, predict_crpc_section

# Function to detect language using langdetect library
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# Function to transcribe audio and get the transcribed text
def transcribe_audio():
    recognizer = sr.Recognizer()

    print("Terminal: Initializing Recognizer.") # Trace
    try:
        with sr.Microphone() as source:
            print("Terminal: Microphone source obtained.") # Trace
            st.info("Speak clearly into your microphone. The app will process after you stop speaking.")
            print("Terminal: Listening for audio...") # Terminal output
            
            print("Terminal: Adjusting for ambient noise...") # Trace
            recognizer.adjust_for_ambient_noise(source, duration=5)
            print("Terminal: Adjustment complete. Listening...")
            
            try:
                print("Terminal: Calling recognizer.listen()...") # Trace before listen
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=120)
                print("Terminal: recognizer.listen() finished.") # Trace after listen
                st.session_state['audio_state'] = 'Processing...' # Update status before transcription
                st.rerun() # Rerun to show processing status
            except sr.WaitTimeoutError:
                st.warning("No speech detected within the timeout period.")
                print("Terminal: No speech detected within timeout.")
                st.session_state['audio_state'] = 'Audio processing finished (no speech detected).'
                st.rerun()
                return None, None
            except Exception as e:
                 st.error(f"An error occurred during audio capture: {e}")
                 print(f"Terminal: Audio capture error: {e}")
                 st.session_state['audio_state'] = f'Audio processing finished (capture error: {e}).'
                 st.rerun()
                 return None, None
    except Exception as e:
        st.error(f"Failed to access microphone: {e}")
        print(f"Terminal: Microphone access error: {e}")
        st.session_state['audio_state'] = f'Microphone access failed: {e}'
        st.rerun()
        return None, None

    try:
        print("Terminal: Recognizing speech...") # Terminal output
        text = recognizer.recognize_google(audio)
        print(f"Terminal: Recognized text: {text}") # Terminal output
        detected_language = detect(text)  # Detect the language of the transcribed text
        print(f"Terminal: Detected language: {detected_language}") # Terminal output
        
        # Return the transcribed text and language
        return text, detected_language
    except sr.UnknownValueError:
        st.warning("Speech Recognition could not understand audio.")
        print("Terminal: Speech Recognition could not understand audio.") # Terminal output
        st.session_state['audio_state'] = 'Audio processing finished (could not understand speech).'
        st.rerun()
        return None, None
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        print(f"Terminal: Speech Recognition API error: {e}") # Terminal output
        st.session_state['audio_state'] = f'Audio processing finished (API error: {e}).'
        st.rerun()
        return None, None

# New/Updated function to process uploaded PDF file using PyMuPDF
def process_uploaded_file(pdf_path):
    """Extract text from PDF file using PyMuPDF."""
    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        extracted_text = []

        # Extract text from each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():  # Only add non-empty pages
                extracted_text.append(text.strip())

        doc.close()
        return extracted_text
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        print(f"Terminal: PDF extraction error: {e}")
        return None

# Updated Function to translate text to English using deep-translator with fallback
def translate_to_english(text, source_language="auto"):
    """Translate text to English using multiple free translation services with fallback."""
    if not text or len(text.strip()) == 0:
        return None

    # Try different translation services in order
    translators = [
        ("Google", lambda t: GoogleTranslator(source=source_language, target='en').translate(t)),
        ("MyMemory", lambda t: MyMemoryTranslator(source=source_language, target='en').translate(t))
    ]

    for service_name, translator_func in translators:
        try:
            # Split text into smaller chunks if needed (some services have length limits)
            chunks = [text[i:i+450] for i in range(0, len(text), 450)]
            translated_chunks = []

            for chunk in chunks:
                translated_chunk = translator_func(chunk)
                if translated_chunk:
                    translated_chunks.append(translated_chunk)
                else:
                    # If a chunk fails, the service might be down or hit a limit.
                    # Skip this service for this text and try the next one.
                    print(f"Terminal: {service_name} failed to translate a chunk.")
                    raise Exception(f"{service_name} chunk translation failed") # Raise to move to the next service

            if translated_chunks:
                return " ".join(translated_chunks)
        except Exception as e:
            st.error(f"Terminal: {service_name} translation error: {e}")
            continue

    st.error("All translation services failed. Please try again later.")
    return None

# Function to translate text to Hindi (keeping the structure, but will use 'deep-translator')
def translate_to_hindi(text, source_language="auto"):
    try:
        translator = GoogleTranslator(source=source_language, target="hi") # Using Google via deep-translator
        translated_text = translator.translate(text)
        return translated_text
    except Exception as e:
        st.error(f"Translation error to Hindi: {e}")
        print(f"Terminal: Translation error to Hindi: {e}")
        return None

# 1) Prompt entering section with Audio ico
# Load IPC data
ipc_data = pd.read_json(r'data\ipc.json').fillna('UNKNOWN')
label_encoder_ipc = LabelEncoder()
ipc_data['label'] = label_encoder_ipc.fit_transform(ipc_data['section_desc'])
train_data_ipc, _ = train_test_split(ipc_data, test_size=0.2, random_state=42)
vectorizer_ipc = TfidfVectorizer(max_features=1000)
X_train_ipc = vectorizer_ipc.fit_transform(train_data_ipc['section_desc']).toarray()
y_train_ipc = train_data_ipc['label'].values
model_ipc = Sequential([
    Input(shape=(1000,)),
    Dense(32, activation='relu'),
    Dense(len(label_encoder_ipc.classes_), activation='softmax')
])
model_ipc.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_ipc.fit(X_train_ipc, y_train_ipc, epochs=100, batch_size=32)

# Load CRPC data
crpc_data = pd.read_json(r'data\crpc.json').fillna('UNKNOWN')
label_encoder_crpc = LabelEncoder()
crpc_data['label'] = label_encoder_crpc.fit_transform(crpc_data['section_desc'])
train_data_crpc, _ = train_test_split(crpc_data, test_size=0.2, random_state=42)
vectorizer_crpc = TfidfVectorizer(max_features=1000)
X_train_crpc = vectorizer_crpc.fit_transform(train_data_crpc['section_desc']).toarray()
y_train_crpc = train_data_crpc['label'].values
model_crpc = Sequential([
    Input(shape=(1000,)),
    Dense(32, activation='relu'),
    Dense(len(label_encoder_crpc.classes_), activation='softmax')
])
model_crpc.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_crpc.fit(X_train_crpc, y_train_crpc, epochs=100, batch_size=32)

# Streamlit UI
st.set_page_config(page_title="LegalAssist", page_icon="🔍", layout="wide")
st.header("🔍 LegalAssist - AI for Legal Section Suggestions")

from rajastan.components.sidebar import sidebar
from rajastan.core.caching import bootstrap_caching

# Enable caching for expensive functions
bootstrap_caching()

sidebar()
    
import nltk
import streamlit as st
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.feature_extraction.text import TfidfVectorizer

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Function to perform sentiment analysis
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return sentiment_score

# Function to create embedding matrix
def create_embedding_matrix(model, word_index):
    embedding_matrix = np.zeros((len(word_index) + 1, model.vector_size))
    for word, i in word_index.items():
        if word in model.wv:
            embedding_matrix[i] = model.wv[word]
    return embedding_matrix
# Function to train IPC model
def train_ipc_model(user_input):
    # Load IPC dataset (adjust file path accordingly)
    with open(r'ipc.json', encoding='utf-8') as f:
        ipc_data = pd.read_json(f)

    # Handle missing values (replace NaN with a placeholder)
    ipc_data = ipc_data.fillna('UNKNOWN')

    # Encode categorical labels
    label_encoder_ipc = LabelEncoder()
    ipc_data['label'] = label_encoder_ipc.fit_transform(ipc_data['section_desc'])

    # Split the data into training and testing sets
    train_data_ipc, test_data_ipc = train_test_split(ipc_data, test_size=0.2, random_state=42)

    # Tokenize text data (using TF-IDF for simplicity, you may need a more sophisticated approach)
    vectorizer_ipc = TfidfVectorizer(max_features=1000)
    X_train_ipc = vectorizer_ipc.fit_transform(train_data_ipc['section_desc']).toarray()
    y_train_ipc = train_data_ipc['label'].values

    # Define the model for IPC
    model_ipc = Sequential([
        Input(shape=(1000,)),
        Dense(32, activation='relu'),
        Dense(len(label_encoder_ipc.classes_), activation='softmax')
    ])

    # Compile the model
    model_ipc.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model_ipc.fit(X_train_ipc, y_train_ipc, epochs=75, batch_size=32)

    # Tokenize and predict IPC sections for user input
    user_input_vectorized = vectorizer_ipc.transform([user_input]).toarray()
    predicted_ipc_probs = model_ipc.predict(user_input_vectorized)

    # Decode the predicted IPC section
    predicted_ipc_label = label_encoder_ipc.inverse_transform([predicted_ipc_probs.argmax()])[0]

    return model_ipc, vectorizer_ipc, label_encoder_ipc, predicted_ipc_label
import streamlit as st
import google.generativeai as genai
import textwrap
from IPython.display import display, Markdown
import os

# Set up Google Generative AI with your API key
GOOGLE_API_KEY = os.getenv('AIzaSyCkkja3hTj8nwDwbZWcT7eEzQUYK2UsvVE')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Function to generate content using the entered text
def generate_content(text):
    response = model.generate_content([text+"semantic representation like uml"])
    response.resolve()
    return response.generated_content[0]

def generate_content(text):
    response = model.generate_content([text+"semantic representation like uml diagram"])
    response.resolve()
    return response.generated_content[0]

# 1) Prompt entering section with Audio icon
with st.form(key="prompt_form"):
    st.subheader("Prompt Section")

    # Initialize session state for the prompt text
    if 'prompt_text' not in st.session_state:
        st.session_state['prompt_text'] = "Enter text or click 'Start Recording' to record..."

    if 'audio_state' not in st.session_state:
        st.session_state['audio_state'] = 'Ready'

    audio_button_clicked = st.form_submit_button("Start Recording 🎤")
    
    # Handle the audio recording process when the button is clicked
    if audio_button_clicked:
        st.session_state['audio_state'] = "Starting recording..."
        st.rerun() # Rerun to show starting status

    # This block initiates the actual recording after the first rerun triggered by button click
    if st.session_state['audio_state'] == "Starting recording...":
         print("Terminal: Calling transcribe_audio...") # Trace
         transcribed_text_data = transcribe_audio() # This call is blocking
         print(f"Terminal: transcribe_audio returned: {transcribed_text_data}") # Trace
         # Update session state with transcribed text after recording finishes
         if transcribed_text_data and transcribed_text_data[0]:
             st.session_state['prompt_text'] = transcribed_text_data[0]
             print(f"Terminal: Updated session_state['prompt_text'] to: {st.session_state['prompt_text']}") # Trace
             st.session_state['audio_state'] = 'Audio recorded and processed successfully!'
         else:
              # Keep the original prompt text if transcription failed or was empty
              st.session_state['audio_state'] = st.session_state.get('audio_state', 'Audio processing finished.') # Use existing error status or default
         st.rerun() # Rerun to update UI with transcribed text and final status

    # Display the text input, its value is controlled by session state
    gen_inp = st.text_input("Enter some text:", value=st.session_state['prompt_text'], key='prompt_input') # Added a key for reliability
    st.write("You entered:", gen_inp)
    prompt_submit = st.form_submit_button("Submit Prompt")

    # Display audio status message from session state
    st.write("Audio Recording Status:", st.session_state['audio_state'])

if st.button("Generate Answer"):
    # Get the current value from the text input using its key
    user_input = st.session_state.get('prompt_input', '')

    # Check if the input is empty or just whitespace before processing
    if not user_input or user_input.isspace():
        st.warning("Please enter some text or record audio before generating an answer.")
    else:
        st.spinner("Generating answer...")

        # Detect the language of the user input
        detected_language = detect_language(user_input)
        st.info(f"Detected input language: {detected_language}")

        # Translate to English if the detected language is not English
        if detected_language != "en":
            st.info(f"Translating {detected_language} text to English...")
            processed_input = translate_to_english(user_input, detected_language)
            # Handle potential translation errors
            if processed_input is None:
                 st.error("Translation failed. Cannot proceed with analysis.")
            st.info(f"Translated English text: {processed_input}")
        else:
            # If already English, use the original input
            processed_input = user_input
            st.info("Input is already English, proceeding with analysis.")

        # Use the processed_input (English text) for prediction
        new_description_ipc = processed_input
        new_description_vectorized_ipc = vectorizer_ipc.transform([new_description_ipc]).toarray()
        predicted_ipc_probs = model_ipc.predict(new_description_vectorized_ipc)
        predicted_ipc_label = label_encoder_ipc.inverse_transform([predicted_ipc_probs.argmax()])[0]
            
        
        new_description_crpc = processed_input
        new_description_vectorized_crpc = vectorizer_crpc.transform([new_description_crpc]).toarray()
        predicted_crpc_probs = model_crpc.predict(new_description_vectorized_crpc)
        predicted_crpc_label = label_encoder_crpc.inverse_transform([predicted_crpc_probs.argmax()])[0] 


        # Display the results
        st.write("Predicted IPC Section Information:")
        # Find matching rows based on the predicted label
        matching_rows_ipc = ipc_data[ipc_data['section_desc'] == predicted_ipc_label]
        if not matching_rows_ipc.empty:
            st.write(matching_rows_ipc.iloc[0])
        else:
            st.write("No matching IPC section found in data.")

        st.write("Predicted CRPC Section Information:")
        matching_rows_crpc = crpc_data[crpc_data['section_desc'] == predicted_crpc_label]
        if not matching_rows_crpc.empty:
            st.write(matching_rows_crpc.iloc[0])
        else:
            st.write("No matching CRPC section found in data.")




# Check if the audio icon button is clicked
#if audio_button_clicked:
    # Start transcribing the recorded audio
   # transcribed_text = transcribe_audio()
   # st.subheader("Transcription:")
   # st.write(transcribed_text)
   # st.text("You entered: " + transcribed_text)

# 4) File upload section
with st.form(key="file_upload_form"):
    st.subheader("File Upload Section")
    uploaded_file = st.file_uploader(
        "Upload a pdf, docx, or txt file",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT",
    )
    file_submit = st.form_submit_button("Submit File")

    if file_submit and uploaded_file is not None:
        file_suffix = uploaded_file.name.split(".")[-1].lower()

        if file_suffix == "pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            st.write("Extracting text from PDF...")
            # Use the updated process_uploaded_file function for PDFs
            text_result_pages = process_uploaded_file(temp_file_path)

            if text_result_pages:
                st.subheader("Extracted and Translated Text:")
                for i, page_text in enumerate(text_result_pages, start=1):
                    st.text_area(f"Original Text (Page {i}):", page_text, height=150)

                    # Use the updated translate_to_english function
                    translated_english_text = translate_to_english(page_text, source_language="auto")

                    if translated_english_text:
                        st.text_area(f"Translated English Text (Page {i}):", translated_english_text, height=150)
                    else:
                        st.warning(f"Translation failed for Page {i}.")
            else:
                st.warning("Could not extract text from the PDF file.")

            # Clean up the temporary file
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"Terminal: Cleaned up temporary file: {temp_file_path}")

        elif file_suffix in ["docx", "txt"]:
             st.warning("DOCX and TXT processing not yet implemented. Please upload a PDF.")
            # You would add logic here for DOCX and TXT extraction and then translation
        else:
            st.warning("Unsupported file type. Please upload a PDF, DOCX, or TXT.")

if 'text_hindi' not in st.session_state:
    st.session_state.text_hindi = ""

with st.form(key="image_upload_form"):
    st.subheader("Image Upload Section")
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        help="Supported image formats: JPG, JPEG, PNG",
    )
    image_submit = st.form_submit_button("Submit Image")

    if image_submit and uploaded_image is not None:
        try:
            img = Image.open(uploaded_image).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)
            # Perform OCR
            extracted_text = pytesseract.image_to_string(img, lang='hin') # Assuming Hindi for OCR, adjust if needed

            st.header("Extracted Text (Hindi):")
            st.write(extracted_text)

            if extracted_text:
                # Translate extracted text to English using the updated function
                translated_english_text = translate_to_english(extracted_text, source_language="hi") # Assuming Hindi source
                st.header("English Translation:")
                if translated_english_text:
                    st.write(translated_english_text)
                else:
                    st.warning("English translation failed.")
            else:
                st.warning("Could not extract text from the image.")
        except Exception as e:
             st.error(f"An error occurred during image processing: {e}")
             print(f"Terminal: Image processing error: {e}")

# 3) Advanced sections
with st.expander("Advanced Options"):
    st.subheader("Advanced Options Section")
    return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
    show_full_doc = st.checkbox("Show parsed contents of the document")

def load_and_prep(file):
    img = Image.open(file).convert("RGB")
    return img

def translate_hindi_to_english(text):
    # This function is no longer needed as translate_to_english handles it
    pass