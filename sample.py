import streamlit as st
import fitz  # PyMuPDF
from deep_translator import GoogleTranslator, MyMemoryTranslator
import tempfile
import os

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
        print(f"Terminal: PDF extraction error: {e}")
        return None

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
            print(f"Terminal: {service_name} translation error: {e}")
            continue
    
    st.error("All translation services failed. Please try again later.")
    return None

st.set_page_config(page_title="PDF Hindi to English Translator", layout="wide")
st.header("📄 PDF Hindi to English Translator")

st.write("Upload a Hindi PDF file and get the extracted text translated to English.")

uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"],
    help="Please upload a PDF file containing Hindi text.",
)

if uploaded_file is not None:
    st.info("File uploaded successfully! Processing...")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        st.write("Extracting text from PDF...")
        text_result_pages = process_uploaded_file(temp_file_path)

        if text_result_pages:
            st.subheader("Extracted and Translated Text:")
            for i, page_text in enumerate(text_result_pages, start=1):
                st.text_area(f"Original Text (Page {i}):", page_text, height=150)

                # Translate the extracted page text to English
                translated_english_text = translate_to_english(page_text, source_language="auto")

                if translated_english_text:
                    st.text_area(f"Translated English Text (Page {i}):", translated_english_text, height=150)
                else:
                    st.warning(f"Translation failed for Page {i}.")
        else:
            st.warning("Could not extract text from the PDF file.")

    except Exception as e:
        st.error(f"An error occurred during file processing: {e}")
        print(f"Terminal: File processing error: {e}")

    finally:
        # Clean up the temporary file
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Terminal: Cleaned up temporary file: {temp_file_path}")

# The original langdetect and translation examples are removed
# to focus sample.py on the PDF translation functionality.
# You can keep them if you prefer sample.py to be a demo of multiple features.

# from langdetect import detect, detect_langs
# from google.cloud import translate_v3
# import os
# PROJECT_ID = "translation-api-access-461408"

# def detect_language_examples():
#     # ... (previous langdetect examples) ...

# def translate_to_english_sample(text, project_id):
#     # ... (previous Google Cloud translation code) ...

# if __name__ == "__main__":
#     try:
#         print("Testing langdetect library...")
#         detect_language_examples()

#         print("\nTesting Google Cloud Translation API (Hindi to English)... ")
#         # ... (previous Google Cloud translation test code) ...

#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
