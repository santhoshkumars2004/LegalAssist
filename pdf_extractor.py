import fitz  # PyMuPDF
import streamlit as st

def process_uploaded_file(pdf_path):
    try:
        pdf_document = fitz.open(pdf_path)
        text_result = []

        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text_result.append(page.get_text())

        return text_result

    except Exception as e:
        # Log the error for debugging
        st.error(f"Error processing PDF file: {e}")
        return None

    finally:
        if pdf_document:
            pdf_document.close()
