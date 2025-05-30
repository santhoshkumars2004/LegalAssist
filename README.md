# LegalAssist - AI-Powered Legal Section Suggestion System

A Streamlit-based web application that helps users identify relevant IPC and CRPC sections based on their input. The application supports multiple input methods including text, audio, PDF documents, and images.

## Features

- Text input with language detection and translation
- Audio recording and transcription
- PDF document processing
- Image OCR with Hindi text extraction
- Automatic translation from Hindi to English
- IPC and CRPC section suggestions
- Sentiment analysis
- Advanced search options

## Setup

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run main.py
```

## Environment Variables

Create a `.env` file with the following variables:
```
GOOGLE_API_KEY=your_google_api_key
```

## Deployment

The application is deployed on Vercel. Visit the live site at: [Your Vercel URL]

## Technologies Used

- Streamlit
- TensorFlow
- NLTK
- PyMuPDF
- Deep Translator
- Google Generative AI
- Tesseract OCR
- Pandas
- NumPy
- scikit-learn

## License

MIT License 