from flask import Flask, render_template, request, url_for, jsonify
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import DataRequired
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from heapq import nlargest
import string
from collections import Counter
from googletrans import Translator
from gtts import gTTS
import os
import speech_recognition as sr
import smtplib
from email.message import EmailMessage

from chat import get_response

# Initialize Flask App
app = Flask(__name__)
app.secret_key = 'b83a1e0ea4e74d22c5d6a3a0ff5e6e66'

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Download NLTK Data
nltk.download('stopwords')
nltk.download('punkt')
import fitz  # PyMuPDF for PDF text extraction
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from pydub import AudioSegment
AudioSegment.converter = "C:\\path\\to\\ffmpeg.exe"
# Configure Upload Folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit to 16MB
# ✅ Function to Convert Speech to Text (Renamed to Avoid Conflict)

# ✅ Function to Convert Speech to Text (Direct WAV Processing)
def process_speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError:
        return "Error with the speech recognition service."
    except Exception as e:
        return f"Error processing audio: {str(e)}"
# Function to Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"
    return text

# Define Forms
class SummarizationForm(FlaskForm):
    text = StringField('Enter the text', validators=[DataRequired()])
    num_words = StringField('Enter number of words for summary', validators=[DataRequired()])
    language = SelectField('Select Language', choices=[
        ('en', 'English'), ('fr', 'French'), ('es', 'Spanish'),
        ('de', 'German'), ('hi', 'Hindi'), ('zh-cn', 'Chinese'),
        ('it', 'Italian'), ('pt', 'Portuguese'), ('ja', 'Japanese'),
        ('ru', 'Russian'), ('ko', 'Korean'), ('ar', 'Arabic'),
        ('tr', 'Turkish'), ('nl', 'Dutch'), ('th', 'Thai'),
        ('sv', 'Swedish'), ('te', 'Telugu'), ('ta', 'Tamil'),
        ('kn', 'Kannada'), ('bn', 'Bengali'), ('ur', 'Urdu'),
        ('fa', 'Persian'), ('vi', 'Vietnamese'), ('pl', 'Polish'),
        ('he', 'Hebrew'), ('id', 'Indonesian')
    ], validators=[DataRequired()])
    submit = SubmitField('Summarize & Translate')


# ✅ Fixed Summarization Route
@app.route('/summarization', methods=['GET', 'POST'])
def summarization():
    form = SummarizationForm()
    summary, translated_text, audio_file, pdf_text = None, None, None, None
    selected_language = 'en'  # Default to English if no language is selected

    if request.method == 'POST':
        if 'pdf_file' in request.files and request.files['pdf_file'].filename != '':
            pdf_file = request.files['pdf_file']
            filename = secure_filename(pdf_file.filename)
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            pdf_file.save(pdf_path)

            # Extract text from PDF
            pdf_text = extract_text_from_pdf(pdf_path)
            num_words = request.form.get('num_words', 100)  # Default word count if not provided
            selected_language = request.form.get('language', 'en')  # Default to English
            summary = summarize_text(pdf_text, num_words)

        elif form.validate_on_submit():
            text = form.text.data
            num_words = form.num_words.data
            selected_language = form.language.data
            summary = summarize_text(text, num_words)

        if summary:
            summarized_text = " ".join(summary)
            translated_text = translate_text(summarized_text, selected_language)
            audio_file = generate_audio(translated_text, selected_language)

    return render_template('summarization.html', form=form, summary=summary, translated_text=translated_text,
                           audio_file=audio_file, pdf_text=pdf_text)


class EmailForm(FlaskForm):
    recipient = StringField('Recipient Email', validators=[DataRequired()])
    subject = StringField('Subject', validators=[DataRequired()])
    message = StringField('Message', validators=[DataRequired()])
    submit = SubmitField('Send Email')

import asyncio

def translate_text(text, target_language):
    translator = Translator()
    try:
        # Run the async translate coroutine in a synchronous manner
        translated = asyncio.run(translator.translate(text, dest=target_language))
        return translated.text
    except Exception as e:
        return f"Translation Error: {str(e)}"

# Function to Summarize Text
def summarize_text(text, num_words):
    try:
        num_words = int(num_words)
    except ValueError:
        return ["Invalid input: Please enter a numeric value."]

    def clean_text(text):
        stop_words = set(stopwords.words('english'))
        words = [word for word in word_tokenize(text) if word.isalnum() and word.lower() not in stop_words]
        return " ".join(words)

    text = clean_text(text)
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ['PROPN', 'ADJ', 'NOUN', 'VERB']]
    freq = Counter(keywords)
    max_freq = freq.most_common(1)[0][1] if freq else 1
    for word in freq.keys():
        freq[word] = freq[word] / max_freq

    sent_strength = {sent: sum(freq.get(word.text, 0) for word in sent) for sent in doc.sents}
    summarized_sentences = nlargest(4, sent_strength, key=sent_strength.get)

    summary = []
    word_count = 0
    for sentence in summarized_sentences:
        words = str(sentence).split()
        if word_count + len(words) <= num_words:
            summary.append(str(sentence))
            word_count += len(words)
        else:
            remaining = num_words - word_count
            if remaining > 0:
                summary.append(" ".join(words[:remaining]))
            break

    return summary

# Function to Convert Speech to Text
def speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError:
        return "Error with the speech recognition service."

# Function to Convert Text to Speech
def generate_audio(text, language='en'):
    tts = gTTS(text=text, lang=language)
    audio_file = "static/audio_output.mp3"
    tts.save(audio_file)
    return audio_file


# Function to Send Email
def send_email(recipient, subject, message):
    sender_email = "sathiya.adventure@gmail.com"
    sender_password = "fulp qcpt nbtq msma"

    msg = EmailMessage()
    msg.set_content(message)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return "Email sent successfully!"
    except Exception as e:
        return f"Error sending email: {str(e)}"


# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')
@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)
# ✅ Route for Speech-to-Text (Direct WAV Processing)
@app.route('/speech_to_text', methods=['GET', 'POST'])
def speech_to_text_page():
    converted_text = None
    if request.method == 'POST':
        if 'audio' not in request.files:
            return jsonify({"converted_text": "No audio file provided."})

        audio_file = request.files['audio']
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], "recorded_audio.wav")  # ✅ Save directly as WAV
        audio_file.save(audio_path)

        # ✅ Process the WAV file directly
        converted_text = process_speech_to_text(audio_path)

    return render_template('speech_to_text.html', converted_text=converted_text)

@app.route('/text_to_speech', methods=['GET', 'POST'])
def text_to_speech_page():
    audio_file = None
    if request.method == 'POST':
        text = request.form['text']
        audio_file = generate_audio(text)
    return render_template('text_to_speech.html', audio_file=audio_file)


@app.route('/email', methods=['GET', 'POST'])
def email_page():
    form = EmailForm()
    response = None
    if form.validate_on_submit():
        response = send_email(form.recipient.data, form.subject.data, form.message.data)
    return render_template('email.html', form=form, response=response)


if __name__ == '__main__':
    app.run(debug=True)
