import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pdfplumber
import easyocr
import torch
from transformers import pipeline
import pyttsx3
import numpy as np
import re
import logging
import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import nltk

# ======================================
# 1. SETUP LOGGING
# ======================================

logging.basicConfig(filename="advanced_audiobook_generator.log", level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# ======================================
# 2. NLTK SETUP
# ======================================

# Specify NLTK data path
nltk.data.path.append(r"C:\\Users\\raj turkar\\AppData\\Roaming\\nltk_data")

# Ensure 'punkt' resource is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading 'punkt' tokenizer...")
    nltk.download('punkt')

# ======================================
# 3. EMOTION DETECTION MODULE
# ======================================

class EmotionDetector:
    def __init__(self):
        # Attempt to load model locally; fallback to Hugging Face hub if not available
        model_path = "./emotion_model"  # Local directory for the model
        try:
            if os.path.exists(model_path):
                logging.info("Loading model from local directory.")
                self.pipeline = pipeline('text-classification', model=model_path, return_all_scores=True)
            else:
                logging.info("Downloading model from Hugging Face.")
                self.pipeline = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base',
                                         return_all_scores=True)
        except Exception as e:
            logging.error(f"Error loading emotion model: {e}")
            raise RuntimeError("Failed to load emotion model.")

    def analyze_emotions(self, text, progress_callback=None):
        """Analyze text structure to identify emotion-rich sections."""
        sentences = sent_tokenize(text)  # More accurate sentence segmentation
        emotion_data = []
        batch_size = 5
        total_sentences = len(sentences)

        try:
            for i in range(0, total_sentences, batch_size):
                batch = sentences[i:i + batch_size]
                batch_results = self.pipeline(batch)
                for sentence, result in zip(batch, batch_results):
                    sorted_emotions = sorted(result, key=lambda x: x['score'], reverse=True)
                    top_emotion = sorted_emotions[0]['label']
                    top_score = sorted_emotions[0]['score']
                    emotion_data.append((sentence, top_emotion, top_score))

                if progress_callback:
                    progress = int(((i + batch_size) / total_sentences) * 40) + 10
                    progress_callback(progress)

            return emotion_data
        except Exception as e:
            logging.error(f"Error in analyzing emotions: {e}")
            return [(text, 'neutral', 0.0)]

    def visualize_emotions(self, emotion_data):
        """Visualize the emotion analysis."""
        emotion_counts = defaultdict(int)
        for _, emotion, _ in emotion_data:
            emotion_counts[emotion] += 1

        plt.figure(figsize=(10, 6))
        plt.bar(emotion_counts.keys(), emotion_counts.values(), color='skyblue')
        plt.title("Emotion Distribution")
        plt.xlabel("Emotions")
        plt.ylabel("Frequency")
        plt.show()

# ======================================
# 4. IMPROVED TTS MODULE
# ======================================

class AdvancedTTS:
    def __init__(self):
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        # Select a high-quality female storytelling voice
        self.engine.setProperty('voice', voices[1].id)  # Typically index 1 is female voice
        self.engine.setProperty('rate', 150)  # Set a slower pace for storytelling
        self.engine.setProperty('volume', 1.0)

    def synthesize_audio(self, emotion_data, output_path):
        try:
            for text, emotion, score in emotion_data:
                self.engine.say(text)
                self.engine.runAndWait()

            self.engine.save_to_file("".join([text for text, _, _ in emotion_data]), output_path)
            self.engine.runAndWait()

            logging.info(f"Audiobook saved at {output_path}")

        except Exception as e:
            logging.error(f"Error during TTS synthesis: {e}")

# ======================================
# 5. PDF HANDLING MODULE
# ======================================

def extract_text_from_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages)
        if not text.strip():
            reader = easyocr.Reader(['en'], gpu=False)
            from pdf2image import convert_from_path
            images = convert_from_path(file_path)
            for img in images:
                text += " ".join(reader.readtext(np.array(img), detail=0))
        logging.info("Text extraction from PDF completed.")
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        messagebox.showerror("Error", f"Failed to extract text from PDF: {e}")
        return ""
# ======================================
# 6. ENHANCED PIPELINE INTEGRATION
# ======================================

class AdvancedAudiobookPipeline:
    def __init__(self):
        self.emotion_detector = EmotionDetector()
        self.tts = AdvancedTTS()

    def process_text(self, text, progress_callback):
        try:
            progress_callback(10)
            logging.info("Starting text analysis...")

            # Analyze emotions in the text
            emotion_data = self.emotion_detector.analyze_emotions(text, progress_callback)
            logging.info(f"Text analyzed into {len(emotion_data)} emotion segments.")

            # Visualize emotion analysis
            self.emotion_detector.visualize_emotions(emotion_data)

            progress_callback(50)

            # Synthesize audiobook
            output_dir = "./audiobooks"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = int(time.time())
            audio_path = os.path.join(output_dir, f"audiobook_{timestamp}.mp3")

            self.tts.synthesize_audio(emotion_data, audio_path)

            progress_callback(100)
            messagebox.showinfo("Success", f"Audiobook created at: {audio_path}")
        except Exception as e:
            logging.error(f"Error during audiobook generation: {e}")
            messagebox.showerror("Error", f"Failed to process audiobook: {e}")

# ======================================
# 7. GUI APPLICATION
# ======================================

class AudiobookApp:
    def __init__(self, root):
        self.root = root
        self.pipeline = AdvancedAudiobookPipeline()
        self.root.title("Advanced Emotional Audiobook Generator")
        self.root.geometry("600x400")
        self.create_widgets()

    def create_widgets(self):
        self.upload_button = tk.Button(self.root, text="Upload PDF", command=self.upload_pdf, font=("Helvetica", 16))
        self.upload_button.pack(pady=20)

        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(pady=20)

        self.status_label = tk.Label(self.root, text="", font=("Helvetica", 12))
        self.status_label.pack(pady=10)

    def update_progress(self, value):
        self.progress_var.set(value)
        self.status_label.config(text=f"Progress: {value}%")

    def upload_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if not file_path:
            logging.warning("No file selected!")
            messagebox.showerror("Error", "No file selected!")
            return

        self.status_label.config(text="Extracting text from PDF...")

        def process_pdf():
            try:
                text = extract_text_from_pdf(file_path)
                if text:
                    self.status_label.config(text="Generating audiobook...")
                    self.pipeline.process_text(text, self.update_progress)
            except Exception as e:
                logging.error(f"Error processing PDF: {e}")
                messagebox.showerror("Error", f"Failed to process PDF: {e}")

        thread = threading.Thread(target=process_pdf)
        thread.start()

# ======================================
# 8. MAIN APPLICATION LAUNCHER
# ======================================

if __name__ == "__main__":
    root = tk.Tk()
    app = AudiobookApp(root)
    root.mainloop()
