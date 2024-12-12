# pdf-to-audiobook-V2

Here is a more detailed and comprehensive README for your project:

```markdown
# Advanced Emotional Audiobook Generator

The **Advanced Emotional Audiobook Generator** is a Python-based application that converts PDF files into audiobooks while incorporating emotional expressions into the narration. This tool leverages cutting-edge natural language processing and text-to-speech technologies to create a rich and engaging audiobook experience.

## Author Details

- **Author Name**: Samir Yogendra Meshram  
- **Email**: [sameerymeshram@gmail.com](mailto:sameerymeshram@gmail.com)  
- **GitHub**: [SamirYMeshram](https://github.com/SamirYMeshram)  

---

## Key Features

1. **Emotion Detection**  
   - Uses a pre-trained model (`j-hartmann/emotion-english-distilroberta-base`) from Hugging Face to analyze emotions within the text.  
   - Identifies key emotions like *joy*, *sadness*, *anger*, and more.  

2. **Text-to-Speech (TTS)**  
   - Converts text to speech using `pyttsx3` for offline and customizable synthesis.  
   - Narrates the text in a clear and expressive tone based on detected emotions.  

3. **PDF Text Extraction**  
   - Extracts text from PDFs using `pdfplumber`.  
   - Handles image-based PDFs with OCR capabilities provided by `easyocr`.  

4. **Emotion Visualization**  
   - Displays a bar chart showing the frequency distribution of emotions using Matplotlib.  

5. **Graphical User Interface (GUI)**  
   - Simple and user-friendly interface built with Tkinter.  
   - Features include file upload, progress tracking, and status notifications.  

6. **Threading for Performance**  
   - Ensures the GUI remains responsive during processing with multithreading.

---

## Installation Guide

### Prerequisites

Ensure you have the following software installed:  
- **Python**: Version 3.8 or higher  
- **Git**: To clone the repository  

Install the required Python packages listed below.  

### Installation Steps

1. **Clone the Repository**  
   Open a terminal and run:  
   ```bash
   git clone https://github.com/SamirYMeshram/AdvancedAudiobookGenerator.git
   cd AdvancedAudiobookGenerator
   ```

2. **Install Required Libraries**  
   Install the dependencies using pip:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK Resources**  
   Download the tokenizer resource used for splitting text into sentences:  
   ```bash
   python -m nltk.downloader punkt
   ```

4. **Emotion Model Setup**  
   - Place the pre-trained model in the `./emotion_model` directory (optional).  
   - Alternatively, the application will download the model automatically during runtime.  

---

## Usage Instructions

1. **Run the Application**  
   Start the application by running the following command:  
   ```bash
   python main.py
   ```

2. **Using the GUI**  
   - **Upload a PDF**: Click the "Upload PDF" button and select a PDF file from your system.  
   - **Track Progress**: Monitor the progress using the progress bar and status messages.  
   - **View Visualizations**: Observe the emotion distribution bar chart displayed during processing.  

3. **Output**  
   - The generated audiobook will be saved in the `./audiobooks` directory.  
   - The filename will follow the format: `audiobook_<timestamp>.mp3`.

---

## File Structure

```
AdvancedAudiobookGenerator/
│
├── audiobooks/                  # Directory for saving generated audiobooks
├── emotion_model/               # Local directory for emotion detection model (optional)
├── advanced_audiobook_generator.log  # Log file for tracking application events
├── requirements.txt             # List of required Python libraries
├── main.py                      # Main application script
├── README.md                    # Project documentation
└── nltk_data/                   # Directory for NLTK data (downloaded dynamically)
```

---

## Requirements

Install the following Python libraries using pip:  

```text
- pdfplumber
- easyocr
- torch
- transformers
- pyttsx3
- nltk
- matplotlib
- numpy
- scikit-learn
- Pillow
- pdf2image
- tk
```

To install all dependencies at once:  
```bash
pip install -r requirements.txt
```

---

## Known Issues

1. **GPU Usage**:  
   If a GPU is available, ensure proper configuration for optimal performance with `torch`.  

2. **OCR Performance**:  
   For image-based PDFs, ensure `easyocr` is configured correctly and dependencies like `tesseract` are installed if required.  

3. **Voice Quality**:  
   `pyttsx3` may offer limited voice options depending on the platform. Ensure suitable voices are installed on your system.  

---

## Future Enhancements

- **Multilingual Support**: Expand the tool to handle multiple languages in text extraction, emotion detection, and TTS.  
- **Custom Voice Selection**: Allow users to choose voices and adjust parameters like pitch and speed dynamically.  
- **Improved Visualization**: Add advanced visualizations like timelines or heatmaps for emotions.  
- **Web Version**: Develop a web-based interface for broader accessibility.  

---

## Contributions

Contributions, issues, and feature requests are welcome!  
Feel free to open an issue or submit a pull request on the GitHub repository.

---

## License

This project is open-source and available under the [MIT License](LICENSE).  

---

## Author

**Samir Yogendra Meshram**  
- Email: [sameerymeshram@gmail.com](mailto:sameerymeshram@gmail.com)  
- GitHub: [SamirYMeshram](https://github.com/SamirYMeshram)  
```

This README provides detailed project information and can guide users and contributors effectively. Let me know if you'd like any changes!
