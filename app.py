from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import pytesseract
import fitz
from transformers import BartTokenizer, BartForConditionalGeneration
from googletrans import Translator
import os
import asyncio
import re

# Initialize Flask app
app = Flask(__name__)

# OCR Setup
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize BART model for summarization
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Initialize the Translator for Translation
translator = Translator()

# Function to process OCR from PDF or image
def process_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == ".pdf":
        return process_pdf(file_path)
    elif file_extension in [".png", ".jpg", ".jpeg"]:
        return process_image(file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, image, or Word document.")

def process_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    extracted_text = ""

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap(dpi=300)
        image_path = f'temp_page_{page_num + 1}.png'
        pix.save(image_path)
        extracted_text += process_image(image_path)

    pdf_document.close()
    return extracted_text

def process_image(image_path):
    image = Image.open(image_path)
    gray_image = image.convert('L')
    enhanced_image = ImageOps.autocontrast(gray_image)
    return pytesseract.image_to_string(enhanced_image)

# Function for abstractive summarization using BART
def abstractive_summary_bart(text, max_new_tokens=130, min_length=30):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=max_new_tokens, min_length=min_length, num_beams=5, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function for translation
async def translate_text(text, dest_lang, src_lang='en'):
    translated = await translator.translate(text, src=src_lang, dest=dest_lang)
    return translated.text

@app.route('/process_file', methods=['POST'])
def process_file_route():
    file = request.files['file']
    class_type = request.form.get('class', type=int)
    lang=request.form.get('language',type=str)
    if not file or class_type not in [1, 2, 3]:
        return jsonify({"error": "Invalid input. Provide a file and a valid class (1=OCR, 2=Summarization, 3=Translation)."}), 400

    file_path = f"./{file.filename}"
    file.save(file_path)

    try:
        # OCR
        if class_type == 1:
            extracted_text = process_file(file_path)
            return jsonify({"extracted_text": extracted_text})

        # Summarization
        elif class_type == 2:
            extracted_text = process_file(file_path)
            summary = abstractive_summary_bart(extracted_text)
            return jsonify({"summary": summary})

        # Translation
        elif class_type == 3:
            extracted_text = process_file(file_path)
            translated_text = asyncio.run(translate_text(extracted_text,lang))
            return jsonify({"translated_text": translated_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/extract_dates', methods=['POST'])
def extract_dates_route():
    file = request.files['file']
    if not file not in [1, 2, 3]:
        return jsonify({"error": "Invalid input. Provide a file and a valid class (1=OCR, 2=Summarization, 3=Translation)."}), 400

    file_path = f"./{file.filename}"
    file.save(file_path)
    text=process_file(file_path)
    if not text:
        return jsonify({"error": "No text provided for date extraction."}), 400

    # Extract dates from the text
    date_patterns = [
        r"\d{1,2}[-/]\d{1,2}[-/]\d{4}",  # Matches formats like 15/08/2025 or 15-08-2025
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",  # Matches formats like 2025/08/15
        r"\d{1,2}(th|st|nd|rd)?\s\w+\s\d{4}"  # Matches formats like 15th August 2025
    ]
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text))

    return jsonify({"dates": dates})

if __name__ == "__main__":
    app.run(debug=True)
