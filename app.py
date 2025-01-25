import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image, ImageOps
import fitz
from docx import Document
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import joblib
from googletrans import Translator

app = Flask(__name__)

# Load the necessary models
bart_model_name = "facebook/bart-large"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)

translator = Translator()

SUPPORTED_EXTENSIONS = ['.pdf', '.png', '.jpg', '.jpeg', '.docx']

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

def process_word_doc(docx_path):
    doc = Document(docx_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def process_file(file_path, processing_type):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    if processing_type == 1:
        if file_extension == ".pdf":
            return process_pdf(file_path)
        elif file_extension in [".png", ".jpg", ".jpeg"]:
            return process_image(file_path)
        else:
            raise ValueError("Invalid file type for OCR.")
    elif processing_type == 2:
        return abstractive_summary_bart(file_path)
    elif processing_type == 5:
        return translate_text(file_path)
    else:
        raise ValueError("Unsupported processing type.")

def abstractive_summary_bart(text, max_new_tokens=130, min_length=30):
    inputs = bart_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = bart_model.generate(inputs['input_ids'], max_length=max_new_tokens + len(inputs['input_ids'][0]), min_length=100, num_beams=5, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def translate_text(text):
    translated = translator.translate(text, src='en', dest='hi')
    return translated.text

@app.route('/process', methods=['POST'])
def process_uploaded_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    try:
        file_path = secure_filename(file.filename)
        file.save(file_path)
        print(f"File saved locally: {file_path}")
        processing_type = int(request.form.get('processing_type', 1))
        processed_text = process_file(file_path, processing_type)
        os.remove(file_path)
        return jsonify({"processed_text": processed_text})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
