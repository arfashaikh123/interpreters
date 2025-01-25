# OCR.py
import os
import fitz
from PIL import Image, ImageOps
import pytesseract
from docx import Document

# Set the Tesseract path (adjust based on your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def process_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == ".pdf":
        return process_pdf(file_path)
    elif file_extension in [".png", ".jpg", ".jpeg"]:
        return process_image(file_path)
    elif file_extension == ".docx":
        return process_word_doc(file_path)
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

def process_word_doc(docx_path):
    doc = Document(docx_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])
