from googletrans import Translator

# Initialize the Translator
translator = Translator()

def translate_text(text, src_lang='en', dest_lang='hi'):
    translated = translator.translate(text, src=src_lang, dest=dest_lang)
    return translated.text
