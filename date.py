from flask import Flask, request, jsonify
from transformers import BartForConditionalGeneration, BartTokenizer

# Initialize the Flask app
app = Flask(__name__)

# Load the BART model and tokenizer
model_name = "facebook/bart-large"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def classify_sentences(text):
    # Split text into sentences
    sentences = text.split('\n')
    classifications = []

    # Iterate over sentences to classify each one
    for sentence in sentences:
        if sentence.strip():  # Ignore empty lines
            # Preprocess the sentence for the model
            inputs = tokenizer("Classify the following: " + sentence, return_tensors="pt", max_length=1024, truncation=True)

            # Generate the output
            outputs = model.generate(inputs['input_ids'], max_length=50, num_beams=5, early_stopping=True)

            # Decode the output
            classification = tokenizer.decode(outputs[0], skip_special_tokens=True)
            classifications.append({
                'sentence': sentence,
                'classification': classification
            })

    return classifications

@app.route('/classify_text', methods=['POST'])
def classify_text():
    # Get text from the POST request
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Get the classification results
    classifications = classify_sentences(text)

    return jsonify({"classifications": classifications})

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)
