# summarizer.py
from transformers import BartTokenizer, BartForConditionalGeneration

def abstractive_summary_bart(text, max_new_tokens=130, min_length=30):
    model_name = "facebook/bart-large-cnn"  
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    input_text = f"{text}" 
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    summary_ids = model.generate(inputs['input_ids'], 
                                 max_length=max_new_tokens + len(inputs['input_ids'][0]), 
                                 min_length=min_length, 
                                 num_beams=5, 
                                 early_stopping=True)
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
