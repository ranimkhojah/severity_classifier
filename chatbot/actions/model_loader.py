from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

# Define the global variables
tokenizer = None
model = None
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_model():
    global tokenizer, model
    model_path = '../bert_hazard_classifier'
    
    if os.path.exists(model_path):
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.to(device)
    else:
        print("Model directory contents:", os.listdir())
        raise Exception(f"Model not found in {model_path}")

# Load the model when this module is imported
load_model()
