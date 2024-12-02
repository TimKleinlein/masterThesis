from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle as pkl

with open('data/transcriptions.pkl', 'rb') as file:
    transcriptions = pkl.load(file)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("best_model")

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("best_model")
model.to("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(text):
    return tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")

# Example input text
for i in ['I am an impostor, you got me.', 'Indeed i am a crewmate.']:
    input_text = i
    encoded_input = preprocess(input_text)
    encoded_input = {key: value.to("cuda" if torch.cuda.is_available() else "cpu") for key, value in encoded_input.items()}

    with torch.no_grad():
        outputs = model(**encoded_input)
        logits = outputs.logits

    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)

    # Get the predicted class
    predicted_class = torch.argmax(probabilities, dim=1).item()

    # Print the predicted class
    print(f"Predicted class: {predicted_class}")
