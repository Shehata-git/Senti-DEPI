from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle

# Load your tokenizer and model (BERT for sentiment classification)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Assuming your `tuned_sentiment_model.pkl` is a saved Hugging Face model
with open('tuned_sentiment_model.pkl', 'rb') as file:
    sentiment_model = pickle.load(file)

app = Flask(__name__)

# Define a function to predict sentiment using the model
def predict_sentiment(text):
    # Tokenize the input text for the BERT model
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    
    # Perform inference using the loaded model
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    
    # Convert the predicted class (e.g., 0 or 1) into a human-readable label
    sentiment_label = "positive" if predicted_class == 1 else "negative"  # Adjust according to your class labels
    
    return sentiment_label

# Create an API endpoint for sentiment prediction
@app.route('/predict-sentiment', methods=['POST'])
def predict():
    try:
        # Get the input text from the request
        data = request.get_json()
        input_text = data.get('text', '')

        if not input_text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Get the sentiment prediction
        sentiment = predict_sentiment(input_text)
        
        # Return the result
        return jsonify({'sentiment': sentiment})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
