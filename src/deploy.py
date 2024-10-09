from flask import Flask, request, jsonify
import pickle

# Load your tuned sentiment model
with open('tuned_sentiment_model.pkl', 'rb') as file:
    sentiment_model = pickle.load(file)

app = Flask(__name__)

# Define the API endpoint to process sentiment analysis
@app.route('/predict-sentiment', methods=['POST'])
def predict_sentiment():
    try:
        # Get the input text from the request
        data = request.get_json()
        input_text = data.get('text', '')

        if not input_text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Predict the sentiment using the loaded model
        # Assuming the model has a `predict` or similar method
        sentiment = sentiment_model.predict([input_text])[0]
        
        # Return the result
        return jsonify({'sentiment': sentiment})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
