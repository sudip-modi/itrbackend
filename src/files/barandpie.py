from flask import Flask, request, jsonify
from transformers import pipeline
import nltk
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Set up the inference pipeline using a model from the 🤗 Hub
sentiment_analysis = pipeline(model="distilbert-base-uncased")

def analyze_sentiments(feedback_data, aspect):
    sentiments = {'Positive': 0, 'Neutral': 0, 'Negative': 0}

    sentences = nltk.sent_tokenize(feedback_data)

    for sentence in sentences:
        # Modify this part based on the actual logic for different aspects
        if aspect == 'Content':
            if 'informative' in sentence.lower():
                sentiments['Positive'] += 1
            elif 'confusing' in sentence.lower():
                sentiments['Negative'] += 1
            else:
                sentiments['Neutral'] += 1
        elif aspect == 'Instructor':
            if 'knowledgeable' in sentence.lower():
                sentiments['Positive'] += 1
            elif 'unhelpful' in sentence.lower():
                sentiments['Negative'] += 1
            else:
                sentiments['Neutral'] += 1
        elif aspect == 'Pacing':
            if 'too fast' in sentence.lower():
                sentiments['Negative'] += 1
            elif 'well-paced' in sentence.lower():
                sentiments['Positive'] += 1
            else:
                sentiments['Neutral'] += 1
        elif aspect == 'Practical Application':
            if 'lacks practicality' in sentence.lower():
                sentiments['Negative'] += 1
            elif 'practical' in sentence.lower():
                sentiments['Positive'] += 1
            else:
                sentiments['Neutral'] += 1
        elif aspect == 'Engagement':
            if 'engaging' in sentence.lower():
                sentiments['Positive'] += 1
            elif 'boring' in sentence.lower():
                sentiments['Negative'] += 1
            else:
                sentiments['Neutral'] += 1

    return sentiments

@app.route('/analyze', methods=['POST'])
def analyze_feedback():
    try:
        file = request.files['file']
        aspect = request.form.get('selectedAspect')  # Get the selected aspect from the form data

        # Your logic to read the file, perform sentiment analysis, and return results
        feedback_data = file.read().decode('utf-8')

        # Perform sentiment analysis using distilbert
        sentiments_distilbert = analyze_sentiments(feedback_data, aspect)

        # Perform sentiment analysis using BERT
        results = []
        
        print(nltk.sent_tokenize(feedback_data))
        
        for text in nltk.sent_tokenize(feedback_data):
            try:
                sentiment = sentiment_analysis(text)
                results.append(sentiment[0]['label'])
            except Exception as e:
                print(f"Error analyzing sentiment for '{text}': {str(e)}")

        # Aggregate counts
        positive_count = results.count('POSITIVE')
        neutral_count = results.count('NEUTRAL')
        negative_count = results.count('NEGATIVE')

        # Combine the results from distilbert and BERT
        combined_sentiments = {
            'Positive': sentiments_distilbert['Positive'] + positive_count,
            'Neutral': sentiments_distilbert['Neutral'] + neutral_count,
            'Negative': sentiments_distilbert['Negative'] + negative_count
        }

        return jsonify(combined_sentiments)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
