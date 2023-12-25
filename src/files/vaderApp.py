from flask import Flask, request, jsonify
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os
from flask_cors import CORS


# nltk.download('vader_lexicon')

app = Flask(__name__)
CORS(app)

def analyze_sentiments(feedback_data, aspect):
    sid = SentimentIntensityAnalyzer()
    sentiments = {'Positive': 0, 'Neutral': 0, 'Negative': 0}

    sentences = nltk.sent_tokenize(feedback_data)

    for sentence in sentences:
        print(sentence)
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
        # This is a placeholder using NLTK for simplicity
        feedback_data = file.read().decode('utf-8')
        sentiments = analyze_sentiments(feedback_data, aspect)

        return jsonify(sentiments)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
