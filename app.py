from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
def analyze_sentiments(feedback_data, aspect):
    sentiment_analyzer = pipeline("sentiment-analysis",model="distilbert-base-uncased")
    sentiments = {'Positive': 0, 'Neutral': 0, 'Negative': 0}

 
    result = sentiment_analyzer(feedback_data)

    for prediction in result:
        label = prediction['label'].lower()
        score = prediction['score']

    
        if aspect == 'Content':
            if 'positive' in label:
                sentiments['Positive'] += score
            elif 'negative' in label:
                sentiments['Negative'] += score
            else:
                sentiments['Neutral'] += score
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
        aspect = request.form.get('selectedAspect')  
        feedback_data = file.read().decode('utf-8')
        sentiments = analyze_sentiments(feedback_data, aspect)

        return jsonify(sentiments)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
