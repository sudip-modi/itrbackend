from flask import Flask, request, jsonify
from transformers import pipeline
import nltk
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Set up the inference pipeline using a model from the 🤗 Hub
sentiment_analysis = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")

def analyze_sentiments(sentence):
    try:
        # Perform sentiment analysis using BERT
        sentiment = sentiment_analysis(sentence)
        print(sentiment)
        return sentiment[0]['label']
    except Exception as e:
        print(f"Error analyzing sentiment for '{sentence}': {str(e)}")
        return 'ERROR'

@app.route('/analyze', methods=['POST'])
def analyze_feedback():
    try:
        file = request.files['file']
        aspect = request.form.get('selectedAspect')  # Get the selected aspect from the form data

        # Your logic to read the file and perform sentiment analysis on each sentence
        df = pd.read_csv(file)

        # Perform sentiment analysis on each review
        results = []
        for index, row in df.iterrows():
            text = row['review']
            try:
                sentiment = sentiment_analysis(text)
                print(sentiment)
                print(index)
                results.append({'text': text, 'sentiment': sentiment[0]['label']})
            except Exception as e:
                print(f"Error analyzing sentiment for '{text}': {str(e)}")

        # Display the results
        # for result in results:
        #     print(f"Text: {result['text']}\nSentiment: {result['sentiment']}\n")

        # Let's count the number of reviews by sentiments
        sentiment_counts = pd.DataFrame(results)['sentiment'].value_counts()
        
        print(results)
        
        print(sentiment_counts)
        # # Assuming your CSV file has a column named 'review' that contains the text data
        # sentences = []
        # for review in df['review'].astype(str):
        #     sentences.extend(nltk.sent_tokenize(review))

        # # Perform sentiment analysis on each sentence
        # results = {'Positive': 0, 'Neutral': 0, 'Negative': 0, 'Error': 0}
        # for sentence in sentences:
        #     sentiment = analyze_sentiments(sentence)
        #     results[sentiment] += 1

        return sentiment_counts.to_json()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
