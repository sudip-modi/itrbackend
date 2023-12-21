from flask import Flask, request, jsonify
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Download NLTK data (run this once)
import nltk
nltk.download('vader_lexicon')

# Initialize the Sentiment Intensity Analyzer
sid = SentimentIntensityAnalyzer()

def perform_sentiment_analysis(text):
  # Use NLTK's Sentiment Intensity Analyzer
  sentiment_score = sid.polarity_scores(text)['compound']

  # Classify the sentiment
  if sentiment_score >= 0.05:
    return 'Positive'
  elif sentiment_score <= -0.05:
    return 'Negative'
  else:
    return 'Neutral'

def aspect_based_sentiment_analysis(comment):
# Add your custom logic to analyze sentiments for specific aspects
# For simplicity, this example uses the same sentiment analysis logic as perform_sentiment_analysis
  return perform_sentiment_analysis(comment)

@app.route('/analyze', methods=['POST'])
def analyze_reviews():
  try:
    file = request.files['file']
    df = pd.read_csv(file)
    
    # Assuming 'Comment' is the column containing the text data for sentiment analysis
    aspects = ['Content', 'Instructor', 'Pacing', 'PracticalApplication', 'Engagement']
    
    for aspect in aspects:
      df[aspect] = df['Comment'].apply(aspect_based_sentiment_analysis)
    
    analyzed_data = df.to_dict(orient='records')
    
    return jsonify({'success': True, 'data': analyzed_data})
  except Exception as e:
    return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
  app.run(debug=True)