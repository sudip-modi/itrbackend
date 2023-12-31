from flask import Flask, request, jsonify
from transformers import pipeline
import pandas as pd
from flask_cors import CORS
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
CORS(app)
sentiment_analysis = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")

@app.route('/analyze', methods=['POST'])
def category_analyze_feedback():
    try:
        file = request.files['file']
        df = pd.read_csv(file)
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

        sentiment_counts = pd.DataFrame(results)['sentiment'].value_counts()
        return sentiment_counts.to_json()
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/analyze_datewise', methods=['POST'])
def analyze_datewise():
    try:
        file = request.files['file']
        df = pd.read_csv(file)
        results = []
        sentimentLabelColumn=[]

        for index, row in df.iterrows():
            text = row['review']
            try:
                sentiment = sentiment_analysis(text)
                sentimentLabel = sentiment[0]['label']
                sentimentLabelColumn.append(sentimentLabel)
                results.append({'text': text, 'sentimentLabel': sentimentLabel})
            except Exception as e:
                sentimentLabelColumn.append("N/A")
                print(f"Error analyzing sentiment for '{text}': {str(e)}")

        for index, rows in df.iterrows():
            df.at[index, 'sentimentLabel'] = sentimentLabelColumn[index]

        df['date'] = pd.to_datetime(df['reviewdate'],dayfirst=True)
        df['month'] = df['date'].dt.month_name()
        df1 = df.pivot_table(index='month', columns='sentimentLabel', values='review', aggfunc='count').fillna(0)
        df2 = df1.transpose()
        print("======================")
        print("DATAFRAME")
        print(df2)
        return df2.to_json()
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_aspect', methods=['POST'])
def analyze_aspect():
    try:
        # Read the CSV file from the request
        file = request.files['file']
        df = pd.read_csv(file)
        
        # Aspects to analyze
        aspects = ['Quality', 'Content', 'Instructor', 'Material', 'Engagement', 'Application', 'Structure']

        # Initialize sentiment analyzer
        sid = SentimentIntensityAnalyzer()

        # Analyze sentiment for each aspect across all reviews
        aspect_sentiments = {aspect: {'positive': 0, 'neutral': 0, 'negative': 0} for aspect in aspects}

        # Analyze sentiment for each review and aspect
        for review in df['review']:
            for aspect in aspects:
                if aspect.lower() in review.lower():
                    aspect_text = review.lower().split(aspect.lower())[1].split('.')[0]
                    analysis = sid.polarity_scores(aspect_text)['compound']

                    if analysis >= 0.05:
                        aspect_sentiments[aspect]['positive'] += 1
                    elif -0.05 < analysis < 0.05:
                        aspect_sentiments[aspect]['neutral'] += 1
                    else:
                        aspect_sentiments[aspect]['negative'] += 1

        # Prepare data for response
        response_data = {
            'aspects': aspects,
            'positive_counts': [aspect_sentiments[aspect]['positive'] for aspect in aspects],
            'neutral_counts': [aspect_sentiments[aspect]['neutral'] for aspect in aspects],
            'negative_counts': [aspect_sentiments[aspect]['negative'] for aspect in aspects]
        }

        print(jsonify(response_data))
        return jsonify(response_data)

    except Exception as e:
        print(jsonify({'error': str(e)}))
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
