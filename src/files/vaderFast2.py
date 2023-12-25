from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import pipeline
import pandas as pd
from typing import Optional

app = FastAPI()

# Set up the sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")

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

@app.post("/analyze")
async def analyze_feedback(file: UploadFile = File(...), selectedAspect: Optional[str] = 'Content'):
    try:
        # Read CSV file into a DataFrame
        df = pd.read_csv(file.file)

        # Assuming your CSV file has a column named 'review' that contains the text data
        feedback_data = ' '.join(df['review'].astype(str).tolist())

        # Perform sentiment analysis using distilbert
        sentiments = analyze_sentiments(feedback_data, selectedAspect)

        # Perform sentiment analysis using BERT
        results = []
        for text in nltk.sent_tokenize(feedback_data):
            try:
                sentiment = sentiment_analysis(text)
                results.append({'text': text, 'sentiment': sentiment[0]['label']})
            except Exception as e:
                print(f"Error analyzing sentiment for '{text}': {str(e)}")

        # Display the results
        for result in results:
            print(f"Text: {result['text']}\nSentiment: {result['sentiment']}\n")
        
        return JSONResponse(content=sentiments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
