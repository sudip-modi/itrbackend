from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

app = FastAPI()

# Initialize NLTK (uncomment if not already downloaded)
# nltk.download('vader_lexicon')

class SentimentRequest(BaseModel):
    selectedAspect: str

def analyze_sentiments(feedback_data, aspect):
    sid = SentimentIntensityAnalyzer()
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

@app.post('/analyze')
async def analyze_feedback(sentiment_request: SentimentRequest, file: UploadFile = File(...)):
    try:
        # Your logic to read the file, perform sentiment analysis, and return results
        feedback_data = file.file.read().decode('utf-8')
        sentiments = analyze_sentiments(feedback_data, sentiment_request.selectedAspect)

        return JSONResponse(content=sentiments)
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)

if __name__ == "__main__":
    app.run(debug=True)
