from nltk.sentiment import SentimentIntensityAnalyzer

sentiment = SentimentIntensityAnalyzer()

text = "Very Good service offerd by team"

response = sentiment.polarity_scores(text)

print(response['compound'])