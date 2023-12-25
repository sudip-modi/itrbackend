from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer,PatternAnalyzer

text = "Very bad Service offered by team"
category = ""

# ===========================

response = TextBlob(text,analyzer=PatternAnalyzer()).sentiment

response = round(response.polarity)
# ===========================

if response == -1.0:
    category = "Negative"
elif response == 0:
    category = "Neutral"
elif response == 1.0:
    category = "Positive"
    

print(category)