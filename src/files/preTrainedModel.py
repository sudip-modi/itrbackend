from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy 
import numpy as np
from scipy.special import softmax
import tensorflow as tf

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

#distilbert

model_name_d="distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name_d)

config = AutoConfig.from_pretrained(model_name_d)

model = AutoModelForSequenceClassification.from_pretrained(model_name_d)
#bert

from transformers import BertTokenizer, BertForSequenceClassification,BertConfig

model_name_b='bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(model_name_b)

config = BertConfig.from_pretrained(model_name_b)

model = BertForSequenceClassification.from_pretrained(model_name_b)

def sentiment_labels(text):
    encoded_input = tokenizer(text, padding=True,truncation=True,max_length=512, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    return config.id2label[ranking[0]]

text=" Very Good Service offered by Team."

response = sentiment_labels(text)

print(response)