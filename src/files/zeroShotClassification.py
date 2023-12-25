from transformers import pipeline
classifier = pipeline(task="zero-shot-classification",
#device=0,
model="facebook/bart-large-mnli"
)
classifier(text,["positive","negative",'neutral'],multi_class=True)