from transformers import pipeline
import torch
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
import numpy as np
from datasets import load_metric
from huggingface_hub import notebook_login
notebook_login()
# we have to define a trainer
from transformers import TrainingArguments, Trainer


data = ["I love you","I hate you"]

specific_model = pipeline(model="nlptown/bert-base-multilingual-uncased-sentiment")

response = specific_model(data)

# response = sentiment_pipeline(data)

for res in response:
    print("SCORE")
    print(res['score'])
    
# =====================================
imdb = load_dataset("imdb")

torch.cuda.is_available()

small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(3000))])
small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(300))])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples['text'],truncation=True)

tokenized_train = small_train_dataset.map(preprocess_function,batched=True)

tokenized_test = small_test_dataset.map(preprocess,batched=True)

# Let's use a data_collator to convert your training samples to PyTorch tensors and concatenate them with the correct amount of padding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)


# Define a function to compute metrics during evaluation
def compute_metrics(eval_pred):
    # Load accuracy and f1 metrics from the datasets library
    # These metrics help evaluate the performance of the model
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")

    # Unpack predictions and labels from the evaluation prediction
    logits, labels = eval_pred

    # Predict the class with the highest probability
    # This is done by selecting the index with the maximum value along the last axis
    predictions = np.argmax(logits, axis=-1)

    # Compute accuracy and f1 score using the loaded metrics
    # The accuracy metric compares predicted labels with true labels
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    # The f1 score is a measure of a model's precision and recall
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]

    # Return the computed metrics
    # Metrics are returned as a dictionary for easy interpretation and reporting
    return {"accuracy": accuracy, "f1": f1}

# define triner instance

repo_name = "finetuning-sentiment-model-3000-samples"

training_args = TrainingArguments(
    output_dir=repo_name,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch",
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
