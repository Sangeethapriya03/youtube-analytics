import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

class SentimentModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

    def predict(self, comments, batch_size=32):
        predictions = []
        for start in range(0, len(comments), batch_size):
            batch = comments[start:start + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            with torch.inference_mode():
                output = self.model(**inputs)
            classes = output.logits.argmax(dim=1).tolist()
            predictions.extend(self.labels[index] for index in classes)
        return predictions

def load_sentiment_model():
    return SentimentModel()

def predict_sentiment(comment, sentiment_model):
    return sentiment_model.predict([str(comment)])[0]

def add_sentiment(comments_df, sentiment_model):
    comments_df = comments_df.copy()
    comments = comments_df["comment"].astype(str).tolist()
    comments_df["sentiment"] = sentiment_model.predict(comments)
    return comments_df
