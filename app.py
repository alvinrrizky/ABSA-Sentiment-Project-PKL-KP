# Load the libraries
from fastapi import FastAPI, HTTPException
from joblib import load
import predictor
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ABSA_SentimentMultiEmiten.model.bert import bert_ABSA

# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to Your Sentiment Classification FastAPI"}

@app.post("/predict_sentiment")
def predict_sentiment(news, aspect):
    outputs = []
    polarity = ""

    if(not(news)):
        raise HTTPException(status_code=400, 
                            detail = "Please Provide a valid text message")

    # Preprocessing
    final_sentences = predictor.preprocessing_text(news, aspect)
    
    pretrain_model_name = "indolem/indobert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name)

    #Predicts
    i = 0
    sentiments = ["Negative", "Neutral", "Positive"]
    while (i < (len(final_sentences))):
        x, y, z = predictor.predict(final_sentences[i] , aspect, tokenizer)

        y_str = str(y)
        sentiment = sentiments[int(y_str[8])]
        outputs.append({
            "sentence": final_sentences[i],
            "sentiment": sentiment
        })

        i = i+1
        
    return outputs