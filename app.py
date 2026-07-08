from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Sarcasm Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's actual origin before deploying
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = DistilBertTokenizerFast.from_pretrained("./sarcasm_model")
model = DistilBertForSequenceClassification.from_pretrained("./sarcasm_model")
model.eval()

class Input(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok", "message": "Sarcasm Detection API is running"}

@app.post("/predict")
def predict(inp: Input):
    if not inp.text.strip():
        return {"error": "Empty input"}

    tokens = tokenizer(inp.text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        logits = model(**tokens).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred = int(torch.argmax(probs))

    return {
        "text": inp.text,
        "is_sarcastic": bool(pred),
        "confidence": round(float(probs[pred]), 4),
        "probabilities": {
            "not_sarcastic": round(float(probs[0]), 4),
            "sarcastic": round(float(probs[1]), 4),
        }
    }
