import json
import pandas as pd

def get_data(path="Sarcasm_Headlines_Dataset_v2.json"):
    records = [json.loads(line) for line in open(path)]
    df = pd.DataFrame(records)[["headline", "is_sarcastic"]]
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    return train_df, test_df

train_df, test_df = get_data()

# !pip install transformers datasets scikit-learn -q
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

train_ds = Dataset.from_pandas(train_df).map(
    lambda x: tokenizer(x["headline"], truncation=True, padding="max_length", max_length=64),
    batched=True
)
test_ds = Dataset.from_pandas(test_df).map(
    lambda x: tokenizer(x["headline"], truncation=True, padding="max_length", max_length=64),
    batched=True
)
train_ds = train_ds.rename_column("is_sarcastic", "labels")
test_ds = test_ds.rename_column("is_sarcastic", "labels")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds, compute_metrics=compute_metrics)
trainer.train()
model.save_pretrained("./sarcasm_model")
tokenizer.save_pretrained("./sarcasm_model")