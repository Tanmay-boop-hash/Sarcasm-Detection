import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from data import get_data

# load the fine-tuned model from disk
tokenizer = DistilBertTokenizerFast.from_pretrained("./sarcasm_model")
model = DistilBertForSequenceClassification.from_pretrained("./sarcasm_model")
model.eval()

# reload test set (same random_state=42 split as training, so it's consistent)
_, test_df = get_data()

test_texts = test_df["headline"].tolist()
test_labels = test_df["is_sarcastic"].tolist()

# batch this instead of feeding all 5.7k examples at once (memory-safer)
batch_size = 32
all_preds = []

for i in range(0, len(test_texts), batch_size):
    batch = test_texts[i:i+batch_size]
    inputs = tokenizer(batch, truncation=True, padding=True, return_tensors="pt", max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = torch.argmax(logits, dim=-1).tolist()
    all_preds.extend(preds)

# collect misclassified examples
errors = [(t, l, p) for t, l, p in zip(test_texts, test_labels, all_preds) if l != p]

print(f"Total errors: {len(errors)} out of {len(test_texts)} ({len(errors)/len(test_texts)*100:.1f}%)\n")

for text, true, pred in errors[:15]:
    label_str = lambda x: "sarcastic" if x == 1 else "not sarcastic"
    print(f"TEXT: {text}")
    print(f"TRUE: {label_str(true)} | PRED: {label_str(pred)}\n")

