# Sarcasm Detection: TF-IDF Baseline vs. Fine-Tuned DistilBERT

A sarcasm classifier for news headlines, built to compare a classical NLP baseline against a fine-tuned transformer — with a focus on understanding *why* the model succeeds and fails, not just reporting an accuracy number.

- **Live demo:** [https://sarcasm-detection-ruddy.vercel.app](https://sarcasm-detection-ruddy.vercel.app)
- **API (Hugging Face Space):** `https://Tanmay-24-sarcasm-detector-api.hf.space`

> Note: the backend runs on a free-tier CPU Space and may take 20-30 seconds to respond on the first request after a period of inactivity (cold start).

---

## Overview

Given a news headline, the model predicts whether it's sarcastic (Onion-style satire) or genuine (real news). Two models were built and compared:

1. **Baseline**: TF-IDF + Logistic Regression
2. **Main model**: Fine-tuned DistilBERT

The project is served through a FastAPI backend and a React frontend, with both deployed live.

## Results

| Model | Accuracy | F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 84% | 0.84 |
| Fine-tuned DistilBERT | 91.9% | 0.917 |

The transformer improves ~8 points over the baseline, confirming that sarcasm detection benefits meaningfully from contextual, pretrained language representations over pure lexical/n-gram features — though as the error analysis below shows, a sizeable chunk of the remaining errors aren't really fixable by a bigger model alone.

### A note on checkpoint selection

The model was fine-tuned for 3 epochs. Training loss dropped steadily (0.25 → 0.03) while validation loss *increased* after epoch 1 (0.21 → 0.34 → 0.39) — a classic overfitting signature, even though raw accuracy kept ticking up slightly (91.9% → 93.0%). I used Hugging Face's `load_best_model_at_end` with validation loss as the selection metric, which picked the **epoch-1 checkpoint** (91.9% accuracy, lowest validation loss) over the epoch-3 checkpoint (93.0% accuracy, but visibly more overfit and less calibrated). I chose to keep this selection rather than force epoch 3, prioritizing a better-calibrated model over a ~1-point accuracy gain that came with clear overfitting.

---

## Error Analysis

462 of 5,724 test examples were misclassified (8.1% error rate). Manually reviewing these surfaced two recurring patterns:

**1. False positives from stylistically "absurd" real news (Poe's Law).**
A large share of errors were real headlines misclassified as sarcastic — e.g. *"clay aiken gained 30 pounds eating bojangles chicken during his campaign"* or *"chris christie suggests hillary clinton was to blame for boko haram's kidnapping of hundreds of schoolgirls."* These headlines are genuinely true but read as bizarre or exaggerated, which is enough to fool the model. This suggests the model has partly learned to recognize *stylistic* patterns correlated with the source (Onion vs. HuffPost) rather than sarcasm as a human would judge it — the dataset's labels come from source, not annotated intent, so this ambiguity is baked into the data itself.

**2. False negatives requiring world knowledge or compressed irony.**
Sarcastic headlines that were missed tended to rely on context the model has no access to from the text alone — e.g. *"non-priest arrested on charges of child molestation"* (the irony depends on knowing about Catholic Church abuse scandals) or *"jury finds man guilty of murdering wife and children, but gets it"* (hinges entirely on the phrase "but gets it" implying leniency). These are hard cases even for humans without context, and point to a real ceiling on what a text-only, headline-scale model can infer.

## Limitations

- **Domain-specific.** The model was trained exclusively on news headlines and performs noticeably worse on conversational/spoken-style sarcasm. For example, *"Oh great, another Monday, exactly what I needed"* is misclassified as not sarcastic — a sentence structure that essentially never appears in the training data's headline format (third-person, no first-person conversational tone). Accuracy numbers above should be read as domain-specific, not general-purpose.
- **Label noise.** Labels are derived from source (Onion = sarcastic, HuffPost = not) rather than human sarcasm annotation, which is a reasonable proxy but not perfect — see the Poe's Law false positives above.

---

## Future Work

Extending this to conversational or social-media sarcasm (e.g. Twitter/Reddit) would require a different training distribution and likely more context-aware modeling (e.g. including reply-thread context, not just standalone text) — published results on social-media sarcasm datasets typically land in the 65-75% accuracy range, reflecting how much harder the task becomes outside the headline domain.

---

## Tech Stack

- **Modeling**: PyTorch, Hugging Face Transformers (DistilBERT), scikit-learn (TF-IDF + Logistic Regression baseline)
- **Backend**: FastAPI, served via Docker on Hugging Face Spaces
- **Frontend**: React (Vite), deployed on Vercel
- **Training**: Google Colab (T4 GPU)

## Dataset

[News Headlines Dataset for Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) by Rishabh Mishra — ~28.6k headlines sourced from The Onion (sarcastic) and HuffPost (not sarcastic). `Sarcasm_Headlines_Dataset_v2.json` is included in this repo.

---

## Project Structure

```
.
├── data.py                  # dataset loading + train/test split
├── train_baseline.py        # TF-IDF + Logistic Regression baseline
├── train_transformer.py     # DistilBERT fine-tuning (run on Colab/GPU)
├── error_analysis.py        # misclassification inspection
├── app.py                   # FastAPI inference server
├── requirements.txt
├── Sarcasm_Headlines_Dataset_v2.json
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   └── SarcasmDetector.jsx
    └── package.json
```

Note: the fine-tuned model weights (`sarcasm_model/`, ~268MB) are not included in this repo due to GitHub's file size limits, and the baseline's `.pkl` artifacts are excluded as generated files. Both are reproducible via the scripts below, and the fine-tuned weights are hosted alongside the deployed API on Hugging Face Spaces.

---

## Running Locally

**Backend**

```bash
pip install -r requirements.txt

# regenerate the baseline
python train_baseline.py

# fine-tune DistilBERT (recommended: run train_transformer.py on Google Colab with a GPU,
# then download the resulting sarcasm_model/ folder into the project root)

uvicorn app:app --reload --port 8000
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

Update `API_URL` in `src/SarcasmDetector.jsx` to point at `http://localhost:8000/predict` for local development, or the deployed Hugging Face Space URL for production.

---
