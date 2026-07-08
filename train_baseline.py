from data import get_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

train_df, test_df = get_data()

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train = vectorizer.fit_transform(train_df["headline"])
X_test = vectorizer.transform(test_df["headline"])

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_df["is_sarcastic"])

preds = clf.predict(X_test)
print(classification_report(test_df["is_sarcastic"], preds))

joblib.dump(clf, "baseline_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
