import time

import kick_zscaler
from sklearn.metrics import classification_report


def evaluate_performance(y_true, y_pred):
    performance = classification_report(
        y_true, y_pred,
        target_names=["Nagative Review", "Positive Review"]
    )
    print(performance)

from datasets import load_dataset
data = load_dataset("rotten_tomatoes")


from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="mps")

train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

print(train_embeddings.shape)
print(test_embeddings.shape)

clf = LogisticRegression(random_state=42)
clf.fit(train_embeddings, data["train"]["label"])

y_pred = clf.predict(test_embeddings)
evaluate_performance(data["test"]["label"], y_pred)