from sklearn.metrics import classification_report

import kick_zscaler

def evaluate_performance(y_true, y_pred):
    performance = classification_report(
        y_true, y_pred,
        target_names=["Nagative Review", "Positive Review"]
    )
    print(performance)

from datasets import load_dataset
data = load_dataset("rotten_tomatoes")

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="mps")
label_embeddings = model.encode(["A negative review", "A positive review"])

test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)
sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
y_pred = np.argmax(sim_matrix, axis=1)

evaluate_performance(data["test"]["label"], y_pred)