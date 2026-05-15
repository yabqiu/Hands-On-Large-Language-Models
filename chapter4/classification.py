import kick_zscaler

from transformers import pipeline
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import classification_report

def evaluate_performance(y_true, y_pred):
    performance = classification_report(
        y_true, y_pred,
        target_names=["Nagative Review", "Positive Review"]
    )
    print(performance)

data = load_dataset("rotten_tomatoes")
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

pipe = pipeline(
    model=model_path,
    tokenizer=model_path,
    top_k=None,
    return_all_scores=True,
    device_map="mps"
)

y_pred = []
classified = pipe(KeyDataset(data["test"], "text"))  # 对测试集文本进行分类, 1 或 0
for output in tqdm(classified, total=len(data["test"])):
    scores = {item["label"]: item["score"] for item in output}
    negative_score = scores["negative"]
    positive_score = scores["positive"]
    assignment = np.argmax([negative_score, positive_score]) # negative_score >= positive_score -> 0, 反之为 1
    y_pred.append(assignment)

evaluate_performance(data["test"]["label"], y_pred)  # 对测试集文本评出的 1, 0 与测试集 label 对比
