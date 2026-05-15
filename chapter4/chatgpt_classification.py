from sklearn.metrics import classification_report


def evaluate_performance(y_true, y_pred):
    performance = classification_report(
        y_true, y_pred,
        target_names=["Nagative Review", "Positive Review"]
    )
    print(performance)


from datasets import load_dataset
data = load_dataset("rotten_tomatoes")

import openai
from transformers.pipelines.pt_utils import KeyDataset

client = openai.OpenAI(...)

def chatgpt_generate(prompt, model="gpt-5.4-mini"):
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    ).choices[0].message.content

prompt = """Predict whether the following document is a positive or negative movie review:"

{}

If it is positive return 1 and if it is negative return 0. Do not give any other answers.
"""

y_pred = []

cc = 0
for text in KeyDataset(data["test"], "text"):
    text = chatgpt_generate(prompt.format(text))
    y_pred.append(int(text))

evaluate_performance(data["test"]["label"], y_pred)





