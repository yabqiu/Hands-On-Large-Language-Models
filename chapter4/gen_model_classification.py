import kick_zscaler
import transformers
from sklearn.metrics import classification_report

transformers.logging.set_verbosity_error()


def evaluate_performance(y_true, y_pred):
    performance = classification_report(
        y_true, y_pred,
        target_names=["Nagative Review", "Positive Review"]
    )
    print(performance)


from datasets import load_dataset

data = load_dataset("rotten_tomatoes")

from transformers.pipelines.pt_utils import KeyDataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)

model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    # device_map="mps",
    torch_dtype="auto",
)

prompt = "Is the following sentence positive or negative?"

y_pred = []

for text in KeyDataset(data["test"], "text"):
    inputs = tokenizer(f"{prompt} {text}", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=10)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    y_pred.append(0 if text == "negative" else 1)

evaluate_performance(data["test"]["label"], y_pred)
