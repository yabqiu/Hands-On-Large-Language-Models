import transformers
from transformers import pipeline

transformers.logging.set_verbosity_error()

pipe = pipeline(
    "text-generation",
    model="microsoft/Phi-3-mini-4k-instruct",
    device_map="mps",
    torch_dtype="auto",
    max_new_tokens=100,
)

messages = [{
    "role": "user",
    "content": "Write an email apologizing to Sarah for the tragic gardening mishap. Explain hwo it happened."
}]

output = pipe(messages)
print(output[0]["generated_text"][-1]["content"])
