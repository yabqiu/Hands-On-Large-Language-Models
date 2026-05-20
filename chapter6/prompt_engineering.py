import kick_zscaler

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig

model_name = "google/gemma-4-E4B-it"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="mps",
    # trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pip = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False,
)

messages = [
    {"role": "user", "content": "Create a funny joke about chickens."}
]

output = pipe(messages)
print(output[0]["generated_text"])