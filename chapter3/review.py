import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

transformers.logging.set_verbosity_error()

model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="mps",
    torch_dtype="auto",
)

print(model) # print out model layers

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=50,
    do_sample=False,
)

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain hwo it happened."

output = generator(prompt)

print(output[0]["generated_text"])