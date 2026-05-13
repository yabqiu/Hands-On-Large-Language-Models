from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="mps",  # "cuda" if Nvidia, "cpu" if neither
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Write an email apologizing to Sarah for the tragic garding mishap. Explain ho it happened.<|assistant|>"
print(tokenizer.tokenize(prompt))
tokenized = tokenizer(prompt, return_tensors="pt").to("mps")
print(tokenized.input_ids)

generate_ouput = model.generate(
    input_ids=tokenized.input_ids,
    attention_mask=tokenized.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=20
)

print(tokenizer.decode(generate_ouput[0]))