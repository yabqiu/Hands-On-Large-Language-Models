from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="mps",  # "cuda" if Nvidia, "cpu" if neither
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False
)

output = generator([
    {"role": "user", "content": "Create a funny joke about chickens."}
])

print(output[0]["generated_text"]) # Why did the chicken join the band? Because it had the drumsticks!

print(output) # [{'generated_text': ' Why did the chicken join the band? Because it had the drumsticks!'}]

# messages = [
#     {"role": "user", "content": "Create a funny joke about chickens."}
# ]
#
# inputs = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     return_tensors="pt",
#     return_dict=True,
# ).to(model.device)
#
# output = model.generate(**inputs, max_new_tokens=200)
#
# prompt_len = inputs["input_ids"].shape[-1]
# response = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
# print(response)
