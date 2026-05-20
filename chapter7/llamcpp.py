import kick_zscaler

from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
    filename="Phi-3-mini-4k-instruct-q4.gguf",
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=4096,
    verbose=False,
)

response = llm.create_chat_completion(messages=[
    {"role": "user", "content": "Hi! My name is Maarten. What is 1 + 1?"}
])

print(response['choices'][0]['message']['content'])
