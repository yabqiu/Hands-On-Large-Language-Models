from tensorflow.python.eager.context import num_gpus

import kick_zscaler

from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
    filename="*fp16.gguf",
    num_gpus=-1,
    n_ctx=4096,
    verbose=False
)

question = "Create a warrior with fields 'name', 'class', and 'level' for an RPG in JSON for mat."

# output = llm.create_chat_completion(
#     messages=[{"role": "user", "content": question}],
#     response_format={"type": "json_object"},
#     temperature=0
# )['choices'][0]['message']['content']
#
# print(output)
#
# llm.close()

output = llm(
    f"<|user|>\n{question}<|end|>\n<|assistant|>",
    max_tokens=256,
    stop=["<|end|>"],
    # response_format={"type": "json_object"},
    # temperature=0
)

print(output['choices'][0]['text'])
llm.close()