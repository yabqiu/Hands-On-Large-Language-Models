from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

tokens = tokenizer('Hello world')
for token in tokens.input_ids:
    print(tokenizer.decode(token), ':', token)

tokens = tokenizer('Hello world', return_tensors="pt")

output = model(**tokens)

print(output)
print(output[0].shape)
