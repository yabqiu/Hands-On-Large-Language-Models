from PIL import Image

import kick_zscaler

from transformers import AutoProcessor, Blip2ForConditionalGeneration

model_name = "Salesforce/blip2-opt-2.7b"
blip_processor = AutoProcessor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(model_name).to("mps")
# print(model.vision_model, model.language_model)
print(model)

image = Image.open("car.png").convert("RGB")
image_pixels = blip_processor(images=image, return_tensors="pt").to("mps")["pixel_values"]
print(image_pixels.shape) # torch.Size([1, 3, 224, 224])

text = "Her vocalization was remarkably melodic"
token_ids = blip_processor(text=text, return_tensors="pt").to("mps")["input_ids"][0]
tokens = blip_processor.tokenizer.convert_ids_to_tokens(token_ids)
print(tokens) # ['</s>', 'Her', 'Ġvocal', 'ization', 'Ġwas', 'Ġremarkably', 'Ġmel', 'odic']