from PIL import Image

import kick_zscaler

from transformers import AutoProcessor, Blip2ForConditionalGeneration

model_name = "Salesforce/blip2-opt-2.7b"
blip_processor = AutoProcessor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(model_name).to("mps")

image = Image.open("car.png").convert("RGB")
inputs = blip_processor(images=image, return_tensors="pt").to("mps")

generated_ids = model.generate(**inputs, max_new_tokens=50)
generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_text[0].strip())