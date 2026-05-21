from PIL import Image

import kick_zscaler

from transformers import AutoProcessor, Blip2ForConditionalGeneration

model_name = "Salesforce/blip2-opt-2.7b"
blip_processor = AutoProcessor.from_pretrained(model_name)
blip_model = Blip2ForConditionalGeneration.from_pretrained(model_name).to("mps")

image = Image.open("cat.png").convert("RGB")

prompt = "Question: Write down what you see in this picture. Answer:"
inputs = blip_processor(images=image, text=prompt, return_tensors="pt").to("mps")

generated_ids = blip_model.generate(**inputs, max_new_tokens=50)
generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_text[0].strip())