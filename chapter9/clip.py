from PIL import Image

import kick_zscaler

from sentence_transformers import SentenceTransformer, util

image = Image.open("puppy.png").convert("RGB")
caption = "a puppy playing in the snow"

model = SentenceTransformer('openai/clip-vit-base-patch32')

image_embeddings = model.encode(image)
text_embeddings = model.encode(caption)
