import kick_zscaler

image_files = ["puppy.png", "cat.png", "car.png"]
captions = [
    "a puppy playing in the snow",
    "a pixelated image of a cute cat",
    "A supercar on the road \nwith the sunset in the background"
]

from PIL import Image

images = [Image.open(image_file).convert("RGB") for image_file in image_files]

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("clip-ViT-B-32")

image_embeddings = model.encode(images)
text_embeddings = model.encode(captions)

sim_matrix = util.cos_sim(
    image_embeddings, text_embeddings
)

print(sim_matrix)