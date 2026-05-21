import numpy as np

import kick_zscaler

from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import torch.nn.functional as F

# 加载模型与输入
model_id="openai/clip-vit-base-patch32"
clip_tokenizer = CLIPTokenizer.from_pretrained(model_id)
clip_processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id) # CLIPModel 内含两个编码器：文本编码器和图像编码器(ViT)

def image_caption_score(image_file: str, caption: str):
    image = Image.open(image_file).convert("RGB")

    inputs = clip_tokenizer(caption, return_tensors="pt")
    print("caption inputs: ", inputs)
    # caption inputs:  {'input_ids': tensor([[49406,   320,  6829,  1629,   530,   518,  2583, 49407]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]])}

    print("caption tokens: ", clip_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
    # caption tokens:  ['<|startoftext|>', 'a</w>', 'puppy</w>', 'playing</w>', 'in</w>', 'the</w>', 'snow</w>', '<|endoftext|>']

    # 整个文本转换成了一个维度为 512 的嵌入向量，这是一个文本嵌入操作
    text_embedding = model.get_text_features(**inputs).pooler_output
    print("caption embedding shape: ", text_embedding.shape)
    # caption embedding shape:  torch.Size([1, 512])

    processed_image = clip_processor(text=None, images=image, return_tensors="pt")["pixel_values"]
    print("processed_image shape: ", processed_image.shape)
    # processed_image shape:  torch.Size([1, 3, 224, 224])
    # 1: batch, 3: RGB 三个通道，224*224: 图像被缩小/裁剪成了 224*224 的大小

    # 输出采样的图片区域，是一个 224*224 的 RGB 图片
    # 为什么是 224*224, 因为模型 openai/clip-vit-base-patch32", patch 大小是 32*32, 而 224*224 包含 7*7 个 patch
    save_sample_image(processed_image, image_file)

    # 上面 49 个 patch 铺平后被编码成了一个维度为 512 的嵌入向量，这是一个图像嵌入操作
    image_embedding = model.get_image_features(processed_image).pooler_output
    print("image_embedding shape: ", image_embedding.shape)
    # image_embedding shape:  torch.Size([1, 512])

    # 计算它们的相似度
    score = F.cosine_similarity(image_embedding, text_embedding)
    print(f"{image_file}->{caption}: {score.tolist()[0]:.2f}")


def save_sample_image(processed_image: np.ndarray, image_file: str):
    img = processed_image.squeeze(0)          # [3, 224, 224] 去掉了第一维
    img = img.permute(1, 2, 0).numpy()        # [224, 224, 3]，CHW → HWC, 通道优先变换为高度优先, C: Channel

    img_np = (img - img.min()) / (img.max() - img.min()) * 255
    img_np = img_np.astype(np.uint8)

    Image.fromarray(img_np).save("sample-" + image_file)


if __name__ == '__main__':
    image_files = ["vit-split-image.png", "cat.png", "car.png"]
    captions = [
        "a puppy playing in the snow",
        "a pixelated image of a cute cat",
        "A supercar on the road \nwith the sunset in the background"
    ]
    for image_file in image_files:
        for caption in captions:
            image_caption_score(image_file, caption)