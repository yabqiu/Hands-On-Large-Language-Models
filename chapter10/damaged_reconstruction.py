import torch
import nltk
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.datasets import DenoisingAutoEncoderDataset
from transformers import AutoTokenizer

# 加载已训练的模型
embedding_model = SentenceTransformer("tsdae_embedding_model", device="cuda")

# 重建解码器
train_loss = losses.DenoisingAutoEncoderLoss(
    embedding_model,
    decoder_name_or_path="bert-base-uncased",
    tie_encoder_decoder=False
)
train_loss.decoder = train_loss.decoder.to("cuda")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def reconstruct_sentence(damaged_sentence: str) -> str:
    with torch.no_grad():
        # 受损句子 → 句向量
        embedding = embedding_model.encode(
            damaged_sentence,
            convert_to_tensor=True,
            device="cuda"
        ).unsqueeze(0)                        # [1, hidden_dim]

        # 句向量 → 还原句子
        outputs = train_loss.decoder.generate(
            encoder_hidden_states=embedding.unsqueeze(1),  # [1, 1, hidden_dim]
            decoder_input_ids=torch.tensor(
                [[tokenizer.cls_token_id]], device="cuda"
            ),
            max_length=64,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


mnli = load_dataset(
    "nyu-mll/glue", "mnli", split="train"
).select(range(50))
test_sentences = list(mnli["premise"]) + list(mnli["hypothesis"])

damaged_dataset = DenoisingAutoEncoderDataset(test_sentences)

for data in damaged_dataset:
    damaged, original = data.texts[0], data.texts[1]
    reconstructed = reconstruct_sentence(damaged)
    print(f"受损句子: {damaged}")
    print(f"原始句子: {original}")
    print(f"重建句子: {reconstructed}")
    print("-" * 60)