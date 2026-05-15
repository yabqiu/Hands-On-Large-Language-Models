from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from umap import UMAP  # 依赖 uv add umap-learn
from hdbscan import HDBSCAN # 依赖 uv add hdbscan

dataset = load_dataset("maartengr/arxiv_nlp")["train"] # [Titles, Abstracts, Years, Categories]

abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

# text embeddings
embedding_model = SentenceTransformer("thenlper/gte-small", device="cuda")
embeddings = embedding_model.encode(abstracts, batch_size=10, device="cuda", show_progress_bar=True)
print(embeddings.shape)  # (44949, 384)

# dimensionality reduction, 从 384 降到 5
umap_model = UMAP(n_components=5, min_dist=0.0, metric="cosine", random_state=42)
reduced_embeddings = umap_model.fit_transform(embeddings)
print(reduced_embeddings.shape)  # (44949, 5)

# clustering
hdbscan_model = HDBSCAN(
    min_cluster_size=50, metric="euclidean", cluster_selection_method="eom"
).fit(reduced_embeddings)
clusters = hdbscan_model.labels_  # clusters 中每条数据对应的聚类标签，[1,3,-1,143]
print(len(set(clusters)))  # 156 个 cluster, cluster {0,1,2,3,...,153,154,-1}