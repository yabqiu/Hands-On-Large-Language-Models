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

import pandas as pd
import matplotlib.pyplot as plt

# dimensionality reduction, 从 384 降到 2
umap_model = UMAP(n_components=2, min_dist=0.0, metric="cosine", random_state=42)
reduced_embeddings = umap_model.fit_transform(embeddings)
print(reduced_embeddings.shape)  # (44949, 2)

hdbscan_model = HDBSCAN(
    min_cluster_size=50, metric="euclidean", cluster_selection_method="eom"
).fit(reduced_embeddings)
clusters = hdbscan_model.labels_

df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
df["title"] = titles
df['cluster'] = [str(c) for c in clusters]

# 选择离群点和非离群点(聚类)
clusters_df = df.loc[df.cluster != "-1", :]
outliers_df = df.loc[df.cluster == "-1", :]

# 生成图片
plt.scatter(outliers_df.x, outliers_df.y, alpha=0.05, s=2, c="grey")
plt.scatter(
    clusters_df.x, clusters_df.y, c=clusters_df.cluster.astype(int),
    alpha=0.6, s=2, cmap="tab20b"
)
plt.axis("off")

plt.savefig("cluster_chart.png", dpi=300, bbox_inches="tight", format="png", transparent=True)
