import gensim.downloader as api

# 下载 Embedding (66M, glove,训练数据来自维基百科，向量大小：50）
# 其他选项如: 'word2vec-google-news-300', 可查看 https://github.com/piskvorky/gensim-data
model = api.load("glove-wiki-gigaword-50")

similarities = model.most_similar([model['king']], topn=5)
for word, score in similarities:
    print(f"{word:<10} {score:.4f}")
