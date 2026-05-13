import gzip

import pandas as pd
from pathlib import Path

import time
from gensim.models import Word2Vec

current_folder = Path(__file__).parent

# https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt
with gzip.open(current_folder.joinpath('train.txt.gz'), 'rt') as f:
    lines = f.readlines()[2:]
    playlist = [s.rstrip().split() for s in lines if len(s.split())>1]
    print(f"number of playlist: {len(playlist)}")
    for ids in playlist[:2]:
        print(f"number of song in this playlist: {len(ids)}: {str(ids[:10]).rstrip(']')}, ....")

total_songs = {id for ids in playlist for id in ids}
print(f"total songs: {len(total_songs)}")

# https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt
with gzip.open(current_folder.joinpath('song_hash.txt.gz'), 'rt') as f:
    lines = f.readlines()
    songs = [[field.strip() for field in s.split('\t')] for s in lines]
    songs_df = pd.DataFrame(songs, columns=['id', 'title', 'artist'])
    songs_df = songs_df.set_index('id')
    print(f"songs_df shape: {songs_df.shape}")
    print(f"songs_df sample: \n {songs_df.head()}")

time0 = time.time()
model = Word2Vec(sentences=playlist, vector_size=32, window=20, min_count=1, workers=4)
print(f"training time: {time.time() - time0:.2f}s")
print(f"model parameters: {model.wv.vectors.shape}\n")

def recommend_songs(song_id: str, topn=5):
    similarities = model.wv.most_similar(positive=song_id, topn=topn)

    print(f"similar songs of [{songs_df.loc[song_id]['title']}]:\n")

    print(f"{'No':<2} {'Id':<8} {'Title':<50} {'Score'}")
    for idx, (similar_id, score) in enumerate(similarities):
        print(f"{idx+1:<2} {similar_id:<8} {songs_df.loc[similar_id]['title']:<50} {score:.4f}")

recommend_songs('2172')
print(f"\n{"*" * 50}\n")
recommend_songs('26320', 20)
recommend_songs('36739', 10)
