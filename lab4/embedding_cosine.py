import numpy as np
from scipy import spatial
import sys


def spatial_smallest(dist, list_of_close, compared_word):
    if dist != 0.0:
        new_close = [(compared_word, dist) if dist < x[1] == max([y[1] for y in list_of_close]) else x for x in
                     list_of_close]
        return new_close
    else:
        return list_of_close


filepath = "../datasets/embeddings/glove.6B/glove.6B.100d.txt"

emb_dict = {}

with open(filepath, encoding="utf8") as f:
    for line in f:
        word, *embedding = line.split()
        emb_dict[word] = np.array(embedding, dtype=np.float32)

table_e = np.array(emb_dict['table'], dtype=np.float32)
france_e = np.array(emb_dict['france'], dtype=np.float32)
sweden_e = np.array(emb_dict['sweden'], dtype=np.float32)

table_close = [("", sys.maxsize), ("", sys.maxsize - 1), ("", sys.maxsize - 2), ("", sys.maxsize - 3),
               ("", sys.maxsize - 4)]
france_close = table_close
sweden_close = table_close

print(table_close)

for word, embedding in emb_dict.items():
    table_close = spatial_smallest(spatial.distance.cosine(table_e, embedding), table_close, word)

    france_close = spatial_smallest(spatial.distance.cosine(france_e, embedding), france_close, word)

    sweden_close = spatial_smallest(spatial.distance.cosine(sweden_e, embedding), sweden_close, word)

print("tabel: ", table_close)
print("france: ", france_close)
print("sweden: ", sweden_close)
