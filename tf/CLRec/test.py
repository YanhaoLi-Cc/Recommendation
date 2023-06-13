from openTSNE import TSNE
import pandas as pd
import numpy as np
from openTSNE import utils
import matplotlib.pyplot as plt
x = np.genfromtxt('Light.csv', delimiter=',')

# x = x[:1000, :]
# print(len(x))
embedding = TSNE().fit(x)
embedding_train_X = embedding[:, 0]
embedding_train_Y = embedding[:, 1]
angel = np.arctan2(embedding_train_Y, embedding_train_X)
# emb_y = np.arctan(embedding_train_Y)
# plt.plot(emb, emb+0, 'o')
r = 2
x = r * np.sin(angel)
y = r * np.cos(angel)
plt.figure(figsize=(10, 10))
plt.scatter(x, y, s=1, color='lightgreen',  alpha=0.05, marker='o',)

# plt.scatter(embedding_train_X, embedding_train_Y, s=1, color='lightgreen',  alpha=1, marker='o',)


plt.show()