from sklearn import datasets
from openTSNE import TSNE
import pandas as pd
import numpy as np
from openTSNE import utils
import matplotlib.pyplot as plt
# iris = datasets.load_iris()
# x, y = iris["data"], iris["target"]
x = np.genfromtxt('SGL_user.csv', delimiter=',')
# x = x[:10000, :]

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

# plt.scatter(embedding_train_X, embedding_train_Y, s=2, alpha=0.3, marker='o',)
plt.show()
# print(embedding)
# utils.plot(embedding, colors=utils.MACOSKO_COLORS)