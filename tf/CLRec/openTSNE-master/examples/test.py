from openTSNE import TSNE
from examples import utils
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
x = np.genfromtxt('./../../self.U1.csv', delimiter=',')
tsne = TSNE(
    perplexity=20,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
)
embedding_train = tsne.fit(x)
embedding_train_X = embedding_train[:, 0]
embedding_train_Y = embedding_train[:, 1]
emb = np.arctan2(embedding_train_X, embedding_train_Y)

plt.plot(emb, emb+0, 'o')
plt.show()
# print(len(embedding_train))
# utils.plot(embedding_train, x, colors=utils.MACOSKO_COLORS)