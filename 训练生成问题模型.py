import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging
from matplotlib.font_manager import FontProperties

pd.options.mode.chained_assignment = None
font = FontProperties(fname=r"c:/windiws/fonts/simsun.ttc", size=14)  # 字体格式及其大小

logging.basicConfig(format='%(asctimes)s : %(levelname)s : %(message)s', level=logging.INFO)
new_vec = open('../data/Seg_The_smiling_Proud_Wanderer.txt', 'r', encoding='utf-8')

# 训练生成词向量模型
model = Word2Vec(LineSentence(new_vec), sg=0, size=200, window=10, min_count=40, workers=6)
print('模型训练完成')


def tsne_plot(model):
    """Creates and TSBe model and plots it"""
    labels = []
    tokens = []
    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=1000, random_state=20)
    new_values = tsne_model.fit_transform(tokens)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i]),
        fontproperties = font,
        xy = (x[i], y[i]),
        xytext = (5, 2),
        textcoords = 'offset points',
        ha = 'right',
        va = 'bottom',
    plt.show()


tsne_plot(model)
