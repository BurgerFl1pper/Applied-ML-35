from data_processing import openAndRefactor, processData

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns


def vectorization(data):
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X = tfidf.fit_transform(data['text']).toarray()
    return X


def pcaTest(data):
    test = PCA(n_components=2)
    x_pca = test.fit_transform(data)
    return x_pca


def plot(data, x_pca):
    plt.figure(figsize=(10, 6))
    data['Primary Genre'] = data['Genre'].apply(lambda x: x[0] if isinstance(x, list) else x)

    sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=data['Primary Genre'],
                    palette='Set1', s=10)
    plt.title('PCA Visualization of Song Lyrics by Genre')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Main Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def main():
    data = openAndRefactor('finished_data.json')

    data = processData(data)
    X = vectorization(data)
    x_pca = pcaTest(X)
    plot(data, x_pca)

main()
