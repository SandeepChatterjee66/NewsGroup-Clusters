# -*- coding: utf-8 -*-
"""clustering by k means
"""

from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize #Used to extract words from documents
from nltk.stem import WordNetLemmatizer #Used to lemmatize words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans

import sys
from time import time

import pandas as pd
import numpy as np

# Selected 3 categories from the 20 newsgroups dataset

categories = [
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

print("Loading 20 newsgroups dataset for categories:")
print(categories)

df = fetch_20newsgroups(subset='all', categories=categories, 
                             shuffle=False, remove=('headers', 'footers', 'quotes'))

labels = df.target
true_k = len(np.unique(labels)) ## This should be 3 in this example
print(true_k)

"""### Perform Lemmatization"""

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
for i in range(len(df.data)):
    word_list = word_tokenize(df.data[i])
    lemmatized_doc = ""
    for word in word_list:
        lemmatized_doc = lemmatized_doc + " " + lemmatizer.lemmatize(word)
    df.data[i] = lemmatized_doc

print(df.data[1])

"""We next convert our corpus into tf-idf vectors. We remove common stop words, terms with very low document frequency (many of them are numbers or misspells), accents. """

vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words='english', min_df=2) ## Corpus is in English
X = vectorizer.fit_transform(df.data)

print(X.shape)

"""### Clustering using standard k-means"""
print("clustering using k = 3")
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100)
t0 = time()
km.fit(X)
print("ran upto 100 iterations")
print("done in %0.3fs" % (time() - t0))
print()

"""### Standard measures of cluster quality"""
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    
#print("F1: %0.3f" % metrics.f1_score(labels, km.labels_ , average='samples'))
print("F1: %0.3f" % metrics.f1_score(labels, km.labels_ , average='macro'))
print("Accuracy: %0.3f" % metrics.accuracy_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("Hemogenioty: %0.3f" % metrics.homogeneity_score(labels, km.labels_))

print("Contigency matrix: ", metrics.cluster.contingency_matrix)
print("Purity: ", purity_score(labels, km.labels_))

#print("Precision: %0.3f" % metrics.average_precision_score(labels, km.labels_))
#print("Recall: %0.3f" % metrics.recall_score(labels, km.labels_))
print()

"""### Identify the 10 most relevant terms in each cluster"""

centroids = km.cluster_centers_.argsort()[:, ::-1] ## Indices of largest centroids' entries in descending order
terms = vectorizer.get_feature_names_out()
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    for ind in centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()
