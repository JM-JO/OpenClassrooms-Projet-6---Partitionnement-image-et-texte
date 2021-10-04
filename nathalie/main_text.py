from pathlib import Path

import nltk
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from nltk.stem.snowball import EnglishStemmer
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

nltk.download('punkt')
nltk.download('stopwords')

# Load data
filepath = Path(__file__).parent.parent / 'flipkart_com-ecommerce_sample_1050.csv'
df = pd.read_csv(filepath)
df["main_category"] = df.product_category_tree.apply(lambda s: s.split(' >> ')[0].replace('["', ""))
categories = list(df.main_category.unique())


# Function to extract words from product descriptions
def tokenize(text: str):
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    stemmer = EnglishStemmer()
    stopwords = nltk.corpus.stopwords.words('english')
    return [stemmer.stem(word) for word in tokenizer.tokenize(text) if word not in stopwords]


# Compute word weights for each item (use TF-IDF weights)
word_weighter = TfidfVectorizer(tokenizer=tokenize, lowercase=True)
word_weights = word_weighter.fit_transform(df.description)

word_weights_df = pd.DataFrame(word_weights.toarray())
word_weights_df.columns = [x[0] for x in sorted(word_weighter.vocabulary_.items(), key=lambda item: item[1])]
print(f'Shape of word weights:', word_weights.shape)

# Train / Test split
x, y = word_weights, df.main_category
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)
df_train, df_test = df.loc[y_train.index].copy(), df.loc[y_test.index].copy()

# Classification (SVM)
clf = SVC().fit(x_train, y_train)
df_test['main_category_svm'] = clf.predict(x_test)

# Clustering (KNN)
# 1) find clusters  with KNN
# 2) map cluster ids to product categories
# 3) predict product categories on test set
n_categories = len(categories)
clustering = KMeans(n_clusters=n_categories, init='k-means++')
clustering.fit(x_train)
df_train['cluster_id'] = clustering.labels_
conf_matrix = df_train.groupby(['main_category', 'cluster_id']).uniq_id.count().unstack('cluster_id')
conf_matrix = conf_matrix.fillna(0).apply(lambda x: x / sum(x) * 100, axis=1).copy()
cluster_to_category = {}
while len(conf_matrix) > 0:
    category = conf_matrix.max(axis=1).idxmax()
    cluster_id = conf_matrix.loc[category].idxmax()
    cluster_to_category[cluster_id] = category
    conf_matrix = conf_matrix.drop(category, axis=0).drop(cluster_id, axis=1)

df_test['cluster_id'] = clustering.predict(x_test)
df_test['main_category_knn'] = df_test.cluster_id.map(cluster_to_category)

# Display scores
knn_ari = adjusted_rand_score(df_train.main_category, df_train.cluster_id)
knn_accuracy = accuracy_score(y_true=df_test.main_category, y_pred=df_test.main_category_knn)
svm_accuracy = accuracy_score(y_true=df_test.main_category, y_pred=df_test.main_category_svm)
print(f"KNN - Adjusted Rand Index: {round(knn_ari, 2)}", )
print(f'KNN- Classification accuracy on test set: {round(knn_accuracy, 2)}%', )
print(f'SVM - Classification accuracy on test set: {round(svm_accuracy, 2)}%', )

# Plot T-SNE visualization of word weights for each product
df_test[['tsne_0', 'tsne_1']] = manifold.TSNE(n_components=2).fit_transform(x_test)
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 8))
seaborn.scatterplot(x="tsne_0", y="tsne_1", data=df_test, hue="main_category", hue_order=categories, ax=ax0)
seaborn.scatterplot(x="tsne_0", y="tsne_1", data=df_test, hue="main_category_svm", hue_order=categories, ax=ax1)
seaborn.scatterplot(x="tsne_0", y="tsne_1", data=df_test, hue="main_category_knn", hue_order=categories, ax=ax2)
ax0.set_title('Projection TSNE du jeu de test - catégorie observée')
ax1.set_title('Projection TSNE du jeu de test - catégorie prédite par SVM')
ax2.set_title('Projection TSNE du jeu de test - catégorie prédite par KNN')
for ax in [ax0, ax1, ax2]:
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    ax.axis('off')
plt.tight_layout()
plt.show()
