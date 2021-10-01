from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# File paths
ROOT = Path(__file__).parent.parent / 'Images_Flipkart' / 'Flipkart'
descriptors_filepath = ROOT / 'descriptors.csv'
clusters_filepath = ROOT / 'clusters.csv'

# Load table with all products
df = pd.read_csv(ROOT / 'flipkart_com-ecommerce_sample_1050.csv')
df = df.rename({'uniq_id': 'img_id'}, axis=1)
df = df.set_index('img_id').sort_index()
print(f'... found {len(df)} product images')

# Extract ground truth product category for each image
df['label'] = df.product_category_tree.apply(lambda s: s.split('>>')[0][2:])
print(f'... found {len(df.label.unique())} product categories')


# Compute SIFT descriptors for each image in the dataset
def get_sift_descriptors(image_id: str):
    filepath = ROOT / 'Images' / f'{image_id}.jpg'
    img = cv2.imread(str(filepath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    sift = cv2.SIFT_create()
    (key_points, img_descriptors) = sift.detectAndCompute(img, None)
    return pd.DataFrame(img_descriptors).astype(int)


if not descriptors_filepath.exists():
    descriptors = [get_sift_descriptors(image_id=img_id) for img_id in df.index]
    descriptors = pd.concat(descriptors, keys=df.index).droplevel(1, axis=0)
    descriptors.to_csv(descriptors_filepath)
    print(f'... found {len(descriptors)} SIFT keypoints over all product images')

# Group SIFT descriptors into clusters
if not clusters_filepath.exists():
    descriptors = pd.read_csv(descriptors_filepath).set_index('img_id')
    n_clusters = int(np.sqrt(len(descriptors)))
    model = MiniBatchKMeans(n_clusters=n_clusters).fit(X=descriptors.values)
    clusters = model.predict(X=descriptors)
    clusters = pd.Series(clusters, index=descriptors.index, name='cluster_id')
    clusters.to_csv(clusters_filepath)
else:
    clusters = pd.read_csv(clusters_filepath).set_index('img_id').cluster_id
print(f'... found a vocabulary of {len(clusters.unique())} visual words from SIFT keypoint descriptors')

# Compute "bag of visual words" for each image, with TF-IDF weighting
words_counts = clusters.groupby('img_id').value_counts().unstack('cluster_id').fillna(0).astype(int).sort_index()
words_inverse_data_freq = len(df) / (words_counts > 0).sum(axis=0)
bag_of_words = words_counts.apply(lambda row: row / row.sum() * np.log(words_inverse_data_freq), axis=1)

bag_of_words = PCA(n_components=25).fit_transform(bag_of_words)
df['label_prediction'] = KMeans(n_clusters=len(df.label.unique())).fit_predict(bag_of_words)

# T-SNE visualization based on bag of visual words, colored by image ground-truth category
df[['bags_tsne_0', 'bags_tsne_1']] = TSNE(n_components=2).fit_transform(X=bag_of_words)
fig, ax = plt.subplots()
seaborn.scatterplot(x='bags_tsne_0', y='bags_tsne_1', data=df, hue='label', ax=ax)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.axis('off')
plt.tight_layout()
ax.set_axis_off()
plt.show()
