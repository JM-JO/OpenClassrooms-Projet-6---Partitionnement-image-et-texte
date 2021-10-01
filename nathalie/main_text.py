from pathlib import Path

import nltk
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from nltk.stem.snowball import EnglishStemmer
from sklearn import manifold
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

# Load data
filepath = Path(__file__).parent.parent / 'flipkart_com-ecommerce_sample_1050.csv'
df = pd.read_csv(filepath)
df["main_category"] = df.product_category_tree.apply(lambda s: s.split(' >> ')[0].replace('["', ""))


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

# Plot T-SNE visualization of word weights for each product
df[['tsne_0', 'tsne_1']] = manifold.TSNE(n_components=2).fit_transform(word_weights)
fig, ax = plt.subplots(figsize=(8, 5))
seaborn.scatterplot(x="tsne_0", y="tsne_1", data=df, hue="main_category", ax=ax)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.title('Visualisation T-SNE des fréquences de mots (TF-IDF) par article')
plt.axis('off')
plt.tight_layout()
plt.show()


from gensim.models import Word2Vec
import nltk
from gensim.models import KeyedVectors

# Import d'une base word2vec en francais deja entrainee
import gensim.downloader as api

word_vectors = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data
#w2v = KeyedVectors.load_word2vec_format(
#http://embeddings.net/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin,
#    binary=True)
a=1
df.description.apply()
#w2v.wv[mot]
#X.append(np.mean(vec_phrase,0))