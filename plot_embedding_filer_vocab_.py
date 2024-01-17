import gensim.downloader as dl
from dim_reduction_utils import reduce_dim_fit,reduce_dim_infer
import matplotlib.pyplot as plt

def filter_vocab(model):
    return model.index_to_key[1:5000]

def plot_emd(ed_emb,ing_emb):
    ed_x, ed_y = zip(*ed_emb)
    ing_x, ing_y = zip(*ing_emb)

    # Plot the points
    plt.scatter(ed_x, ed_y, marker='o', label='ed',c='blue')
    plt.scatter(ing_x, ing_y, marker='o', label='ing', c='green')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Word Embeddings')

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()


if __name__=='__main__':

    model = dl.load("word2vec-google-news-300")
    vocab = filter_vocab(model)
    vocab = list(filter(lambda x: (x.endswith("ing") or x.endswith("ed")), vocab))
    vocab_emb= [ model[word] for word in vocab]

    pca=reduce_dim_fit(vocab_emb)
    ed_words = list(filter(lambda x: x.endswith("ed"), vocab))
    ing_words = list(filter(lambda x: (x.endswith("ing")), vocab))
    ed_emb = reduce_dim_infer(pca,[ model[word] for word in ed_words])
    ing_emb = reduce_dim_infer(pca, [ model[word] for word in ing_words])
    plot_emd(ed_emb,ing_emb)
