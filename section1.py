import gensim.downloader as dl
model_google = dl.load("word2vec-google-news-300")
model_wiki = dl.load("glove-wiki-gigaword-200")
# this will take a while on first load as it downloads a 1.6G file.
# later calls will be cached.
# You can now use various methods of the “model“ object.
target_words = ["car","dog","apple","face","vice"]
poly_words=["bank","bat","apple"]
def similiarity_score(emb_model,word1, word2):
    return emb_model.similarity(word1, word2)

def most_similar_words(emb_model,words):
    for target_word in words:
        print(target_word)
        # Check if the target word is in the vocabulary
        if target_word in emb_model:
            # Find the most similar words
            similar_words = emb_model.most_similar(target_word, topn=20)

            # Print the results
            print(f"The most similar words to '{target_word}' are:")
            for word, similarity in similar_words:
                print(f"{word}: {similarity}")

w1='Happy'
w2='Gay'
w3='Sad'
print("sim w1,w2",similiarity_score(model_google,w1,w2))
print("sim w1,w3",similiarity_score(model_google,w1,w3))

