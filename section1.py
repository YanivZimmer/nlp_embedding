import gensim.downloader as dl
model = dl.load("word2vec-google-news-300")
# this will take a while on first load as it downloads a 1.6G file.
# later calls will be cached.
# You can now use various methods of the “model“ object.
# you can access the vocabulary like so:
vocab = model.index_to_key
target_words = ["car","dog","apple","face","vice"]
poly_words=["bank","bat","apple"]
def similiarity_score(emb_model,word1, word2):
    return emb_model.similarity(word1, word2)

for target_word in []:
    print(target_word)
    # Check if the target word is in the vocabulary
    if target_word in model:
        # Find the most similar words
        similar_words = model.most_similar(target_word, topn=20)

        # Print the results
        print(f"The most similar words to '{target_word}' are:")
        for word, similarity in similar_words:
            print(f"{word}: {similarity}")

w1='Happy'
w2='Gay'
w3='Sad'
print("sim w1,w2",similiarity_score(model,w1,w2))
print("sim w1,w3",similiarity_score(model,w1,w3))
