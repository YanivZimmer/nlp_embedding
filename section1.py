import gensim.downloader as dl
model = dl.load("word2vec-google-news-300")
# this will take a while on first load as it downloads a 1.6G file.
# later calls will be cached.
# You can now use various methods of the “model“ object.
# you can access the vocabulary like so:
vocab = model.index_to_key
target_words = ["car","dog","apple","face","vice"]
for target_word in target_words:
    print(target_word)
    # Check if the target word is in the vocabulary
    if target_word in model:
        # Find the most similar words
        similar_words = model.most_similar(target_word, topn=20)

        # Print the results
        print(f"The most similar words to '{target_word}' are:")
        for word, similarity in similar_words:
            print(f"{word}: {similarity}")