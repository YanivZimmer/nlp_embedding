import gensim.downloader as dl
model_google = dl.load("word2vec-google-news-300")
model_wiki = dl.load("glove-wiki-gigaword-200")
model_twitter = dl.load("glove-twitter-200")
# this will take a while on first load as it downloads a 1.6G file.
# later calls will be cached.
# You can now use various methods of the “model“ object.

# examples from: https://www.eltconcourse.com/training/inservice/lexicogrammar/polysemy.html
# pen (a writing implement) and pen (the action of writing)
# dish (a kind of plate) and dish (a meal)
# frame (a surrounding for a picture) and frame (falsely incriminate)
# cap (the top of a pen or bottle) and cap (a type of headgear)
# drive (control a vehicle) and drive (a short private roadway)
# satellite (an orbiting space craft) and satellite (a country or state dependent on a more powerful one)
# bank (a commercial enterprise concerned with money) and bank (the side of a river)
polysemous_words = ["apple", "mouse", "pen", "dish", "frame", "cap", "drive", "satellite", "bat", "bank", "rose"]
def similarity_score(model, word1, word2):
    return model.similarity(word1, word2)

def most_similar_words(model, words, num):
    for target_word in words:
        # Check if the target word is in the vocabulary
        if target_word in model:
            # Find the most similar words
            similar_words = model.most_similar(target_word, topn=num)

            # Print the results
            print(f"The most similar words to '{target_word}' are:")
            for word, similarity in similar_words:
                print(f"{word}: {similarity}")

        print("\n")

def most_similar_words_two_models(model1, model1_name, model2, model2_name, words, num):
    for target_word in words:
        # Check if the target word is in the vocabulary
        for model, model_name in [(model1, model1_name), (model2, model2_name)]:
            print(f"Using {model_name} model:")
            if target_word in model:
                # Find the most similar words
                similar_words = model.most_similar(target_word, topn=num)

                # Print the results
                print(f"The most similar words to '{target_word}' are:")
                for word, similarity in similar_words:
                    print(f"{word}: {similarity}")

        print("\n")

if __name__ == "__main__":
    # Generating lists of the most similar words
    target_words = ["car", "dog", "apple", "face", "vice"]
    #most_similar_words(model_google, target_words, 20)

    # Polysemous Words
    #most_similar_words(model_google, polysemous_words, 10)

    # Synonyms and Antonyms
    '''
    w1, w2, w3 = 'Happy', 'Gay', 'Sad'
    print("sim w1, w2:", similarity_score(model_google, w1, w2))
    print("sim w1, w3:", similarity_score(model_google, w1, w3))
    w1, w2, w3 = 'rose', 'soar', 'decreased'
    print("sim w1, w2:", similarity_score(model_google, w1, w2))
    print("sim w1, w3:", similarity_score(model_google, w1, w3))
    '''

    # The Effect of Different Corpora
    target_words = ["car", "dog", "apple", "rain", "weekend", "report",
                    "pillow", "outlet", "manufacture", "course", "truck", "thanks"
                    , "king", "panic", "smiley", "haha", "crazy", "love"]
    most_similar_words_two_models(model_wiki, "wiki", model_twitter, "twitter", target_words, 10)
