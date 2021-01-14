# Define a function to perform both stemming and tokenization
import nltk
import re
# For stemming, importing the SnowballStemmer (PorterStemmer can also be used)
from nltk.stem.snowball import SnowballStemmer


# Create an English language SnowballStemmer object
stemmer = SnowballStemmer("english")


def tokenize_and_stem(text):
    try:
        tokens = [word for sent in nltk.sent_tokenize(
            text) for word in nltk.word_tokenize(sent)]

        # Filter out raw tokens to remove noise
        filtered_tokens = [
            token for token in tokens if re.search('[a-zA-Z]', token)]

        # Stem the filtered_tokens
        stems = [stemmer.stem(word) for word in filtered_tokens]

        # Tokenize by sentence, then by word
        return stems
    except:
        return 0
