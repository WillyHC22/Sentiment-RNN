from string import punctuation
import numpy as np
from collections import Counter

def word_mapping_from_raw_reviews(reviews):
    """returns the dictionnary encoding every existing word in the reviews into an integer"""
    reviews = reviews.lower()
    text_punct_filter = ''.join([char for char in reviews if char not in punctuation])
    reviews_split = text_punct_filter.split('\n')
    all_text = ' '.join(reviews_split)
    words = all_text.split()
    vocab_to_int = {word:integer for integer, word in enumerate(Counter(words), 1)}
    return vocab_to_int


def label_encoding(labels):
    encoded_labels = [1 if label == "positive" else 0 for label in labels.split("\n")]
    return encoded_labels


def tokenize_map_reviews(reviews, vocab_to_int:dict):
    reviews = reviews.lower()
    all_text = ''.join([c for c in reviews if c not in punctuation])
    reviews_split = all_text.split('\n')
    reviews_ints = [[vocab_to_int[word] for word in review.split()] for review in reviews_split]
    return reviews_ints


def remove_zero_length(reviews_ints, encoded_labels):
    for review, label in zip(reviews_ints, encoded_labels):
        if len(review) == 0:
            reviews_ints.remove(review)
            encoded_labels.remove(label)
    return reviews_ints, encoded_labels


def pad_features(args, reviews_ints):
    ''' Return features of review_ints, where each review is padded with 0's 
        or truncated to the input seq_length.
    '''
    seq_length = args["sequence_length"]
    features = np.empty([len(reviews_ints), seq_length])
    for i, review in enumerate(reviews_ints):
        if len(review) >= seq_length:
            features[i] = review[:seq_length]
        else:
            nb_zero = seq_length - len(review)
            features[i] = [0] * nb_zero + review
            
    return features