from string import punctuation
import numpy as np
from collections import Counter

def create_words():
    """returns list of every words in the review after punctuation filtering"""
    with open("../../data/raw/reviews.txt", "r") as f:
        labels = f.read()
    with open("../../data/raw/labels.txt", "r") as f:
        reviews = f.read()
    reviews = reviews.lower()
    text_punct_filter = ''.join([char for char in reviews if char not in punctuation])
    reviews_split = text_punct_filter.split('\n')
    all_text = ' '.join(reviews_split)
    words = all_text.split()
    return words