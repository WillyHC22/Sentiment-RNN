
from src.data.download_data import download_file
from src.get_parser import get_parser
from src.data.data_preprocessing import *

args = get_parser()

def download_data(args):
    if args["download"]:
        download_file(args)

if __name__ == '__main__':
    args = get_parser()
    download_data(args)


    reviews_file = "data/raw/reviews.txt"
    labels_file = "data/raw/labels.txt"
    with open(reviews_file, "r") as f:
        reviews = f.read()
    with open(labels_file, 'r') as f:
        labels = f.read()

    vocab_to_int = word_mapping_from_raw_reviews(reviews_file)
    encoded_labels = label_encoding(labels_file)
    reviews_int = tokenize_map_reviews()