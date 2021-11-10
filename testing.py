import argparse
from src.data.download_data import download_file
from src.get_parser import get_parser
from src.data.data_preprocessing import *
from models.model import sentimentRNN

args = get_parser()

parser = argparse.ArgumentParser()
parser.add_argument("--vocab_size", type=int, help="Give the filename for the downloaded data")
parser.add_argument("--output_size", type=int, default=1, help="Output size for the rnn")
parser.add_argument("--embedding_dim", type=int, default=400, help="Size of the embeddings for rnn")
parser.add_argument("--hidden_dim", type=int, default=256, help="hidden dimension of the rnn")
parser.add_argument("--n_layers", type=int, default=2, help="number of layers for the rnn")
args1 = vars(parser.parse_args())

def download_data(args):
    if args["download"]:
        download_file(args)

def process_data(reviews_file, labels_file):
    with open(reviews_file, "r") as f:
        reviews = f.read()
    with open(labels_file, 'r') as f:
        labels = f.read()

    vocab_to_int = word_mapping_from_raw_reviews(reviews)
    encoded_labels = label_encoding(labels)
    reviews_ints = tokenize_map_reviews(reviews, vocab_to_int)
    reviews_ints, encoded_labels = remove_zero_length(reviews_ints, encoded_labels)

    return reviews_ints, encoded_labels, vocab_to_int


if __name__ == '__main__':

    #Download some data if needed
    args = get_parser()
    download_data(args)

    reviews_ints, encoded_labels, vocab_to_int = process_data("data/raw/reviews.txt", "data/raw/labels.txt")
    #features = pad_features(args, reviews_ints)

    if args['vocab_size'] is None:
        args['vocab_size'] = len(vocab_to_int) + 1

    #net = sentimentRNN(**args1)
    print(args["vocab_size"], args["output_size"], args["embedding_dim"], args["hidden_dim"], args["n_layers"])
    net = sentimentRNN(args["vocab_size"], args["output_size"], args["embedding_dim"], args["hidden_dim"], args["n_layers"])
    print(net)


