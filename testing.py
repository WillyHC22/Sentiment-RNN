import argparse
import torch
import torch.nn as nn
from training import DataLoadSplit, TrainingRNN
from src.data.download_data import download_file
from src.get_parser import get_parser
from src.data.data_preprocessing import *
from models.model import sentimentRNN

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
    features = pad_features(args, reviews_ints)
    vocab_size = len(vocab_to_int) + 1
    loader = DataLoadSplit(encoded_labels, features, args)

    model = sentimentRNN(vocab_size, args)
    print(f"This is the model we are using : {model}")
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])
    print("Loading the data...")
    loaders = loader.load_split_data()
    train = TrainingRNN(model, loaders, args)
    args_train = [criterion, optimizer, args["epochs"], args["batch_size"], args["print_every"]]
    print("Start training")
    if args["train"]:
        train.trainRNN(criterion, optimizer)
        torch.save(model.state_dict(), "models/saved_models/model_test.pth")

    