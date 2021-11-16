import argparse

from torch.serialization import storage_to_tensor_type

def get_parser():
    parser = argparse.ArgumentParser(description = "Download file from url")
    parser.add_argument("-u", "--url", type=str, help="Give the url link of the data you want to download")
    parser.add_argument("-p", "--save_path", type=str, default="./data/raw/", help="Give the path where you want the data to be downloaded")
    parser.add_argument("-n", "--file_name", type=str, help="Give the filename for the downloaded data")
    parser.add_argument("-d", "--download", action="store_true", help="Use -d to download data. Need -u and -n arguments")
    parser.add_argument("-sl", "--sequence_length", type=int, default=200, help="The maximum length sequence for every batch, we will either truncate down or pad to fill the sequence length")
    parser.add_argument("--output_size", type=int, default=1, help="Output size for the rnn")
    parser.add_argument("--embedding_dim", type=int, default=400, help="Size of the embeddings for rnn")
    parser.add_argument("--hidden_dim", type=int, default=256, help="hidden dimension of the rnn")
    parser.add_argument("--n_layers", type=int, default=2, help="number of layers for the rnn")
    parser.add_argument("--drop_rate", type=int, default=0.5, help="probability for the dropout")
    parser.add_argument("-lr", "--learning_rate", default=0.001, type=int, help="number of layers for the rnn")
    parser.add_argument("-e", "--epochs", default=3, type=int, help="number of epochs for training")
    parser.add_argument("--print_every", default=100, type=int, help="Number of step for one update of the results")
    parser.add_argument("-bs", "--batch_size", default=50, type=int, help="number of epochs for training")
    parser.add_argument("--train",  action="store_true", help="Use this argument if you want training")
    parser.add_argument("--load",  action="store_true", help="Use this argument if you want to load the model")
    parser.add_argument("--eval", action="store_true", help="Use this argument if you want evaluation")
    parser.add_argument("--predict", action="store_true", help="Use this argument if you want inference test")
    args = vars(parser.parse_args())

    if args["download"] and (args["url"] is None or args["file_name"] is None):
        parser.error("--download requires --url/-u and --file_name/-n.")

    return args