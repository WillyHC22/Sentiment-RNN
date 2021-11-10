import argparse

def get_parser():
    parser = argparse.ArgumentParser(description = "Download file from url")
    parser.add_argument("-u", "--url", type=str, help="Give the url link of the data you want to download")
    parser.add_argument("-p", "--save_path", type=str, default="./data/raw/", help="Give the path where you want the data to be downloaded")
    parser.add_argument("-n", "--file_name", type=str, help="Give the filename for the downloaded data")
    parser.add_argument("-d", "--download", action="store_true", help="Use -d to download data. Need -u and -n arguments")
    parser.add_argument("-sl", "--sequence_length", type=int, default=200, help="The maximum length sequence for every batch, we will either truncate down or pad to fill the sequence length")
    parser.add_argument("--vocab_size", type=int, help="Give the filename for the downloaded data")
    parser.add_argument("--output_size", type=int, default=1, help="Output size for the rnn")
    parser.add_argument("--embedding_dim", type=int, default=400, help="Size of the embeddings for rnn")
    parser.add_argument("--hidden_dim", type=int, default=256, help="hidden dimension of the rnn")
    parser.add_argument("--n_layers", type=int, default=2, help="number of layers for the rnn")


    args = vars(parser.parse_args())

    if args["download"] and (args["url"] is None or args["file_name"] is None):
        parser.error("--download requires --url/-u and --file_name/-n.")

    return args