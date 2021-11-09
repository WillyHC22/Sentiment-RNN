
from src.data.download_data import download_file
from src.get_parser import get_parser

args = get_parser()

if __name__ == '__main__':
    if args["download"]:
        download_file(args)