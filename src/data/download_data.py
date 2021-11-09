import requests
import argparse

#c:\Users\willy\Desktop\Course\sentiment-rnn\src\data
def get_parser():
    parser = argparse.ArgumentParser(description = "Download file from url")
    parser.add_argument("-u", "--url", type=str, default=None, required=True, help="Give the url link of the data you want to download")
    parser.add_argument("-p", "--save_path", type=str, default="./data/raw/", required=False, help="Give the path where you want the data to be downloaded")
    parser.add_argument("-n", "--file_name", type=str, default=None, required=True, help="Give the filename for the downloaded data")
    parser.add_argument("-d", "--download", action="store_true", help="Do you want to download data or not ?")
    args = vars(parser.parse_args())
    return args


def download_file(args):

    url = args["url"] 
    file_path = args["save_path"] + args["file_name"]

    print(f"Downloading data from {url} into {file_path}")

    with open(file_path, "wb") as file:
        response = requests.get(url)
        file.write(response.content)