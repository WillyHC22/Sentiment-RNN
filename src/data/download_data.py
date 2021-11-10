import requests

def download_file(args):

    url = args["url"] 
    file_path = args["save_path"] + args["file_name"]

    print(f"Downloading data from {url} into {file_path}")

    with open(file_path, "wb") as file:
        response = requests.get(url)
        file.write(response.content)