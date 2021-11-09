import argparse

def get_parser():
    parser = argparse.ArgumentParser(description = "Download file from url")
    parser.add_argument("-u", "--url", type=str, default=None, help="Give the url link of the data you want to download")
    parser.add_argument("-p", "--save_path", type=str, default="./data/raw/", required=False, help="Give the path where you want the data to be downloaded")
    parser.add_argument("-n", "--file_name", type=str, default=None, help="Give the filename for the downloaded data")
    parser.add_argument("-d", "--download", action="store_true", help="Use -d to download data. Need -u and -n arguments")
    args = vars(parser.parse_args())

    if args["download"] and (args["url"] is None or args["file_name"] is None):
        parser.error("--download requires --url/-u and --file_name/-n.")

    return args