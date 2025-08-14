from copernicus import read_credentials, make_request


def main():

    credentials = read_credentials()

    make_request(credentials)

if __name__ == '__main__':
    main()

