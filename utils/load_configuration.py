import configparser, os

def load_config(path=None):
    if path is None:
        path = os.path.join(os.getcwd(), 'config.ini')
    config = configparser.ConfigParser()
    config.read(path)
    return config
