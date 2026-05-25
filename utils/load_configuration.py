import configparser, os
from pathlib import Path

def load_config(path_1=None):
    try:
        # Create a ConfigParser object
        config = configparser.ConfigParser()
        # Read the config.ini file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = Path(base_dir).parent
        config_path = os.path.join(parent_dir, path_1)
        config.read(config_path)
        return config
    except FileNotFoundError as fnfe:
        raise (fnfe)
    except Exception as e:
        raise (e)

