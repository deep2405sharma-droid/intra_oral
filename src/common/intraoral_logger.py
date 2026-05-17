import logging

def getLogger(log_filename):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(log_filename)), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()
