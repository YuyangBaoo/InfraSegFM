import logging
import os


def get_logger(log_file=None, log_level=logging.INFO, rank=0, file_mode="a"):
    logger = logging.getLogger(name="ViT")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.setLevel(log_level if rank == 0 else logging.ERROR)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level if rank == 0 else logging.ERROR)
    logger.addHandler(console_handler)
    if log_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(filename=log_file, mode=file_mode, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level if rank == 0 else logging.ERROR)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger
