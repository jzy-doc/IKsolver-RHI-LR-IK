import logging


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        # "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        # "[%(asctime)s] %(message)s"
        # "[%(asctime)s]\t%(message)s", "%Y-%m-%d_%H:%M:%S"  # [2021-01-01_01:10:22]
        "[%(asctime)s][%(filename)s]\t%(message)s", "%Y-%m-%d_%H:%M:%S"  # [2021-01-01_01:10:22]
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger