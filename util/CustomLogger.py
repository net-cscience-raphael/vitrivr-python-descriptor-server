
import logging
import logging.config
from util.CustomFormatter import CustomFormatter


def setup_logging(args):

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("app.log", mode="a", encoding="utf-8")

    logger = logging.getLogger()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
        console_handler.setLevel(logging.ERROR)
        file_handler.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)
        file_handler.setLevel(logging.INFO)


    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    console_handler.setFormatter(CustomFormatter(format))
    file_handler.setFormatter(logging.Formatter(format))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.info(f"Logging configured, level: {logging.getLevelName(logger.level)}")