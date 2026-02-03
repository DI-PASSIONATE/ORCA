import logging
import colorlog
import sys

logger = logging.getLogger("ORCA")

stdout = colorlog.StreamHandler(stream=sys.stdout)

fmt = colorlog.ColoredFormatter(
    fmt="%(name)s (%(module)s:%(lineno)d): %(white)s%(asctime)s%(reset)s | %(log_color)s%(levelname)s%(reset)s >> %(log_color)s%(message)s%(reset)s",
    datefmt="%H:%M:%S",
)

stdout.setFormatter(fmt)
logger.addHandler(stdout)

logger.setLevel(logging.INFO)
