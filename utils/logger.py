import sys
from pathlib import Path
from loguru import logger

ROOT_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Remove default handler
logger.remove()

# Console handler — INFO and above
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    colorize=True,
)

# File handler — DEBUG and above, daily rotation
logger.add(
    LOGS_DIR / "app_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="00:00",
    retention="30 days",
    compression="zip",
    enqueue=True,
)

# Error-only file
logger.add(
    LOGS_DIR / "errors_{time:YYYY-MM-DD}.log",
    level="ERROR",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="10 MB",
    retention="90 days",
    backtrace=True,
    diagnose=True,
    enqueue=True,
)


def get_logger(name: str):
    return logger.bind(name=name)
