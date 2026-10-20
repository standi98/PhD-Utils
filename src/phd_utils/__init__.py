import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())


def hello() -> str:
    return "Hello from phd-utils!"
