import logging

import time    
from typing import Literal


_start_time = time.time() 

class _ElapsedFormatter(logging.Formatter):
    def format(self, record):
        record.elapsed = f"{time.time() - _start_time:.2f}s"
        return super().format(record)

def setup_logging(level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO", format: str|None = None) -> None:
    """"Format logging in a nice way."""
    default_format = "%(elapsed)s - %(name)s - %(funcName)s - Line:%(lineno)d - %(levelname)s - %(message)s"
    
    if format is None:
        format = default_format


    handler = logging.StreamHandler()
    handler.setFormatter(_ElapsedFormatter(default_format))
    
    logging.getLogger().setLevel(level.upper())
    logging.getLogger().addHandler(handler)
    
    logging.basicConfig(level=level, format=format)