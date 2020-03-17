import copy
import logging
import os
import sys


def initialize_logging(log_path=None):
    loggers = {}

    # Check for environmental variables
    logger_env_vars = {
        'feature_extraction_LOG_LEVEL': 'feature_extraction',
        'feature_extraction_ES_LOG_LEVEL': 'feature_extraction.entityset',
        'feature_extraction_BACKEND_LOG_LEVEL': 'feature_extraction.computation_backend'
    }
    for logger_env, logger in logger_env_vars.items():
        log_level = os.environ.get(logger_env, None)
        if log_level is not None:
            loggers[logger] = log_level

    # Set log level to info if not otherwise specified.
    loggers.setdefault('feature_extraction', 'info')
    loggers.setdefault('feature_extraction.computation_backend', 'info')
    loggers.setdefault('feature_extraction.entityset', 'info')

    fmt = '%(asctime)-15s %(name)s - %(levelname)s    %(message)s'
    out_handler = logging.StreamHandler(sys.stdout)
    out_handler.setFormatter(logging.Formatter(fmt))
    if log_path is not None:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(fmt))
    
    for name, level in list(loggers.items()):
        LEVEL = getattr(logging, level.upper())
        logger = logging.getLogger(name)
        logger.setLevel(LEVEL)        
        for _handler in logger.handlers:
            logger.removeHandler(_handler)
        logger.addHandler(out_handler)
        if log_path is not None:
            logger.addHandler(file_handler)





class Config():
    def __init__(self):
        self._data = {}
        self.set_to_default()

    def set_to_default(self):
        PWD = os.path.dirname(__file__)
        primitive_data_folder = os.path.join(PWD, "primitives/data")
        self._data = {
            "primitive_data_folder": primitive_data_folder,
        }

    def get(self, key):
        return copy.deepcopy(self._data[key])

    def get_all(self):
        return copy.deepcopy(self._data)

    def set(self, values):
        self._data.update(values)


config = Config()
