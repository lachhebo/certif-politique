import logging
from datetime import datetime
from pathlib import Path

from certifia import ROOT_PATH
from certifia.utils.metaclass import Singleton


class Logger(metaclass=Singleton):

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MainLogger")

    def set_config(self, root_path: Path = ROOT_PATH):
        filename = str(root_path) + '/logs/{:%Y-%m-%d}.log'.format(datetime.now())
        file_handler = logging.FileHandler(filename)
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        self.logger.addHandler(file_handler)

    @property
    def warning(self):
        self.set_config()
        return self.logger.warning

    @property
    def info(self):
        self.set_config()
        return self.logger.info

    @property
    def critical(self):
        self.set_config()
        return self.logger.critical

    @property
    def debug(self):
        self.set_config()
        return self.logger.debug
