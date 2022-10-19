import logging.config
import os
import sys
import yaml

import coloredlogs
import verboselogs

from configs.pathes import conf_path


def get_logging_dict():
    config_dict = None

    if os.path.exists(conf_path["LOGGING_YAML"]):
        with open(conf_path["LOGGING_YAML"], 'rt') as f:
            config_dict = yaml.safe_load(f.read())

    return config_dict


def setup_logging(config_dict, default_level=logging.DEBUG):
    """Setup logging configuration

    """
    verboselogs.install()
    coloredlogs.install()

    if config_dict is not None:
        logging.config.dictConfig(config_dict)
    else:
        logging.basicConfig(level=default_level)

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)

        module = os.getcwd().split("/")[-1]
        path_name = record.pathname.split("/")

        if module in path_name:
            id_mod = path_name.index(module)
            record.shortpath = "/".join(path_name[id_mod:])
        else:
            record.shortpath = module + "/" + path_name[-1]

        record.shortpath = ("..."+record.shortpath[-35:]) if len(record.shortpath) > 35 else record.shortpath
        
        return record

    logging.setLogRecordFactory(record_factory)
