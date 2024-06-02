import logging
import pprint
from enum import Enum
from math import acos, cos, radians, sin

import yaml


def parse_configs(configs_path: str) -> dict:
    """Parse configs from the YAML file.

    Args:
        configs_path (str): Path to the YAML file

    Returns:
        dict: Parsed configs
    """
    configs = yaml.safe_load(open(configs_path, "r"))
    logger.info(f"Configs: {pprint.pformat(configs)}")
    return configs


def get_distance(source_country: list[float], target_country: list[float]) -> float:
    """Calculate the distance between two countries.

    Args:
        source_country (list[float]): Source country coordinates
        target_country (list[float]): Target country coordinates

    Returns:
        float: Distance in KM
    """
    source_lat = radians(source_country[0])
    source_long = radians(source_country[1])
    target_lat = radians(target_country[0])
    target_long = radians(target_country[1])
    dist = 6371.01 * acos(
        sin(source_lat) * sin(target_lat)
        + cos(source_lat) * cos(target_lat) * cos(source_long - target_long)
    )
    return dist


class HintType(Enum):
    AUDIO = "Audio"
    TEXT = "Text"
    IMAGE = "Image"


CONFIGS_PATH = "configs.yaml"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

configs = parse_configs(CONFIGS_PATH)
