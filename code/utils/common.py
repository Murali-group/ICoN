import typer
from pathlib import Path

import torch
import itertools


def extend_path(path: Path, extension: str) -> Path:
    """Extends a path by adding an extension to the stem.

    Args:
        path (Path): Full path.
        extension (str): Extension to add. This will replace the current extension.

    Returns:
        Path: New path with extension.
    """
    return path.parent / (path.stem + extension)


def cyan(string: str, **kwargs) -> str:
    return typer.style(string, fg=typer.colors.CYAN, bold=True, **kwargs)


def magenta(string: str, **kwargs) -> str:
    return typer.style(string, fg=typer.colors.MAGENTA, bold=True, **kwargs)


class Device:
    """Returns the currently used device by calling `Device()`.

    Returns:
        str: Either "cuda" or "cpu".
    """

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    def __new__(cls) -> str:
        return cls._device

def wrapper_generate_combination_dict(config_dict):
    '''
    Given a dictionary where a key is a paramater name. We can pass list of values for a certain parameter.
    Now, our goal is to generate a list of dicts from the original dict such that the keys will be the same,
    however, each key will have a single value, not a list of values. Also the new list of dicts will cover all combination of
    parameter values given in original dict.
    '''
    #though net_names contain a list of network_names, I do not want to iterate over them separately
    #hence, converting the list of network names into a string to stop the interation.
    config_dict['net_names']=  str(config_dict['net_names'])

    #generate combinations for param values mentioned in the gat_shapes dict.
    config_dict['gat_shapes'] = generate_combination_dict(config_dict['gat_shapes'])

    #generate combinations for param values mentioned in the config_dict
    configs = generate_combination_dict(config_dict)

    #convert back the list of network names from string to lst
    for config in configs:
        config['net_names'] = eval(config['net_names'])

    return configs

def generate_combination_dict(config_dict):
    '''
    Given a dictionary where a key is a paramater name. We can pass list of values for a certain parameter.
    Now, our goal is to generate a list of dicts from the original dict such that the keys will be the same,
    however, each key will have a single value, not a list of values. Also the new list of dicts will cover all combination of
    parameter values given in original dict.
    '''
    keys = list(config_dict.keys())
    values_lists = [config_dict[key] if isinstance(config_dict[key], list) else [config_dict[key]] for key in keys]
    combinations = list(itertools.product(*values_lists))

    result = []
    for combination in combinations:
        new_dict = {}
        for key, value in zip(keys, combination):
            new_dict[key] = value
        result.append(new_dict)

    return result


