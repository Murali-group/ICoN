import itertools

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


