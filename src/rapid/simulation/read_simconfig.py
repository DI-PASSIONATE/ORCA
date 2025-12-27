import json

def read_simconfig(simconfig_filename: str) -> dict:
    """Reads simulation configuration from a file and returns it as a dictionary.

    Args:
        simconfig_filename (str): Path to the simulation configuration file.
    Returns:
        dict: A dictionary containing simulation configuration parameters.
    """
    with open(simconfig_filename, 'r') as file:
        simconfig = json.load(file)
    return simconfig