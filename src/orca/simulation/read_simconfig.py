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

    # Add e9 suffix to frequency values if they are in GHz
    if 'fstart' in simconfig['saved_values']:
        fstart = simconfig['saved_values']['fstart']
        if fstart < 1e6:  # assuming values less than 1 MHz are in GHz
            simconfig['saved_values']['fstart'] = fstart * 1e9
    if 'fstop' in simconfig['saved_values']:
        fstop = simconfig['saved_values']['fstop']
        if fstop < 1e6:  # assuming values less than 1 MHz are in GHz
            simconfig['saved_values']['fstop'] = fstop * 1e9
    if 'fstep' in simconfig['saved_values']:
        fstep = simconfig['saved_values']['fstep']
        if fstep < 1e6:  # assuming values less than 1 MHz are in GHz
            simconfig['saved_values']['fstep'] = fstep * 1e9
    if "fdump" in simconfig['saved_values']:
        fdump = simconfig['saved_values']['fdump']
        if fdump < 1e6:  # assuming values less than 1 MHz are in GHz
            simconfig['saved_values']['fdump'] = fdump * 1e9
    return simconfig