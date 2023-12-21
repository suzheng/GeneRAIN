import json

class JsonUtils:
    def __init__(self, tmp=None):
        self.tmp = tmp
    
    def save_data_to_file(self, data, output_file):
        with open(output_file, 'w') as fp:
            json.dump(data, fp, indent=4)

    def load_data_from_file(self, input_file):
        with open(input_file, 'r') as fp:
            data = json.load(fp)
        return data

def load_parameters_from_json(json_file, default_parameters):
    # Load parameters from JSON file
    with open(json_file, 'r') as f:
        input_parameters = json.load(f)

    # Create a dictionary to store the parameters
    parameters = {}

    # Iterate through the default parameters
    for key, default_value in default_parameters.items():
        # If the parameter exists in the JSON file, use its value
        if key in input_parameters:
            parameters[key] = input_parameters[key]
        # Otherwise, use the default value
        else:
            parameters[key] = default_value

    return parameters
    
